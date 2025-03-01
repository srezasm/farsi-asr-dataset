from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session
import webvtt
import os
import tarfile
import shutil
from huggingface_hub import HfApi
from utils import SingletonLogger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = SingletonLogger().get_logger()

class FilesIterator:
    def __init__(self):
        self.api = HfApi()
        self.repo_id = 'farsi-asr/farsi-asr-dataset'
        self.processed_repo_id = 'farsi-asr/farsi-youtube-asr-dataset'
        self.tmp_dir = 'tmp'
        self.tar_files = self._get_tar_files()
    
    def _get_tar_files(self):
        try:
            tar_files = self.api.list_repo_files(self.repo_id, repo_type='dataset')
            return [f for f in tar_files if f.startswith('youtube') and f.endswith('.tar.gz')]
        except Exception as e:
            logger.error(f"Error listing repo files: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _download_tar(self, tar_file):
        return self.api.hf_hub_download(
            self.repo_id, tar_file, repo_type='dataset', local_dir=self.tmp_dir
        )

    def _process_tar(self, tar_file):
        chl_id = os.path.basename(tar_file).split('.')[0]
        extracted_dir = os.path.join(self.tmp_dir, chl_id)
        
        try:
            tar_path = self._download_tar(tar_file)
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.tmp_dir)
            
            for root, _, files in os.walk(extracted_dir):
                sub_files = [f for f in files if f.endswith('.vtt')]
                audio_files = [f for f in files if f.endswith('.opus')]
                
                if not sub_files or not audio_files:
                    missing = []
                    if not sub_files: missing.append('subtitles')
                    if not audio_files: missing.append('audio')
                    logger.warning(f"Missing {', '.join(missing)} in {root}")
                    continue
                
                yield (
                    os.path.join(root, sub_files[0]),
                    os.path.join(root, audio_files[0]),
                    chl_id
                )
            
            yield None  # Signal end of processing for this tar
            
            self._create_and_upload_tar(chl_id, tar_file)
            
        except Exception as e:
            logger.error(f"Error processing {tar_file}: {e}")
        finally:
            shutil.rmtree(extracted_dir, ignore_errors=True)
            shutil.rmtree(chl_id, ignore_errors=True)

    def _create_and_upload_tar(self, chl_id, original_tar_name):
        new_tar_name = original_tar_name
        new_tar_path = os.path.join(os.getcwd(), new_tar_name)
        
        if not os.path.exists(chl_id) or not os.listdir(chl_id):
            logger.warning(f"No processed files found for {chl_id}")
            return

        try:
            with tarfile.open(new_tar_path, 'w:gz') as tar:
                tar.add(chl_id, arcname=chl_id)
            logger.info(f"Created processed tar: {new_tar_path}")

            self.api.upload_file(
                path_or_fileobj=new_tar_path,
                path_in_repo=new_tar_name,
                repo_id=self.processed_repo_id,
                repo_type='dataset'
            )
            logger.info(f"Uploaded {new_tar_name} successfully")
        except Exception as e:
            logger.error(f"Failed to process {new_tar_name}: {e}")
        finally:
            if os.path.exists(new_tar_path):
                os.remove(new_tar_path)

    def __iter__(self):
        for tar_file in self.tar_files:
            logger.info(f"Processing archive: {tar_file}")
            yield from self._process_tar(tar_file)

def get_captions(sub_filepath):
    try:
        sub = webvtt.read(sub_filepath)
    except Exception as e:
        logger.error(f"Error reading {sub_filepath}: {e}")
        return []

    def format_time(t): return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000

    return [
        Caption(
            start=format_time(caption.start_time),
            end=format_time(caption.end_time),
            text=caption.text.strip()
        )
        for caption in sub.iter_slice()
    ]

def process_video(chunker, sub_filepath, audio_filepath, chl_id, session):
    vid_id = os.path.basename(sub_filepath).split('.')[0]
    dir_path = os.path.join(chl_id, vid_id)
    
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Processing video {vid_id}")

        captions = get_captions(sub_filepath)
        if not captions:
            logger.warning(f"No captions found for {vid_id}")
            return

        processed = chunker.chunk(
            merge=True,
            audio_file=audio_filepath,
            captions=captions,
            output_dir=dir_path
        )

        create_chunks(session, 'youtube', vid_id, processed)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to process {vid_id}: {e}")
        session.rollback()
        shutil.rmtree(dir_path, ignore_errors=True)
    finally:
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            shutil.rmtree(dir_path)

if __name__ == "__main__":
    init_db()
    chunker = AudioChunker()
    video_iter = FilesIterator()

    with get_db_session() as session:
        for item in video_iter:
            if item is None:  # Signals end of a tar processing
                continue
            sub_filepath, audio_filepath, chl_id = item
            process_video(chunker, sub_filepath, audio_filepath, chl_id, session)