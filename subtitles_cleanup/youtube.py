from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session
import webvtt
import os
import glob
from typing import Generator
import logging
import time
import tqdm
import tarfile
import shutil

# Clear existing handlers and configure fresh
logging.root.handlers = []
file_handler = logging.FileHandler(
    f'{time.strftime("%Y%m%d")}.log', mode='a'
)
formatter = logging.Formatter(
    '%(asctime)s - %(leveln ame)s - %(message)s', '%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
logging.root.addHandler(file_handler)
logging.root.setLevel(logging.INFO)


class FilesIterator:
    def __init__(self):
        self.api = HfApi()
        self.repo_id = 'farsi-asr/farsi-asr-dataset'
        self.tar_files = self._get_tar_files()
    
    def _get_tar_files(self):
        tar_files = self.api.list_repo_files(self.repo_id, repo_type='dataset')
        tar_files = [f for f in tar_files if f.startswith('youtube') and f.endswith('.tar.gz')]
        return tar_files

    def _get_files(self):
        tmp_dir = '/content/tmp'

        for tar_file in self.tar_files:
            os.makedirs(tmp_dir, exist_ok=True)
            
            print(f'Working on {tar_file}...')

            # Download the tar file
            tar_path = self.api.hf_hub_download(
                self.repo_id, tar_file, repo_type='dataset', local_dir=tmp_dir
            )
            
            # Extract the tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(tmp_dir)
            
            # Yield files from the extracted directory
            chl_id = os.path.basename(tar_file).split('_')[0]
            for root, _, files in os.walk(os.path.join(tmp_dir, chl_id)):
                sub_path = [os.path.join(root, f) for f in files if f.endswith('.vtt')]
                audio_path = [os.path.join(root, f) for f in files if f.endswith('.opus')]
                
                if not sub_path or not audio_path:
                    not_found = 'subtitles and audio' if not sub_path and not audio_path else (
                        'subtitles' if not sub_path else 'audio'
                    )
                    logging.warning(f"Could not find {not_found} file in {root}. Skipping...")
                    continue
                
                yield sub_path[0], audio_path[0], os.path.basename(tar_file).split('.')[0]

            # Clean up the temporary directory for this tar file
            shutil.rmtree(tmp_dir)

            batch_files = glob.glob(f'/content/{os.path.basename(tar_file).split(".")[0]}/*')
            # Create new compressed archive with combined content
            with tarfile.open(os.path.basename(tar_path), 'w:gz') as tar:
                for fpath in batch_files:
                    arcname = f'{chl_id}/{os.path.basename(fpath)}'
                    tar.add(fpath, arcname=arcname)
                    shutil.rmtree(fpath, ignore_errors=True)
            
            self.api.upload_file(
                path_or_fileobj=os.path.basename(tar_path),
                path_in_repo=os.path.basename(tar_path),
                repo_id='farsi-asr/farsi-youtube-asr-dataset',
                repo_type='dataset'
            )

    def __iter__(self):
        return self._get_files()


def get_captions(sub_filepath):
    captions = []
    try:
        sub = webvtt.read(sub_filepath)
    except Exception as e:
        logging.error(f"Error reading subtitles from {sub_filepath}: {e}")
        return []

    def format_time(t): return t.hours * 3600 + t.minutes * \
        60 + t.seconds + t.milliseconds / 1000

    for caption in sub.iter_slice():
        start_time = format_time(caption.start_time)
        end_time = format_time(caption.end_time)
        captions.append(
            Caption(
                start=start_time,
                end=end_time,
                text=caption.text.strip()
            )
        )

    return captions


if __name__ == "__main__":
    init_db()

    chunker = AudioChunker()
    video_iter = FilesIterator()

    with get_db_session() as session:

        # p_bar = tqdm.tqdm()
        for sub_filepath, audio_filepath, chl_id in video_iter:
            vid_id = os.path.basename(sub_filepath).split('.')[0]
            print(vid_id)
            
            dir_path = os.path.join(chl_id, vid_id)
            os.makedirs(dir_path, exist_ok=True)

            captions = chunker.chunk(
                merge=True,
                audio_file=audio_filepath,
                captions=get_captions(sub_filepath),
                output_dir=dir_path
            )

            create_chunks(session, 'youtube', vid_id, captions)
            # p_bar.update(1)