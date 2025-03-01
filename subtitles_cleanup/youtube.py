import os
import tarfile
import shutil
from huggingface_hub import HfApi
import webvtt
from os.path import join, basename, isdir, relpath

from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session, chunk_exists
from utils import SingletonLogger

# Set up logging
logger = SingletonLogger().get_logger()

hf_api = HfApi()

class FilesIterator:
    def __init__(self):
        self.repo_id = 'farsi-asr/farsi-asr-dataset'
        self.target_repo_id = 'farsi-asr/farsi-youtube-asr-dataset'
        self.tar_files = self._get_tar_files()
    
    def _get_tar_files(self):
        try:
            tar_files = hf_api.list_repo_files(self.repo_id, repo_type='dataset')
            tar_files = [f for f in tar_files if f.startswith('youtube') and f.endswith('.tar.gz')]
            logger.info(f"Found {len(tar_files)} tar.gz files in repository {self.repo_id}.")
            return tar_files
        except Exception as e:
            logger.error(f"Error listing repository files: {e}")
            return []
    
    def _get_files(self):
        # Base temporary directory for extraction
        base_tmp_dir = '/content/tmp'
        os.makedirs(base_tmp_dir, exist_ok=True)

        for tar_file in self.tar_files:
            logger.info(f"Processing tar file: {tar_file}")

            # Create a unique temporary directory for this tar file
            tmp_dir = join(base_tmp_dir, basename(tar_file).replace('.tar.gz', ''))
            os.makedirs(tmp_dir, exist_ok=True)
            
            try:
                # Download the tar file
                tar_path = hf_api.hf_hub_download(
                    self.repo_id, tar_file, repo_type='dataset', local_dir=tmp_dir
                )
                logger.info(f"Downloaded {tar_file} to {tar_path}")
            except Exception as e:
                logger.error(f"Error downloading {tar_file}: {e}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            try:
                # Extract the tar file into tmp_dir
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(tmp_dir)
                logger.info(f"Extracted {tar_file} into {tmp_dir}")
            except Exception as e:
                logger.error(f"Error extracting {tar_file}: {e}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            # Determine the expected directory (using the first part of tar_file's name)
            tar_chunk_name = basename(tar_file).split('.')[0]
            chl_id = '_'.join(basename(tar_file).split('_')[:-1])
            extracted_dir = join(tmp_dir, chl_id)
            if not isdir(extracted_dir):
                logger.warning(f"Expected directory {extracted_dir} not found. Skipping {tar_file}.")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            found_files = False
            # Walk through the extracted directory to find .vtt and .opus files
            for root, _, files in os.walk(extracted_dir):
                sub_files = [join(root, f) for f in files if f.endswith('.vtt')]
                audio_files = [join(root, f) for f in files if f.endswith('.opus')]
                
                if not sub_files or not audio_files:
                    missing = []
                    if not sub_files:
                        missing.append("subtitles (.vtt)")
                    if not audio_files:
                        missing.append("audio (.opus)")
                    logger.warning(f"Missing {', '.join(missing)} in {root}. Skipping this directory.")
                    continue
                
                found_files = True
                yield sub_files[0], audio_files[0], tar_chunk_name

            if not found_files:
                logger.warning(f"No valid file pairs found in {extracted_dir} for {tar_file}.")

            # Archive processed output directory
            # Here we assume that processing (chunking) writes output under a directory named after 'chl_id'
            output_dir = join(os.getcwd(), tar_chunk_name)
            if isdir(output_dir):
                try:
                    archive_path = join(os.getcwd(), f"{tar_chunk_name}.tar.gz")
                    with tarfile.open(archive_path, 'w:gz') as tar_archive:
                        for root, _, files in os.walk(output_dir):
                            for f in files:
                                file_path = join(root, f)
                                arcname = relpath(file_path, os.getcwd())
                                tar_archive.add(file_path, arcname=arcname)
                    logger.info(f"Created archive {archive_path} from {output_dir}")

                    hf_api.upload_file(
                        path_or_fileobj=archive_path,
                        path_in_repo=basename(archive_path),
                        repo_id=self.target_repo_id,
                        repo_type='dataset'
                    )
                    logger.info(f"Uploaded archive {archive_path} to repository {self.target_repo_id}")

                    # Remove the output directory after archiving
                    shutil.rmtree(output_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(f"Error archiving/uploading processed files from {output_dir}: {e}")
            else:
                logger.info(f"No output directory {output_dir} found to archive for {tar_file}.")

            # Clean up the temporary extraction directory
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory {tmp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {tmp_dir}: {e}")

    def __iter__(self):
        return self._get_files()


def get_captions(sub_filepath):
    captions = []
    try:
        vtt_captions = webvtt.read(sub_filepath)
    except Exception as e:
        logger.error(f"Error reading subtitles from {sub_filepath}: {e}")
        return []

    def time_to_seconds(time_str):
        try:
            h, m, s = time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception as e:
            logger.error(f"Error converting time string '{time_str}': {e}")
            return 0

    for caption in vtt_captions:
        try:
            start_time = time_to_seconds(caption.start)
            end_time = time_to_seconds(caption.end)
            captions.append(
                Caption(
                    start=start_time,
                    end=end_time,
                    text=caption.text.strip()
                )
            )
        except Exception as e:
            logger.error(f"Error processing caption in {sub_filepath}: {e}")
            continue

    return captions


if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()

    chunker = AudioChunker()
    video_iter = FilesIterator()

    with get_db_session() as session:
        for sub_filepath, audio_filepath, chl_id in video_iter:
            try:
                vid_id = basename(sub_filepath).split('.')[0]
                logger.info(f"Processing video ID: {vid_id}")

                if chunk_exists(session, vid_id, 'youtube'):
                    logger.info(f"Video {vid_id} already processed. Skipping.")
                    continue

                # Create an output directory for the chunked audio
                output_dir = join(chl_id, vid_id)
                os.makedirs(output_dir, exist_ok=True)

                captions = get_captions(sub_filepath)
                if not captions:
                    logger.warning(f"No captions extracted from {sub_filepath}. Skipping video {vid_id}.")
                    continue

                # Process audio chunking
                processed_captions = chunker.chunk(
                    merge=True,
                    audio_file=audio_filepath,
                    captions=captions,
                    output_dir=output_dir
                )
                logger.info(f"Created {len(processed_captions)} audio chunks for video {vid_id}")

                # Record processed chunks in the database
                create_chunks(session, 'youtube', vid_id, processed_captions)

                hf_api.upload_file(
                    path_or_fileobj=archive_path,
                    path_in_repo=basename(archive_path),
                    repo_id=self.target_repo_id,
                    repo_type='dataset'
                )
            except Exception as e:
                logger.error(f"Error processing video {sub_filepath}: {e}")
                continue

    logger.info("Processing complete.")
