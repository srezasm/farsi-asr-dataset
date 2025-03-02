import tarfile
import shutil
from huggingface_hub import HfApi
import webvtt
from os.path import join, basename, isdir, relpath
from os import makedirs, listdir, remove
from itertools import groupby

from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session, chunk_exists
from utils import SingletonLogger

logger = SingletonLogger().get_logger()

hf_api = HfApi()
repo_id = 'farsi-asr/farsi-asr-dataset'
target_repo_id = 'farsi-asr/farsi-youtube-asr-dataset'
tmp_dir = 'tmp'

def get_captions(sub_path):
    def format_time(t):
        return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000
    
    try:
        sub = webvtt.read(sub_path)
    except Exception as e:
        logger.error(f"Error reading subtitles from {sub_path}: {e}")
        return []

    captions = []
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

def get_channel_id(tar_file):
    return basename(tar_file)[:24]

def download_tar_file(tar_files):
    for tar_file in tar_files:
        try:
            tar_path = hf_api.hf_hub_download(
                repo_id, tar_file, repo_type='dataset', local_dir=tmp_dir
            )
            logger.info(f"Downloaded {tar_file}")
        except Exception as e:
            logger.error(f"Error downloading {tar_file}: {e}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(tmp_dir)
            logger.info(f"Extracted {tar_file} into {tmp_dir}")
        except Exception as e:
            logger.error(f"Error extracting {tar_file}: {e}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

    return get_channel_id(tar_files[0])

if __name__ == '__main__':
    # check if db exists in target repo
    if 'data.db' not in hf_api.list_repo_files(target_repo_id, repo_type='dataset'):
        logger.info("Initializing database...")
        init_db()
    else:
        hf_api.hf_hub_download(
            target_repo_id, 'data.db', repo_type='dataset', local_dir='.'
        )
        logger.info("Downloaded database")

    chunker = AudioChunker()

    # Get youtube tar files from repo
    tar_files = hf_api.list_repo_files(repo_id, repo_type='dataset')
    tar_files = [f for f in tar_files if f.startswith('youtube') and f.endswith('.tar.gz')]
    logger.info(f"Found {len(tar_files)} tar.gz files in repository {repo_id}.")
    tar_files = [list(group) for _, group in groupby(sorted(tar_files), key=get_channel_id)]

    for channel_tar_files in tar_files:
        makedirs(tmp_dir, exist_ok=True)

        # download and extract channel tar files
        channel_id = download_tar_file(channel_tar_files)

        # process channel videos
        for vid_id in listdir(join(tmp_dir, channel_id)):
            vid_files = listdir(join(tmp_dir, channel_id, vid_id))
            
            sub_file = [f for f in vid_files if f.endswith('.vtt')]
            audio_file = [f for f in vid_files if f.endswith('.opus')]

            if not sub_file or not audio_file:
                logger.error(f"Missing subtitles or audio file for video {vid_id}")
                continue

            sub_path = join(tmp_dir, channel_id, vid_id, sub_file[0])
            audio_path = join(tmp_dir, channel_id, vid_id, audio_file[0])

            vid_id = basename(sub_path).split('.')[0]
            logger.info(f"Processing video ID: {vid_id}")

            with get_db_session() as session:
                if chunk_exists(session, vid_id, 'youtube'):
                    logger.info(f"Video {vid_id} already processed. Skipping.")
                    continue

            output_dir = join(channel_id, vid_id)
            makedirs(output_dir, exist_ok=True)

            captions = get_captions(sub_path)
            if not captions:
                logger.warning(f"No captions extracted from {sub_path}.")
                continue

            processed_captions = chunker.chunk(
                merge=True,
                audio_file=audio_path,
                captions=captions,
                output_dir=output_dir
            )
            logger.info(f"Created {len(processed_captions)} audio chunks for video {vid_id}")

            with get_db_session() as session:
                create_chunks(session, 'youtube', vid_id, processed_captions)
            logger.info(f"Recorded processed chunks in the database for video {vid_id}")

        if not isdir(channel_id):
            logger.info(f"No chunks created for channel {channel_id}. Skipping.")
            continue
        
        # create archive and upload to target repo
        archive_path = shutil.make_archive(channel_id, 'gztar', root_dir='.', base_dir=channel_id)
        logger.info(f"Created archive {archive_path}")

        hf_api.upload_file(
            path_or_fileobj=archive_path,
            path_in_repo=basename(archive_path),
            repo_id=target_repo_id,
            repo_type='dataset'
        )
        logger.info(f"Uploaded archive {archive_path}")

        hf_api.upload_file(
            path_or_fileobj='data.db',
            path_in_repo='data.db',
            repo_id=target_repo_id,
            repo_type='dataset'
        )
        logger.info("Uploaded database")

        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(channel_id, ignore_errors=True)
        remove(archive_path)