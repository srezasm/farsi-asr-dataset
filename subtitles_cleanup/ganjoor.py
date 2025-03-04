import tarfile
import shutil
from huggingface_hub import HfApi
from os.path import join, basename, isdir, relpath
import json
from os import makedirs, listdir, remove
from tenacity import retry, stop_after_attempt, wait_exponential
from normalizer import ValidationStatus
import re

from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session, chunk_exists
from utils import SingletonLogger

logger = SingletonLogger().get_logger()

hf_api = HfApi()
chunker = AudioChunker()
repo_id = 'farsi-asr/ganjoor-dataset'
target_repo_id = 'farsi-asr/ganjoor-chunked-asr-dataset'
tmp_dir = 'tmp'

def get_captions(sub_path):
    try:
        with open(sub_path, 'r') as f:
            captions_dict = json.load(f)
    except Exception as e:
        logger.error(f"Error reading subtitles from {sub_path}: {e}")
        return []

    captions = []
    for caption in captions_dict:
        # check if all values are present
        if not all(k in caption.keys() for k in ['start', 'end', 'text']):
            logger.error(f"Invalid caption: {caption}")
            continue

        captions.append(
            Caption(
                start=caption['start'],
                end=caption['end'],
                text=caption['text']
            )
        )

    return captions

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_archive(archive_path):
    hf_api.upload_file(
        path_or_fileobj=archive_path,
        path_in_repo=basename(archive_path),
        repo_id=target_repo_id,
        repo_type='dataset'
    )
    logger.info(f"Uploaded archive {archive_path}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_db():
    hf_api.upload_file(
        path_or_fileobj='data.db',
        path_in_repo='data.db',
        repo_id=target_repo_id,
        repo_type='dataset'
    )
    logger.info("Uploaded database")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _download_tar_file(tar_file):
    return hf_api.hf_hub_download(
        repo_id, tar_file, repo_type='dataset', local_dir=tmp_dir
    )

list_repo_files = hf_api.list_repo_files(target_repo_id, repo_type='dataset')
def download_and_extract_tar_file(tar_file: str):
    if basename(tar_file) in list_repo_files:
        return None

    artist_id = basename(tar_file).replace('.tar.gz', '')

    try:
        tar_path = _download_tar_file(tar_file)
        logger.info(f"Downloaded {tar_file}")
    except Exception as e:
        logger.error(f"Error downloading {tar_file}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    try:
        tar_dir = join(tmp_dir, artist_id)
        makedirs(tar_dir, exist_ok=True)
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(tar_dir)
        logger.info(f"Extracted {tar_file} into {tmp_dir}")
    except Exception as e:
        logger.error(f"Error extracting {tar_file}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    return artist_id

def validate_captions(captions: list[Caption], filename: str):
    valid_captions = []
    invalid_captions = []

    for i in range(len(captions)):
        caption = captions[i]

        if caption.start >= caption.end:
            logger.warning(f"caption {caption.text} in {filename} has invalid start/end time.")
            invalid_captions.append(caption.copy_with(status=ValidationStatus.UNKNOWN))
        elif i != 0 and caption.start == 0:
            logger.warning(f"caption \"{caption.text}\" for {filename} starts at 0.")
            invalid_captions.append(caption.copy_with(status=ValidationStatus.UNKNOWN))
        elif i != len(captions) - 1 and caption.end == 0:
            logger.warning(f"caption \"{caption.text}\" for {filename} ends at 0.")
            invalid_captions.append(caption.copy_with(status=ValidationStatus.UNKNOWN))
        elif i == len(captions) - 1 and caption.end == 0:
            logger.warning(f"last caption for {filename} ends at 0.")
            try:
                duration = chunker._get_audio_duration(filename.replace('.json', '.mp3'))
                invalid_captions.append(caption.copy_with(end=duration))
                valid_captions.append(caption.copy())
            except Exception as e:
                logger.error(f"Error getting audio duration for replacing end time of {filename} last caption: {e}")
                invalid_captions.append(caption.copy_with(status=ValidationStatus.UNKNOWN))
        else:
            valid_captions.append(caption.copy())

    return valid_captions, invalid_captions

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

    # Get youtube tar files from repo
    tar_files = hf_api.list_repo_files(repo_id, repo_type='dataset')
    tar_files = [f for f in tar_files if re.match(r'^ganjoor/[a-zA-Z0-9-]+\.tar\.gz$', f)]
    logger.info(f"Found {len(tar_files)} tar.gz files in repository {repo_id}.")
    
    for tar_file in tar_files:
        makedirs(tmp_dir, exist_ok=True)

        artist_id = download_and_extract_tar_file(tar_file)
        if not artist_id:
            logger.info(f'Already processed {artist_id}. Skipping.')
            continue

        file_ids = listdir(join(tmp_dir, artist_id))
        file_ids = [f.split('.')[0] for f in file_ids if f.endswith('mp3')]
        for file_id in file_ids:
            logger.info(f"Processing video ID: {file_id}")

            audio_file = file_id + '.mp3'
            sub_file = file_id + '.json'

            with get_db_session() as session:
                if chunk_exists(session, file_id, 'ganjoor'):
                    logger.info(f"Already processed {audio_file}. Skipping.")
                    continue

            audio_path = join(tmp_dir, artist_id, audio_file)
            sub_path = join(tmp_dir, artist_id, sub_file)

            output_dir = join(artist_id, file_id)
            makedirs(output_dir, exist_ok=True)

            captions = get_captions(sub_path)
            if not captions:
                logger.warning(f"No captions extracted from {sub_path}.")
                continue

            captions, invalid_captions = validate_captions(captions, audio_file)
            if not captions:
                logger.warning(f"No valid captions extracted from {sub_path}.")
                continue

            processed_captions = chunker.chunk(
                merge=False,
                audio_file=audio_path,
                captions=captions,
                output_dir=output_dir
            )
            logger.info(
                f"Created {len(processed_captions)} audio chunks for {file_id}")

            with get_db_session() as session:
                create_chunks(session, 'ganjoor', file_id, processed_captions)
                create_chunks(session, 'ganjoor', file_id, invalid_captions)
            logger.info(
                f"Recorded processed chunks in the database for {file_id}")

        if not isdir(artist_id):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.warning(
                f"No chunks created for channel {artist_id}. Skipping.")
            continue

        # create archive and upload to target repo
        archive_path = shutil.make_archive(
            artist_id, 'gztar', root_dir='.', base_dir=artist_id)
        logger.info(f"Created archive {archive_path}")

        upload_archive(archive_path)

        upload_db()

        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(artist_id, ignore_errors=True)
        remove(archive_path)
