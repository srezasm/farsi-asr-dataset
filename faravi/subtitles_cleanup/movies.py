import shutil
from huggingface_hub import HfApi
import webvtt
from os.path import join, basename, isdir, dirname
from os.path import exists as file_exists
from os import makedirs, remove
from tenacity import retry, stop_after_attempt, wait_fixed

from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session, chunk_exists
from utils import SingletonLogger

logger = SingletonLogger().get_logger()

hf_api = HfApi()
chunker = AudioChunker()
repo_id = 'farsi-asr/filimo-asr-dataset'
target_repo_id = 'farsi-asr/filimo-chunked-asr-dataset'
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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1800))
def download_movie_dirs(movie_names):
    return hf_api.snapshot_download(repo_id, repo_type='dataset', local_dir='tmp', allow_patterns=movie_names)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1800))
def upload_results(archive_path):
    makedirs('upload', exist_ok=True)

    shutil.copy('data.db', join('upload', 'data.db'))
    shutil.move(archive_path, join('upload', basename(archive_path)))

    hf_api.upload_folder(
        repo_id=target_repo_id,
        folder_path='upload',
        repo_type='dataset'
    )
    logger.info(f"Uploaded archive and db files to target repo")

    shutil.rmtree('upload', ignore_errors=True)


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

    # get videos from repo
    movies = hf_api.list_repo_files(repo_id, repo_type='dataset')
    movie_patterns = list(set([dirname(m) + '/*' for m in movies if m.startswith('filimo') and 'db' not in m]))
    movies = [m.split('/')[1] for m in movie_patterns]
    # sort lists
    movies = sorted(movies)
    movie_patterns = sorted(movie_patterns)
    logger.info(f"Found {len(movies)} movies in repository {repo_id}.")

    batch_size = 100
    for i in range(0, len(movies), batch_size):
        makedirs(tmp_dir, exist_ok=True)

        current_movies = movies[i:i+batch_size]
        current_patterns = movie_patterns[i:i+batch_size]
        
        batch_number = str(int(i / batch_size) + 1)
        with get_db_session() as session:
            if chunk_exists(session, current_movies[0], 'filimo'):
                logger.info(f"Batch {batch_number} already processed. Skipping.")
                continue

        # download and extract channel tar files
        download_movie_dirs(current_patterns)

        # process channel videos
        for vid_id in current_movies:
            logger.info(f"Processing video ID: {vid_id}")

            current_dir = join(tmp_dir, 'filimo', vid_id)
            
            if not isdir(current_dir):
                logger.error(f"Missing directory for video {vid_id}")
                continue

            sub_path = join(current_dir, vid_id + '.srt')
            audio_path = join(current_dir, vid_id + '.mp3')

            if not file_exists(sub_path) or not file_exists(audio_path):
                logger.error(f"Missing subtitles or audio file for video {vid_id}")
                continue

            output_dir = join('filimo', vid_id)
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
                create_chunks(session, 'filimo', vid_id, processed_captions)
            logger.info(f"Recorded processed chunks in the database for video {vid_id}")

        if not isdir('filimo'):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.warning(f"No tar file created for {int(batch_number):02d} batch. Skipping.")
            continue
        
        # create archive and upload to target repo
        archive_path = shutil.make_archive(f'batch_{int(batch_number):02d}', 'gztar', root_dir='.', base_dir='filimo')
        logger.info(f"Created archive {archive_path}")

        upload_results(archive_path)

        # cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree('filimo', ignore_errors=True)
