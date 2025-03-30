import multiprocessing
import time
import os
import shutil
import subprocess
from huggingface_hub import HfApi, login
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from tenacity import retry, before_sleep_log, after_log, wait_exponential, stop_after_attempt
from faravi.subtitles_cleanup.utils import SingletonLogger
from faravi.subtitles_cleanup.chunker import Caption, AudioChunker
import logging

login()

logger = SingletonLogger().get_logger()
target_repo_id = 'farsi-asr/PerSets-tarjoman-chunked'
source_repo_id = 'PerSets/tarjoman-persian-asr'

model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
)

pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)

def upload_results(tar_file):
    @retry(
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        wait=wait_exponential(2),
        stop=stop_after_attempt(10),
    )
    def upload(tar_file):
        HfApi().upload_file(
            path_or_fileobj=tar_file,
            path_in_repo=os.path.basename(tar_file),
            repo_id=target_repo_id,
            repo_type='dataset'
        )
        logger.info(f"Uploaded archive {tar_file}")

    upload(tar_file)

def get_wav_file(mp3_file):
    logger.info(f"Converting {mp3_file} to wav")

    wav_file = os.path.basename(mp3_file).replace('.MP3', '.wav')
    cmd = ['ffmpeg', '-hwaccel', 'cuda', '-y', '-i', mp3_file, wav_file]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed with return code {result.returncode}: {result.stderr}")
            return None

        return wav_file

    except subprocess.SubprocessError as e:
        logger.error(f"Subprocess error while running FFmpeg: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while running FFmpeg: {str(e)}")
        return None

def get_captions(vad) -> list[Caption]:
    captions = []
    for segment, _, label in vad.itertracks(yield_label=True):
        if label == 'SPEECH':
            start = segment.start
            end = segment.end
            caption = Caption(start, end, '')
            captions.append(caption)

    return captions

def process_audio(mp3_file, json_file):
    s = time.time()
    wav_file = get_wav_file(mp3_file)

    if wav_file is None:
        return

    audio_name = os.path.basename(wav_file).split('.')[0]
    os.makedirs(audio_name, exist_ok=True)

    logger.info(f"Running segment model {wav_file}")
    vad = pipeline(wav_file)
    logger.info(f"Done in {time.time() - s}. Splitting audio...")
    
    captions = get_captions(vad)
    if len(captions) == 0:
        shutil.rmtree(audio_name)
        os.remove(wav_file)
        logger.warning(f"No speech found in {wav_file}")
        return
    
    chunker = AudioChunker()
    chunker.chunk(True, wav_file, captions, audio_name)

    shutil.copy(json_file, audio_name)

    logger.info(f"Creating archive {audio_name}")
    tar_file = shutil.make_archive(audio_name, 'gztar', root_dir='.', base_dir=audio_name)
    upload_results(tar_file)

    shutil.rmtree(audio_name)
    os.remove(tar_file)
    os.remove(wav_file)

    print(f"Done in {time.time() - s}")

def download_and_process(filename):
    local_dir = 'tarjoman-persian-asr'
    try:
        HfApi().snapshot_download(
            repo_id=source_repo_id,
            repo_type='dataset',
            allow_patterns=f'train/{filename}*',
            local_dir=local_dir
        )

        mp3_file = os.path.join(local_dir, f'train/{filename}.MP3')
        json_file = os.path.join(local_dir, f'train/{filename}.json')

        if not os.path.exists(mp3_file) or not os.path.exists(json_file):
            raise FileNotFoundError(f'{filename} not found in {local_dir}')

        process_audio(mp3_file, json_file)
    except Exception as e:
        logger.error(f'Error processing {filename}: {e}')
    finally:
        shutil.rmtree(local_dir)

def get_filename(filepath):
    basename = os.path.basename(filepath)
    filename = basename.split('.')[0]
    return filename

if __name__ == '__main__':
    source_files = HfApi().list_repo_files(source_repo_id, repo_type='dataset')
    source_files = set([get_filename(f) for f in source_files if f.endswith('.MP3')])
    
    dest_files = HfApi().list_repo_files(target_repo_id, repo_type='dataset')
    dest_files = set([get_filename(f) for f in dest_files if f.endswith('.tar.gz')])

    new_files = list(source_files - dest_files)

    for i, file in enumerate(new_files):
        print(f'audio {i + 1:03d}/{len(new_files)}')
        download_and_process(file)
