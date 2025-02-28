from chunker import AudioChunker, Caption
from db import create_chunks, init_db, get_db_session
import webvtt
import os
import glob
from typing import Generator
import logging
import time
import tqdm

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
        self.dir_path = '/home/srezas/Programming/projects/negar/samples/UCa-ZgzdNQYDOet7t_yrcPHw'
        self.files = list(self._get_files())

    def _get_files(self) -> Generator[tuple[str, str], None, None]:
        dir_list = os.listdir(self.dir_path)
        for sub_dir in dir_list:
            sub_path = glob.glob(os.path.join(self.dir_path, sub_dir, '*.vtt'))
            audio_path = glob.glob(os.path.join(
                self.dir_path, sub_dir, '*.opus'))

            if not sub_path or not audio_path:
                not_found = 'subtitles and audio' if not sub_path and not audio_path else (
                    'subtitles' if not sub_path else 'audio')
                logging.error(
                    f"Could not find {not_found} file for {sub_dir} directory")
                continue

            yield sub_path[0], audio_path[0]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)


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

        p_bar = tqdm.tqdm(total=len(video_iter))
        for sub_filepath, audio_filepath in video_iter:
            vid_id = os.path.basename(sub_filepath).split('.')[0]
            os.makedirs(vid_id, exist_ok=True)

            captions = chunker.chunk(
                merge=True,
                audio_file=audio_filepath,
                captions=get_captions(sub_filepath),
                output_dir=vid_id
            )

            create_chunks(session, 'youtube', vid_id, captions)

            p_bar.update(1)
