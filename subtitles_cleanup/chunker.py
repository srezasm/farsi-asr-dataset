import os
from dataclasses import dataclass
from typing import Optional, List
from normalizer import ValidationStatus, TextNormalizer
import subprocess
from utils import SingletonLogger

logger = SingletonLogger().get_logger()

@dataclass
class Caption:
    start: float
    end: float
    text: str
    status: Optional[ValidationStatus] = None
    filename: Optional[str] = None

    def copy(self) -> 'Caption':
        return Caption(**self.__dict__)


class AudioChunker:
    def __init__(self):
        self.normalizer = TextNormalizer()

    def _filter_captions(self, captions: List[Caption]) -> tuple[List[Caption], List[Caption]]:
        filtered_captions = []

        for cap in captions:
            result = self.normalizer.normalize(cap.text)

            cap.status = result.status
            cap.text = result.text

            filtered_captions.append(cap)

        return filtered_captions

    def _merge(self, captions: List[Caption]) -> List[Caption]:
        """
        Merges consecutive captions if they are within a certain range.
        """
        MAX_CHUNK_DURATION = 30.0
        CAPTION_OVERLAP_THRESHOLD = 0.25

        if not captions:
            return []

        # Create a copy of the first caption
        merged_captions = []
        current_caption = captions[0].copy()

        for caption in captions[1:]:
            # Merge if within range
            if current_caption.end + CAPTION_OVERLAP_THRESHOLD >= caption.start and\
                    current_caption.status == caption.status:
                if caption.end - current_caption.start >= MAX_CHUNK_DURATION:
                    merged_captions.append(current_caption)
                    current_caption = caption.copy()
                else:
                    current_caption.end = caption.end
                    current_caption.text += ' ' + caption.text
            else:
                merged_captions.append(current_caption)
                current_caption = caption.copy()
        merged_captions.append(current_caption)
        return merged_captions

    def _adjust_start_end(self, captions: List[Caption], duration: float) -> List[Caption]:
        """
        Adjusts the start and end times of captions to ensure they do not
        overlap and fit within the given audio length.
        """
        RANGE = 0.25  # seconds
        adjusted = []
        for caption in captions:
            adjusted.append(Caption(caption.start - RANGE,
                            caption.end + RANGE, caption.text, caption.status))

        # Resolve overlaps between consecutive captions
        for i in range(len(adjusted) - 1):
            curr_end = adjusted[i].end
            next_start = adjusted[i + 1].start
            if curr_end > next_start:
                avg = round((curr_end + next_start) / 2.0, 3)
                adjusted[i].end = avg
                adjusted[i + 1].start = avg

        # Ensure the first caption starts at 0 or later
        if adjusted[0].start < 0:
            adjusted[0].start = 0

        # Ensure the last caption does not extend beyond audio length
        if adjusted[-1].end > duration:
            adjusted[-1].end = duration

        return adjusted

    def _slice_audio(self, audio_file: str, start: float, end: float, output_file: str):
        """
        Slices the audio file from start to end and writes the output to the
        given file.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", audio_file,
            "-t", str(end - start),
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            output_file
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _get_audio_duration(self, audio_file: str) -> float:
        """
        Returns the duration of the audio file in seconds.
        """
        cmd = [
            'ffprobe',
            '-i', audio_file,
            '-show_entries',
            'format=duration',
            '-v',
            'quiet',
            '-of',
            'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def chunk(self, merge: bool, audio_file: str, captions: List[Caption], output_dir: str) -> tuple[List[Caption], List[Caption]]:
        """
        Slices the audio file according to the given captions and writes the
        audio chunks to the output directory. If merge is True, captions will be merged.
        """
        try:
            # Ensure captions are sorted by start time
            captions.sort(key=lambda c: c.start)

            captions = self._filter_captions(captions)

            if merge:
                captions = self._merge(captions)

            # Convert audio length to seconds for consistency
            captions = self._adjust_start_end(
                captions,
                self._get_audio_duration(audio_file)
            )

            for i, cap in enumerate(captions):
                filename = f'{os.path.basename(audio_file).split(".")[0]}_{i+1:04d}.opus'

                self._slice_audio(
                    audio_file,
                    cap.start,
                    cap.end,
                    os.path.join(output_dir, filename)
                )

                cap.filename = filename

            return captions
        
        except Exception as e:
            logger.error(f'Failed to chunk audio "{audio_file}", Error message: {e}')
            return []
