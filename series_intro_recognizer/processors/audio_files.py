import logging
from typing import Iterator, Any

import librosa
import numpy as np

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_samples import recognise_from_audio_samples
from series_intro_recognizer.tp.interval import Interval

logger = logging.getLogger(__name__)


def _load(file: str,
          offset: float | None,
          duration: float | None,
          cfg: Config) -> np.ndarray[Any, np.dtype[np.float64]]:
    offset = offset or 0
    audio, rate = librosa.load(file, sr=cfg.rate, mono=True, offset=offset, duration=duration)
    if rate != cfg.rate:
        raise ValueError(f'Wrong rate: {rate} != {cfg.rate}')

    logger.debug('Audio loaded to memory: %s (%.1fs)', file, audio.shape[0] / cfg.rate)

    return audio


def recognise_from_audio_files(files: Iterator[str], cfg: Config) -> list[Interval]:
    """
    Recognises series openings from audio files passed as file paths or file-like objects.
    :param files: list of file paths
    :param cfg: configuration
    :return: list of recognised intervals
    """
    audio_samples_iter = map(lambda file: _load(file, None, None, cfg), files)
    results = recognise_from_audio_samples(audio_samples_iter, cfg)
    return results


def recognise_from_audio_files_with_offsets(files: Iterator[tuple[str, float | None, float | None]],
                                            cfg: Config) -> list[Interval]:
    """
    Recognises series openings from audio files passed as file paths or file-like objects.
    If the offset or duration are passed, the audio is analysed from the offset to the offset + duration.
    WARNING: The passed offset ARE NOT added to the recognised intervals. Please add them manually if needed.
    :param files: list of tuples with file path, offset (in seconds) and duration (in seconds)
    :param cfg: configuration
    :return: list of recognised intervals
    """
    audio_samples_iter = map(lambda file_data: _load(file_data[0], file_data[1], file_data[2], cfg), files)
    results = recognise_from_audio_samples(audio_samples_iter, cfg)
    return results
