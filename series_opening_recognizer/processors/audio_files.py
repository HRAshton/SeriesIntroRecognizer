import logging
from typing import List, Dict

import librosa
import numpy as np

from series_opening_recognizer.configuration import RATE
from series_opening_recognizer.processors.audio_samples import recognise_from_audio_samples
from series_opening_recognizer.tp.interval import Interval

logger = logging.getLogger(__name__)


def _load(file: str) -> np.ndarray:
    audio, rate = librosa.load(file, sr=RATE, mono=True)
    if rate != RATE:
        raise ValueError(f'Wrong rate: {rate} != {RATE}')

    logger.debug(f'Audio loaded to memory: {file} ({audio.shape[0] / RATE:.1f}s)')

    return audio


def recognise_from_audio_files(file_paths: List[str]) -> Dict[str, Interval]:
    audio_samples_iter = map(lambda path: _load(path), file_paths)
    results = recognise_from_audio_samples(audio_samples_iter)
    return {file_paths[idx]: interval for idx, interval in results.items()}
