import logging
from typing import List, Dict

import librosa
import numpy as np

from series_opening_recognizer.config import Config
from series_opening_recognizer.processors.audio_samples import recognise_from_audio_samples
from series_opening_recognizer.tp.interval import Interval

logger = logging.getLogger(__name__)


def _load(file: str, cfg: Config) -> np.ndarray:
    audio, rate = librosa.load(file, sr=cfg.RATE, mono=True)
    if rate != cfg.RATE:
        raise ValueError(f'Wrong rate: {rate} != {cfg.RATE}')

    logger.debug(f'Audio loaded to memory: {file} ({audio.shape[0] / cfg.RATE:.1f}s)')

    return audio


def recognise_from_audio_files(file_paths: List[str], cfg: Config) -> Dict[str, Interval]:
    audio_samples_iter = map(lambda path: _load(path, cfg), file_paths)
    results = recognise_from_audio_samples(audio_samples_iter, cfg)
    return {file_paths[idx]: interval for idx, interval in results.items()}
