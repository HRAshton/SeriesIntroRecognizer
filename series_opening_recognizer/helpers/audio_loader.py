import logging
from typing import Tuple

import librosa
import numpy as np

from series_opening_recognizer.configuration import RATE

logger = logging.getLogger(__name__)


def load_audio(file: str) -> Tuple[str, np.ndarray]:
    audio, rate = librosa.load(file, sr=RATE, mono=True)
    if rate != RATE:
        raise ValueError(f'Wrong rate: {rate} != {RATE}')

    logger.debug(f'Audio loaded to memory: {file} ({audio.shape[0] / RATE:.1f}s)')

    return file, audio
