import logging
import os
from typing import Iterator, Any, cast

import librosa
import numpy as np
import soundfile as sf  # type: ignore

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_samples import recognise_from_audio_samples
from series_intro_recognizer.tp.interval import Interval

logger = logging.getLogger(__name__)

AudioFile = str | os.PathLike[str]


def _load(file: AudioFile,
          offset: float | None,
          duration: float | None,
          cfg: Config) -> np.ndarray[Any, np.dtype[np.float64]]:
    offset = offset or 0
    with sf.SoundFile(file) as sound_file:
        rate = sound_file.samplerate
        if offset:
            sound_file.seek(int(offset * rate))

        frames = int(duration * rate) if duration is not None else -1
        audio = sound_file.read(frames=frames, dtype='float32', always_2d=False).T

    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    if rate != cfg.rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=cfg.rate)

    logger.debug('Audio loaded to memory: %s (%.1fs)', file, audio.shape[0] / cfg.rate)

    return cast(np.ndarray[Any, np.dtype[np.float64]], audio)


def recognise_from_audio_files(files: Iterator[AudioFile], cfg: Config) -> list[Interval]:
    """
    Recognises series openings from audio files passed as file paths or file-like objects.
    :param files: list of file paths
    :param cfg: configuration
    :return: list of recognised intervals
    """
    audio_samples_iter = map(lambda file: _load(file, None, None, cfg), files)
    results = recognise_from_audio_samples(audio_samples_iter, cfg)
    return results


def recognise_from_audio_files_with_offsets(files: Iterator[tuple[AudioFile, float | None, float | None]],
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
