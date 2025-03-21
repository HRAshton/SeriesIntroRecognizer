import pytest

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files_with_offsets


@pytest.mark.skip(reason="test is not implemented")
def test_recognise_from_audio_files() -> None:
    """
    Copy 8 6-minute audio files to assets/audio_files/ directory and run the test.
    :return:
    """
    cfg = Config()
    files = [(f'../assets/audio_files/{i}.wav', None, None) for i in range(1, 8)]
    recognised = recognise_from_audio_files_with_offsets(iter(files), cfg)

    for interval in recognised:
        print(interval, interval.end - interval.start)
        assert interval.start >= 0
        assert interval.end >= 0
        assert interval.end - interval.start > 0
        assert 90 - (interval.end - interval.start) <= 1
