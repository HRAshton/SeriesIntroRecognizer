from pathlib import Path

import pytest

from series_intro_recognizer.config import Config
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files
from tests.manual.harness import smoke_rows_to_markdown

LOCAL_AUDIO_DIR = Path(__file__).parent / "audio_files"


def test_recognise_from_audio_files() -> None:
    files = sorted(LOCAL_AUDIO_DIR.glob("*.wav"), key=lambda path: int(path.stem.split("_")[0]))
    if not files:
        pytest.skip(f"No wav files found under {LOCAL_AUDIO_DIR}")

    cfg = Config(save_intermediate_results=True)
    recognised = recognise_from_audio_files(iter(files), cfg)

    rows = [
        (file_path.name, interval.start, interval.end, interval.end - interval.start)
        for file_path, interval in zip(files, recognised)
    ]

    print(smoke_rows_to_markdown(rows))

    for _, start, end, duration in rows:
        assert start >= 0
        assert end >= 0
        assert duration > 0
        assert 90 - duration <= 1
