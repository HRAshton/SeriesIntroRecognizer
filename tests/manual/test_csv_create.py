import pytest

from series_intro_recognizer.config import Config
from series_intro_recognizer.helpers.telemetry import telemetry
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files
from tests.manual.harness import (
    ManualOptions,
    audio_files_for_series,
    build_created_rows,
    created_rows_to_markdown,
    discover_series,
    select_series,
    write_created_rows,
)


def test_csv_create(manual_options: ManualOptions) -> None:
    if manual_options.telemetry:
        telemetry.enable(lambda name, secs: print(f"{name}: {secs:.3f}s"))

    series_ids = select_series(discover_series(manual_options.audio_root), manual_options)
    if not series_ids:
        pytest.skip(f"No series folders found under {manual_options.audio_root}")

    cfg = Config()
    for series_id in series_ids:
        audio_files = audio_files_for_series(manual_options.audio_root, series_id)
        recognised = recognise_from_audio_files(iter(audio_file.path for audio_file in audio_files), cfg)
        rows = build_created_rows(audio_files, recognised)

        write_created_rows(manual_options.output_csv, rows)
        print(created_rows_to_markdown(series_id, rows))
