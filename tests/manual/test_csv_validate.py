import pytest

from series_intro_recognizer.config import Config
from series_intro_recognizer.helpers.telemetry import telemetry
from series_intro_recognizer.processors.audio_files import recognise_from_audio_files
from tests.manual.harness import (
    ManualOptions,
    audio_files_for_series,
    build_validation_report,
    expected_series_ids,
    read_expected_csv,
    select_series,
)


def test_csv_validate(manual_options: ManualOptions) -> None:
    if not manual_options.expected_csv.exists():
        pytest.fail(f"CSV not found: {manual_options.expected_csv}")

    if manual_options.telemetry:
        telemetry.enable(lambda name, secs: print(f"{name}: {secs:.3f}s"))

    expected_rows = read_expected_csv(manual_options.expected_csv)
    series_ids = select_series(expected_series_ids(manual_options.expected_csv), manual_options)
    if not series_ids:
        pytest.skip(f"No series found in {manual_options.expected_csv}")

    cfg = Config.preset_anime_ending() if manual_options.validates_endings else Config()
    mismatched_series: list[int] = []

    for series_id in series_ids:
        audio_files = audio_files_for_series(manual_options.audio_root, series_id)
        recognised = recognise_from_audio_files(iter(audio_file.path for audio_file in audio_files), cfg)
        report = build_validation_report(series_id, audio_files, expected_rows, recognised, manual_options)

        print(report.to_markdown())
        if report.mismatches:
            mismatched_series.append(series_id)

    assert mismatched_series == []
