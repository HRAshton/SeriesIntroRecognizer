from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]
from tabulate import tabulate

from series_intro_recognizer.tp.interval import Interval

CSV_COLUMNS = ["series", "episode", "total_ep_length", "wav_length", "start", "end"]
COMPARISON_COLUMNS = [
    "episode",
    "audio",
    "exp_start",
    "exp_end",
    "exp_dur",
    "act_start",
    "act_end",
    "act_dur",
    "start_delta",
    "end_delta",
    "dur_delta",
]


@dataclass(frozen=True)
class ManualOptions:
    audio_root: Path
    expected_csv: Path
    output_csv: Path
    kind: str
    series_ids: tuple[int, ...]
    series_skip: int
    tolerance_secs: float
    telemetry: bool

    @property
    def validates_endings(self) -> bool:
        if self.kind != "auto":
            return self.kind == "ending"
        return "ending" in self.expected_csv.name or "endings" in self.expected_csv.name


@dataclass(frozen=True)
class EpisodeAudio:
    series_id: int
    episode: int
    total_length: float
    path: Path

    @property
    def wav_length(self) -> float:
        with sf.SoundFile(self.path) as sound_file:
            frames = cast(int, sound_file.frames)
            samplerate = cast(int, sound_file.samplerate)
            return frames / samplerate


@dataclass(frozen=True)
class ExpectedEpisode:
    series_id: int
    episode: int
    total_length: float
    wav_length: float
    start: float
    end: float

    @classmethod
    def from_csv_row(cls, row: list[str]) -> ExpectedEpisode:
        return cls(
            series_id=int(row[0]),
            episode=int(row[1]),
            total_length=float(row[2]),
            wav_length=float(row[3]),
            start=float(row[4]),
            end=float(row[5]),
        )


@dataclass(frozen=True)
class CreatedIntervalRow:
    series_id: int
    episode: int
    total_length: float
    wav_length: float
    start: float
    end: float

    def as_csv_row(self) -> tuple[int, int, float, float, float, float]:
        return (self.series_id, self.episode, self.total_length, self.wav_length, self.start, self.end)


@dataclass(frozen=True)
class ComparisonRow:
    episode: int
    audio_name: str
    expected: Interval
    actual: Interval

    @property
    def expected_duration(self) -> float:
        return self.expected.end - self.expected.start

    @property
    def actual_duration(self) -> float:
        return self.actual.end - self.actual.start

    @property
    def start_delta(self) -> float:
        return self.actual.start - self.expected.start

    @property
    def end_delta(self) -> float:
        return self.actual.end - self.expected.end

    @property
    def duration_delta(self) -> float:
        return self.actual_duration - self.expected_duration

    def matches(self, tolerance_secs: float) -> bool:
        return (
            _matches_value(self.actual.start, self.expected.start, tolerance_secs)
            and _matches_value(self.actual.end, self.expected.end, tolerance_secs)
        )

    def as_display_row(self) -> tuple[int, str, str, str, str, str, str, str, str, str, str]:
        return (
            self.episode,
            self.audio_name,
            _fmt_time(self.expected.start),
            _fmt_time(self.expected.end),
            f"{self.expected_duration:.1f}",
            _fmt_time(self.actual.start),
            _fmt_time(self.actual.end),
            f"{self.actual_duration:.1f}",
            _fmt_delta(self.start_delta),
            _fmt_delta(self.end_delta),
            _fmt_delta(self.duration_delta),
        )


@dataclass(frozen=True)
class ValidationReport:
    series_id: int
    rows: tuple[ComparisonRow, ...]
    mismatches: tuple[ComparisonRow, ...]
    tolerance_secs: float

    def to_markdown(self) -> str:
        display_rows = [row.as_display_row() for row in self.rows]
        sections = [
            "",
            f"Series {self.series_id}: compared {len(self.rows)} episodes.",
            tabulate(display_rows, headers=COMPARISON_COLUMNS, tablefmt="github"),
        ]

        if self.mismatches:
            mismatch_rows = [row.as_display_row() for row in self.mismatches]
            sections.extend(
                [
                    "",
                    f"Mismatches beyond {self.tolerance_secs:.1f}s:",
                    tabulate(mismatch_rows, headers=COMPARISON_COLUMNS, tablefmt="github"),
                ]
            )

        return "\n".join(sections)


def discover_series(audio_root: Path) -> list[int]:
    if not audio_root.exists():
        return []
    return sorted(int(path.name) for path in audio_root.iterdir() if path.is_dir() and path.name.isdigit())


def select_series(all_series: list[int], options: ManualOptions) -> list[int]:
    selected = list(options.series_ids) if options.series_ids else all_series
    return selected[options.series_skip:]


def read_expected_csv(expected_csv: Path) -> list[ExpectedEpisode]:
    if not expected_csv.exists():
        return []

    with open(expected_csv, newline="", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        return [ExpectedEpisode.from_csv_row(row) for row in csv_reader]


def expected_series_ids(expected_csv: Path) -> list[int]:
    return sorted({row.series_id for row in read_expected_csv(expected_csv)})


def audio_files_for_series(audio_root: Path, series_id: int) -> list[EpisodeAudio]:
    audio_dir = audio_root / str(series_id)
    return [
        _episode_audio(series_id, path)
        for path in sorted(audio_dir.iterdir(), key=lambda item: int(item.stem.split("_")[0]))
    ]


def write_created_rows(output_csv: Path, rows: list[CreatedIntervalRow]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if output_csv.stat().st_size == 0:
            writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(row.as_csv_row())
            csvfile.flush()


def created_rows_to_markdown(series_id: int, rows: list[CreatedIntervalRow]) -> str:
    return "\n".join(
        [
            "",
            f"Series {series_id}: processed {len(rows)} episodes.",
            tabulate([row.as_csv_row() for row in rows], headers=CSV_COLUMNS, floatfmt=".3f", tablefmt="github"),
        ]
    )


def build_created_rows(audio_files: list[EpisodeAudio], recognised: list[Interval]) -> list[CreatedIntervalRow]:
    return [
        CreatedIntervalRow(
            series_id=audio_file.series_id,
            episode=audio_file.episode,
            total_length=audio_file.total_length,
            wav_length=audio_file.wav_length,
            start=interval.start,
            end=interval.end,
        )
        for audio_file, interval in zip(audio_files, recognised)
    ]


def build_validation_report(
    series_id: int,
    audio_files: list[EpisodeAudio],
    expected_rows: list[ExpectedEpisode],
    recognised: list[Interval],
    options: ManualOptions,
) -> ValidationReport:
    expected_by_episode = {
        row.episode: row
        for row in expected_rows
        if row.series_id == series_id
    }

    rows = tuple(
        _comparison_row(audio_file, expected_by_episode.get(audio_file.episode), interval, options.validates_endings)
        for audio_file, interval in zip(audio_files, recognised)
    )
    mismatches = tuple(row for row in rows if not row.matches(options.tolerance_secs))
    return ValidationReport(series_id, rows, mismatches, options.tolerance_secs)


def smoke_rows_to_markdown(rows: list[tuple[str, float, float, float]]) -> str:
    return "\n" + tabulate(rows, headers=["audio", "start", "end", "duration"], floatfmt=".3f", tablefmt="github")


def _episode_audio(series_id: int, path: Path) -> EpisodeAudio:
    stem_parts = path.stem.split("_")
    episode = int(stem_parts[0])
    total_length = float(stem_parts[1]) if len(stem_parts) > 1 else math.nan
    return EpisodeAudio(series_id, episode, total_length, path)


def _comparison_row(
    audio_file: EpisodeAudio,
    expected: ExpectedEpisode | None,
    actual: Interval,
    validates_endings: bool,
) -> ComparisonRow:
    if expected is None:
        expected_interval = Interval(math.nan, math.nan)
        actual_interval = actual
    elif validates_endings and not (math.isnan(expected.total_length) or math.isnan(expected.wav_length)):
        expected_interval = _to_episode_interval(expected.total_length, expected.wav_length, expected.start,
                                                 expected.end)
        actual_interval = _to_episode_interval(expected.total_length, expected.wav_length, actual.start, actual.end)
    else:
        expected_interval = Interval(expected.start, expected.end)
        actual_interval = actual

    return ComparisonRow(audio_file.episode, audio_file.path.name, expected_interval, actual_interval)


def _to_episode_interval(total_length: float, wav_length: float, start: float, end: float) -> Interval:
    return Interval(total_length + start - wav_length, total_length + end - wav_length)


def _matches_value(actual: float, expected: float, tolerance_secs: float) -> bool:
    if math.isnan(actual) and math.isnan(expected):
        return True
    if math.isnan(actual) or math.isnan(expected):
        return False
    return abs(actual - expected) < tolerance_secs


def _fmt_time(secs: float) -> str:
    if math.isnan(secs):
        return "nan"
    sign = "-" if secs < 0 else ""
    secs = abs(secs)
    minutes = int(secs) // 60
    seconds = int(secs) % 60
    return f"{sign}{minutes}:{seconds:02d}"


def _fmt_delta(secs: float) -> str:
    if math.isnan(secs):
        return "nan"
    return f"{secs:+.1f}"
