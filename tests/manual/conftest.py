from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.manual.harness import ManualOptions

REPORTS_DIR = Path(__file__).parent / "reports"
LATEST_DIR = REPORTS_DIR / "latest"
DEBUG_LOG = LATEST_DIR / "debug.log"
JUNIT_XML = LATEST_DIR / "junit.xml"
REPORT_DIR_KEY = pytest.StashKey[Path]()
DEBUG_LOG_HANDLER_KEY = pytest.StashKey[logging.FileHandler]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("manual")
    group.addoption(
        "--run-manual",
        action="store_true",
        default=os.getenv("SIR_RUN_MANUAL") == "1",
        help="Run tests under tests/manual.",
    )
    group.addoption(
        "--manual-audio-root",
        default=os.getenv("SIR_MANUAL_AUDIO_ROOT", "audio_op_6min"),
        help="Root directory containing per-series audio folders.",
    )
    group.addoption(
        "--manual-expected-csv",
        default=os.getenv("SIR_MANUAL_EXPECTED_CSV", "all_found_results_unique.csv"),
        help="CSV with expected intervals for validation.",
    )
    group.addoption(
        "--manual-output-csv",
        default=os.getenv("SIR_MANUAL_OUTPUT_CSV", str(REPORTS_DIR / "manual-results.csv")),
        help="CSV written by test_csv_create.",
    )
    group.addoption(
        "--manual-kind",
        default=os.getenv("SIR_MANUAL_KIND", "auto"),
        choices=("auto", "opening", "ending"),
        help="How to interpret recognised intervals during validation.",
    )
    group.addoption(
        "--manual-series",
        action="append",
        default=[],
        help="Series id to run. Can be repeated or comma-separated.",
    )
    group.addoption(
        "--manual-series-skip",
        default=int(os.getenv("SIR_MANUAL_SERIES_SKIP", "0")),
        type=int,
        help="Skip this many discovered or selected series.",
    )
    group.addoption(
        "--manual-tolerance-secs",
        default=float(os.getenv("SIR_MANUAL_TOLERANCE_SECS", "1.0")),
        type=float,
        help="Allowed start/end difference for CSV validation.",
    )
    group.addoption(
        "--manual-telemetry",
        action="store_true",
        default=os.getenv("SIR_MANUAL_TELEMETRY") == "1",
        help="Print telemetry timings during manual tests.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: Any) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = REPORTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    config.stash[REPORT_DIR_KEY] = report_dir

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if not getattr(config.option, "xmlpath", None):
        config.option.xmlpath = JUNIT_XML

    file_handler = logging.FileHandler(DEBUG_LOG, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s:%(lineno)d %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(min(root_logger.level, logging.DEBUG))
    config.stash[DEBUG_LOG_HANDLER_KEY] = file_handler


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-manual"):
        return

    skip_manual = pytest.mark.skip(reason="manual tests require --run-manual")
    manual_dir = Path(__file__).parent
    for item in items:
        # Only apply manual test rules to tests in the manual folder
        if manual_dir in Path(item.fspath).parents or Path(item.fspath).parent == manual_dir:
            item.add_marker(skip_manual)


@pytest.fixture
def manual_options(pytestconfig: pytest.Config) -> ManualOptions:
    return ManualOptions(
        audio_root=Path(pytestconfig.getoption("--manual-audio-root")),
        expected_csv=Path(pytestconfig.getoption("--manual-expected-csv")),
        output_csv=Path(pytestconfig.getoption("--manual-output-csv")),
        kind=pytestconfig.getoption("--manual-kind"),
        series_ids=_parse_series_ids(pytestconfig.getoption("--manual-series")),
        series_skip=pytestconfig.getoption("--manual-series-skip"),
        tolerance_secs=pytestconfig.getoption("--manual-tolerance-secs"),
        telemetry=pytestconfig.getoption("--manual-telemetry"),
    )


def pytest_report_header(config: Any) -> str:
    return f"manual test reports: {LATEST_DIR}"


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    report_dir = session.config.stash[REPORT_DIR_KEY]
    report_dir.mkdir(parents=True, exist_ok=True)

    debug_log_handler = session.config.stash[DEBUG_LOG_HANDLER_KEY]
    logging.getLogger().removeHandler(debug_log_handler)
    debug_log_handler.close()

    for report_file in ("junit.xml", "debug.log"):
        source = LATEST_DIR / report_file
        if source.exists():
            shutil.copy2(source, report_dir / report_file)

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(terminal, "stats", {}) if terminal is not None else {}
    lines = [
        f"exitstatus: {exitstatus}",
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "results:",
    ]
    for outcome in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        count = len(stats.get(outcome, []))
        if count:
            lines.append(f"{outcome}: {count}")
    if len(lines) == 4:
        lines.append("no tests collected")

    (report_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    terminalreporter.write_sep("=", f"manual test report saved to {config.stash[REPORT_DIR_KEY]}")


def _parse_series_ids(values: list[str]) -> tuple[int, ...]:
    series_ids: list[int] = []
    for value in values:
        series_ids.extend(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(series_ids)
