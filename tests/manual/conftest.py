from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest


REPORTS_DIR = Path(__file__).parent / "reports"
LATEST_DIR = REPORTS_DIR / "latest"
DEBUG_LOG = LATEST_DIR / "debug.log"
JUNIT_XML = LATEST_DIR / "junit.xml"
REPORT_DIR_KEY = pytest.StashKey[Path]()
DEBUG_LOG_HANDLER_KEY = pytest.StashKey[logging.FileHandler]()


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
