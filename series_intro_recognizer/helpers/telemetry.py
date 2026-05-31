"""
Lightweight telemetry for measuring GPU step durations.

By default, telemetry is disabled and all measure() calls are zero-overhead no-ops.
Enable it by calling telemetry.enable(callback) where callback receives (name, seconds).

Usage::

    from series_intro_recognizer.helpers.telemetry import telemetry

    records = []
    telemetry.enable(lambda name, secs: records.append((name, secs)))
    # ... run pipeline ...
    telemetry.disable()
"""

from __future__ import annotations

import time
from typing import Callable, Literal


# ---------------------------------------------------------------------------
# Noop context – reused singleton, zero-allocation when telemetry is off.
# ---------------------------------------------------------------------------

class _NoopContext:
    """Returned by Telemetry.measure() when telemetry is disabled."""

    __slots__ = ()

    def __enter__(self) -> "_NoopContext":
        return self

    def __exit__(self, *_: object) -> Literal[False]:
        return False


_NOOP = _NoopContext()


# ---------------------------------------------------------------------------
# Active context – created only when telemetry is enabled.
# ---------------------------------------------------------------------------

class _ActiveContext:
    """Returned by Telemetry.measure() when telemetry is enabled."""

    __slots__ = ("_name", "_callback", "_start")

    def __init__(self, name: str, callback: Callable[[str, float], None]) -> None:
        self._name = name
        self._callback = callback
        self._start = 0.0

    def __enter__(self) -> "_ActiveContext":
        _sync_gpu()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> Literal[False]:
        _sync_gpu()
        self._callback(self._name, time.perf_counter() - self._start)
        return False


def _sync_gpu() -> None:
    try:
        import cupy as cp  # type: ignore
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public Telemetry class
# ---------------------------------------------------------------------------

class Telemetry:
    """
    Thread-safe-enough telemetry gate for sequential GPU pipelines.

    Production overhead when disabled: one bool attribute read per measure() call.
    """

    __slots__ = ("_enabled", "_callback")

    def __init__(self) -> None:
        self._enabled: bool = False
        self._callback: Callable[[str, float], None] | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def enable(self, callback: Callable[[str, float], None]) -> None:
        """Enable telemetry. *callback(name, seconds)* is called after each step."""
        self._callback = callback
        self._enabled = True

    def disable(self) -> None:
        """Disable telemetry and clear the callback."""
        self._enabled = False
        self._callback = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, name: str) -> _NoopContext | _ActiveContext:
        """
        Return a context manager that times the enclosed block.

        When disabled, returns a singleton noop object (no allocation, no timing).
        When enabled, GPU streams are synchronised before and after the block.
        """
        if not self._enabled:
            return _NOOP
        assert self._callback is not None
        return _ActiveContext(name, self._callback)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global telemetry instance used by all pipeline steps.
telemetry: Telemetry = Telemetry()

