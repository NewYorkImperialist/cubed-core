"""Low-overhead, opt-in wall-time tracing for the unified E2E benchmark.

The tracer is inert unless ``CUBED_E2E_ACTIVE=1`` and
``CUBED_E2E_TIMING_PATH`` are both set.  Spans are aggregated in memory and
written once, so profiling does not turn a per-frame loop into an I/O loop.
Inclusive and self time are both retained; callers may therefore report a
critical-path parent without double-counting nested GPU/decode work.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time


class PerfTrace:
    """One process-local aggregate trace."""

    def __init__(self, process: str):
        self.process = process
        self.path = os.environ.get("CUBED_E2E_TIMING_PATH")
        self.enabled = (
            os.environ.get("CUBED_E2E_ACTIVE", "0") == "1" and bool(self.path)
        )
        self.parent_start_ns = _optional_int(
            os.environ.get("CUBED_E2E_PARENT_START_NS")
        )
        self._stages: dict[str, dict[str, int]] = {}
        self._marks: dict[str, dict[str, int]] = {}
        self._meta: dict[str, object] = {}
        self._stack: list[dict[str, int | str]] = []
        if self.enabled:
            self.mark("process_import")
            atexit.register(self.write)

    @contextmanager
    def span(self, name: str):
        """Aggregate one nested wall span under ``name``."""
        if not self.enabled:
            yield
            return
        frame: dict[str, int | str] = {
            "name": name,
            "start_ns": time.perf_counter_ns(),
            "child_ns": 0,
        }
        self._stack.append(frame)
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            current = self._stack.pop()
            elapsed = end_ns - int(current["start_ns"])
            self_ns = max(0, elapsed - int(current["child_ns"]))
            self._add(name, elapsed, self_ns)
            if self._stack:
                self._stack[-1]["child_ns"] = (
                    int(self._stack[-1]["child_ns"]) + elapsed
                )

    def start_ns(self) -> int:
        """Return a monotonic start token for a manually delimited child span."""
        return time.perf_counter_ns() if self.enabled else 0

    def finish_child(self, name: str, start_ns: int) -> None:
        """Finish a manual span and charge it as a child of the active parent."""
        if not self.enabled:
            return
        elapsed = time.perf_counter_ns() - start_ns
        self._add(name, elapsed, elapsed)
        if self._stack:
            self._stack[-1]["child_ns"] = (
                int(self._stack[-1]["child_ns"]) + elapsed
            )

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        self._marks[name] = {
            "realtime_ns": time.time_ns(),
            "monotonic_ns": time.perf_counter_ns(),
        }

    def set_meta(self, **values: object) -> None:
        if self.enabled:
            self._meta.update(values)

    def increment(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        counters = self._meta.setdefault("counters", {})
        assert isinstance(counters, dict)
        counters[name] = int(counters.get(name, 0)) + int(value)

    def _add(self, name: str, inclusive_ns: int, self_ns: int) -> None:
        rec = self._stages.setdefault(
            name, {"count": 0, "inclusive_ns": 0, "self_ns": 0}
        )
        rec["count"] += 1
        rec["inclusive_ns"] += int(inclusive_ns)
        rec["self_ns"] += int(self_ns)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cubed/perf-trace/v1",
            "process": self.process,
            "pid": os.getpid(),
            "parent_start_ns": self.parent_start_ns,
            "parent_clock": "time.perf_counter_ns",
            "stages": {
                name: {
                    "count": rec["count"],
                    "inclusive_ms": rec["inclusive_ns"] / 1_000_000.0,
                    "self_ms": rec["self_ns"] / 1_000_000.0,
                }
                for name, rec in sorted(self._stages.items())
            },
            "marks": dict(sorted(self._marks.items())),
            "meta": self._meta,
        }

    def write(self) -> None:
        if not self.enabled or not self.path:
            return
        # An atexit write refreshes the same file after an earlier explicit
        # write.  This is intentional: failures after the main result still
        # leave the most complete trace possible.
        payload = self.payload()
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, target)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


_TRACES: dict[str, PerfTrace] = {}


def get_trace(process: str) -> PerfTrace:
    """Return the singleton trace shared by modules in this process."""
    if process not in _TRACES:
        _TRACES[process] = PerfTrace(process)
    return _TRACES[process]
