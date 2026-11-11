"""Time alignment of asynchronous streams (no future leakage).

Every source publishes into a StreamHub with ``recv_ns`` = this process's
``time.monotonic_ns()`` when the message was received, the sender's own timestamp
(``source_ns``, in the sender's clock domain; None if the message has none) and the
sender's sequence number if it has one.

Alignment is "as of": for a policy tick at time t the sample of each stream is the newest
one with ``recv_ns <= t``. A message that arrives after t can never reach that tick's
observation or action. A stream whose newest eligible sample is older than its limit is
reported as stale instead of being filled in.
"""

from __future__ import annotations

import bisect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Sample:
    stream: str
    value: Any
    recv_ns: int
    source_ns: Optional[int] = None
    seq: Optional[int] = None
    capture_ns: Optional[int] = None     # when the data was taken, in this process's clock, if known


class StreamHub:
    """Thread-safe latest/history store plus listeners (the raw recorder subscribes here)."""

    def __init__(self, history_s: float = 3.0) -> None:
        self._lock = threading.Lock()
        self._history_ns = int(history_s * 1e9)
        self._streams: dict[str, deque] = {}
        self._listeners: list[Callable[[Sample], None]] = []
        self._max_len: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.gaps: dict[str, int] = {}        # missing sequence numbers per stream
        self._last_seq: dict[str, int] = {}

    def limit_history(self, stream: str, max_len: int) -> None:
        """Keep at most ``max_len`` samples of a stream (large payloads such as images)."""

        self._max_len[stream] = max_len

    def add_listener(self, listener: Callable[[Sample], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Sample], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def publish(self, sample: Sample) -> None:
        with self._lock:
            history = self._streams.setdefault(sample.stream, deque())
            if history and sample.recv_ns < history[-1].recv_ns:
                return  # out-of-order receive time: keep the history monotonic
            history.append(sample)
            horizon = sample.recv_ns - self._history_ns
            while len(history) > 1 and history[0].recv_ns < horizon:
                history.popleft()
            limit = self._max_len.get(sample.stream)
            while limit is not None and len(history) > limit:
                history.popleft()
            self.counts[sample.stream] = self.counts.get(sample.stream, 0) + 1
            if sample.seq is not None:
                last = self._last_seq.get(sample.stream)
                if last is not None and sample.seq > last + 1:
                    self.gaps[sample.stream] = self.gaps.get(sample.stream, 0) + sample.seq - last - 1
                self._last_seq[sample.stream] = sample.seq
        for listener in list(self._listeners):
            listener(sample)

    def asof(self, stream: str, t_ns: int) -> Optional[Sample]:
        """Newest sample of ``stream`` received at or before ``t_ns``."""

        with self._lock:
            history = self._streams.get(stream)
            if not history:
                return None
            times = [s.recv_ns for s in history]
            index = bisect.bisect_right(times, t_ns) - 1
            return history[index] if index >= 0 else None

    def first_at_or_after(self, stream: str, t_ns: int, max_gap_ns: int) -> Optional[Sample]:
        """Earliest sample received in [t_ns, t_ns + max_gap_ns]: the command issued for tick t."""

        with self._lock:
            history = self._streams.get(stream)
            if not history:
                return None
            times = [s.recv_ns for s in history]
            index = bisect.bisect_left(times, t_ns)
            if index < len(history) and history[index].recv_ns - t_ns <= max_gap_ns:
                return history[index]
            return None

    def latest(self, stream: str) -> Optional[Sample]:
        with self._lock:
            history = self._streams.get(stream)
            return history[-1] if history else None


def age_ms(sample: Optional[Sample], t_ns: int) -> Optional[float]:
    """Age of the data at t: uses the capture time when known, else the receive time."""

    if sample is None:
        return None
    reference = sample.capture_ns if sample.capture_ns is not None else sample.recv_ns
    return (t_ns - reference) / 1e6


def gather(hub: StreamHub, t_ns: int, limits_ms: dict[str, float]) -> tuple[dict, list[str]]:
    """As-of samples for every stream in ``limits_ms`` and the reasons some are unusable."""

    samples: dict[str, Sample] = {}
    problems: list[str] = []
    for stream, limit in limits_ms.items():
        sample = hub.asof(stream, t_ns)
        if sample is None:
            problems.append(f"{stream}: no data yet")
            continue
        age = age_ms(sample, t_ns)
        if age is not None and age > limit:
            problems.append(f"{stream}: stale ({age:.0f} ms > {limit:.0f} ms)")
            continue
        samples[stream] = sample
    return samples, problems
