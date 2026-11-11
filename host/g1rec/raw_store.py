"""Raw per-episode recording: every received message with its timestamps, at its own rate.

recordings/raw/episode_000001/
    robot_state.parquet    arm_state, hand_state_left/right   (rt/lowstate, rt/dex3/*/state)
    robot_action.parquet   arm_cmd, hand_cmd_left/right       (rt/arm_sdk|rt/lowcmd, rt/dex3/*/cmd)
    tactile.parquet        tactile_left/right                 (press sensors, original sampling rate)
    camera_state.parquet   neck                               (active camera joints, UDP from PC2)
    timestamps.parquet     image frame stamps and the recorder's tick decisions

Each row: topic, recv_ns (recorder monotonic clock), source_ns (sender's clock domain, may be
null), seq (sender sequence number, may be null), values (flat float list), layout (how
``values`` is laid out). While recording, rows are flushed to ``<name>.part-NNNN.parquet``
every few seconds, so a crash loses at most that interval; ``close()`` merges the parts
into ``<name>.parquet`` and ``recover()`` does the same for a crashed episode.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .sync import Sample

logger = logging.getLogger(__name__)

FILE_OF_STREAM = {
    "arm_state": "robot_state", "hand_state_left": "robot_state", "hand_state_right": "robot_state",
    "arm_cmd": "robot_action", "hand_cmd_left": "robot_action", "hand_cmd_right": "robot_action",
    "tactile_left": "tactile", "tactile_right": "tactile",
    "neck": "camera_state",
    "image": "timestamps", "tick": "timestamps",
}
FILES = ("robot_state", "robot_action", "tactile", "camera_state", "timestamps")


def flatten_payload(value: Any) -> tuple[list[float], str]:
    """A dict of numbers / lists / lists of lists -> (flat floats, "key:len,key:len" layout)."""

    values: list[float] = []
    layout: list[str] = []

    def add(name: str, item: Any) -> None:
        if isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
            item = item.tolist() if hasattr(item, "tolist") else item
            flat: list[float] = []
            stack = [item]
            while stack:
                cur = stack.pop(0)
                if isinstance(cur, (list, tuple)):
                    stack = list(cur) + stack
                else:
                    flat.append(float(cur))
            values.extend(flat)
            layout.append(f"{name}:{len(flat)}")
        elif isinstance(item, (int, float, bool)):
            values.append(float(item))
            layout.append(f"{name}:1")
        # strings / None are not numeric: not part of the raw values

    if isinstance(value, dict):
        for key in value:
            add(str(key), value[key])
    else:
        add("value", value)
    return values, ",".join(layout)


def sample_to_row(sample: Sample) -> dict:
    values, layout = flatten_payload(sample.value)
    return {"topic": sample.stream, "recv_ns": int(sample.recv_ns),
            "source_ns": None if sample.source_ns is None else int(sample.source_ns),
            "seq": None if sample.seq is None else int(sample.seq),
            "capture_ns": None if sample.capture_ns is None else int(sample.capture_ns),
            "values": values, "layout": layout}


def _schema():
    import pyarrow as pa

    return pa.schema([("topic", pa.string()), ("recv_ns", pa.int64()), ("source_ns", pa.int64()),
                      ("seq", pa.int64()), ("capture_ns", pa.int64()),
                      ("values", pa.list_(pa.float64())), ("layout", pa.string())])


def merge_parts(episode_dir: Path) -> dict[str, int]:
    """Merge ``<name>.part-*.parquet`` into ``<name>.parquet``; returns row counts."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    counts: dict[str, int] = {}
    for name in FILES:
        parts = sorted(episode_dir.glob(f"{name}.part-*.parquet"))
        if not parts:
            continue
        table = pa.concat_tables([pq.read_table(p) for p in parts])
        pq.write_table(table, episode_dir / f"{name}.parquet")
        counts[name] = table.num_rows
        for part in parts:
            part.unlink()
    return counts


def recover(episode_dir) -> dict[str, int]:
    """Make a crashed episode readable (merge leftover parts)."""

    return merge_parts(Path(episode_dir))


class RawEpisodeStore:
    """Writes the samples published on the hub into one raw episode directory."""

    def __init__(self, root, episode_index: int, flush_s: float = 2.0) -> None:
        self.dir = Path(root) / f"episode_{episode_index:06d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._flush_s = flush_s
        self._queue: "queue.SimpleQueue[Optional[Sample]]" = queue.SimpleQueue()
        self._rows: dict[str, list] = {}
        self._part = {name: 0 for name in FILES}
        self.counts: dict[str, int] = {}
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="raw-store", daemon=True)
        self._thread.start()

    def on_sample(self, sample: Sample) -> None:       # hub listener; never blocks the source
        if sample.stream in FILE_OF_STREAM:
            self._queue.put(sample)

    def _flush(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for name, rows in list(self._rows.items()):
            if not rows:
                continue
            self._part[name] += 1
            table = pa.Table.from_pylist(rows, schema=_schema())
            pq.write_table(table, self.dir / f"{name}.part-{self._part[name]:04d}.parquet")
            self._rows[name] = []

    def _run(self) -> None:
        next_flush = time.monotonic() + self._flush_s
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = False
            try:
                if item is None:
                    self._flush()
                    return
                if item is not False:
                    name = FILE_OF_STREAM[item.stream]
                    self._rows.setdefault(name, []).append(sample_to_row(item))
                    self.counts[item.stream] = self.counts.get(item.stream, 0) + 1
                if time.monotonic() >= next_flush:
                    self._flush()
                    next_flush = time.monotonic() + self._flush_s
            except OSError as error:       # disk full etc.: stop writing, report through .error
                self.error = error
                logger.error("raw store write failed: %s", error)
                self._rows.clear()
                return
            except Exception as error:     # keep the thread alive for unexpected row problems
                self.error = self.error or error
                logger.exception("raw store failed")

    def close(self) -> dict[str, int]:
        self._queue.put(None)
        self._thread.join(timeout=30)
        try:
            merged = merge_parts(self.dir)
        except Exception as error:
            self.error = self.error or error
            logger.error("raw merge failed (parts are kept for recover()): %s", error)
            merged = {}
        return merged
