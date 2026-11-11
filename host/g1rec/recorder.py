"""Episode recorder: turns the asynchronous streams into 30 Hz LeRobot frames.

For tick time t (this process's monotonic clock):
  - observations: as-of join, the newest sample received at or before t (never a later one);
  - action: the first command the teleop published at or after t (what it did next), or the
    newest command at or before t with ``action_alignment: asof``;
  - a tick is dropped (counted, never zero-filled) if any required stream is missing or stale.
Ticks are processed one control period late so the command issued for tick t has arrived.
The original messages are written untouched to the raw store at their own rate.
"""

from __future__ import annotations

import logging
import math
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import schema
from .config import Config
from .raw_store import RawEpisodeStore
from .sync import Sample, StreamHub, age_ms, gather

logger = logging.getLogger(__name__)

STATE_STREAMS = ("arm_state", "hand_state_left", "hand_state_right",
                 "tactile_left", "tactile_right", "neck", "image")
ACTION_STREAMS = ("arm_cmd", "hand_cmd_left", "hand_cmd_right", "neck")


def limits_ms(cfg: Config, streams) -> dict[str, float]:
    age = cfg.max_age_ms
    out = {}
    for stream in streams:
        key = ("hand_state" if stream.startswith("hand_state") else
               "hand_cmd" if stream.startswith("hand_cmd") else
               "tactile" if stream.startswith("tactile") else stream)
        out[stream] = float(age[key])
    return out


def build_state_parts(samples: dict[str, Sample]) -> dict:
    """Aligned samples -> {"left_arm": {...}, ..., "camera": {"qpos": [pitch, yaw]}}."""

    arm = samples["arm_state"].value
    parts = {"left_arm": arm["left_arm"], "right_arm": arm["right_arm"]}
    for side in ("left", "right"):
        motors = samples[f"hand_state_{side}"].value
        rows = samples[f"tactile_{side}"].value.get("pressure") or []
        if not rows or len(rows[0]) < schema.PRESSURE_DIM:
            raise schema.SchemaError(f"{side} hand reports no press sensor with 12 values")
        parts[f"{side}_hand"] = {**motors, "pressure": rows[0][:schema.PRESSURE_DIM]}
    present = samples["neck"].value.get("present_deg")
    if not present or len(present) != 2:
        raise schema.SchemaError("neck encoder angles are missing")
    yaw_pitch_rad = [math.radians(v) for v in present]
    parts["camera"] = {"qpos": schema.neck_to_camera_qpos(yaw_pitch_rad)}
    return parts


def build_action_parts(samples: dict[str, Sample]) -> dict:
    arm = samples["arm_cmd"].value
    parts = {"left_arm": {"qpos": arm["left_arm"]["qpos"]},
             "right_arm": {"qpos": arm["right_arm"]["qpos"]},
             "left_hand": samples["hand_cmd_left"].value,
             "right_hand": samples["hand_cmd_right"].value}
    command = samples["neck"].value.get("command_deg")
    if not command or len(command) != 2:
        raise schema.SchemaError("neck command angles are missing")
    parts["camera"] = {"qpos": schema.neck_to_camera_qpos([math.radians(v) for v in command])}
    return parts


def frame_images(frame: Any, cameras) -> dict[str, Any]:
    """ZED left|right BGR (and extra mono cameras) -> {camera: RGB array}."""

    import numpy as np

    bgr = frame.bgr
    width = bgr.shape[1]
    if bgr.ndim != 3 or bgr.shape[2] != 3 or width % 2:
        raise schema.SchemaError("expected a BGR left|right ZED frame with an even width")
    half = width // 2
    eyes = {"cam_left_active": bgr[:, :half], "cam_right_active": bgr[:, half:]}
    eyes.update(frame.extra)
    missing = [c for c in cameras if c not in eyes]
    if missing:
        raise schema.SchemaError(f"no image for cameras {missing}")
    return {c: np.ascontiguousarray(eyes[c][..., ::-1]) for c in cameras}


class Recorder:
    def __init__(self, cfg: Config, hub: StreamHub, writer: Any,
                 raw_factory: Callable[..., Any] = RawEpisodeStore,
                 now_ns: Callable[[], int] = time.monotonic_ns,
                 emit: Optional[Callable[[dict], None]] = None) -> None:
        self.cfg, self.hub, self.writer = cfg, hub, writer
        self._raw_factory, self._now = raw_factory, now_ns
        self.emit = emit or (lambda event: None)
        self.period_ns = int(1e9 / cfg.fps)
        self.delay_ns = int(1.5 * self.period_ns)
        self._state_limits = limits_ms(cfg, STATE_STREAMS)
        self._action_limits = limits_ms(cfg, ACTION_STREAMS)
        self.recording = False
        self._raw: Any = None
        self._next_tick: Optional[int] = None
        self._episode: dict = {}
        self.episodes: list[dict] = []
        self._raw_dirs: dict[int, str] = {}
        Path(cfg.raw_root).mkdir(parents=True, exist_ok=True)

    # --- readiness ----------------------------------------------------------------

    def readiness(self) -> list[str]:
        """Why recording could not start right now (empty = ready)."""

        t = self._now()
        _, problems = gather(self.hub, t, self._state_limits)
        free_gb = shutil.disk_usage(Path(self.cfg.raw_root)).free / 1e9
        if free_gb < self.cfg.min_free_gb:
            problems.append(f"disk: only {free_gb:.1f} GB free (< {self.cfg.min_free_gb} GB)")
        return problems

    # --- episodes -------------------------------------------------------------------

    def _next_raw_index(self) -> int:
        existing = [int(p.name.split("_")[-1]) for p in Path(self.cfg.raw_root).glob("episode_*")
                    if p.name.split("_")[-1].isdigit()]
        return max(existing, default=0) + 1

    def start_episode(self) -> bool:
        if self.recording:
            return False
        free_gb = shutil.disk_usage(Path(self.cfg.raw_root)).free / 1e9
        if free_gb < self.cfg.min_free_gb:
            self.emit({"event": "error", "msg": f"not enough disk space ({free_gb:.1f} GB free)"})
            return False
        if not self.writer.begin_episode():
            self.emit({"event": "error", "msg": "previous episode is still being saved"})
            return False
        index = self._next_raw_index()
        self._raw = self._raw_factory(self.cfg.raw_root, index, self.cfg.raw_flush_s)
        self.hub.add_listener(self._raw.on_sample)
        self.recording = True
        self._next_tick = None
        self._episode = {"raw_index": index, "frames": 0, "ticks": 0, "dropped": {}, "start_ns": self._now(),
                         "first_tick_ns": None, "last_tick_ns": None,
                         "counts_at_start": dict(self.hub.counts), "gaps_at_start": dict(self.hub.gaps)}
        self.emit({"event": "episode_started", "raw_episode": index})
        return True

    def stop_episode(self) -> Optional[dict]:
        if not self.recording:
            return None
        self.recording = False
        self.hub.remove_listener(self._raw.on_sample)
        handle = self.writer.end_episode()
        raw_counts = {}
        try:
            self._raw.close()
            raw_counts = dict(self._raw.counts)
        except Exception as error:
            self.emit({"event": "error", "msg": f"raw store: {error}"})
        handle.wait(600)
        ep = self._episode
        frames = ep["frames"]
        duration = 0.0 if ep["first_tick_ns"] is None else (ep["last_tick_ns"] - ep["first_tick_ns"]) / 1e9
        saved = self.writer.error is None and frames > 0
        summary = {
            "raw_episode": ep["raw_index"], "saved": saved, "frames": frames, "ticks": ep["ticks"],
            "duration_s": round(duration + 1.0 / self.cfg.fps, 3) if frames else 0.0,
            "dropped_ticks": ep["dropped"],
            "robot_states": raw_counts.get("arm_state", 0),
            "actions": raw_counts.get("arm_cmd", 0),
            "tactile_samples": raw_counts.get("tactile_left", 0) + raw_counts.get("tactile_right", 0),
            "neck_samples": raw_counts.get("neck", 0), "raw_counts": raw_counts,
            "video_streams": list(self.cfg.cameras),
            "sequence_gaps": {k: v - ep["gaps_at_start"].get(k, 0) for k, v in self.hub.gaps.items()
                              if v - ep["gaps_at_start"].get(k, 0)},
            "writer_error": None if self.writer.error is None else str(self.writer.error),
        }
        if saved:
            self._raw_dirs[self.writer.total_episodes() - 1] = f"episode_{ep['raw_index']:06d}"
            self.writer.write_sidecar("raw_episodes.json", {"lerobot_episode_to_raw_dir": self._raw_dirs})
        self.episodes.append(summary)
        self.emit({"event": "episode_saved" if saved else "episode_discarded", **summary})
        return summary

    # --- ticks ------------------------------------------------------------------------

    def _drop(self, reason: str) -> None:
        key = reason.split(" (")[0][:80]          # "arm_state: stale", "image: no data yet", ...
        self._episode["dropped"][key] = self._episode["dropped"].get(key, 0) + 1

    def process_tick(self, t_ns: int) -> bool:
        """Build and queue the frame for tick time t; False if the tick is dropped."""

        ep = self._episode
        ep["ticks"] += 1
        samples, problems = gather(self.hub, t_ns, self._state_limits)
        action_samples: dict[str, Sample] = {}
        for stream in ("arm_cmd", "hand_cmd_left", "hand_cmd_right"):
            if self.cfg.action_alignment == "next":
                sample = self.hub.first_at_or_after(stream, t_ns, self.delay_ns)
            else:
                sample = self.hub.asof(stream, t_ns)
                if sample is not None and age_ms(sample, t_ns) > self._action_limits[stream]:
                    sample = None
            if sample is None:
                problems.append(f"{stream}: no command for this tick")
            else:
                action_samples[stream] = sample
        if problems:
            for problem in problems:
                self._drop(problem)
            self.hub.publish(Sample("tick", {"accepted": 0}, t_ns))
            return False
        try:
            state_parts = build_state_parts(samples)
            action_parts = build_action_parts({**action_samples, "neck": samples["neck"]})
            state = schema.build_state(state_parts)
            action = schema.build_action(action_parts)
            images = frame_images(samples["image"].value, self.cfg.cameras)
        except schema.SchemaError as error:
            self._drop(f"schema: {error}")
            return False
        self.writer.add_frame(images, state, action)
        ep["frames"] += 1
        ep["first_tick_ns"] = t_ns if ep["first_tick_ns"] is None else ep["first_tick_ns"]
        ep["last_tick_ns"] = t_ns
        self.hub.publish(Sample("tick", {"accepted": 1, "frame_index": ep["frames"] - 1}, t_ns))
        return True

    def step(self) -> None:
        """Process every tick that is due (called from the loop)."""

        if not self.recording:
            return
        now = self._now()
        if self._next_tick is None:
            self._next_tick = now
        while now >= self._next_tick + self.delay_ns:
            if now - (self._next_tick + self.delay_ns) > 3 * self.period_ns:
                self._drop("loop: late (skipped ahead)")
                self._next_tick = now - self.delay_ns
            self.process_tick(self._next_tick)
            self._next_tick += self.period_ns
            if self.writer.queue.qsize() > 4 * self.cfg.fps:
                self.emit({"event": "error", "msg": "writer is over 4 s behind; stopping the episode"})
                self.stop_episode()
                return
        if self.writer.error is not None or (self._raw is not None and self._raw.error is not None):
            self.emit({"event": "error", "msg": f"write error: {self.writer.error or self._raw.error}"})
            self.stop_episode()

    # --- loop ------------------------------------------------------------------------

    def run(self, stop: threading.Event, commands: "queue.Queue[str]") -> None:
        last_status = time.monotonic()
        while not stop.is_set():
            try:
                command = commands.get(timeout=0.005)
            except queue.Empty:
                command = None
            if command == "start":
                self.start_episode()
            elif command == "stop":
                self.stop_episode()
            elif command == "quit":
                break
            self.step()
            if time.monotonic() - last_status >= 1.0:
                last_status = time.monotonic()
                if self.recording:
                    ep = self._episode
                    self.emit({"event": "status", "frames": ep["frames"], "ticks": ep["ticks"],
                               "dropped": dict(ep["dropped"])})
        if self.recording:
            self.stop_episode()
