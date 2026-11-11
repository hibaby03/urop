"""Synthetic streams with realistic, different rates, for tests without hardware."""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from g1rec import schema
from g1rec.sources import ImageFrame
from g1rec.sync import Sample, StreamHub


def arm_state_value(t: float) -> dict:
    return {side: {"qpos": [0.1 * i + 0.05 * math.sin(t + i) for i in range(7)],
                   "qvel": [0.01 * i for i in range(7)], "torque": [0.2 * i for i in range(7)]}
            for side in ("left_arm", "right_arm")}


def hand_state_value(t: float) -> dict:
    return {"qpos": [0.3 * math.sin(t + i) for i in range(7)], "qvel": [0.02] * 7, "torque": [0.05] * 7}


def tactile_value(t: float) -> dict:
    rows = [[100.0 + 10 * s + i + math.sin(t) for i in range(12)] for s in range(9)]
    return {"pressure": rows, "temperature": [[30.0] * 12] * 9, "lost": [0] * 9}


def arm_cmd_value(t: float) -> dict:
    return {side: {"qpos": [0.1 * i + 0.05 * math.sin(t + i + 0.03) for i in range(7)], "torque": [0.0] * 7}
            for side in ("left_arm", "right_arm")}


def image_frame(index: int, width: int = 256, height: int = 72) -> ImageFrame:
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, : width // 2] = (0, 0, 40 + index % 200)         # left eye (BGR red)
    bgr[:, width // 2:] = (40 + index % 200, 0, 0)          # right eye (BGR blue)
    return ImageFrame(bgr, {})


class SyntheticRobot:
    """Publishes every stream at its own rate on background threads."""

    RATES = {"arm_state": 500.0, "hand_state": 100.0, "tactile": 100.0, "arm_cmd": 120.0,
             "hand_cmd": 120.0, "neck": 60.0, "image": 30.0}

    def __init__(self, hub: StreamHub, rates: dict | None = None, enabled: set | None = None) -> None:
        self.hub = hub
        self.rates = {**self.RATES, **(rates or {})}
        self.enabled = enabled or set(self.rates)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.t0 = time.monotonic()
        self.hub.limit_history("image", 8)

    def _every(self, name: str, fn) -> None:
        def loop() -> None:
            period = 1.0 / self.rates[name]
            nxt = time.monotonic()
            count = 0
            while not self._stop.is_set():
                fn(count, time.monotonic() - self.t0)
                count += 1
                nxt += period
                delay = nxt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        thread = threading.Thread(target=loop, daemon=True)
        self._threads.append(thread)

    def start(self) -> "SyntheticRobot":
        hub = self.hub

        def now() -> int:
            return time.monotonic_ns()

        if "arm_state" in self.enabled:
            self._every("arm_state", lambda n, t: hub.publish(Sample("arm_state", arm_state_value(t), now(), seq=n)))
        if "hand_state" in self.enabled:
            for side in ("left", "right"):
                self._every("hand_state", lambda n, t, s=side: hub.publish(
                    Sample(f"hand_state_{s}", hand_state_value(t), now())))
        if "tactile" in self.enabled:
            for side in ("left", "right"):
                self._every("tactile", lambda n, t, s=side: hub.publish(
                    Sample(f"tactile_{s}", tactile_value(t), now())))
        if "arm_cmd" in self.enabled:
            self._every("arm_cmd", lambda n, t: hub.publish(Sample("arm_cmd", arm_cmd_value(t), now())))
        if "hand_cmd" in self.enabled:
            for side in ("left", "right"):
                self._every("hand_cmd", lambda n, t, s=side: hub.publish(
                    Sample(f"hand_cmd_{s}", {"qpos": [0.3 * math.sin(t + i + 0.03) for i in range(7)]}, now())))
        if "neck" in self.enabled:
            self._every("neck", lambda n, t: hub.publish(Sample(
                "neck", {"present_deg": [10 * math.sin(t), 5 * math.cos(t)],
                         "command_deg": [10 * math.sin(t + 0.05), 5 * math.cos(t + 0.05)],
                         "torque_enabled": True}, now(), seq=n)))
        if "image" in self.enabled:
            self._every("image", lambda n, t: hub.publish(Sample("image", image_frame(n), now(), seq=n)))
        for thread in self._threads:
            thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
