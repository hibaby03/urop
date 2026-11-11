"""LeRobotDataset writer for the master dataset (video features, UAF schema).

Encoding and disk writes run on a worker thread so the recorder loop is never blocked. The
dataset API calls (create / add_frame / save_episode / finalize) are the ones of the LeRobot
version installed here; verify.py checks the result with that same installed version.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import schema

logger = logging.getLogger(__name__)


def fit_image(rgb: Any, mode: str = "letterbox") -> Any:
    """Resize an RGB image to IMAGE_H x IMAGE_W (letterbox keeps the aspect ratio)."""

    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    if mode == "stretch":
        return cv2.resize(rgb, (schema.IMAGE_W, schema.IMAGE_H), interpolation=cv2.INTER_AREA)
    if mode != "letterbox":
        raise ValueError(f"unknown image fit {mode!r}")
    scale = min(schema.IMAGE_W / width, schema.IMAGE_H / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((schema.IMAGE_H, schema.IMAGE_W, 3), dtype=rgb.dtype)
    top, left = (schema.IMAGE_H - new_h) // 2, (schema.IMAGE_W - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def open_dataset(root: Path, repo_id: str, fps: int, features: dict) -> Any:
    """Create the LeRobotDataset, or open an existing one to append episodes."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (Path(root) / "meta" / "info.json").is_file():
        return LeRobotDataset(repo_id, root=root)
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root,
                                 robot_type=schema.ROBOT_TYPE, use_videos=True,
                                 image_writer_threads=4)


class SaveHandle:
    def __init__(self) -> None:
        self._done = threading.Event()
        self.frames = 0

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)


class LeRobotWriter:
    def __init__(self, root, repo_id: str, task: str, fps: int, cameras: Sequence[str],
                 image_fit: str = "letterbox",
                 dataset_factory: Optional[Callable[..., Any]] = None) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.task = task
        self.fps = int(fps)
        self.cameras = list(cameras)
        self.image_fit = image_fit
        self._factory = dataset_factory or open_dataset
        self._dataset: Any = None
        self._sidecars: dict[str, dict] = {}
        self.error: Optional[BaseException] = None
        self.queue: "queue.Queue[Any]" = queue.Queue()
        self.episodes_saved = 0
        self.frames_in_episode = 0
        self._frames_done = 0
        self._open = False
        self._saving = threading.Event()
        self._worker = threading.Thread(target=self._run, name="lerobot-writer", daemon=True)
        self._worker.start()

    def write_sidecar(self, name: str, content: dict) -> None:
        self._sidecars[name] = content
        if self._dataset is not None:
            self._write_sidecars()

    def _write_sidecars(self) -> None:
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        for name, content in self._sidecars.items():
            (meta / name).write_text(json.dumps(content, indent=2, ensure_ascii=False))

    def begin_episode(self) -> bool:
        """False while the previous episode is still being saved (or after an error)."""

        if self._saving.is_set() or self.error is not None:
            return False
        if self._dataset is None:
            self._dataset = self._factory(self.root, self.repo_id, self.fps,
                                          schema.lerobot_features(self.cameras))
            self._write_sidecars()
        self._open = True
        self.frames_in_episode = 0
        return True

    def add_frame(self, images_rgb: dict[str, Any], state: Sequence[float],
                  action: Sequence[float]) -> None:
        if not self._open:
            raise RuntimeError("add_frame without an open episode")
        if len(state) != schema.STATE_DIM or len(action) != schema.ACTION_DIM:
            raise schema.SchemaError("state/action dimension mismatch")
        missing = [c for c in self.cameras if c not in images_rgb]
        if missing:
            raise schema.SchemaError(f"missing camera images {missing}")
        self.frames_in_episode += 1
        self.queue.put(("frame", images_rgb, list(state), list(action)))

    def end_episode(self) -> SaveHandle:
        handle = SaveHandle()
        if not self._open:
            handle._done.set()
            return handle
        self._open = False
        self._saving.set()
        handle.frames = self.frames_in_episode
        self.queue.put(("save", handle))
        return handle

    def total_episodes(self) -> int:
        meta = getattr(self._dataset, "meta", None)
        return int(getattr(meta, "total_episodes", self.episodes_saved))

    def close(self) -> None:
        if self._open:
            self.end_episode()
        self.queue.put(("stop",))
        self._worker.join()
        dataset = self._dataset
        if dataset is not None and hasattr(dataset, "finalize"):
            try:
                dataset.finalize()
            except Exception as error:
                self.error = self.error or error
        if self.error is not None:
            raise OSError(f"LeRobot writer failed: {self.error}") from self.error

    def _frame(self, images: dict, state: list, action: list) -> dict:
        import numpy as np

        frame: dict[str, Any] = {"observation.state": np.asarray(state, dtype=np.float32),
                                 "action": np.asarray(action, dtype=np.float32), "task": self.task}
        for camera in self.cameras:
            frame[f"observation.images.{camera}"] = fit_image(images[camera], self.image_fit)
        return frame

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            kind = item[0]
            try:
                if kind == "stop":
                    return
                if self.error is not None:
                    continue
                if kind == "frame":
                    self._dataset.add_frame(self._frame(item[1], item[2], item[3]))
                    self._frames_done += 1
                elif kind == "save":
                    if self._frames_done == 0:
                        clear = getattr(self._dataset, "clear_episode_buffer", None)
                        if clear is not None:
                            clear()
                        logger.warning("episode had no accepted frames; nothing saved")
                    else:
                        self._dataset.save_episode()
                        self.episodes_saved += 1
                        self._frames_done = 0
            except Exception as error:
                self.error = error
                logger.exception("LeRobot writer failed")
            finally:
                if kind == "save":
                    self._saving.clear()
                    item[1]._done.set()
