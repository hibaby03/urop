"""Read-only stream sources. This module contains no publisher and no command sender.

- DDS   rt/lowstate, rt/dex3/{left,right}/state      robot state, tactile
        rt/arm_sdk | rt/lowcmd, rt/dex3/{left,right}/cmd   the commands the teleop publishes
- UDP   :5006                                          neck (active camera) state+command from PC2
- ZMQ   PC2 ZED stream via teleimager's ImageClient (a SUB socket) and the PC2 clock-sync probe

The parsers of the active-camera project (neck packet, JPEG stamp, clock sync) come from the
bundled verbatim copy in host/third_party/unitree_g1_urop/host, so the wire formats are never re-implemented.
"""

from __future__ import annotations

import logging
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import schema
from .config import Config
from .sync import Sample, StreamHub

logger = logging.getLogger(__name__)


@dataclass
class ImageFrame:
    """One decoded ZED frame (left|right BGR) plus optional extra mono cameras."""

    bgr: Any
    extra: dict


BUNDLED_HOST = Path(__file__).resolve().parent.parent / "third_party" / "unitree_g1_urop" / "host"


def _import_urop_module(cfg: Config, name: str) -> Any:
    """Stream parsers of the active-camera project: the bundled verbatim copy unless g1_urop_host is set."""

    host = Path(cfg.g1_urop_host).expanduser() if cfg.g1_urop_host else BUNDLED_HOST
    if not host.is_dir():
        raise RuntimeError(f"stream parsers not found at {host} (see third_party/unitree_g1_urop/VENDORED.md)")
    path = str(host.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    import importlib

    return importlib.import_module(name)


def _setup_xr_path(cfg: Config) -> None:
    if cfg.xr_repo:
        root = str(Path(cfg.xr_repo).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


class DdsSources:
    """Subscribers for robot state, tactile and the teleop's command topics."""

    def __init__(self, cfg: Config, hub: StreamHub) -> None:
        self.cfg, self.hub = cfg, hub
        self._subs: list[Any] = []

    def start(self) -> None:
        _setup_xr_path(self.cfg)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_

        dds = self.cfg.dds
        if dds.interface:
            ChannelFactoryInitialize(dds.domain, networkInterface=dds.interface)
        else:
            ChannelFactoryInitialize(dds.domain)

        def subscribe(topic: str, message_type: Any, handler: Callable) -> None:
            subscriber = ChannelSubscriber(topic, message_type)
            # queueLen > 0 runs the handler on the SDK's reader thread, not the DDS listener thread.
            subscriber.Init(handler, 10)
            self._subs.append(subscriber)

        subscribe(dds.arm_state_topic, LowState_, self._on_lowstate)
        arm_cmd_topic = dds.arm_cmd_topic_motion if dds.motion_mode else dds.arm_cmd_topic_debug
        subscribe(arm_cmd_topic, LowCmd_, self._on_lowcmd)
        for side in ("left", "right"):
            topics = dds.hand_topics[side]
            subscribe(topics["state"], HandState_, self._hand_state_handler(side))
            subscribe(topics["cmd"], HandCmd_, self._hand_cmd_handler(side))

    # Handlers must never raise: an exception would end the SDK reader thread.

    def _on_lowstate(self, msg: Any) -> None:
        try:
            now = time.monotonic_ns()
            parts = {}
            for part, indices in (("left_arm", schema.ARM_MOTOR_INDICES_LEFT),
                                  ("right_arm", schema.ARM_MOTOR_INDICES_RIGHT)):
                motors = [msg.motor_state[i] for i in indices]
                parts[part] = {"qpos": [m.q for m in motors], "qvel": [m.dq for m in motors],
                               "torque": [m.tau_est for m in motors]}
            self.hub.publish(Sample("arm_state", parts, now, source_ns=int(msg.tick) * 1_000_000))
        except Exception:
            logger.debug("bad lowstate message", exc_info=True)

    def _on_lowcmd(self, msg: Any) -> None:
        try:
            now = time.monotonic_ns()
            parts = {}
            for part, indices in (("left_arm", schema.ARM_MOTOR_INDICES_LEFT),
                                  ("right_arm", schema.ARM_MOTOR_INDICES_RIGHT)):
                motors = [msg.motor_cmd[i] for i in indices]
                parts[part] = {"qpos": [m.q for m in motors], "torque": [m.tau for m in motors]}
            self.hub.publish(Sample("arm_cmd", parts, now))
        except Exception:
            logger.debug("bad lowcmd message", exc_info=True)

    def _hand_state_handler(self, side: str) -> Callable:
        def handler(msg: Any) -> None:
            try:
                now = time.monotonic_ns()
                motors = [msg.motor_state[i] for i in schema.HAND_MOTOR_INDICES]
                self.hub.publish(Sample(f"hand_state_{side}", {
                    "qpos": [m.q for m in motors], "qvel": [m.dq for m in motors],
                    "torque": [m.tau_est for m in motors]}, now))
                sensors = list(msg.press_sensor_state)
                self.hub.publish(Sample(f"tactile_{side}", {
                    "pressure": [[float(v) for v in s.pressure] for s in sensors],
                    "temperature": [[float(v) for v in s.temperature] for s in sensors],
                    "lost": [int(s.lost) for s in sensors]}, now))
            except Exception:
                logger.debug("bad hand state message", exc_info=True)
        return handler

    def _hand_cmd_handler(self, side: str) -> Callable:
        def handler(msg: Any) -> None:
            try:
                now = time.monotonic_ns()
                self.hub.publish(Sample(f"hand_cmd_{side}", {
                    "qpos": [msg.motor_cmd[i].q for i in schema.HAND_MOTOR_INDICES]}, now))
            except Exception:
                logger.debug("bad hand cmd message", exc_info=True)
        return handler

    def close(self) -> None:
        for subscriber in self._subs:
            try:
                subscriber.Close()
            except Exception:
                pass


class NeckSource:
    """UDP listener for the active camera's joint state and command (sent by PC2)."""

    def __init__(self, cfg: Config, hub: StreamHub, clock: Optional[Callable] = None,
                 decode: Optional[Callable] = None) -> None:
        self.cfg, self.hub = cfg, hub
        self._clock = clock                      # () -> ClockEstimate | None
        self._decode = decode or _import_urop_module(cfg, "neck_state_receiver").decode_neck_state_packet
        self.rejected = 0
        self._stop = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((cfg.neck.bind, cfg.neck.port))
        self._socket.settimeout(0.05)
        self._thread = threading.Thread(target=self._loop, name="neck-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                datagram, _ = self._socket.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            now = time.monotonic_ns()
            try:
                packet = self._decode(datagram)
            except ValueError:
                self.rejected += 1
                continue
            self.hub.publish(neck_sample(packet, now, self._clock() if self._clock else None))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close()


def neck_sample(packet: Any, recv_ns: int, clock: Any = None) -> Sample:
    """Neck packet -> hub sample (angles in degrees as sent; the recorder converts to radians)."""

    present = list(packet.present_deg) if packet.present_deg else []
    capture = None
    if clock is not None and packet.present_pc2_monotonic_ns:
        capture = clock.pc2_to_host_ns(packet.present_pc2_monotonic_ns)
    return Sample("neck", {"present_deg": present, "command_deg": list(packet.command_deg),
                           "torque_enabled": bool(packet.torque_enabled)},
                  recv_ns, source_ns=packet.present_pc2_monotonic_ns or packet.command_pc2_monotonic_ns,
                  seq=packet.pose_sequence, capture_ns=capture)


class ImageSource:
    """Polls teleimager's ZMQ subscriber for new JPEGs and publishes decoded frames."""

    def __init__(self, cfg: Config, hub: StreamHub, poll_hz: float = 120.0) -> None:
        self.cfg, self.hub = cfg, hub
        self._period = 1.0 / poll_hz
        self.clock_sync: Any = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None
        self.shape: Optional[tuple] = None

    def start(self) -> None:
        _setup_xr_path(self.cfg)
        frame_timing = _import_urop_module(self.cfg, "frame_timing")
        from teleimager.image_client import ImageClient

        self._client = ImageClient(host=self.cfg.images.pc2_ip, request_bgr=False)
        head = self._client.get_cam_config()["head_camera"]
        if not head["binocular"] or not head["enable_zmq"]:
            raise RuntimeError("ZED binocular head camera with enable_zmq=true is required")
        self.shape = tuple(head["image_shape"][:2])
        self.clock_sync = frame_timing.ClockSyncClient(self.cfg.images.pc2_ip,
                                                       self.cfg.images.clock_sync_port)
        reader = frame_timing.StampedFrameReader(self._client.get_head_frame)
        extra_getters = {}
        if "cam_left_wrist" in self.cfg.cameras:
            extra_getters["cam_left_wrist"] = self._client.get_left_wrist_frame
        if "cam_right_wrist" in self.cfg.cameras:
            extra_getters["cam_right_wrist"] = self._client.get_right_wrist_frame
        unsupported = [c for c in self.cfg.cameras
                       if c in ("cam_left_high", "cam_right_high")]
        if unsupported:
            raise RuntimeError(f"static head cameras {unsupported} are not supported: the head "
                               "slot of this setup is the active ZED")
        self.hub.limit_history("image", 8)
        self._thread = threading.Thread(target=self._loop, args=(reader, extra_getters),
                                        name="image-source", daemon=True)
        self._thread.start()

    def _loop(self, reader: Any, extra_getters: dict) -> None:
        last_seq = None
        last_obj = None
        while not self._stop.is_set():
            try:
                frame = reader.read()
                now = time.monotonic_ns()
                if frame.bgr is not None and frame.bgr is not last_obj:
                    last_obj = frame.bgr
                    stamp = frame.stamp
                    estimate = self.clock_sync.estimate() if self.clock_sync else None
                    capture = None
                    if stamp is not None and estimate is not None:
                        capture = estimate.pc2_to_host_ns(stamp.capture_pc2_monotonic_ns)
                    extra = {}
                    for name, getter in extra_getters.items():
                        jpg = getter().jpg
                        if jpg:
                            import cv2
                            import numpy as np

                            extra[name] = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    seq = None if stamp is None else stamp.frame_sequence
                    self.hub.publish(Sample(
                        "image", ImageFrame(frame.bgr, extra), now,
                        source_ns=None if stamp is None else stamp.capture_pc2_monotonic_ns,
                        seq=seq, capture_ns=capture))
                    last_seq = seq
            except Exception:
                logger.exception("image source error")
            self._stop.wait(self._period)

    def clock_estimate(self) -> Any:
        return self.clock_sync.estimate() if self.clock_sync is not None else None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for resource in (self.clock_sync, self._client):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
