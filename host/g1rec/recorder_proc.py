"""Recorder process: ``python -m g1rec.recorder_proc --config config.yaml``.

Protocol with the orchestrator (one JSON object per line):
  stdin  : {"cmd": "start" | "stop" | "quit"}
  stdout : events {"event": "ready" | "not_ready" | "episode_started" | "status" |
                   "episode_saved" | "episode_discarded" | "error", ...}
The recorder only subscribes to streams; it publishes nothing to the robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time

from . import config as config_mod
from . import schema
from .lerobot_writer import LeRobotWriter
from .recorder import Recorder
from .sources import DdsSources, ImageSource, NeckSource
from .sync import StreamHub


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


def write_dataset_info(writer: LeRobotWriter, cfg) -> None:
    writer.write_sidecar("g1_master_info.json", {
        "schema": "uaf_g1_dex3_master_v1",
        "source": "layout from kuehnrobin/UAF_unitree_g1 convert_unitree_json_to_lerobot.py / constants.py",
        "robot_type": schema.ROBOT_TYPE, "fps": cfg.fps, "cameras": cfg.cameras,
        "joint_mapping": schema.joint_mapping(),
        "action_semantics": {
            "arm": "q target of rt/arm_sdk (motion mode) or rt/lowcmd (debug mode): what the official "
                   "xr_teleoperate controller actually published (after its IK and velocity limiting), "
                   "position control (kp/kd set by the controller)",
            "hand": "q target of rt/dex3/*/cmd (position control)",
            "camera": "commanded [pitch, yaw] of the active camera neck, from PC2's head_pose_receiver",
            "alignment": cfg.action_alignment,
        },
        "camera_convention": {
            "order": ["pitch", "yaw"], "units": "radians",
            "frame": "relative to the neck center; yaw right-positive, pitch up-positive "
                     "(unitree_g1_urop). UAF's own camera used absolute encoder angles.",
        },
        "pressure": "press_sensor_state[0].pressure[0:12] per hand (as UAF); the raw store keeps all sensors",
        "observation_time": "as-of: newest sample received at or before the tick (no future data)",
        "task": {"goal": cfg.task, "desc": cfg.task_desc},
        "image": {"height": schema.IMAGE_H, "width": schema.IMAGE_W, "fit": cfg.images.fit},
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = config_mod.load(args.config)
    hub = StreamHub()
    images = ImageSource(cfg, hub)
    neck = dds = writer = None
    try:
        dds = DdsSources(cfg, hub)
        dds.start()
        images.start()
        neck = NeckSource(cfg, hub, clock=images.clock_estimate)
        neck.start()
        writer = LeRobotWriter(cfg.dataset_root, cfg.repo_id, cfg.task, cfg.fps, cfg.cameras,
                               image_fit=cfg.images.fit)
        write_dataset_info(writer, cfg)
        recorder = Recorder(cfg, hub, writer, emit=emit)

        commands: "queue.Queue[str]" = queue.Queue()
        stop = threading.Event()

        def read_commands() -> None:
            for line in sys.stdin:
                try:
                    commands.put(json.loads(line)["cmd"])
                except (ValueError, KeyError):
                    continue
            commands.put("quit")           # orchestrator went away: save and exit

        threading.Thread(target=read_commands, daemon=True).start()

        deadline = time.monotonic() + 30
        problems = recorder.readiness()
        while problems and time.monotonic() < deadline:
            time.sleep(0.5)
            problems = recorder.readiness()
        emit({"event": "ready" if not problems else "not_ready", "problems": problems})
        recorder.run(stop, commands)
        return 0
    finally:
        for resource in (images, neck, dds):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                emit({"event": "error", "msg": str(error)})


if __name__ == "__main__":
    raise SystemExit(main())
