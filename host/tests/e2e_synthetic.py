"""End-to-end run without hardware: synthetic streams -> Recorder -> real LeRobotDataset -> verify.

python -m tests.e2e_synthetic [--seconds 3]   (run from the project root, needs lerobot)
Writes into a temporary directory. Two episodes of different length are recorded.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from g1rec import config as config_mod
from g1rec import verify
from g1rec.lerobot_writer import LeRobotWriter
from g1rec.recorder import Recorder
from g1rec.recorder_proc import write_dataset_info
from g1rec.sync import StreamHub

from tests.synthetic import SyntheticRobot


def run(tmp: Path, seconds: float = 3.0) -> dict:
    cfg = config_mod.from_dict({"dataset_root": str(tmp / "master_dataset"), "raw_root": str(tmp / "raw"),
                                "repo_id": "local/e2e", "min_free_gb": 0.0})
    hub = StreamHub()
    robot = SyntheticRobot(hub).start()
    events = []
    writer = LeRobotWriter(cfg.dataset_root, cfg.repo_id, cfg.task, cfg.fps, cfg.cameras)
    write_dataset_info(writer, cfg)
    recorder = Recorder(cfg, hub, writer, emit=events.append)
    try:
        time.sleep(0.6)                                         # let every stream produce data
        assert recorder.readiness() == [], recorder.readiness()
        summaries = []
        for length in (seconds, seconds * 1.7):                 # different episode lengths
            assert recorder.start_episode()
            end = time.monotonic() + length
            while time.monotonic() < end:
                recorder.step()
                time.sleep(0.004)
            summaries.append(recorder.stop_episode())
    finally:
        robot.stop()
        writer.close()
    cfg_args = (cfg.dataset_root, cfg.repo_id, cfg.fps, cfg.cameras)
    result = verify.validate_dataset(*cfg_args)
    act = verify.act_batch_check(*cfg_args)
    masked = verify.act_batch_check(*cfg_args, mask=verify.schema.state_mask(
        {"arm_torques": False, "hand_torques": False, "pressure": False}))
    raw = {p.name: sorted(f.name for f in p.iterdir()) for p in sorted(Path(cfg.raw_root).glob("episode_*"))}
    return {"summaries": summaries, "validate": result, "act": act, "act_masked": masked, "raw": raw,
            "meta": sorted(f.name for f in (Path(cfg.dataset_root) / "meta").iterdir())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="g1rec_e2e_") as tmp:
        out = run(Path(tmp), args.seconds)
    for s in out["summaries"]:
        print({k: s[k] for k in ("raw_episode", "saved", "frames", "duration_s", "dropped_ticks",
                                 "robot_states", "actions", "tactile_samples", "neck_samples")})
    for title in ("validate", "act", "act_masked"):
        print(f"--- {title}: {'PASS' if out[title]['ok'] else 'FAIL'}")
        for name, check in out[title]["checks"].items():
            print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {name}: {check['detail']}")
    print("raw:", json.dumps(out["raw"]))
    print("meta:", out["meta"])
