"""Synthetic streams -> Recorder -> real LeRobotDataset -> LeRobot loader -> ACT batch (needs lerobot)."""

import importlib.util
import sys

import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset", reason="lerobot[dataset] not installed")
pytest.importorskip("torch")


def test_two_episodes_load_and_train_one_batch(tmp_path):
    from tests.e2e_synthetic import run

    out = run(tmp_path, seconds=1.0)
    assert [s["saved"] for s in out["summaries"]] == [True, True]
    assert out["summaries"][0]["frames"] != out["summaries"][1]["frames"]      # different lengths
    failed = {k: v["detail"] for k, v in out["validate"]["checks"].items() if not v["ok"]}
    assert not failed, failed
    assert out["act"]["ok"], out["act"]
    assert out["act_masked"]["ok"], out["act_masked"]          # sensor-ablation (masked) input also trains
    assert set(out["raw"]["episode_000001"]) == {"camera_state.parquet", "robot_action.parquet",
                                                 "robot_state.parquet", "tactile.parquet", "timestamps.parquet"}
