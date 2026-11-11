"""Verification of a recorded master dataset with the LeRobot version installed here.

validate_dataset()  checks 1-9 of the spec: it loads the dataset with LeRobotDataset and checks
                    counts, shapes, joint mapping, video decoding, timestamps, action-chunk
                    padding at episode ends and NaN/zero-filled data.
act_batch_check()   check 10: builds an ACT policy from the dataset's own features, loads a real
                    batch (with action chunks) through LeRobot's DataLoader and runs forward+backward.

Both return {"ok": bool, "checks": {name: {"ok": bool, "detail": str}}}. Nothing is reported as
passed unless it was actually run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from . import schema


def _check(checks: dict, name: str, ok: bool, detail: str = "") -> bool:
    checks[name] = {"ok": bool(ok), "detail": detail}
    return bool(ok)


def _dataset_to_policy_features():
    """Its module moved between LeRobot versions (0.4.x: datasets.utils, 0.7: utils.feature_utils)."""

    try:
        from lerobot.utils.feature_utils import dataset_to_policy_features
    except ImportError:
        from lerobot.datasets.utils import dataset_to_policy_features
    return dataset_to_policy_features


def _load(root: Path, repo_id: str, **kwargs):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, **kwargs)


def validate_dataset(root, repo_id: str, fps: int = schema.DEFAULT_FPS,
                     cameras: Sequence[str] = schema.REQUIRED_CAMERAS, chunk_size: int = 10) -> dict:
    import numpy as np

    root = Path(root).expanduser().resolve()
    checks: dict = {}
    try:
        ds = _load(root, repo_id)
    except Exception as error:
        _check(checks, "1 load", False, f"{type(error).__name__}: {error}")
        return {"ok": False, "checks": checks}
    _check(checks, "1 load", True, f"LeRobotDataset at {root}")

    episode_index = np.asarray(ds.hf_dataset["episode_index"])
    frame_index = np.asarray(ds.hf_dataset["frame_index"])
    timestamp = np.asarray(ds.hf_dataset["timestamp"], dtype=np.float64)
    episodes = sorted(set(int(e) for e in episode_index))
    total_eps, total_frames = ds.meta.total_episodes, ds.meta.total_frames
    ok = (len(episodes) == total_eps and len(episode_index) == total_frames == len(ds)
          and all(int((episode_index == e).sum()) > 0 for e in episodes))
    _check(checks, "2 episodes/frames", ok, f"{total_eps} episodes, {total_frames} frames")

    expected = schema.lerobot_features(cameras)
    problems = []
    for key, feature in expected.items():
        have = ds.meta.features.get(key)
        if have is None:
            problems.append(f"missing {key}")
        elif tuple(have["shape"]) != tuple(feature["shape"]):
            problems.append(f"{key} shape {tuple(have['shape'])} != {tuple(feature['shape'])}")
        elif have["dtype"] != feature["dtype"]:
            problems.append(f"{key} dtype {have['dtype']} != {feature['dtype']}")
    _check(checks, "3 observation shapes", not problems, "; ".join(problems) or "state 110, images 3x480x640 video")

    names = ds.meta.features["action"]["names"]
    info_path = root / "meta" / "g1_master_info.json"
    mapping_ok = list(names) == schema.action_names() and \
        list(ds.meta.features["observation.state"]["names"]) == schema.state_names()
    if info_path.is_file():
        mapping_ok = mapping_ok and "joint_mapping" in json.loads(info_path.read_text())
    else:
        mapping_ok = False
    _check(checks, "4 action dim / joint mapping", mapping_ok and schema.ACTION_DIM == len(names),
           f"action {len(names)}-d; joint mapping stored in meta/g1_master_info.json")

    states = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float64)
    actions = np.asarray(ds.hf_dataset["action"], dtype=np.float64)
    finite = np.isfinite(states).all() and np.isfinite(actions).all()
    zero_rows = int((~states.any(axis=1)).sum())
    _check(checks, "8 missing sensor data", finite and zero_rows == 0,
           f"non-finite: {not finite}, all-zero state rows (zero-filled): {zero_rows}")

    decode_problems = []
    for ep in episodes:
        rows = np.nonzero(episode_index == ep)[0]
        for idx in sorted({int(rows[0]), int(rows[-1])}):
            item = ds[idx]
            for camera in cameras:
                image = item.get(f"observation.images.{camera}")
                if image is None or tuple(image.shape[-2:]) != (schema.IMAGE_H, schema.IMAGE_W) \
                        or not bool(np.isfinite(image.numpy()).all()):
                    decode_problems.append(f"episode {ep} frame {idx}: bad image {camera}")
    _check(checks, "5 video decoding", not decode_problems, "; ".join(decode_problems[:3]) or "first/last frame of every episode decoded")

    ts_problems = []
    for ep in episodes:
        rows = np.nonzero(episode_index == ep)[0]
        fi, ts = frame_index[rows], timestamp[rows]
        if not (fi == np.arange(len(rows))).all():
            ts_problems.append(f"episode {ep}: frame_index not consecutive")
        if not np.allclose(ts, fi / fps, atol=1e-4):
            ts_problems.append(f"episode {ep}: timestamps do not match frame_index/{fps}")
    _check(checks, "6 timestamps", not ts_problems, "; ".join(ts_problems) or f"timestamp == frame_index/{fps} in every episode")

    boundary_problems = []
    try:
        chunked = _load(root, repo_id, delta_timestamps={"action": [i / fps for i in range(chunk_size)]})
        for ep in episodes:
            rows = np.nonzero(episode_index == ep)[0]
            length = len(rows)
            for local in sorted({0, max(0, length - chunk_size), length - 1}):
                item = chunked[int(rows[local])]
                pad = item["action_is_pad"].numpy()
                want = np.arange(chunk_size) + local >= length
                if not (pad == want).all():
                    boundary_problems.append(f"episode {ep} frame {local}: pad {pad.astype(int).tolist()} != {want.astype(int).tolist()}")
        detail = f"chunk {chunk_size}: padding starts exactly at each episode end"
    except Exception as error:
        boundary_problems.append(f"{type(error).__name__}: {error}")
        detail = ""
    _check(checks, "7 chunks stay inside episodes", not boundary_problems, "; ".join(boundary_problems[:3]) or detail)

    try:
        from lerobot.configs.types import FeatureType
        dataset_to_policy_features = _dataset_to_policy_features()

        features = dataset_to_policy_features(ds.meta.features)
        types = {k: v.type for k, v in features.items()}
        mapped = (types.get("action") == FeatureType.ACTION and types.get("observation.state") == FeatureType.STATE
                  and all(types.get(f"observation.images.{c}") == FeatureType.VISUAL for c in cameras))
        _check(checks, "9 ACT feature mapping", mapped, str({k: v.name for k, v in types.items()}))
    except Exception as error:
        _check(checks, "9 ACT feature mapping", False, f"{type(error).__name__}: {error}")

    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def act_batch_check(root, repo_id: str, fps: int = schema.DEFAULT_FPS,
                    cameras: Sequence[str] = schema.REQUIRED_CAMERAS, chunk_size: int = 10,
                    mask: Optional[Any] = None, batch_size: int = 2) -> dict:
    """Real batch -> ACT forward + backward. ``mask`` (bool, 110) emulates a sensor-ablation run."""

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    dataset_to_policy_features = _dataset_to_policy_features()

    root = Path(root).expanduser().resolve()
    checks: dict = {}
    try:
        ds = _load(root, repo_id, delta_timestamps={"action": [i / fps for i in range(chunk_size)]})
        features = dataset_to_policy_features(ds.meta.features)
        output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
        input_features = {k: v for k, v in features.items() if k not in output_features}
        keep = [f"observation.images.{c}" for c in cameras]
        input_features = {k: v for k, v in input_features.items()
                          if k == "observation.state" or k in keep}
        stats = {k: {s: np.asarray(x) for s, x in v.items()} for k, v in ds.meta.stats.items()}
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(int(mask.sum()),))
            stats["observation.state"] = {s: x[mask] if getattr(x, "shape", ()) == (schema.STATE_DIM,) else x
                                          for s, x in stats["observation.state"].items()}
        config = ACTConfig(input_features=input_features, output_features=output_features,
                           chunk_size=chunk_size, n_action_steps=chunk_size, device="cpu",
                           pretrained_backbone_weights=None, dim_model=64, n_heads=4, dim_feedforward=128,
                           n_encoder_layers=1, n_decoder_layers=1, latent_dim=8, use_vae=True)
        policy = ACTPolicy(config)
        preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
        batch = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False)))
        if mask is not None:
            batch["observation.state"] = batch["observation.state"][:, torch.as_tensor(mask)]
        batch = preprocessor(batch)
        policy.train()
        loss, info = policy.forward(batch)
        loss.backward()
        ok = bool(torch.isfinite(loss))
        _check(checks, "10 ACT forward/backward", ok,
               f"batch {batch_size}, state {tuple(batch['observation.state'].shape)}, "
               f"action {tuple(batch['action'].shape)}, loss {float(loss.detach()):.4f}")
    except Exception as error:
        _check(checks, "10 ACT forward/backward", False, f"{type(error).__name__}: {error}")
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def report(cfg, act_check: bool = True) -> dict:
    """Short strings for the orchestrator's episode summary."""

    try:
        import lerobot  # noqa: F401
    except Exception as error:
        text = f"NOT VERIFIED (lerobot not importable: {error})"
        return {"dataset": text, "act": text}
    result = validate_dataset(cfg.dataset_root, cfg.repo_id, cfg.fps, cfg.cameras)
    failed = [f"{k}: {v['detail']}" for k, v in result["checks"].items() if not v["ok"]]
    dataset = "VALID" if result["ok"] else "INVALID (" + "; ".join(failed) + ")"
    if not act_check:
        return {"dataset": dataset, "act": "NOT RUN (--no-act-check)"}
    act = act_batch_check(cfg.dataset_root, cfg.repo_id, cfg.fps, cfg.cameras)
    return {"dataset": dataset, "act": "PASS" if act["ok"] else "FAIL (" + act["checks"]["10 ACT forward/backward"]["detail"] + ")"}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a recorded master dataset with LeRobot")
    parser.add_argument("root")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--fps", type=int, default=schema.DEFAULT_FPS)
    parser.add_argument("--cameras", nargs="*", default=list(schema.REQUIRED_CAMERAS))
    parser.add_argument("--no-act", action="store_true")
    args = parser.parse_args(argv)
    repo_id = args.repo_id or f"local/{Path(args.root).name}"
    result = validate_dataset(args.root, repo_id, args.fps, args.cameras)
    if not args.no_act:
        act = act_batch_check(args.root, repo_id, args.fps, args.cameras)
        result["checks"].update(act["checks"])
        result["ok"] = result["ok"] and act["ok"]
    for name, check in result["checks"].items():
        print(f"[{'PASS' if check['ok'] else 'FAIL'}] {name}: {check['detail']}")
    print("OVERALL:", "PASS" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
