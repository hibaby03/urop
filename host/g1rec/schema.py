"""Master dataset layout, taken from the published UAF code (not guessed).

Source: kuehnrobin/UAF_unitree_g1
  unitree_lerobot/utils/convert_unitree_json_to_lerobot.py  (state/action order, features)
  unitree_lerobot/utils/constants.py                        (G1_DEX3_CONFIG: joint names, cameras)
and kuehnrobin/UAF_lerobot  src/lerobot/datasets/unified_ablation_framework.py
  (the index ranges FeatureFilter uses to mask sensors).

observation.state (110, float32), in this order:
    left_arm   q[7] dq[7] tau[7]                (0-20)
    right_arm  q[7] dq[7] tau[7]                (21-41)
    left_hand  q[7] dq[7] tau[7] pressure[12]   (42-74)
    right_hand q[7] dq[7] tau[7] pressure[12]   (75-107)
    camera     q[2] = [pitch, yaw]              (108-109)
action (30, float32):
    left_arm q[7] | right_arm q[7] | left_hand q[7] | right_hand q[7] | camera q[2]

Feature names are the UAF ones (``left_arm_qpos_0`` ...), because UAF's FeatureFilter
reads them from ``meta.features["observation.state"]["names"]``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

ROBOT_TYPE = "Unitree_G1_Dex3"
DEFAULT_FPS = 30

# Joint names in message-index order (UAF constants.G1_DEX3_CONFIG.motors).
ARM_JOINTS_LEFT = ("kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
                   "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw")
ARM_JOINTS_RIGHT = ("kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
                    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw")
HAND_JOINTS_LEFT = ("kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
                    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1")
HAND_JOINTS_RIGHT = ("kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
                     "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1")
CAMERA_JOINTS = ("kCameraPitch", "kCameraYaw")

# rt/lowstate and rt/lowcmd motor indices of the arm joints (G1_29_JointArmIndex).
G1_29_ARM_MOTOR_OFFSET = 15
ARM_MOTOR_INDICES_LEFT = tuple(range(15, 22))
ARM_MOTOR_INDICES_RIGHT = tuple(range(22, 29))
# rt/dex3/{left,right}/{state,cmd}: motor index i is joint i of HAND_JOINTS_*.
HAND_MOTOR_INDICES = tuple(range(7))

NUM_ARM = 7
NUM_HAND = 7
PRESSURE_DIM = 12           # UAF: press_sensor_state[0].pressure[0:12] per hand
NUM_CAMERA = 2

# (part, [(field, size), ...]) in state order.
STATE_LAYOUT = (
    ("left_arm", (("qpos", 7), ("qvel", 7), ("torque", 7))),
    ("right_arm", (("qpos", 7), ("qvel", 7), ("torque", 7))),
    ("left_hand", (("qpos", 7), ("qvel", 7), ("torque", 7), ("pressure", 12))),
    ("right_hand", (("qpos", 7), ("qvel", 7), ("torque", 7), ("pressure", 12))),
    ("camera", (("qpos", 2),)),
)
ACTION_LAYOUT = (("left_arm", 7), ("right_arm", 7), ("left_hand", 7), ("right_hand", 7), ("camera", 2))

STATE_DIM = sum(size for _, fields in STATE_LAYOUT for _, size in fields)   # 110
ACTION_DIM = sum(size for _, size in ACTION_LAYOUT)                          # 30

# UAF camera keys; ``color_N`` is the index UAF's recorder used for the same camera.
CAMERAS = {
    "cam_left_high": "color_0", "cam_right_high": "color_1",
    "cam_left_wrist": "color_2", "cam_right_wrist": "color_3",
    "cam_left_active": "color_4", "cam_right_active": "color_5",
}
REQUIRED_CAMERAS = ("cam_left_active", "cam_right_active")
OPTIONAL_CAMERAS = ("cam_left_wrist", "cam_right_wrist", "cam_left_high", "cam_right_high")
IMAGE_H, IMAGE_W = 480, 640


class SchemaError(ValueError):
    """A sample lacks a value the master dataset requires."""


def state_names() -> list[str]:
    return [f"{part}_{field}_{i}" for part, fields in STATE_LAYOUT
            for field, size in fields for i in range(size)]


def action_names() -> list[str]:
    return [f"{part}_qpos_{i}" for part, size in ACTION_LAYOUT for i in range(size)]


def state_slices() -> dict[str, slice]:
    """{"left_arm_qpos": slice(0, 7), ..., "camera_qpos": slice(108, 110)} (UAF FeatureFilter keys)."""

    slices, start = {}, 0
    for part, fields in STATE_LAYOUT:
        for field, size in fields:
            slices[f"{part}_{field}"] = slice(start, start + size)
            start += size
    return slices


def joint_mapping() -> dict[str, Any]:
    """Joint names and message indices, stored in the dataset metadata."""

    return {
        "state_layout": state_names(),
        "action_layout": action_names(),
        "joint_names": {
            "left_arm": list(ARM_JOINTS_LEFT), "right_arm": list(ARM_JOINTS_RIGHT),
            "left_hand": list(HAND_JOINTS_LEFT), "right_hand": list(HAND_JOINTS_RIGHT),
            "camera": list(CAMERA_JOINTS),
        },
        "message_indices": {
            "left_arm": {"topic": "rt/lowstate | rt/lowcmd | rt/arm_sdk", "motor_index": list(ARM_MOTOR_INDICES_LEFT)},
            "right_arm": {"topic": "rt/lowstate | rt/lowcmd | rt/arm_sdk", "motor_index": list(ARM_MOTOR_INDICES_RIGHT)},
            "left_hand": {"topic": "rt/dex3/left/state | cmd", "motor_index": list(HAND_MOTOR_INDICES)},
            "right_hand": {"topic": "rt/dex3/right/state | cmd", "motor_index": list(HAND_MOTOR_INDICES)},
        },
        "slices": {k: [v.start, v.stop] for k, v in state_slices().items()},
    }


def lerobot_features(cameras: Sequence[str] = REQUIRED_CAMERAS, mode: str = "video") -> dict[str, dict]:
    """Feature dict for LeRobotDataset.create (same shapes/names/dtypes as UAF's converter)."""

    unknown = [c for c in cameras if c not in CAMERAS]
    if unknown:
        raise ValueError(f"unknown cameras {unknown}; choose from {list(CAMERAS)}")
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": state_names()},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": action_names()},
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": mode, "shape": (3, IMAGE_H, IMAGE_W), "names": ["channels", "height", "width"]}
    return features


def build_state(parts: dict) -> list[float]:
    """Flatten {"left_arm": {"qpos": [...], "qvel": [...], "torque": [...]}, ...} to 110 floats."""

    import math

    out: list[float] = []
    for part, fields in STATE_LAYOUT:
        block = parts.get(part)
        if block is None:
            raise SchemaError(f"state part {part!r} is missing")
        for field, size in fields:
            values = [float(v) for v in block.get(field, [])]
            if len(values) != size:
                raise SchemaError(f"{part}.{field} has {len(values)} values, expected {size}")
            if not all(math.isfinite(v) for v in values):
                raise SchemaError(f"{part}.{field} has a non-finite value")
            out += values
    return out


def build_action(parts: dict) -> list[float]:
    import math

    out: list[float] = []
    for part, size in ACTION_LAYOUT:
        values = [float(v) for v in (parts.get(part) or {}).get("qpos", [])]
        if len(values) != size:
            raise SchemaError(f"action {part} has {len(values)} values, expected {size}")
        if not all(math.isfinite(v) for v in values):
            raise SchemaError(f"action {part} has a non-finite value")
        out += values
    return out


def neck_to_camera_qpos(yaw_pitch_rad: Sequence[float]) -> list[float]:
    """Active-camera neck [yaw, pitch] (unitree_g1_urop) -> UAF camera order [pitch, yaw]."""

    yaw, pitch = yaw_pitch_rad
    return [float(pitch), float(yaw)]


def state_mask(include: Optional[dict] = None) -> Any:
    """Boolean mask over the 110-d state for sensor ablation.

    ``include`` keys (all default True): joint_positions, arm_velocities, hand_velocities,
    arm_torques, hand_torques, pressure, camera_positions, left_arm/right_arm/left_hand/right_hand.
    Mirrors UAF's FeatureFilter, so an ablation can be reproduced from the same master dataset.
    """

    import numpy as np

    opts = {"joint_positions": True, "arm_velocities": True, "hand_velocities": True,
            "arm_torques": True, "hand_torques": True, "pressure": True, "camera_positions": True,
            "left_arm": True, "right_arm": True, "left_hand": True, "right_hand": True}
    opts.update(include or {})
    unknown = set(opts) - {"joint_positions", "arm_velocities", "hand_velocities", "arm_torques",
                           "hand_torques", "pressure", "camera_positions",
                           "left_arm", "right_arm", "left_hand", "right_hand"}
    if unknown:
        raise ValueError(f"unknown mask options {sorted(unknown)}")
    mask = np.zeros(STATE_DIM, dtype=bool)
    for key, sl in state_slices().items():
        part, field = key.rsplit("_", 1)
        if part == "camera":
            keep = opts["camera_positions"]
        elif not opts[part]:
            keep = False
        elif field == "qpos":
            keep = opts["joint_positions"]
        elif field == "qvel":
            keep = opts["arm_velocities" if "arm" in part else "hand_velocities"]
        elif field == "torque":
            keep = opts["arm_torques" if "arm" in part else "hand_torques"]
        else:  # pressure
            keep = opts["pressure"]
        mask[sl] = keep
    return mask
