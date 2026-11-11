"""The layout must equal the published UAF one (UAF_unitree_g1 converter + UAF_lerobot FeatureFilter)."""

import numpy as np
import pytest

from g1rec import schema


def test_dimensions():
    assert schema.STATE_DIM == 110 and schema.ACTION_DIM == 30
    assert len(schema.state_names()) == 110 and len(schema.action_names()) == 30


def test_state_slices_equal_the_uaf_filter_ranges():
    # UAF_lerobot unified_ablation_framework.FeatureFilter index table
    uaf = {"left_arm_qpos": (0, 7), "left_arm_qvel": (7, 14), "left_arm_torque": (14, 21),
           "right_arm_qpos": (21, 28), "right_arm_qvel": (28, 35), "right_arm_torque": (35, 42),
           "left_hand_qpos": (42, 49), "left_hand_qvel": (49, 56), "left_hand_torque": (56, 63),
           "left_hand_pressure": (63, 75), "right_hand_qpos": (75, 82), "right_hand_qvel": (82, 89),
           "right_hand_torque": (89, 96), "right_hand_pressure": (96, 108), "camera_qpos": (108, 110)}
    got = {k: (v.start, v.stop) for k, v in schema.state_slices().items()}
    assert got == uaf


def test_feature_names_are_the_uaf_names():
    names = schema.state_names()
    assert names[0] == "left_arm_qpos_0" and names[7] == "left_arm_qvel_0" and names[14] == "left_arm_torque_0"
    assert names[63] == "left_hand_pressure_0" and names[96] == "right_hand_pressure_0"
    assert names[108] == "camera_qpos_0" and names[109] == "camera_qpos_1"
    assert schema.action_names()[14] == "left_hand_qpos_0" and schema.action_names()[28] == "camera_qpos_0"


def test_image_features_follow_uaf():
    f = schema.lerobot_features()
    assert set(k for k in f if k.startswith("observation.images.")) == {
        "observation.images.cam_left_active", "observation.images.cam_right_active"}
    image = f["observation.images.cam_left_active"]
    assert image["dtype"] == "video" and image["shape"] == (3, 480, 640)
    assert image["names"] == ["channels", "height", "width"]


def test_hand_joint_order_differs_per_hand():
    assert schema.HAND_JOINTS_LEFT[3:5] == ("kLeftHandMiddle0", "kLeftHandMiddle1")
    assert schema.HAND_JOINTS_RIGHT[3:5] == ("kRightHandIndex0", "kRightHandIndex1")
    mapping = schema.joint_mapping()
    assert mapping["message_indices"]["left_arm"]["motor_index"] == list(range(15, 22))
    assert mapping["message_indices"]["right_arm"]["motor_index"] == list(range(22, 29))


def parts(value=0.0):
    block = {"qpos": [value] * 7, "qvel": [value] * 7, "torque": [value] * 7}
    return {"left_arm": dict(block), "right_arm": dict(block),
            "left_hand": {**block, "pressure": [value] * 12},
            "right_hand": {**block, "pressure": [value] * 12}, "camera": {"qpos": [value, value]}}


def test_build_state_order():
    p = parts(0.0)
    p["left_hand"]["pressure"] = list(range(12))
    p["camera"]["qpos"] = [7.0, 8.0]
    state = schema.build_state(p)
    assert len(state) == 110 and state[63:75] == list(map(float, range(12))) and state[108:] == [7.0, 8.0]


def test_missing_or_bad_values_are_rejected_not_filled():
    p = parts()
    del p["right_hand"]
    with pytest.raises(schema.SchemaError):
        schema.build_state(p)
    p = parts()
    p["left_arm"]["torque"] = []
    with pytest.raises(schema.SchemaError):
        schema.build_state(p)
    p = parts()
    p["camera"]["qpos"] = [float("nan"), 0.0]
    with pytest.raises(schema.SchemaError):
        schema.build_state(p)


def test_camera_order_is_pitch_then_yaw():
    assert schema.neck_to_camera_qpos([0.3, -0.1]) == [-0.1, 0.3]


def test_sensor_masks():
    assert int(schema.state_mask().sum()) == 110
    no_torque = schema.state_mask({"arm_torques": False, "hand_torques": False})
    assert int(no_torque.sum()) == 110 - 14 - 14
    only_pos = schema.state_mask({"arm_velocities": False, "hand_velocities": False, "arm_torques": False,
                                  "hand_torques": False, "pressure": False})
    assert int(only_pos.sum()) == 28 + 2          # positions + camera joints (UAF "positions only")
    arms_only = schema.state_mask({"left_hand": False, "right_hand": False, "camera_positions": False})
    assert int(arms_only.sum()) == 42
    with pytest.raises(ValueError):
        schema.state_mask({"nope": True})
