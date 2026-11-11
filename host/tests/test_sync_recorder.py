import threading
import time

import pytest

from g1rec import config as config_mod
from g1rec import schema
from g1rec.recorder import Recorder, build_state_parts
from g1rec.sync import Sample, StreamHub, gather

from tests import synthetic as syn


def test_asof_never_returns_a_future_sample():
    hub = StreamHub()
    for t in (10, 20, 30):
        hub.publish(Sample("s", t, recv_ns=t))
    assert hub.asof("s", 25).value == 20
    assert hub.asof("s", 30).value == 30
    assert hub.asof("s", 9) is None


def test_first_at_or_after_is_the_next_command_within_the_gap():
    hub = StreamHub()
    for t in (10, 40, 70):
        hub.publish(Sample("c", t, recv_ns=t))
    assert hub.first_at_or_after("c", 11, max_gap_ns=50).value == 40
    assert hub.first_at_or_after("c", 41, max_gap_ns=10) is None      # next is 70: too far
    assert hub.first_at_or_after("c", 71, max_gap_ns=50) is None


def test_gather_reports_missing_and_stale():
    hub = StreamHub()
    hub.publish(Sample("a", 1, recv_ns=0))
    hub.publish(Sample("b", 1, recv_ns=int(900e6)))
    samples, problems = gather(hub, int(1e9), {"a": 100.0, "b": 100.0, "c": 100.0})
    assert set(samples) == {"b"}
    assert any("a: stale" in p for p in problems) and any("c: no data" in p for p in problems)


def test_sequence_gaps_are_counted():
    hub = StreamHub()
    for seq in (1, 2, 5, 6):
        hub.publish(Sample("n", 0, recv_ns=seq, seq=seq))
    assert hub.gaps["n"] == 2


def test_image_history_is_bounded():
    hub = StreamHub()
    hub.limit_history("image", 3)
    for i in range(10):
        hub.publish(Sample("image", i, recv_ns=i))
    assert hub.asof("image", 9).value == 9 and hub.asof("image", 7).value == 7
    assert hub.asof("image", 5) is None                  # only the newest 3 frames are kept


class FakeWriter:
    def __init__(self):
        self.frames, self.error = [], None
        self.queue = type("Q", (), {"qsize": lambda self: 0})()
        self.saved = 0

    def begin_episode(self):
        return True

    def add_frame(self, images, state, action):
        self.frames.append((images, state, action))

    def end_episode(self):
        self.saved += 1
        return type("H", (), {"wait": lambda self, t=None: True})()

    def total_episodes(self):
        return self.saved

    def write_sidecar(self, *a):
        pass


class FakeRaw:
    def __init__(self, root, index, flush_s):
        self.counts, self.error = {}, None

    def on_sample(self, sample):
        self.counts[sample.stream] = self.counts.get(sample.stream, 0) + 1

    def close(self):
        return {}


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def make(tmp_path, **cfg_over):
    cfg = config_mod.from_dict({"raw_root": str(tmp_path / "raw"), "dataset_root": str(tmp_path / "ds"),
                                "min_free_gb": 0.0, **cfg_over})
    hub, clock, writer = StreamHub(), Clock(), FakeWriter()
    return cfg, hub, clock, writer, Recorder(cfg, hub, writer, raw_factory=FakeRaw, now_ns=clock)


def feed_all(hub, t_ns, cmd_offset_ns=0):
    t = t_ns / 1e9
    hub.publish(Sample("arm_state", syn.arm_state_value(t), t_ns))
    for side in ("left", "right"):
        hub.publish(Sample(f"hand_state_{side}", syn.hand_state_value(t), t_ns))
        hub.publish(Sample(f"tactile_{side}", syn.tactile_value(t), t_ns))
        hub.publish(Sample(f"hand_cmd_{side}", {"qpos": [0.5] * 7}, t_ns + cmd_offset_ns))
    hub.publish(Sample("arm_cmd", syn.arm_cmd_value(t), t_ns + cmd_offset_ns))
    hub.publish(Sample("neck", {"present_deg": [10.0, 5.0], "command_deg": [11.0, 6.0], "torque_enabled": True}, t_ns))
    hub.publish(Sample("image", syn.image_frame(1), t_ns))


def test_complete_tick_makes_a_frame_and_camera_is_pitch_yaw(tmp_path):
    cfg, hub, clock, writer, rec = make(tmp_path)
    feed_all(hub, 1_000_000_000, cmd_offset_ns=5_000_000)
    rec.start_episode()
    assert rec.process_tick(1_000_000_000 + 1_000_000)        # state <= t, command at/after t
    images, state, action = writer.frames[0]
    assert len(state) == 110 and len(action) == 30
    import math
    assert state[108:] == pytest.approx([math.radians(5.0), math.radians(10.0)])     # [pitch, yaw]
    assert action[28:] == pytest.approx([math.radians(6.0), math.radians(11.0)])
    assert set(images) == {"cam_left_active", "cam_right_active"}


@pytest.mark.parametrize("missing", ["arm_state", "tactile_left", "neck", "image", "hand_state_right"])
def test_missing_stream_drops_the_tick_instead_of_zero_filling(tmp_path, missing):
    cfg, hub, clock, writer, rec = make(tmp_path)
    t = 1_000_000_000
    feed_all(hub, t, cmd_offset_ns=5_000_000)
    hub._streams.pop(missing)
    rec.start_episode()
    assert not rec.process_tick(t + 1_000_000)
    assert writer.frames == []
    assert any(missing in k for k in rec._episode["dropped"])


def test_stale_stream_drops_the_tick(tmp_path):
    cfg, hub, clock, writer, rec = make(tmp_path)
    feed_all(hub, 1_000_000_000, cmd_offset_ns=5_000_000)
    rec.start_episode()
    assert not rec.process_tick(1_000_000_000 + int(2e9))     # 2 s later: everything is stale
    assert writer.frames == []


def test_command_issued_before_the_tick_is_not_used_as_its_action(tmp_path):
    cfg, hub, clock, writer, rec = make(tmp_path)
    t = 2_000_000_000
    feed_all(hub, t, cmd_offset_ns=-10_000_000)               # commands published BEFORE the tick
    rec.start_episode()
    assert not rec.process_tick(t + 1_000_000)
    assert any("no command" in k for k in rec._episode["dropped"])


def test_future_observation_does_not_enter_the_tick(tmp_path):
    cfg, hub, clock, writer, rec = make(tmp_path)
    t = 3_000_000_000
    feed_all(hub, t, cmd_offset_ns=5_000_000)
    rec.start_episode()
    hub.publish(Sample("arm_state", syn.arm_state_value(99.0), t + 10_000_000))   # arrives after the tick
    assert rec.process_tick(t + 1_000_000)
    first_q = writer.frames[0][1][0]
    assert first_q == pytest.approx(syn.arm_state_value(t / 1e9)["left_arm"]["qpos"][0])


def test_step_processes_ticks_one_period_late(tmp_path):
    cfg, hub, clock, writer, rec = make(tmp_path)
    clock.t = 10_000_000_000
    rec.start_episode()
    for k in range(200):
        clock.t += 5_000_000
        feed_all(hub, clock.t, cmd_offset_ns=2_000_000)
        rec.step()
    assert 20 <= len(writer.frames) <= 31                      # 1 s at 30 Hz minus the delay
    summary = rec.stop_episode()
    assert summary["saved"] and summary["frames"] == len(writer.frames)


def test_actual_robot_values_not_commands_are_the_state(tmp_path):
    state_parts = None
    hub = StreamHub()
    feed_all(hub, 1_000_000_000)
    samples = {s: hub.latest(s) for s in ("arm_state", "hand_state_left", "hand_state_right",
                                          "tactile_left", "tactile_right", "neck")}
    state_parts = build_state_parts(samples)
    assert state_parts["left_hand"]["pressure"] == samples["tactile_left"].value["pressure"][0][:12]
