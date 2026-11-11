"""Subscriber-only guarantees and configuration checks."""

import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from g1rec import config as config_mod
from g1rec.config import ProcessSpec
from g1rec.orchestrator import ManagedProcess

ROOT = Path(__file__).resolve().parent.parent


def test_the_recorder_side_has_no_publisher_or_command_sender():
    """sources.py/recorder*.py may subscribe only: no ChannelPublisher, no Write(), no send of commands."""
    forbidden = re.compile(r"ChannelPublisher|\.Write\(|CmdPublisher|motor_cmd\[[^\]]+\]\s*\.(q|dq|tau|kp|kd|mode)\s*=|"
                           r"sendto\(|MotionSwitcher|ArmController|Enter_Debug_Mode")
    for name in ("sources.py", "recorder.py", "recorder_proc.py", "sync.py", "raw_store.py", "lerobot_writer.py"):
        text = (ROOT / "g1rec" / name).read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", '"', "'")))
        hits = forbidden.findall(code)
        assert not hits, f"{name}: {hits}"


def test_sources_only_read_from_the_existing_projects():
    """sources.py is the only module that touches the existing projects (read-only imports)."""
    text = (ROOT / "g1rec" / "sources.py").read_text()
    writes = re.findall(r"open\([^)]*[\"'][wax]b?[\"']|write_text|write_bytes|mkdir|unlink|rmtree|shutil\.|os\.(remove|rename)", text)
    assert not writes, writes


def test_config_requires_the_active_stereo_pair():
    with pytest.raises(ValueError, match="cam_right_active"):
        config_mod.from_dict({"cameras": ["cam_left_active"]})
    with pytest.raises(ValueError):
        config_mod.from_dict({"nonsense": 1})
    cfg = config_mod.from_dict({"cameras": ["cam_left_active", "cam_right_active", "cam_left_wrist"], "fps": 20})
    assert cfg.fps == 20 and cfg.max_age_ms["image"] == 500.0


def test_launch_mode_needs_a_command():
    with pytest.raises(ValueError):
        config_mod.from_dict({"processes": {"xr_teleop": {"mode": "launch"}}})
    cfg = config_mod.from_dict({"processes": {"active_camera": {"mode": "external"}}})
    assert cfg.process("active_camera").mode == "external"


def test_shipped_config_loads():
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    assert cfg.fps == 30 and "xr_teleop" in cfg.processes


def test_managed_process_ready_text_stdin_closed_and_stop(tmp_path):
    spec = ProcessSpec(name="dummy", cmd=[sys.executable, "-u", "-c",
                       "import sys,time; print('booting'); print('SERVER READY'); "
                       "print('stdin:', repr(sys.stdin.read())); time.sleep(60)"],
                       ready_text="SERVER READY", ready_timeout_s=10, stop_timeout_s=3)
    managed = ManagedProcess(spec, tmp_path)
    managed.start()
    try:
        assert managed.wait_ready()
        assert managed.alive()
        time.sleep(0.3)
        assert "stdin: ''" in "\n".join(managed.tail)         # child cannot read the keyboard
    finally:
        managed.stop()
    assert not managed.alive()
    assert "SERVER READY" in (tmp_path / "dummy.log").read_text()


def test_process_that_dies_before_ready_is_reported(tmp_path):
    spec = ProcessSpec(name="bad", cmd=[sys.executable, "-c", "print('boom')"], ready_text="READY", ready_timeout_s=5)
    managed = ManagedProcess(spec, tmp_path)
    managed.start()
    assert not managed.wait_ready()
    assert "boom" in "\n".join(managed.tail)


def test_ssh_wrapper_quotes_the_remote_command():
    spec = ProcessSpec(name="pc2", cmd=["python3", "pc2/zed teleimager_server.py"], ssh="unitree@192.168.123.164")
    argv = ManagedProcess(spec, Path("/tmp")).argv()
    assert argv[:3] == ["ssh", "-tt", "unitree@192.168.123.164"] and "'pc2/zed teleimager_server.py'" in argv[-1]


def test_ssh_command_changes_to_the_remote_directory_first():
    spec = ProcessSpec(name="pc2", cmd=["python3", "pc2/head_pose_receiver.py", "--yaw-limit-deg", "15"],
                       ssh="192.168.123.164", cwd="~/unitree_g1_urop")
    argv = ManagedProcess(spec, Path("/tmp")).argv()
    assert argv == ["ssh", "-tt", "192.168.123.164", "--",
                    "cd ~/unitree_g1_urop && python3 pc2/head_pose_receiver.py --yaw-limit-deg 15"]


def test_local_tilde_is_expanded_and_ready_text_accepts_alternatives(tmp_path):
    import os
    spec = ProcessSpec(name="x", cmd=["python", "~/a.py"])
    assert ManagedProcess(spec, tmp_path).argv()[1] == os.path.expanduser("~/a.py")
    alt = ProcessSpec(name="alt", cmd=[sys.executable, "-u", "-c", "print('Please enter the start signal'); import time; time.sleep(30)"],
                      ready_text="Press [r] to start syncing|Please enter the start signal", ready_timeout_s=10)
    managed = ManagedProcess(alt, tmp_path)
    managed.start()
    try:
        assert managed.wait_ready()
    finally:
        managed.stop()


def test_shipped_config_uses_the_existing_addresses_and_readme_commands():
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    assert list(cfg.processes) == ["active_camera_zed", "active_camera_neck", "xr_teleop"]   # start order
    zed, neck, teleop = (cfg.processes[n] for n in ("active_camera_zed", "active_camera_neck", "xr_teleop"))
    assert zed.ssh == neck.ssh == "192.168.123.164" == cfg.images.pc2_ip
    assert zed.cmd == ["python3", "pc2/zed_teleimager_server.py"]
    assert neck.cmd[:2] == ["python3", "pc2/head_pose_receiver.py"] and "--yaw-limit-deg" in neck.cmd
    assert "--ipc" in teleop.cmd and "192.168.123.164" in teleop.cmd and "run_teleop_with_neck.py" in teleop.cmd[1]
