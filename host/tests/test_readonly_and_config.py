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
    assert "192.168.123.164" in teleop.cmd and "run_teleop_with_neck.py" in teleop.cmd[1]


def test_pty_child_sees_a_terminal_and_only_receives_keys_we_send(tmp_path):
    """The legacy teleop reads the keyboard itself (sshkeyboard needs a tty): give it a pty, type only r/q."""
    code = ("import os,sys,termios\n"
            "print('isatty', os.isatty(0)); termios.tcgetattr(0)\n"            # what sshkeyboard needs
            "import tty; tty.setcbreak(0)\n"                                 # like sshkeyboard: raw mode first
            "print('READY', flush=True)\n"
            "while True:\n"
            "    k = os.read(0, 1).decode()\n"
            "    print('key', k, flush=True)\n"
            "    if k == 'q': break\n")
    spec = ProcessSpec(name="legacy", cmd=[sys.executable, "-u", "-c", code], ready_text="READY", ready_timeout_s=10)
    managed = ManagedProcess(spec, tmp_path, tty=True)
    managed.start()
    try:
        assert managed.wait_ready()
        assert managed.saw("isatty True")
        assert managed.send_key("r") and managed.send_key("q")
        deadline = time.monotonic() + 5
        while not managed.saw("key q") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert managed.saw("key r") and managed.saw("key q")
    finally:
        managed.stop()
    assert not managed.alive() and managed._master is None


def test_child_without_tty_gets_no_input(tmp_path):
    spec = ProcessSpec(name="plain", cmd=[sys.executable, "-u", "-c", "import os,sys; print('tty', os.isatty(0)); print('READY')"],
                       ready_text="READY", ready_timeout_s=10)
    managed = ManagedProcess(spec, tmp_path)
    managed.start()
    try:
        assert managed.wait_ready() and managed.saw("tty False")
        assert not managed.send_key("r")                       # no terminal: nothing can be typed
    finally:
        managed.stop()


def test_r_types_into_the_teleop_terminal_once_and_q_on_shutdown(tmp_path):
    from g1rec.orchestrator import Orchestrator
    cfg = config_mod.from_dict({"dataset_root": str(tmp_path / "ds"), "raw_root": str(tmp_path / "raw"), "teleop_input": "pty",
                                "processes": {"xr_teleop": {"cmd": [sys.executable, "-u", "-c",
                                    "import os,tty\ntty.setcbreak(0)\nprint('Please enter the start signal', flush=True)\n"
                                    "while True:\n    k=os.read(0,1).decode()\n    print('got',k,flush=True)\n"
                                    "    if k=='r': print('start Tracking', flush=True)\n    if k=='q': break\n"],
                                    "ready_text": "start signal", "ready_timeout_s": 10}}})
    lines = []
    orch = Orchestrator(cfg, None, out=lines.append)
    orch.recorder.start = lambda: None
    orch.recorder.wait_ready = lambda timeout: {"event": "ready", "problems": []}
    orch.recorder.send = lambda cmd: None
    orch.recorder.stop = lambda: None
    assert orch.start_all()
    teleop = orch.managed["xr_teleop"]
    try:
        assert orch._ensure_teleop_started()
        assert orch._ensure_teleop_started()                    # second r: already following, nothing typed again
        time.sleep(0.3)
        assert [l for l in teleop.tail if l.startswith("got")].count("got r") == 1
        assert teleop.saw("start Tracking") and not any("did not confirm" in l for l in lines)
        orch.shutdown()
        assert teleop.saw("got q")                              # the teleop's own quit key
    finally:
        teleop.stop()


def test_teleop_input_validation_and_shipped_default():
    import pytest
    with pytest.raises(ValueError):
        config_mod.from_dict({"teleop_input": "stdin"})
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    assert cfg.teleop_input == "pty" and "--ipc" not in cfg.processes["xr_teleop"].cmd
