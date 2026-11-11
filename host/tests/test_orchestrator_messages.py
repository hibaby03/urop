"""What main.py tells the operator: ready detection on wrapped log lines, progress while waiting, instructions."""

import sys
import time

import pytest

from g1rec import config as config_mod
from g1rec import orchestrator as orch_mod
from g1rec.config import ProcessSpec
from g1rec.orchestrator import ManagedProcess, Orchestrator, ready_matches

# The teleop's real output (Rich-style logger): the message is wrapped over three lines and the right-aligned
# `file.py:line` column sits between "start" and "signal".
WRAPPED = [
    "  18:28:01.229794 INFO     [Dex3_1_Controller] Subscribe  robot_hand_unitree.py:94",
    "                           dds ok.",
    "  18:28:01.241436 INFO     Initialize Dex3_1_Controller  robot_hand_unitree.py:101",
    "                           OK!",
    "  ",
    "  18:28:01.241636 INFO     Please enter the start       teleop_hand_and_arm.py:388",
    "                           signal (enter 'r' to start",
    "                           the subsequent program)",
]
TELEOP_READY = "Press [r] to start syncing|Please enter the start signal"


def window(lines):
    return " ".join(x.strip() for x in lines)


def test_the_real_wrapped_teleop_output_is_recognised():
    assert not any(TELEOP_READY.split("|")[1] in line for line in WRAPPED)       # the old per-line check failed
    assert ready_matches(window(WRAPPED), TELEOP_READY)


@pytest.mark.parametrize("text,expected", [
    ("Press [r] to start syncing the robot with your movements.", True),
    ("Please enter the start signal (enter 'r' to start the subsequent program)", True),
    ("Please enter the   start\n  teleop_hand_and_arm.py:388   signal", True),
    ("Initialize Dex3_1_Controller OK!", False),
    ("Please enter your name", False),
    ("Please enter the start of something else entirely " + "x" * 200 + " signal", False),   # gap is bounded
])
def test_ready_matches(text, expected):
    assert ready_matches(text, TELEOP_READY) is expected


def test_ready_text_without_spaces_and_special_characters():
    assert ready_matches('{"event":"listening","port":5005}', '"event":"listening"')
    assert ready_matches("[display] publishing per-eye JPEGs", "[display] publishing")
    assert not ready_matches("nothing here", "[display] publishing")
    assert not ready_matches("anything", "")


def child(code, **kw):
    return ProcessSpec(name="child", cmd=[sys.executable, "-c", code], **kw)


def test_managed_process_becomes_ready_from_wrapped_output(tmp_path):
    code = f"import time\nfor l in [{', '.join(repr(l) for l in WRAPPED)}]:\n    print(l)\ntime.sleep(30)"
    managed = ManagedProcess(child(code, ready_text=TELEOP_READY, ready_timeout_s=15), tmp_path)
    managed.start()
    try:
        assert managed.wait_ready()
    finally:
        managed.stop()


def test_unflushed_child_output_still_arrives(tmp_path):
    """No `python -u` and no flush: PYTHONUNBUFFERED set by main.py makes the line arrive at once."""
    code = "import time\nprint('SERVER READY')\ntime.sleep(30)"
    managed = ManagedProcess(child(code, ready_text="SERVER READY", ready_timeout_s=10), tmp_path)
    started = time.monotonic()
    managed.start()
    try:
        assert managed.wait_ready() and time.monotonic() - started < 8
    finally:
        managed.stop()


def test_wait_reports_what_the_program_is_doing(tmp_path):
    code = "import time\nprint('Initialize Dex3_1_Controller OK!', flush=True)\ntime.sleep(30)"
    managed = ManagedProcess(child(code, ready_text="never printed", ready_timeout_s=1.6), tmp_path)
    reports = []
    managed.start()
    try:
        assert not managed.wait_ready(on_wait=lambda secs, line: reports.append((secs, line)), interval=0.4)
    finally:
        managed.stop()
    assert len(reports) >= 2 and "Initialize Dex3_1_Controller OK!" in reports[-1][1]
    assert reports[-1][0] > reports[0][0]


def make_orchestrator(tmp_path, teleop_code, ready_timeout=15, **cfg_extra):
    cfg = config_mod.from_dict({
        "dataset_root": str(tmp_path / "ds"), "raw_root": str(tmp_path / "raw"), "teleop_input": "pty",
        "processes": {"active_camera_zed": {"mode": "external"},
                      "xr_teleop": {"cmd": [sys.executable, "-c", teleop_code], "ready_text": TELEOP_READY,
                                    "ready_timeout_s": ready_timeout}}, **cfg_extra})
    lines = []
    o = Orchestrator(cfg, None, out=lines.append)
    o.recorder.start = lambda: None
    o.recorder.wait_ready = lambda timeout: {"event": "ready", "problems": []}
    return o, lines


def wrapped_teleop_code(delay=0.0):
    body = "".join(f"    print({l!r}, flush=True)\n" for l in WRAPPED)
    return f"import time, os, tty\ntime.sleep({delay})\nif True:\n{body}time.sleep(30)\n"


def test_start_all_announces_each_step_and_that_r_is_awaited(tmp_path):
    o, lines = make_orchestrator(tmp_path, wrapped_teleop_code())
    try:
        assert o.start_all()
    finally:
        o.managed["xr_teleop"].stop()
    text = "\n".join(lines)
    assert "[OK] Active Camera Running (external)" in text
    assert "[START] xr_teleop 시작 중" in text
    assert "[OK] XR Teleoperation Running  -> 로봇 팔로우 시작(r 입력) 대기 중" in text
    assert "[START] Recorder 시작 중" in text and "[OK] Recorder Ready" in text
    order = [text.index(k) for k in ("Active Camera Running", "XR Teleoperation Running", "Recorder Ready")]
    assert order == sorted(order)


def test_a_slow_teleop_is_never_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "WAIT_REPORT_S", 0.4)
    o, lines = make_orchestrator(tmp_path, wrapped_teleop_code(delay=1.5))
    try:
        assert o.start_all()
    finally:
        o.managed["xr_teleop"].stop()
    waits = [l for l in lines if l.startswith("[WAIT] xr_teleop 준비 대기")]
    assert len(waits) >= 2 and "(no output yet)" in waits[0]


def test_failure_still_shows_the_last_output(tmp_path):
    o, lines = make_orchestrator(tmp_path, "print('stuck in initialisation', flush=True)\nimport time; time.sleep(30)",
                                 ready_timeout=1.0)
    try:
        assert not o.start_all()
    finally:
        o.managed["xr_teleop"].stop()
    assert any("[FAIL] xr_teleop: not ready" in l and "stuck in initialisation" in l for l in lines)


def test_instructions_tell_the_operator_to_press_r():
    o = Orchestrator(config_mod.from_dict({}), None, out=lambda text: None)
    text = o.instructions()
    assert "r 을 누르세요" in text and "Enter VR" in text
    for key_line in ("r - Start Episode", "s - Stop & Save Episode", "q - Quit"):
        assert key_line in text
    assert "[주의]" not in text


def test_instructions_list_missing_streams():
    o = Orchestrator(config_mod.from_dict({}), None, out=lambda text: None)
    o._recorder_problems = ["image: no data yet", "neck: stale (900 ms > 500 ms)"]
    text = o.instructions()
    assert "[주의]" in text and "image: no data yet" in text and "neck: stale" in text
