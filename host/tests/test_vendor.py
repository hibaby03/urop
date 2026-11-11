"""The bundled copy of unitree_g1_urop must stay verbatim, and the code must really use it."""

import hashlib
import subprocess
import sys
from pathlib import Path

from g1rec import config as config_mod
from g1rec.orchestrator import ROOT, expand_root, run_sync, sync_commands
from g1rec.sources import BUNDLED_HOST, _import_urop_module

VENDOR = ROOT / "third_party" / "unitree_g1_urop"          # Host-side verbatim files
PC2 = ROOT.parent / "pc2"                                  # PC2-side verbatim files (sibling of host/)

import pytest

needs_pc2 = pytest.mark.skipif(not PC2.is_dir(), reason="pc2/ is not next to host/ (host/ copied alone)")


def _modified(directory):
    bad = []
    for line in (directory / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split(None, 1)
        if hashlib.sha256((directory / name.strip()).read_bytes()).hexdigest() != digest:
            bad.append(name.strip())
    return bad


def test_bundled_host_files_are_unmodified():
    assert not _modified(VENDOR), f"vendored host files were edited: {_modified(VENDOR)}"


@needs_pc2
def test_pc2_files_are_unmodified():
    assert not _modified(PC2), f"pc2 files were edited: {_modified(PC2)}"


def test_host_dependency_closure_is_present():
    for name in ("run_teleop_with_neck", "head_pose_udp", "neck_trace", "neck_clock", "frame_timing",
                 "neck_state_receiver"):
        assert (VENDOR / "host" / f"{name}.py").is_file(), name


@needs_pc2
def test_pc2_dependency_closure_is_present():
    for name in ("zed_teleimager_server", "head_pose_receiver", "display_stream", "frame_timing", "zed_stereo",
                 "neck_trace", "active_camera_protocol", "dynamixel_neck", "neck_state_sender"):
        assert (PC2 / f"{name}.py").is_file(), name
    assert (PC2 / "cam_config_zed.yaml").is_file()


def test_parsers_are_imported_from_the_bundled_copy_without_any_config():
    cfg = config_mod.from_dict({})
    assert cfg.g1_urop_host is None
    frame_timing = _import_urop_module(cfg, "frame_timing")
    neck = _import_urop_module(cfg, "neck_state_receiver")
    assert Path(frame_timing.__file__).resolve().parent == BUNDLED_HOST.resolve()
    assert Path(neck.__file__).resolve().parent == BUNDLED_HOST.resolve()
    assert callable(neck.decode_neck_state_packet) and hasattr(frame_timing, "ClockSyncClient")


@needs_pc2
def test_the_pc2_programs_import_only_what_is_in_pc2():
    """Run the PC2 entry point's --help with only pc2/ available (no unitree_g1_urop, no host/ on the path)."""
    for script in ("head_pose_receiver.py",):
        result = subprocess.run([sys.executable, str(PC2 / script), "--help"], capture_output=True,
                                text=True, cwd=str(PC2), timeout=60)
        assert result.returncode == 0, result.stderr[-400:]


def test_launcher_dependencies_are_bundled():
    """Its helper modules import from the bundled directory; it then needs only the official repo (external)."""
    code = ("import sys; sys.path.insert(0, %r); import head_pose_udp, neck_trace, neck_clock; print('ok')"
            % str(VENDOR / "host"))
    ok = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0 and "ok" in ok.stdout, ok.stderr[-400:]
    result = subprocess.run([sys.executable, str(VENDOR / "host" / "run_teleop_with_neck.py"),
                             "--xr-repo", "/nonexistent/xr_teleoperate", "--help"],
                            capture_output=True, text=True, timeout=60)
    assert "ModuleNotFoundError" not in result.stderr and "ImportError" not in result.stderr
    assert "teleop script not found" in result.stderr          # stops only for the external official repo


def test_shipped_config_points_at_the_bundled_files():
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    assert cfg.g1_urop_host is None
    launcher = expand_root(cfg.processes["xr_teleop"].cmd[1])
    assert Path(launcher) == VENDOR / "host" / "run_teleop_with_neck.py" and Path(launcher).is_file()
    for name in ("active_camera_zed", "active_camera_neck"):
        spec = cfg.processes[name]
        assert spec.cmd[1].startswith("pc2/")                  # used only if switched to `mode: launch`
        if PC2.is_dir():
            assert (PC2 / spec.cmd[1].split("/", 1)[1]).is_file()


def test_pc2_programs_are_started_by_hand_by_default():
    """The shipped config never connects to PC2: no sync, active camera processes are `external`."""
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    assert cfg.pc2_sync is None
    assert cfg.processes["active_camera_zed"].mode == "external"
    assert cfg.processes["active_camera_neck"].mode == "external"
    assert cfg.processes["xr_teleop"].mode == "launch"          # the Host teleop is still started by main.py


def test_default_start_does_not_run_ssh_or_rsync(tmp_path):
    from g1rec.orchestrator import Orchestrator
    cfg = config_mod.load(str(ROOT / "config.yaml"))
    cfg.processes.pop("xr_teleop")                              # nothing to launch in this test
    cfg.dataset_root, cfg.raw_root = str(tmp_path / "ds"), str(tmp_path / "raw")
    lines = []
    orch = Orchestrator(cfg, None, out=lines.append)
    orch.recorder.start = lambda: None
    orch.recorder.wait_ready = lambda timeout: {"event": "ready", "problems": []}
    assert orch.start_all()
    assert any("Active Camera Running (external)" in line for line in lines)
    assert not any("PC2 programs copied" in line or "copying the PC2" in line for line in lines)


def test_sync_commands_use_a_new_remote_directory_and_never_delete():
    sync = {"ssh": "192.168.123.164", "src": "{ROOT}/../pc2", "dest": "~/g1rec_pc2/pc2"}
    (rsync,) = sync_commands(sync, have_rsync=True)
    assert rsync[0] == "rsync" and "--delete" not in rsync
    assert rsync[-2] == str(ROOT) + "/../pc2/" and rsync[-1] == "192.168.123.164:~/g1rec_pc2/pc2/"
    assert "mkdir -p ~/g1rec_pc2/pc2 && rsync" in rsync
    mkdir, scp = sync_commands(sync, have_rsync=False)
    assert mkdir[-1] == "mkdir -p ~/g1rec_pc2/pc2" and scp[0] == "scp" and scp[-1].endswith(":~/g1rec_pc2/pc2/")


def test_run_sync_reports_failures_and_stops_at_the_first_error():
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv[0])
        return type("R", (), {"returncode": 255 if argv[0] == "ssh" else 0, "stderr": "Permission denied", "stdout": ""})()

    sync = {"ssh": "h", "src": "/x", "dest": "~/d"}
    ok, error = run_sync(sync, run=fake_run, which=lambda name: None)
    assert not ok and "Permission denied" in error and calls == ["ssh"]
    ok, error = run_sync(sync, run=lambda argv, **kw: type("R", (), {"returncode": 0, "stderr": "", "stdout": ""})(),
                         which=lambda name: "/usr/bin/rsync")
    assert ok and error == ""


def test_pc2_sync_validation():
    import pytest
    with pytest.raises(ValueError):
        config_mod.from_dict({"pc2_sync": {"ssh": "h"}})


def test_host_works_without_pc2_next_to_it(tmp_path):
    """host/ copied alone: the sync step is skipped with a note, not an error."""
    from g1rec.orchestrator import Orchestrator
    cfg = config_mod.from_dict({"pc2_sync": {"ssh": "h", "src": str(tmp_path / "missing_pc2"), "dest": "~/d"},
                                "dataset_root": str(tmp_path / "ds"), "raw_root": str(tmp_path / "raw"),
                                "processes": {"active_camera_zed": {"mode": "external"}}})
    lines = []
    orch = Orchestrator(cfg, None, out=lines.append)
    orch.recorder.start = lambda: None                      # no real recorder in this test
    orch.recorder.wait_ready = lambda timeout: {"event": "ready", "problems": []}
    orch.start_all()
    assert any("assuming the PC2 programs are already on PC2" in line for line in lines)
    assert any("Active Camera Running" in line for line in lines)
