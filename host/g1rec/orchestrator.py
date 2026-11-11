"""Orchestrator: starts the existing programs as independent processes, owns the keyboard.

Responsibilities (and nothing else): process start/stop/health, the r/s/q keys, forwarding the
operator's first `r` to the official teleop over its IPC server, and reporting recorder events.
It contains no teleoperation, IK, retargeting or camera control, and it never reads the
programs' internal state: the recorder subscribes to their data streams.
"""

from __future__ import annotations

import collections
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .config import Config, ProcessSpec


ROOT = Path(__file__).resolve().parent.parent


def expand_root(text: str) -> str:
    return text.replace("{ROOT}", str(ROOT))


def sync_commands(sync: dict, have_rsync: bool) -> list[list[str]]:
    """Commands that copy the bundled pc2/ directory to PC2 (a new directory; nothing is deleted)."""

    src = expand_root(sync["src"]).rstrip("/")
    dest, ssh = sync["dest"].rstrip("/"), sync["ssh"]
    if have_rsync:
        return [["rsync", "-az", "--exclude", "__pycache__", "-e", "ssh -o BatchMode=yes",
                 "--rsync-path", f"mkdir -p {dest} && rsync", src + "/", f"{ssh}:{dest}/"]]
    return [["ssh", "-o", "BatchMode=yes", ssh, f"mkdir -p {dest}"],
            ["scp", "-r", "-o", "BatchMode=yes", src + "/.", f"{ssh}:{dest}/"]]


def run_sync(sync: dict, run=subprocess.run, which=shutil.which) -> tuple[bool, str]:
    for argv in sync_commands(sync, which("rsync") is not None):
        try:
            result = run(argv, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"{argv[0]}: {error}"
        if result.returncode != 0:
            return False, f"{argv[0]} failed ({result.returncode}): {(result.stderr or result.stdout).strip()[-300:]}"
    return True, ""


class ManagedProcess:
    def __init__(self, spec: ProcessSpec, log_dir: Path) -> None:
        self.spec = spec
        self.proc: Optional[subprocess.Popen] = None
        self.tail: collections.deque = collections.deque(maxlen=200)
        self.ready = threading.Event()
        self.log_path = log_dir / f"{spec.name}.log"
        self._log = None

    def argv(self) -> list[str]:
        if self.spec.ssh:
            remote = shlex.join(self.spec.cmd)
            if self.spec.cwd:                       # remote directory; ~ is expanded by the remote shell
                cwd = self.spec.cwd
                remote = f"cd {cwd if cwd.startswith('~') and ' ' not in cwd else shlex.quote(cwd)} && {remote}"
            return ["ssh", "-tt", self.spec.ssh, "--", remote]
        return [os.path.expanduser(expand_root(a)) if a.startswith(("~", "{ROOT}")) else a for a in self.spec.cmd]

    def start(self) -> None:
        if self.spec.mode == "external":
            self.ready.set()
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        env = {**os.environ, **{k: str(v) for k, v in self.spec.env.items()}}
        cwd = str(Path(self.spec.cwd).expanduser()) if self.spec.cwd and not self.spec.ssh else None
        # stdin is closed on purpose: the orchestrator owns the keyboard, so no child can
        # read keys. New session: Ctrl-C in the terminal reaches only the orchestrator.
        self.proc = subprocess.Popen(self.argv(), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        threading.Thread(target=self._pump, name=f"{self.spec.name}-out", daemon=True).start()
        if not self.spec.ready_text:
            self.ready.set()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            self._log.write(raw)
            self._log.flush()
            line = raw.decode(errors="replace").rstrip()
            self.tail.append(line)
            if self.spec.ready_text and any(t in line for t in self.spec.ready_text.split("|")):
                self.ready.set()

    def alive(self) -> bool:
        return self.spec.mode == "external" or (self.proc is not None and self.proc.poll() is None)

    def wait_ready(self) -> bool:
        deadline = time.monotonic() + self.spec.ready_timeout_s
        while time.monotonic() < deadline:
            if self.ready.wait(0.2):
                return True
            if not self.alive():
                return False
        return False

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, self.spec.stop_timeout_s), (signal.SIGTERM, 5.0),
                          (signal.SIGKILL, 2.0)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue


class RecorderProcess:
    """The recorder subprocess and its JSON-lines protocol."""

    def __init__(self, config_path: Optional[str], log_dir: Path,
                 on_event: Callable[[dict], None]) -> None:
        self.on_event = on_event
        self.proc: Optional[subprocess.Popen] = None
        self.ready: Optional[dict] = None
        self._ready_evt = threading.Event()
        self._config_path = config_path
        self._log_path = log_dir / "recorder.log"

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "g1rec.recorder_proc"]
        if self._config_path:
            argv += ["--config", self._config_path]
        self._errlog = open(self._log_path, "ab")
        root = str(Path(__file__).resolve().parent.parent)
        self.proc = subprocess.Popen(argv, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._errlog, start_new_session=True)
        threading.Thread(target=self._pump, name="recorder-out", daemon=True).start()

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("event") in ("ready", "not_ready"):
                self.ready = event
                self._ready_evt.set()
            self.on_event(event)

    def wait_ready(self, timeout: float) -> Optional[dict]:
        self._ready_evt.wait(timeout)
        return self.ready

    def send(self, cmd: str) -> None:
        if self.proc is not None and self.proc.poll() is None and self.proc.stdin:
            try:
                self.proc.stdin.write((json.dumps({"cmd": cmd}) + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        self.send("quit")
        if self.proc is not None:
            try:
                self.proc.wait(timeout=120)      # saving a running episode can take a while
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)


class Keyboard:
    """Single-key input from the terminal (cbreak), or lines when stdin is not a terminal."""

    def __init__(self) -> None:
        self._tty = sys.stdin.isatty()
        self._old = None

    def __enter__(self) -> "Keyboard":
        if self._tty:
            import termios
            import tty

            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc) -> None:
        if self._tty and self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)

    def read(self, timeout: float) -> Optional[str]:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        text = os.read(sys.stdin.fileno(), 1).decode(errors="ignore") if self._tty else sys.stdin.readline()[:1]
        return text.lower() or None


class Orchestrator:
    def __init__(self, cfg: Config, config_path: Optional[str], act_check: bool = True,
                 out: Callable[[str], None] = print) -> None:
        self.cfg, self.out, self.act_check = cfg, out, act_check
        self.log_dir = Path(cfg.dataset_root).parent / "logs"
        self.managed: dict[str, ManagedProcess] = {}
        self.recorder = RecorderProcess(config_path, self.log_dir, self._on_event)
        self.ipc = None
        self.recording = False
        self._events: "collections.deque[dict]" = collections.deque()
        self._lock = threading.Lock()

    # --- recorder events ---------------------------------------------------------------

    def _on_event(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)

    def _drain_events(self) -> None:
        while True:
            with self._lock:
                if not self._events:
                    return
                event = self._events.popleft()
            kind = event.get("event")
            if kind == "episode_started":
                self.out(f"[REC] Episode {event['raw_episode']:06d} started")
            elif kind in ("episode_saved", "episode_discarded"):
                self.recording = False
                self._report_episode(event)
            elif kind == "error":
                self.out(f"[ERROR] {event.get('msg')}")
            elif kind == "status" and event.get("dropped"):
                self.out(f"[REC] frames={event['frames']} dropped={event['dropped']}")

    def _report_episode(self, e: dict) -> None:
        from . import verify

        if not e.get("saved"):
            self.out(f"Episode {e['raw_episode']:06d} NOT saved (no accepted frames or a write error: "
                     f"{e.get('writer_error')}). Raw data is in {self.cfg.raw_root}.")
            return
        self.out(f"\nEpisode {e['raw_episode']:06d} Saved\n")
        self.out(f"Frames: {e['frames']}\nDuration: {e['duration_s']} s\nRobot States: {e['robot_states']}\n"
                 f"Actions: {e['actions']}\nTactile Samples: {e['tactile_samples']}\n"
                 f"Video Streams: {len(e['video_streams'])} ({', '.join(e['video_streams'])})")
        if e.get("dropped_ticks"):
            self.out(f"Dropped ticks: {e['dropped_ticks']}")
        if e.get("sequence_gaps"):
            self.out(f"Missing sequence numbers: {e['sequence_gaps']}")
        result = verify.report(self.cfg, act_check=self.act_check)
        self.out(f"\nLeRobotDataset: {result['dataset']}")
        self.out(f"ACT Training Compatibility: {result['act']}\n")

    # --- lifecycle -----------------------------------------------------------------------

    def start_all(self) -> bool:
        sync_src = Path(expand_root(self.cfg.pc2_sync["src"])).resolve() if self.cfg.pc2_sync else None
        if sync_src is not None and not sync_src.is_dir():
            self.out(f"[INFO] {sync_src} not found (host/ was copied without pc2/): "
                     "assuming the PC2 programs are already on PC2")
        elif self.cfg.pc2_sync and any(n.startswith("active_camera") and p.mode == "launch"
                                       for n, p in self.cfg.processes.items()):
            ok, error = run_sync(self.cfg.pc2_sync)
            if not ok:
                self.out(f"[FAIL] copying the PC2 programs to {self.cfg.pc2_sync['ssh']}: {error}")
                return False
            self.out(f"[OK] PC2 programs copied to {self.cfg.pc2_sync['ssh']}:{self.cfg.pc2_sync['dest']}")
        groups = (("active_camera", "Active Camera Running"), ("xr_teleop", "XR Teleoperation Running"))
        for prefix, label in groups:
            names = [n for n in self.cfg.processes if n.startswith(prefix)]      # config order = start order
            if not names:
                self.out(f"[SKIP] {prefix}: not configured")
                continue
            for name in names:
                managed = ManagedProcess(self.cfg.processes[name], self.log_dir)
                self.managed[name] = managed
                managed.start()
                if not managed.wait_ready():
                    self.out(f"[FAIL] {name}: not ready. Last output:\n  " + "\n  ".join(list(managed.tail)[-8:]))
                    return False
            external = all(self.cfg.processes[n].mode == "external" for n in names)
            self.out(f"[OK] {label}" + (" (external)" if external else ""))
        self.recorder.start()
        ready = self.recorder.wait_ready(60)
        if ready is None:
            self.out("[FAIL] Recorder did not start; see " + str(self.log_dir / "recorder.log"))
            return False
        if ready["event"] == "ready":
            self.out("[OK] Recorder Ready")
        else:
            self.out("[WAIT] Recorder started but streams are missing: " + "; ".join(ready["problems"]))
        try:
            from .ipc_client import TeleopIpcClient

            self.ipc = TeleopIpcClient()
        except Exception as error:
            self.out(f"[WARN] teleop IPC client unavailable ({error}); `r` will not start the teleop")
        return True

    def _ensure_teleop_started(self) -> bool:
        if self.ipc is None:
            return True
        state = self.ipc.state()
        if state.get("START"):
            return True
        if not state:
            self.out("[WARN] no heartbeat from the official teleop: was it started with --ipc?")
            return False
        self.out("Starting teleop following (official `r`)...")
        if not self.ipc.start_following():
            self.out("[WARN] the teleop did not accept the start command (hands tracked? READY?)")
            return False
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.ipc.state().get("START"):
                return True
            time.sleep(0.1)
        self.out("[WARN] teleop did not report START")
        return False

    def loop(self) -> None:
        self.out("\nPress:\n  r - Start Episode\n  s - Stop & Save Episode\n  q - Quit\n")
        with Keyboard() as keyboard:
            while True:
                self._drain_events()
                for name, managed in self.managed.items():
                    if not managed.alive():
                        self.out(f"[ERROR] {name} exited. Recording continues only if its streams are still alive.")
                if not self.recorder.alive():
                    self.out("[ERROR] Recorder exited")
                    return
                key = keyboard.read(0.1)
                if key == "r" and not self.recording:
                    if self._ensure_teleop_started():
                        self.recording = True
                        self.recorder.send("start")
                elif key == "s" and self.recording:
                    self.recorder.send("stop")
                    self.out("Saving episode...")
                elif key == "q":
                    return

    def shutdown(self) -> None:
        self.out("Shutting down (a running episode is saved first)...")
        self.recorder.stop()
        self._drain_events()
        if self.ipc is not None:
            try:
                self.ipc.stop_teleop()
            except Exception:
                pass
            self.ipc.close()
        for name in reversed(list(self.managed)):
            self.managed[name].stop()
