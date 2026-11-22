#!/usr/bin/env python3
"""Run the real teleop (same arguments as run_teleop_with_neck.py) with the right Dex3 hand's
STATE pinned to a fixed value, so Dex3_1_Controller no longer blocks at start-up waiting for
`rt/dex3/right/state`. The teleop script and the xr_teleoperate repo are not modified.

Why: Dex3_1_Controller.__init__ waits until BOTH hands' "state received" Events are set
("[Dex3_1_Controller] Waiting to subscribe dds..."; older versions: both state arrays non-zero).
With no right-hand state it waits forever,
so the teleop never reaches its start prompt and the arms never follow.

What this does (in this process only):
  * subclasses Dex3_1_Controller; its state-reader thread first writes a small non-zero constant into
    right_hand_state_array and sets _right_hand_state_received, then runs the original reader. If a real right-hand state ever arrives
    it simply overwrites the constant.
Consequences:
  * left hand + arms work normally.
  * right-hand COMMANDS are still published on rt/dex3/right/cmd. If the right hand is powered and only its
    state publisher is broken, it will still move; if it is not connected, the commands go nowhere.
  * the right-hand state seen by the teleop (and any recording of it) is the constant, not real data.

Usage (replaces run_teleop_with_neck.py; kill stale teleop/Vuer processes first, port 8012 must be free):
    cd ~/hckang/xr_teleoperate
    python ~/hckang/jslee/urop/host/run_teleop_fake_right_hand.py \
        --xr-repo ~/hckang/xr_teleoperate --neck-pose-ip 192.168.123.164 \
        --neck-pose-port 5005 --arm=G1_29 --ee=dex3 [--fake-right-q 0.001]

    python3 run_teleop_fake_right_hand.py --selftest      # checks the patch mechanism, no robot
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

DEFAULT_FAKE_Q = 0.001   # any non-zero value satisfies the controller's `any(state_array)` check


def make_stub(base, fake_q=DEFAULT_FAKE_Q, log=print, left_index=None, right_index=None):
    """Subclass of Dex3_1_Controller whose reader thread pins the right-hand state first.

    With left_index/right_index (the Dex3_1_*_JointIndex enums) the two hands are also read INDEPENDENTLY:
    the original loop calls Left.Read() and Right.Read() in lockstep, so a right Read() that never returns
    keeps the left state from ever being processed.
    """

    class RightHandStateStub(base):
        def _subscribe_hand_state(self):
            try:
                array = self.right_hand_state_array
                with array.get_lock():
                    for index in range(len(array)):
                        array[index] = fake_q
                log(f"[right-stub] right_hand_state_array pinned to {fake_q} (no real right-hand state needed)")
            except Exception as error:  # noqa: BLE001
                log(f"[right-stub] could not pin the right-hand state: {type(error).__name__}: {error}")
            # Newer controllers wait on a "received" Event instead of a non-zero array.
            event = getattr(self, "_right_hand_state_received", None)
            if event is not None:
                event.set()
                log("[right-stub] _right_hand_state_received set (the controller's start-up wait is released)")
            self._start_wait_report(log)
            can_split = (left_index is not None and right_index is not None
                         and hasattr(self, "LeftHandState_subscriber") and hasattr(self, "RightHandState_subscriber")
                         and hasattr(self, "_left_hand_state_received"))
            try:
                if can_split:
                    return self._independent_readers(log)
                return super()._subscribe_hand_state()
            except BaseException as error:  # the reader thread would otherwise die without a trace
                log(f"[right-stub] original reader thread raised {type(error).__name__}: {error}")
                raise

        def _independent_readers(self, log):
            """Left and right read in separate threads, so one silent topic cannot block the other."""
            import threading
            import time

            def copy(msg, indices, array, event):
                for position, joint in enumerate(indices):
                    array[position] = msg.motor_state[joint].q
                event.set()

            def right_loop():
                try:
                    while True:
                        msg = self.RightHandState_subscriber.Read()
                        if msg is not None:
                            copy(msg, right_index, self.right_hand_state_array, self._right_hand_state_received)
                        time.sleep(0.002)
                except BaseException as error:  # noqa: BLE001
                    log(f"[right-stub] right reader thread raised {type(error).__name__}: {error}")

            threading.Thread(target=right_loop, daemon=True).start()
            log("[right-stub] left and right hand states are now read independently")
            first = True
            while True:
                msg = self.LeftHandState_subscriber.Read()
                if msg is not None:
                    copy(msg, left_index, self.left_hand_state_array, self._left_hand_state_received)
                    if first:
                        log("[right-stub] first LEFT hand state received")
                        first = False
                time.sleep(0.002)

        def _start_wait_report(self, log):
            """Once a second until the start-up wait is released: which side is still missing."""
            import threading
            import time

            def report():
                waited = 0
                while waited < 600:
                    left = getattr(self, "_left_hand_state_received", None)
                    right = getattr(self, "_right_hand_state_received", None)
                    left_set = left.is_set() if left is not None else None
                    right_set = right.is_set() if right is not None else None
                    if left_set and right_set:
                        log("[right-stub] both state events are set; start-up wait is over")
                        return
                    try:
                        left_q = list(self.left_hand_state_array[:])
                        right_q = list(self.right_hand_state_array[:])
                    except Exception:  # noqa: BLE001
                        left_q = right_q = "?"
                    log(f"[right-stub] still waiting: left_received={left_set} right_received={right_set} "
                        f"left_q={left_q} right_q={right_q}")
                    time.sleep(1.0)
                    waited += 1

            threading.Thread(target=report, daemon=True).start()

    RightHandStateStub.__name__ = base.__name__
    RightHandStateStub.__qualname__ = base.__qualname__
    return RightHandStateStub


def check_shape(cls):
    """Return None if the class looks like the expected one, else a message."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError) as error:
        return f"cannot read the source of {cls}: {error}"
    missing = [name for name in ("_subscribe_hand_state", "right_hand_state_array", "_right_hand_state_received")
               if name not in source]
    if missing:
        return f"expected names not found in Dex3_1_Controller: {missing}"
    return None


def run_and_report_swallowed(main_callable, argv):
    """Run the launcher; if the teleop ends through its own `finally: exit(0)`, print the exception that
    exit(0) replaced. Without this, an error inside the follow loop leaves no traceback at all."""
    import traceback
    try:
        return main_callable(argv)
    except SystemExit as stop:
        swallowed = stop.__context__
        if swallowed is not None and not isinstance(swallowed, KeyboardInterrupt):
            print("\n[swallowed-exception] the teleop stopped because of this error (its finally: exit(0) hid it):",
                  file=sys.stderr, flush=True)
            traceback.print_exception(type(swallowed), swallowed, swallowed.__traceback__)
            sys.stderr.flush()
        raise


# --------------------------------------------------------------------------- self-test (no SDK)

def selftest():
    import threading
    import time
    from multiprocessing import Array

    class FakeDex3:                      # mimics the start-up wait of the real controller
        def __init__(self, right_arrives=False):
            self.right_arrives = right_arrives
            self.left_hand_state_array = Array("d", 7)
            self.right_hand_state_array = Array("d", 7)
            self._left_hand_state_received = threading.Event()
            self._right_hand_state_received = threading.Event()
            threading.Thread(target=self._subscribe_hand_state, daemon=True).start()
            for _ in range(300):
                if self._left_hand_state_received.is_set() and self._right_hand_state_received.is_set():
                    return
                time.sleep(0.01)
            raise TimeoutError("still waiting for both hand states")

        def _subscribe_hand_state(self):
            for _ in range(2000):
                self.left_hand_state_array[0] = 0.5
                self._left_hand_state_received.set()
                if self.right_arrives:
                    time.sleep(0.2)
                    self.right_hand_state_array[0] = 0.9
                    self._right_hand_state_received.set()
                time.sleep(0.005)

    results = []

    def case(name, ok):
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}")

    try:
        FakeDex3()
        case("unpatched blocks without right-hand state", False)
    except TimeoutError:
        case("unpatched blocks without right-hand state", True)

    stub = make_stub(FakeDex3, 0.001, log=lambda *_: None)
    try:
        controller = stub()
        case("patched starts without right-hand state", True)
        case("right state is the pinned constant", abs(controller.right_hand_state_array[0] - 0.001) < 1e-12)
    except TimeoutError:
        case("patched starts without right-hand state", False)

    controller = stub(right_arrives=True)
    time.sleep(0.5)
    case("a real right-hand state overwrites the constant", abs(controller.right_hand_state_array[0] - 0.9) < 1e-12)
    case("shape check accepts the expected layout", check_shape(FakeDex3) is None)

    # A controller like the real one: lockstep Left.Read()/Right.Read(), the right Read() blocks forever.
    class FakeMsg:
        def __init__(self, q):
            self.motor_state = {0: type("M", (), {"q": q})(), 1: type("M", (), {"q": q})()}

    class Left:
        def Read(self):
            time.sleep(0.001)
            return FakeMsg(0.5)

    class SilentRight:
        def Read(self):
            threading.Event().wait()      # never returns: no right-hand publisher

    class Lockstep:
        def __init__(self):
            self.left_hand_state_array = Array("d", 2)
            self.right_hand_state_array = Array("d", 2)
            self.LeftHandState_subscriber, self.RightHandState_subscriber = Left(), SilentRight()
            self._left_hand_state_received = threading.Event()
            self._right_hand_state_received = threading.Event()
            threading.Thread(target=self._subscribe_hand_state, daemon=True).start()
            for _ in range(300):
                if self._left_hand_state_received.is_set() and self._right_hand_state_received.is_set():
                    return
                time.sleep(0.01)
            raise TimeoutError("still waiting for both hand states")

        def _subscribe_hand_state(self):
            while True:
                left, right = self.LeftHandState_subscriber.Read(), self.RightHandState_subscriber.Read()
                if left is not None:
                    self._left_hand_state_received.set()
                if right is not None:
                    self._right_hand_state_received.set()

    try:
        Lockstep.right_hand_state_array = None
        Lockstep()
        case("lockstep reader blocks on a silent right topic (unpatched)", False)
    except TimeoutError:
        case("lockstep reader blocks on a silent right topic (unpatched)", True)
    quiet = lambda *_: None
    try:
        patched = make_stub(Lockstep, 0.001, log=quiet, left_index=[0, 1], right_index=[0, 1])()
        case("independent readers start despite a silent right topic", True)
        case("left state is the real one", abs(patched.left_hand_state_array[0] - 0.5) < 1e-12)
        case("right state is still the pinned constant", abs(patched.right_hand_state_array[0] - 0.001) < 1e-12)
    except TimeoutError:
        case("independent readers start despite a silent right topic", False)

    import io
    import contextlib

    def teleop_like(_argv):               # a loop error followed by `finally: exit(0)` like teleop_hand_and_arm.py
        try:
            raise ValueError("boom in the follow loop")
        finally:
            exit(0)

    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        try:
            run_and_report_swallowed(teleop_like, [])
        except SystemExit:
            pass
    case("an error hidden by the teleop's exit(0) is printed",
         "boom in the follow loop" in captured.getvalue() and "swallowed-exception" in captured.getvalue())

    def clean_exit(_argv):
        try:
            pass
        finally:
            exit(0)

    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        try:
            run_and_report_swallowed(clean_exit, [])
        except SystemExit:
            pass
    case("a normal exit prints nothing", captured.getvalue() == "")

    class Different:
        def __init__(self): pass
    case("shape check rejects a different layout", check_shape(Different) is not None)
    return 0 if all(results) else 1


# --------------------------------------------------------------------------- main

def main():
    if "--selftest" in sys.argv:
        return selftest()

    argv = sys.argv[1:]
    fake_q = DEFAULT_FAKE_Q
    cleaned = []
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
        elif arg == "--fake-right-q" and i + 1 < len(argv):
            fake_q, skip = float(argv[i + 1]), True
        elif arg.startswith("--fake-right-q="):
            fake_q = float(arg.split("=", 1)[1])
        else:
            cleaned.append(arg)
    argv = cleaned
    if fake_q == 0.0:
        print("ERROR: --fake-right-q must be non-zero (the controller waits for a non-zero state)")
        return 2

    repo = None
    for i, arg in enumerate(argv):
        if arg == "--xr-repo" and i + 1 < len(argv):
            repo = argv[i + 1]
        elif arg.startswith("--xr-repo="):
            repo = arg.split("=", 1)[1]
    if repo is None:
        print("ERROR: pass --xr-repo ~/hckang/xr_teleoperate (same arguments as run_teleop_with_neck.py)")
        return 2
    repo = Path(repo).expanduser().resolve()
    launcher_dir = Path(__file__).resolve().parent / "third_party" / "unitree_g1_urop" / "host"
    if not (launcher_dir / "run_teleop_with_neck.py").is_file():
        print(f"ERROR: launcher not found in {launcher_dir}")
        return 2

    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    # Imports below create loggers; later basicConfig() calls (launcher, teleop script) would raise.
    logging_mp.basicConfig = lambda *args, **kwargs: None
    sys.path.insert(0, str(launcher_dir))
    sys.path.insert(0, str(repo))

    import teleop.robot_control.robot_hand_unitree as hand_module
    problem = check_shape(hand_module.Dex3_1_Controller)
    if problem:
        print(f"ERROR: {problem}\nThis Host's Dex3_1_Controller differs from the expected layout; "
              "nothing was patched. Paste the source of Dex3_1_Controller.__init__ and its state-reader method "
              f"from {hand_module.__file__}.")
        return 2
    hand_module.Dex3_1_Controller = make_stub(
        hand_module.Dex3_1_Controller, fake_q, log=lambda msg: print(msg, flush=True),
        left_index=getattr(hand_module, "Dex3_1_Left_JointIndex", None),
        right_index=getattr(hand_module, "Dex3_1_Right_JointIndex", None))
    print(f"[right-stub] Dex3_1_Controller patched (fake right state {fake_q}); real teleop starts now.", flush=True)

    import run_teleop_with_neck as launcher
    return run_and_report_swallowed(launcher.main, argv)


if __name__ == "__main__":
    sys.exit(main())
