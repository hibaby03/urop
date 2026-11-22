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


def make_stub(base, fake_q=DEFAULT_FAKE_Q, log=print):
    """Subclass of Dex3_1_Controller whose reader thread pins the right-hand state first."""

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
            return super()._subscribe_hand_state()

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
    hand_module.Dex3_1_Controller = make_stub(hand_module.Dex3_1_Controller, fake_q,
                                              log=lambda msg: print(msg, flush=True))
    print(f"[right-stub] Dex3_1_Controller patched (fake right state {fake_q}); real teleop starts now.", flush=True)

    import run_teleop_with_neck as launcher
    return launcher.main(argv)


if __name__ == "__main__":
    sys.exit(main())
