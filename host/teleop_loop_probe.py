#!/usr/bin/env python3
"""Run the REAL teleop unchanged, and log what happens inside its follow loop.

Same command line as third_party/unitree_g1_urop/host/run_teleop_with_neck.py
(same display relay, neck UDP, etc.); it only wraps three calls the teleop makes
every cycle, logs their values once per second, and calls the original:

  1. TeleVuerWrapper.get_motion_state_data()   Quest hands -> left/right wrist pose
  2. G1_29_ArmIK.solve_ik()                     wrist poses -> joint targets
  3. G1_29_ArmController.ctrl_dual_arm()        joint targets -> DDS to the robot

It changes nothing the teleop sends. Run it INSTEAD of run_teleop_with_neck.py,
press r in this terminal, then move your arms for ~20 s, then Ctrl+C/q. A verdict is
printed at exit:

    cd ~/hckang/xr_teleoperate
    python ~/hckang/jslee/urop/host/teleop_loop_probe.py \
        --xr-repo ~/hckang/xr_teleoperate --neck-pose-ip 192.168.123.164 \
        --neck-pose-port 5005 --arm=G1_29

    # add --fake-right-hand when the right Dex3 hand publishes no state (the controller would wait forever):
    #   same patch as run_teleop_fake_right_hand.py, so the loop above can be observed at all.

    python3 teleop_loop_probe.py --selftest      # verdict logic only, no robot
"""
from __future__ import annotations

import atexit
import sys
import threading
import time
from pathlib import Path

import numpy as np

POSE_MOVE_M = 0.02      # wrist translation span (m) below this = "pose is constant"
IK_MOVE_RAD = 0.02      # IK solution span (rad) below this = "IK output is constant"


class Stats:
    """Per-window and cumulative numbers; one instance, updated from the teleop loop thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total = self._new()
        self.window = self._new()
        self.first_error = {}

    @staticmethod
    def _new():
        return {"td_calls": 0, "td_bad": 0, "td_lo": None, "td_hi": None,
                "ik_calls": 0, "ik_exc": 0, "ik_nonfinite": 0, "ik_lo": None, "ik_hi": None, "ik_step_max": 0.0,
                "ctrl_calls": 0, "ctrl_exc": 0, "ctrl_dev_max": 0.0}

    @staticmethod
    def _range(d, lo, hi, v):
        d[lo] = v.copy() if d[lo] is None else np.minimum(d[lo], v)
        d[hi] = v.copy() if d[hi] is None else np.maximum(d[hi], v)

    def _each(self):
        return (self.total, self.window)

    def td(self, left, right):
        with self.lock:
            for d in self._each():
                d["td_calls"] += 1
                if left is None or right is None:
                    d["td_bad"] += 1
                    continue
                pos = np.concatenate([np.asarray(left, float)[:3, 3], np.asarray(right, float)[:3, 3]])
                if not np.isfinite(pos).all() or not pos.any():
                    d["td_bad"] += 1
                    continue
                self._range(d, "td_lo", "td_hi", pos)

    def ik(self, current_q, sol_q):
        with self.lock:
            for d in self._each():
                d["ik_calls"] += 1
                sol = np.asarray(sol_q, float)
                if not np.isfinite(sol).all():
                    d["ik_nonfinite"] += 1
                    continue
                self._range(d, "ik_lo", "ik_hi", sol)
                if current_q is not None:
                    d["ik_step_max"] = max(d["ik_step_max"], float(np.abs(sol - np.asarray(current_q, float)).max()))

    def ctrl(self, target, current):
        with self.lock:
            for d in self._each():
                d["ctrl_calls"] += 1
                if current is not None:
                    d["ctrl_dev_max"] = max(d["ctrl_dev_max"], float(np.abs(np.asarray(target, float) - np.asarray(current, float)).max()))

    def exception(self, kind, error):
        with self.lock:
            key = "ik_exc" if kind == "ik" else "ctrl_exc"
            for d in self._each():
                d[key] += 1
            self.first_error.setdefault(kind, f"{type(error).__name__}: {error}")

    def take_window(self):
        with self.lock:
            window, self.window = self.window, self._new()
            return window


def span(lo, hi):
    return 0.0 if lo is None else float(np.max(hi - lo))


def judge(total, first_error=None):
    out = []
    first_error = first_error or {}
    if total["td_calls"] == 0:
        return [("FAIL", "get_motion_state_data() was never called: the teleop never entered its follow loop. "
                 "It is still waiting at the start prompt (r not received by THIS terminal), or it failed "
                 "before the loop (see the traceback above).")]
    out.append(("OK", f"Follow loop is running ({total['td_calls']} cycles)."))
    pose = span(total["td_lo"], total["td_hi"])
    if total["td_bad"] == total["td_calls"]:
        out.append(("FAIL", "Every get_motion_state_data() returned an unusable wrist pose (None / NaN / all zeros): "
                    "the Quest hand data is not producing arm poses in TeleVuerWrapper."))
        return out
    if pose < POSE_MOVE_M:
        out.append(("FAIL", f"Wrist poses from TeleVuerWrapper never changed (translation span {pose:.3f} m) although the "
                    "loop ran: your hand motion is not reaching the teleop (constant/stale pose). The cause is "
                    "between the Quest event handler and get_motion_state_data (hand/head arrays not updated)."))
    else:
        out.append(("OK", f"Wrist poses change with your hands (span {pose:.2f} m)."))
    if total["ik_calls"] == 0:
        out.append(("FAIL", "solve_ik() was never called although the loop ran (different arm/IK path: "
                    "--upperbody-ik / --use-head-waist?)."))
    if total["ik_exc"]:
        out.append(("FAIL", f"solve_ik() raised {total['ik_exc']}x; first: {first_error.get('ik')}"))
    if total["ik_nonfinite"]:
        out.append(("FAIL", f"solve_ik() returned NaN/inf {total['ik_nonfinite']}x."))
    ik = span(total["ik_lo"], total["ik_hi"])
    if total["ik_calls"] and not total["ik_exc"]:
        if ik < IK_MOVE_RAD and pose >= POSE_MOVE_M:
            out.append(("FAIL", f"Wrist poses move but the IK joint targets do not (span {ik:.3f} rad): IK cannot "
                        "reach/follow these poses (workspace, frame/offset, or IK stuck at a limit)."))
        elif ik >= IK_MOVE_RAD:
            out.append(("OK", f"IK targets change (span {ik:.2f} rad, largest step vs current {total['ik_step_max']:.2f} rad)."))
    if total["ctrl_exc"]:
        out.append(("FAIL", f"ctrl_dual_arm() raised {total['ctrl_exc']}x; first: {first_error.get('ctrl')}"))
    elif total["ik_calls"] and total["ctrl_calls"] == 0:
        out.append(("FAIL", "IK ran but ctrl_dual_arm() was never called: targets are computed and never sent."))
    elif total["ctrl_calls"]:
        out.append(("OK", f"ctrl_dual_arm() sends targets (largest |target - current| {total['ctrl_dev_max']:.2f} rad). "
                    "If all of the above is OK but the robot still stays put, check the robot side with "
                    "teleop_chain_probe.py running alongside."))
    return out


def format_window(w, elapsed):
    if w["td_calls"] == 0:
        return f"[loop] t={elapsed:4.0f}s no follow-loop cycles (waiting for r / not started)"
    return (f"[loop] t={elapsed:4.0f}s cycles/s={w['td_calls']} bad_pose={w['td_bad']} | "
            f"wrist span L+R={span(w['td_lo'], w['td_hi']):.3f}m | "
            f"IK calls={w['ik_calls']} exc={w['ik_exc']} nan={w['ik_nonfinite']} span={span(w['ik_lo'], w['ik_hi']):.3f}rad "
            f"| ctrl calls={w['ctrl_calls']} max|tgt-cur|={w['ctrl_dev_max']:.2f}rad")


def install_hooks(stats, wrapper_cls, ik_cls, ctrl_cls):
    """Wrap the three methods in place; each wrapper logs then calls the original unchanged."""
    original_td = wrapper_cls.get_motion_state_data
    original_ik = ik_cls.solve_ik
    original_ctrl = ctrl_cls.ctrl_dual_arm

    def get_motion_state_data(self, *args, **kwargs):
        data = original_td(self, *args, **kwargs)
        try:
            stats.td(getattr(data, "left_arm_pose", None), getattr(data, "right_arm_pose", None))
        except Exception:  # noqa: BLE001 - a probe must never break the teleop
            pass
        return data

    def solve_ik(self, left, right, current_q=None, current_dq=None, *args, **kwargs):
        try:
            result = original_ik(self, left, right, current_q, current_dq, *args, **kwargs)
        except Exception as error:  # noqa: BLE001
            stats.exception("ik", error)
            raise
        try:
            stats.ik(current_q, result[0])
        except Exception:  # noqa: BLE001
            pass
        return result

    def ctrl_dual_arm(self, q_target, tauff_target, *args, **kwargs):
        try:
            try:
                current = self.get_current_dual_arm_q()
            except Exception:  # noqa: BLE001
                current = None
            stats.ctrl(q_target, current)
        except Exception:  # noqa: BLE001
            pass
        try:
            return original_ctrl(self, q_target, tauff_target, *args, **kwargs)
        except Exception as error:  # noqa: BLE001
            stats.exception("ctrl", error)
            raise

    wrapper_cls.get_motion_state_data = get_motion_state_data
    ik_cls.solve_ik = solve_ik
    ctrl_cls.ctrl_dual_arm = ctrl_dual_arm


def start_reporter(stats):
    start = time.monotonic()

    def run():
        while True:
            time.sleep(1.0)
            print(format_window(stats.take_window(), time.monotonic() - start), flush=True)

    threading.Thread(target=run, name="loop-probe-report", daemon=True).start()


def print_verdict(stats):
    print("\n=== verdict (teleop follow loop) ===", flush=True)
    for level, text in judge(stats.total, stats.first_error):
        print(f"[{level}] {text}", flush=True)


# --------------------------------------------------------------------------- self-test

def selftest():
    def pose(x):
        m = np.eye(4); m[:3, 3] = [x, 0.2, 0.3]
        return m

    class Data:
        def __init__(self, x): self.left_arm_pose, self.right_arm_pose = pose(x), pose(-x)

    class Wrapper:
        x = 0.0
        def get_motion_state_data(self): return Data(Wrapper.x)

    class IK:
        mode = "follow"
        def solve_ik(self, l, r, q, dq):
            if IK.mode == "raise": raise RuntimeError("boom")
            sol = np.zeros(14) if IK.mode == "const" else np.full(14, l[0, 3])
            return sol, np.zeros(14)

    class Ctrl:
        def get_current_dual_arm_q(self): return np.zeros(14)
        def ctrl_dual_arm(self, q, tau): pass

    def run(scenario):
        stats = Stats()
        install_hooks(stats, type("W", (Wrapper,), {}), type("I", (IK,), {}), type("C", (Ctrl,), {}))
        return stats

    results = []

    def scenario(name, cycles, x_fn, ik_mode, expect, ctrl=True):
        class W(Wrapper): pass
        class I(IK): pass
        class C(Ctrl): pass
        stats = Stats()
        install_hooks(stats, W, I, C)
        IK.mode = ik_mode
        w, i, c = W(), I(), C()
        for k in range(cycles):
            Wrapper.x = x_fn(k)
            d = w.get_motion_state_data()
            try:
                sol, tau = i.solve_ik(d.left_arm_pose, d.right_arm_pose, np.zeros(14), np.zeros(14))
            except RuntimeError:
                continue
            if ctrl:
                c.ctrl_dual_arm(sol, tau)
        text = " ".join(m for _, m in judge(stats.total, stats.first_error))
        ok = expect in text
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}")

    scenario("healthy", 50, lambda k: 0.01 * k, "follow", "ctrl_dual_arm() sends targets")
    scenario("loop never entered", 0, lambda k: 0.0, "follow", "never entered its follow loop")
    scenario("pose constant", 50, lambda k: 0.3, "follow", "never changed")
    scenario("IK does not follow", 50, lambda k: 0.01 * k, "const", "IK joint targets do not")
    scenario("IK raises", 50, lambda k: 0.01 * k, "raise", "solve_ik() raised")
    scenario("targets never sent", 50, lambda k: 0.01 * k, "follow", "never called: targets", ctrl=False)
    return 0 if all(results) else 1


# --------------------------------------------------------------------------- main

def main():
    if "--selftest" in sys.argv:
        return selftest()

    launcher_dir = Path(__file__).resolve().parent / "third_party" / "unitree_g1_urop" / "host"
    if not (launcher_dir / "run_teleop_with_neck.py").is_file():
        print(f"ERROR: launcher not found in {launcher_dir}")
        return 2
    argv = sys.argv[1:]
    fake_right = "--fake-right-hand" in argv
    argv = [a for a in argv if a != "--fake-right-hand"]
    repo = None
    for i, a in enumerate(argv):
        if a == "--xr-repo" and i + 1 < len(argv):
            repo = argv[i + 1]
        elif a.startswith("--xr-repo="):
            repo = a.split("=", 1)[1]
    if repo is None:
        print("ERROR: pass --xr-repo ~/hckang/xr_teleoperate (same arguments as run_teleop_with_neck.py)")
        return 2
    repo = Path(repo).expanduser().resolve()

    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    # The imports below create loggers, so a later basicConfig() (the launcher and the teleop script
    # both call it) would raise "Logging already started". It is already configured: make it a no-op.
    logging_mp.basicConfig = lambda *args, **kwargs: None
    sys.path.insert(0, str(launcher_dir))
    sys.path.insert(0, str(repo))
    import televuer
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    if fake_right:
        import run_teleop_fake_right_hand as stub_module
        import teleop.robot_control.robot_hand_unitree as hand_module
        problem = stub_module.check_shape(hand_module.Dex3_1_Controller)
        if problem:
            print(f"ERROR: {problem}", flush=True)
            return 2
        hand_module.Dex3_1_Controller = stub_module.make_stub(
            hand_module.Dex3_1_Controller, stub_module.DEFAULT_FAKE_Q, log=lambda msg: print(msg, flush=True),
            left_index=getattr(hand_module, "Dex3_1_Left_JointIndex", None),
            right_index=getattr(hand_module, "Dex3_1_Right_JointIndex", None))
        print("[loop-probe] right Dex3 hand state is faked (--fake-right-hand)", flush=True)

    stats = Stats()
    install_hooks(stats, televuer.TeleVuerWrapper, G1_29_ArmIK, G1_29_ArmController)
    start_reporter(stats)
    atexit.register(print_verdict, stats)
    print("[loop-probe] hooks installed (G1_29 arm only). Press r in this terminal, then move your arms.", flush=True)

    import run_teleop_with_neck as launcher
    return launcher.main(argv)


if __name__ == "__main__":
    sys.exit(main())
