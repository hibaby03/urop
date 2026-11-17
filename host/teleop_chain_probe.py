#!/usr/bin/env python3
"""Passive monitor: why does the robot not follow the hands during teleop?

Run it on the Host in a SECOND terminal while the real teleop (main.py /
xr_teleoperate) is running. It only SUBSCRIBES to DDS (plus one read-only
motion-switcher query); it never publishes, never changes robot mode.

It watches, on the robot DDS network, every hop after the Quest:

  1. rt/lowstate            robot -> Host   can this Host see the robot at all?
  2. rt/lowcmd | rt/arm_sdk Host  -> robot  is the teleop publishing arm commands?
  3. command joints change  hands -> IK     do the commanded targets follow the hands?
  4. lowstate joints change robot reacts    does the robot actually move to them?
  5. rt/dex3/{left,right}/{cmd,state}       the same for the Dex3 hands
  6. MotionSwitcher mode                    is the built-in controller still active?

and prints which hop breaks first.

Usage:
    1) python3 teleop_chain_probe.py --seconds 40 [--interface enp3s0] [--domain 0]
    2) start the teleop in another terminal, press r, then move BOTH arms/hands
       slowly for the whole run (the teleop ramps arm speed up for the first seconds).
    python3 teleop_chain_probe.py --selftest     # checks only the verdict logic (no robot/SDK)

Interface/domain must match what the teleop uses (config.yaml: dds.interface/domain).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np

ARM_IDX = list(range(15, 29))   # G1_29 left arm 15-21, right arm 22-28 (G1_29_JointArmIndex)
ARM_SDK_WEIGHT_IDX = 29         # rt/arm_sdk: motor_cmd[29].q is the blend weight

MOVE_RAD = 0.05                 # command span above this = "the command is moving"
FOLLOW_RATIO = 0.3              # state span / command span below this = "robot not following"
TRACK_ERR_RAD = 0.2             # mean |cmd - state| above this = poor tracking
MIN_RATE_HZ = 5.0


# --------------------------------------------------------------------------- analysis (pure, no SDK)

def follow_metrics(cmd, state):
    """cmd/state: (T, N) arrays sampled at the same instants."""
    cmd, state = np.asarray(cmd, float), np.asarray(state, float)
    if cmd.ndim != 2 or len(cmd) < 3 or cmd.shape != state.shape:
        return None
    tail = slice(int(len(cmd) * 0.7), None)
    return {
        "samples": int(len(cmd)),
        "cmd_span": float(np.ptp(cmd, axis=0).max()),
        "state_span": float(np.ptp(state, axis=0).max()),
        "mean_err": float(np.abs(cmd[tail] - state[tail]).mean()),
    }


def judge(s):
    """s: summary dict -> list of (level, message). level: FAIL / WARN / OK / INFO."""
    out = []
    seconds = max(s["seconds"], 1e-9)
    ls_hz = s["lowstate_count"] / seconds

    if s["lowstate_count"] == 0:
        out.append(("FAIL", "No rt/lowstate received: this Host cannot see the robot over DDS "
                    "(wrong --interface/--domain, Wi-Fi/multicast blocked, robot off). The teleop "
                    "cannot control anything either; fix this first (lowstate_probe.py / wireless_dds_probe.py)."))
        return out
    if ls_hz < MIN_RATE_HZ:
        out.append(("WARN", f"rt/lowstate only {ls_hz:.1f} Hz: the DDS link is very lossy/slow."))
    else:
        out.append(("OK", f"rt/lowstate received ({ls_hz:.0f} Hz): the Host sees the robot."))

    mode = s.get("mode")
    if mode is not None and mode != "":
        out.append(("WARN", f"Motion switcher reports an ACTIVE built-in controller: '{mode}'. In plain "
                    "lowcmd mode the robot is expected to be in debug/idle (no built-in controller); otherwise it "
                    "can ignore or fight arm lowcmd. Release it, or run the teleop with --motion (rt/arm_sdk)."))
    elif "mode" in s and mode == "":
        out.append(("OK", "Motion switcher: no built-in controller active (debug/idle)."))

    cmd_topic = s.get("cmd_topic")
    if cmd_topic is None:
        out.append(("FAIL", "No arm command seen on rt/lowcmd or rt/arm_sdk: the teleop is NOT publishing. "
                    "Likely: the teleop never got 'r' (look for 'Program start signal received' in its log; "
                    "main.py types r into a pty), it is still blocked at start-up, it crashed, or it uses a "
                    "different --domain/--interface than this monitor."))
        return out
    out.append(("OK", f"Arm commands are published on {cmd_topic} ({s['cmd_count'] / seconds:.0f} Hz)."))

    if s.get("cmd_kp_max") is not None and s["cmd_kp_max"] <= 0:
        out.append(("FAIL", "Arm commands carry kp=0 for every joint: no stiffness, the arm cannot move."))
    if cmd_topic == "rt/arm_sdk" and s.get("arm_sdk_weight_max") is not None and s["arm_sdk_weight_max"] < 0.05:
        out.append(("FAIL", "rt/arm_sdk blend weight stays ~0: the robot ignores the arm commands."))

    m = s.get("arm")
    if m is None:
        out.append(("WARN", "Too few samples to judge arm motion (is rt/lowstate also flowing while commands arrive?)."))
    elif m["cmd_span"] < MOVE_RAD:
        out.append(("FAIL", f"Arm command targets never changed (span {m['cmd_span']:.3f} rad): the teleop "
                    "publishes a constant pose, so the hands are not reaching IK. Quest/TeleVuer side: "
                    "run hand_tracking_probe.py (hand events, hand/arm pose changing). Also check the tracking "
                    "gate and that you actually moved your arms during this run."))
    elif m["state_span"] < FOLLOW_RATIO * m["cmd_span"]:
        out.append(("FAIL", f"Commands move (span {m['cmd_span']:.2f} rad) but the robot does not "
                    f"(state span {m['state_span']:.2f} rad): the robot receives and ignores the arm commands. "
                    "Check the motion-switcher mode (above), mode_machine/CRC mismatch, e-stop/damping state, "
                    "and --motion vs debug mode."))
    elif m["mean_err"] > TRACK_ERR_RAD:
        out.append(("WARN", f"Robot moves but tracks poorly (mean error {m['mean_err']:.2f} rad): low gains/"
                    "speed limit, joint limits, or an obstacle."))
    else:
        out.append(("OK", f"Arm follows its commands (cmd span {m['cmd_span']:.2f}, state span "
                    f"{m['state_span']:.2f}, mean error {m['mean_err']:.3f} rad)."))

    for side in ("left", "right"):
        h = s["hand"][side]
        if h["cmd_count"] == 0:
            out.append(("WARN", f"Dex3 {side}: no command on rt/dex3/{side}/cmd (hand controller not publishing)."))
        elif h["state_count"] == 0:
            out.append(("WARN", f"Dex3 {side}: commands sent but no rt/dex3/{side}/state (hand off/not connected)."))
        elif h["metrics"] is not None:
            hm = h["metrics"]
            if hm["cmd_span"] < MOVE_RAD:
                out.append(("WARN", f"Dex3 {side}: commands never changed (span {hm['cmd_span']:.3f}): hand pose "
                            "not reaching the retargeter (or you did not move that hand)."))
            elif hm["state_span"] < FOLLOW_RATIO * hm["cmd_span"]:
                out.append(("WARN", f"Dex3 {side}: commands move but the hand does not."))
            else:
                out.append(("OK", f"Dex3 {side}: follows its commands."))
    return out


# --------------------------------------------------------------------------- DDS side

class Feed:
    def __init__(self, name, extract):
        self.name, self.extract = name, extract
        self.lock = threading.Lock()
        self.count = self.errors = 0
        self.last = None
        self.latest = None
        self.last_error = None

    def callback(self, message):
        try:
            data = self.extract(message)
        except Exception as error:  # noqa: BLE001
            with self.lock:
                self.errors += 1
                self.last_error = f"{type(error).__name__}: {error}"
            return
        with self.lock:
            self.count += 1
            self.last = time.monotonic()
            self.latest = data

    def snapshot(self):
        with self.lock:
            return self.count, self.latest


def _lowstate(m):
    return {"q": np.array([m.motor_state[i].q for i in ARM_IDX], float)}


def _lowcmd(m):
    cmd = m.motor_cmd
    return {"q": np.array([cmd[i].q for i in ARM_IDX], float),
            "kp": np.array([cmd[i].kp for i in ARM_IDX], float),
            "weight": float(cmd[ARM_SDK_WEIGHT_IDX].q) if len(cmd) > ARM_SDK_WEIGHT_IDX else None}


def _handstate(m):
    return {"q": np.array([x.q for x in m.motor_state], float)}


def _handcmd(m):
    return {"q": np.array([x.q for x in m.motor_cmd], float)}


def mode_check():
    """Read-only motion-switcher query. Returns the active mode name, '' if none, None if unavailable."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    client = MotionSwitcherClient()
    client.SetTimeout(3.0)
    client.Init()
    status, result = client.CheckMode()
    if status != 0 or not isinstance(result, dict):
        raise RuntimeError(f"CheckMode status={status} result={result!r}")
    return result.get("name") or ""


def monitor(args):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, HandState_, HandCmd_

    if args.interface:
        ChannelFactoryInitialize(args.domain, args.interface)
    else:
        ChannelFactoryInitialize(args.domain)

    feeds = {
        "lowstate": ("rt/lowstate", LowState_, _lowstate),
        "lowcmd": ("rt/lowcmd", LowCmd_, _lowcmd),
        "arm_sdk": ("rt/arm_sdk", LowCmd_, _lowcmd),
        "l_state": ("rt/dex3/left/state", HandState_, _handstate),
        "r_state": ("rt/dex3/right/state", HandState_, _handstate),
        "l_cmd": ("rt/dex3/left/cmd", HandCmd_, _handcmd),
        "r_cmd": ("rt/dex3/right/cmd", HandCmd_, _handcmd),
    }
    feed, subs = {}, []
    for key, (topic, msg_type, extract) in feeds.items():
        feed[key] = Feed(topic, extract)
        sub = ChannelSubscriber(topic, msg_type)
        sub.Init(feed[key].callback, 10)
        subs.append(sub)

    summary = {"mode": None}
    if not args.no_mode_check:
        try:
            summary["mode"] = mode_check()
            print(f"[mode] motion switcher active mode: {summary['mode']!r}", flush=True)
        except Exception as error:  # noqa: BLE001
            summary.pop("mode")
            print(f"[mode] check unavailable ({type(error).__name__}: {error}); skipped", flush=True)

    print(f"Monitoring {args.seconds:.0f}s on domain {args.domain}, "
          f"interface {args.interface or 'default'}. Start the teleop, press r and MOVE your arms/hands.\n", flush=True)
    series = {"lowcmd": ([], []), "arm_sdk": ([], []), "l": ([], []), "r": ([], [])}
    start = time.monotonic()
    next_print = start + 1.0
    last_counts = {k: 0 for k in feed}
    try:
        while time.monotonic() - start < args.seconds:
            time.sleep(0.1)
            _, ls = feed["lowstate"].snapshot()
            for key in ("lowcmd", "arm_sdk"):
                _, cmd = feed[key].snapshot()
                if cmd is not None and ls is not None:
                    series[key][0].append(cmd["q"]); series[key][1].append(ls["q"])
            for side, state_key, cmd_key in (("l", "l_state", "l_cmd"), ("r", "r_state", "r_cmd")):
                _, st = feed[state_key].snapshot()
                _, cm = feed[cmd_key].snapshot()
                if st is not None and cm is not None and len(st["q"]) == len(cm["q"]):
                    series[side][0].append(cm["q"]); series[side][1].append(st["q"])
            if time.monotonic() >= next_print:
                next_print += 1.0
                counts = {k: f.snapshot()[0] for k, f in feed.items()}
                rate = {k: counts[k] - last_counts[k] for k in counts}
                last_counts = counts
                arm = follow_metrics(*[np.array(x) for x in series["lowcmd" if counts["lowcmd"] >= counts["arm_sdk"] else "arm_sdk"]]) \
                    if len(series["lowcmd"][0]) > 3 or len(series["arm_sdk"][0]) > 3 else None
                print(f"t={time.monotonic() - start:3.0f}s lowstate +{rate['lowstate']}/s | "
                      f"lowcmd +{rate['lowcmd']}/s arm_sdk +{rate['arm_sdk']}/s | "
                      f"dex3 cmd L/R +{rate['l_cmd']}/{rate['r_cmd']}/s state +{rate['l_state']}/{rate['r_state']}/s | "
                      + (f"arm cmd span {arm['cmd_span']:.2f} state span {arm['state_span']:.2f} "
                         f"err {arm['mean_err']:.2f}" if arm else "arm: no samples yet"), flush=True)
    finally:
        elapsed = time.monotonic() - start
        for sub in subs:
            try:
                sub.Close()
            except Exception:  # noqa: BLE001
                pass

    counts = {k: f.snapshot()[0] for k, f in feed.items()}
    errors = {f.name: f.last_error for f in feed.values() if f.errors}
    if errors:
        print(f"\nmessage decode errors (SDK message layout differs?): {errors}")
    cmd_key = "lowcmd" if counts["lowcmd"] >= counts["arm_sdk"] else "arm_sdk"
    cmd_topic = feed[cmd_key].name if counts[cmd_key] else None
    latest = feed[cmd_key].snapshot()[1]
    weights = [c for c in ([latest["weight"]] if latest and latest["weight"] is not None else [])]
    summary.update(
        seconds=elapsed, lowstate_count=counts["lowstate"], cmd_topic=cmd_topic, cmd_count=counts[cmd_key],
        cmd_kp_max=float(latest["kp"].max()) if latest else None,
        arm_sdk_weight_max=max(weights) if weights else None,
        arm=follow_metrics(*[np.array(x) for x in series[cmd_key]]) if series[cmd_key][0] else None,
        hand={side: {"cmd_count": counts[f"{side}_cmd"], "state_count": counts[f"{side}_state"],
                     "metrics": follow_metrics(*[np.array(x) for x in series[side]]) if series[side][0] else None}
              for side in ("l", "r")},
    )
    summary["hand"] = {"left": summary["hand"]["l"], "right": summary["hand"]["r"]}
    return summary


def report(summary):
    print("\n=== verdict ===")
    results = judge(summary)
    for level, text in results:
        print(f"[{level}] {text}")
    return 1 if any(level == "FAIL" for level, _ in results) else 0


# --------------------------------------------------------------------------- self-test (no SDK)

def selftest():
    t = np.linspace(0, 6, 60)[:, None]
    moving = 0.4 * np.sin(t) * np.ones((1, 14))
    lag = 0.4 * np.sin(t - 0.2) * np.ones((1, 14))
    still = np.zeros((60, 14))
    hand = lambda c, s, m=None: {"cmd_count": c, "state_count": s, "metrics": m}
    ok_hand = hand(500, 500, follow_metrics(moving[:, :7], lag[:, :7]))
    base = dict(seconds=10.0, lowstate_count=5000, cmd_topic="rt/lowcmd", cmd_count=2500, cmd_kp_max=60.0,
                arm_sdk_weight_max=None, mode="", hand={"left": ok_hand, "right": ok_hand},
                arm=follow_metrics(moving, lag))
    cases = [
        ("healthy", base, None),
        ("no lowstate", dict(base, lowstate_count=0), "No rt/lowstate"),
        ("teleop not publishing", dict(base, cmd_topic=None, cmd_count=0), "NOT publishing"),
        ("constant command", dict(base, arm=follow_metrics(still, still)), "never changed"),
        ("robot ignores command", dict(base, arm=follow_metrics(moving, still)), "does not"),
        ("kp = 0", dict(base, cmd_kp_max=0.0), "kp=0"),
        ("builtin controller active", dict(base, mode="ai"), "ACTIVE built-in controller"),
        ("hand not publishing", dict(base, hand={"left": hand(0, 500), "right": ok_hand}), "Dex3 left: no command"),
    ]
    failed = 0
    for name, summary, expect in cases:
        text = " ".join(m for _, m in judge(summary))
        levels = [lv for lv, _ in judge(summary)]
        good = ("FAIL" not in levels and "WARN" not in levels) if expect is None else expect in text
        failed += not good
        print(f"{'ok  ' if good else 'FAIL'} {name}")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--interface", default=None, help="DDS NIC (config.yaml dds.interface); default: SDK default")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--no-mode-check", action="store_true", help="skip the read-only motion-switcher query")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    try:
        summary = monitor(args)
    except ImportError as error:
        print(f"ERROR: unitree_sdk2py not importable ({error}). Use the teleop's Python environment.")
        return 2
    except KeyboardInterrupt:
        print("stopped by user")
        return 130
    print(json.dumps({k: v for k, v in summary.items() if k != "hand"}, default=str))
    return report(summary)


if __name__ == "__main__":
    sys.exit(main())
