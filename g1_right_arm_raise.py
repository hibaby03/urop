"""G1-29: raise the right arm via rt/arm_sdk; preview unless --execute is set."""

import argparse
from dataclasses import dataclass
import json
import math
import threading
import time


# G1 arm_sdk has one shared upper-body blend weight, not a per-arm weight.
# Hold the left arm and waist at their initial measured positions throughout.
UPPER_JOINTS = tuple(range(12, 29))
RIGHT_JOINTS = tuple(range(22, 29))
WEIGHT_JOINT = 29
CONTROL_DT = 0.02  # 50 Hz, as in Unitree's arm_sdk example.
STATE_TIMEOUT = 0.25
MAX_SPEED = 0.4  # rad/s, bound on the commanded smoothstep trajectory.


def finite_range(low, high):
    def parse(value):
        number = float(value)
        if not math.isfinite(number) or not low <= number <= high:
            raise argparse.ArgumentTypeError("must be finite and in [%s, %s]" % (low, high))
        return number
    return parse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", help="Robot-facing DDS interface, e.g. eth0 on PC2")
    parser.add_argument("--domain", type=int, choices=range(233), default=0, metavar="0..232")
    parser.add_argument("--raise-deg", type=finite_range(10, 90), default=90.0,
                        help="Absolute forward shoulder elevation from zero (default: 90)")
    parser.add_argument("--elbow-deg", type=finite_range(0, 90), default=20.0)
    parser.add_argument("--move-seconds", type=finite_range(4, 60), default=6.0,
                        help="Minimum duration of each lift/return; extended for speed limit")
    parser.add_argument("--hold-seconds", type=finite_range(0, 30), default=2.0)
    parser.add_argument("--state-wait", type=finite_range(1, 60), default=10.0)
    parser.add_argument("--execute", action="store_true", help="Publish real motor commands")
    parser.add_argument("--probe", action="store_true",
                        help="Read robot state and query modes without motor commands")
    args = parser.parse_args(argv)
    if (args.execute or args.probe) and not args.interface:
        parser.error("--execute/--probe requires --interface")
    if args.execute and args.probe:
        parser.error("--probe and --execute cannot be used together")
    return args


def right_target(args):
    # Shoulder pitch about +Y: negative rotates the hanging arm forward.
    # Slight outward roll and elbow flexion avoid a fully locked straight arm.
    return (-math.radians(args.raise_deg), -0.1, 0.0,
            math.radians(args.elbow_deg), 0.0, 0.0, 0.0)


def blend(start, end, fraction):
    fraction = min(1.0, max(0.0, fraction))
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return tuple(a + (b - a) * smooth for a, b in zip(start, end))


@dataclass(frozen=True)
class Snapshot:
    q: tuple
    mode_machine: int
    tick: int
    received_at: float


class StateReceiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.error = None

    def receive(self, message):
        try:
            if len(message.motor_state) < 29:
                raise ValueError("G1-29 motor_state needs at least 29 entries")
            q = tuple(float(message.motor_state[joint].q) for joint in UPPER_JOINTS)
            if not all(math.isfinite(value) for value in q):
                raise ValueError("non-finite upper-body position")
            sample = Snapshot(q, int(message.mode_machine), int(message.tick), time.monotonic())
            with self.lock:
                if self.state is not None:
                    delta = (sample.tick - self.state.tick) & 0xFFFFFFFF
                    if not 0 < delta < 0x80000000:
                        return  # Repeated/old packets must not refresh the watchdog.
                self.state = sample
        except Exception as error:
            with self.lock:
                self.error = str(error)

    def snapshot(self, mode_machine=None):
        with self.lock:
            sample, error = self.state, self.error
        if error:
            raise RuntimeError("Invalid lowstate: " + error)
        if sample is None:
            raise RuntimeError("No lowstate received")
        if time.monotonic() - sample.received_at > STATE_TIMEOUT:
            raise RuntimeError("Lowstate stopped advancing for more than 0.25 s")
        if mode_machine is not None and sample.mode_machine != mode_machine:
            raise RuntimeError("Robot mode_machine changed during arm control")
        return sample

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.state is not None or self.error is not None
            if ready:
                return self.snapshot()
            time.sleep(CONTROL_DT)
        raise RuntimeError("No G1 lowstate within %.1f s; no command was sent" % timeout)


class ArmMotion:
    def __init__(self, publisher, command, crc, receiver, initial):
        self.publisher = publisher
        self.command = command
        self.crc = crc
        self.receiver = receiver
        self.initial = initial
        self.last_q = initial.q
        self.weight = 0.0
        self.attempted = False
        self.report_at = 0.0
        command.mode_pr = 0  # PR joint control, not ankle AB control.
        command.mode_machine = initial.mode_machine
        for joint in UPPER_JOINTS:
            motor = command.motor_cmd[joint]
            motor.mode = 1
            motor.kp = 40.0 if joint in (19, 20, 21, 26, 27, 28) else 60.0
            motor.kd = 1.5
            motor.dq = 0.0
            motor.tau = 0.0

    def write(self, q, weight):
        for joint, position in zip(UPPER_JOINTS, q):
            self.command.motor_cmd[joint].q = position
        self.command.motor_cmd[WEIGHT_JOINT].q = weight
        self.command.crc = self.crc.Crc(self.command)
        self.attempted = True
        if not self.publisher.Write(self.command):
            raise RuntimeError("DDS command write failed")
        self.last_q, self.weight = tuple(q), weight

    def segment(self, start, end, duration, weight_start, weight_end):
        elapsed = 0.0
        while True:
            sample = self.receiver.snapshot(self.initial.mode_machine)
            fraction = min(1.0, elapsed / duration)
            self.write(blend(start, end, fraction),
                       weight_start + (weight_end - weight_start) * fraction)
            if time.monotonic() >= self.report_at:
                shoulder = UPPER_JOINTS.index(22)
                print("[feedback] tick=%d mode_machine=%d weight=%.2f "
                      "right_shoulder_pitch: cmd=%.1f deg actual=%.1f deg" % (
                          sample.tick, sample.mode_machine, self.weight,
                          math.degrees(self.last_q[shoulder]),
                          math.degrees(sample.q[shoulder])), flush=True)
                self.report_at = time.monotonic() + 1.0
            if fraction >= 1.0:
                return
            # No catch-up bursts if scheduling is delayed.
            time.sleep(CONTROL_DT)
            elapsed += CONTROL_DT

    def run(self, args):
        initial = self.initial.q
        target = list(initial)
        for joint, position in zip(RIGHT_JOINTS, right_target(args)):
            target[UPPER_JOINTS.index(joint)] = position
        # max derivative of cubic smoothstep is 1.5; use the largest joint delta.
        duration = max(args.move_seconds,
                       1.5 * max(abs(a - b) for a, b in zip(initial, target)) / MAX_SPEED)
        print("Taking arm_sdk control; lift/return duration %.2f s each" % duration, flush=True)
        self.segment(initial, initial, 2.0, 0.0, 1.0)
        print("Raising right arm", flush=True)
        self.segment(initial, target, duration, 1.0, 1.0)
        shoulder = UPPER_JOINTS.index(22)
        measured = self.receiver.snapshot(self.initial.mode_machine).q[shoulder]
        if (abs(target[shoulder] - initial[shoulder]) > math.radians(5)
                and abs(measured - initial[shoulder]) < math.radians(1)):
            raise RuntimeError(
                "NO_ARM_RESPONSE: shoulder feedback changed less than 1 degree despite "
                "a target change greater than 5 degrees. DDS Write success does not confirm "
                "robot acceptance. Check high-level mode, arm_sdk reader/network, other "
                "arm publishers and firmware. Run --probe without --execute for mode queries.")
        if args.hold_seconds:
            print("Holding right arm", flush=True)
            self.segment(target, target, args.hold_seconds, 1.0, 1.0)
        print("Returning to initial pose", flush=True)
        self.segment(target, initial, duration, 1.0, 1.0)
        print("Releasing arm_sdk control", flush=True)
        self.segment(initial, initial, 2.0, 1.0, 0.0)

    def abort(self):
        # Do not continue the trajectory with missing feedback or after Ctrl+C.
        # Request immediate hand-back to the active high-level controller.
        # Delivery is best-effort; this is NOT the robot's emergency stop.
        if not self.attempted:
            return
        for _ in range(5):
            try:
                self.write(self.last_q, 0.0)
            except Exception as error:
                print("WARNING: could not release arm_sdk: %s" % error, flush=True)
            time.sleep(CONTROL_DT)


def query_modes():
    # Get-only RPC calls. Do not call ReleaseMode, SelectMode or SetFsmId.
    queries = (
        ("motion_switcher", "unitree_sdk2py.comm.motion_switcher.motion_switcher_client",
         "MotionSwitcherClient", "CheckMode"),
        ("loco_fsm", "unitree_sdk2py.g1.loco.g1_loco_client", "LocoClient", "GetFsmId"),
    )
    import importlib
    for name, module_name, class_name, method in queries:
        try:
            client = getattr(importlib.import_module(module_name), class_name)()
            client.SetTimeout(2.0)
            client.Init()
            code, data = getattr(client, method)()
            print(json.dumps({"query": name, "code": code, "data": data}), flush=True)
        except Exception as error:
            print(json.dumps({"query": name, "error": str(error)}), flush=True)


def execute(args):
    # Delay SDK imports: preview and --help work on a computer without DDS.
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(args.domain, networkInterface=args.interface)
    receiver = StateReceiver()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    publisher = None
    motion = None
    try:
        subscriber.Init(receiver.receive, 1)
        print("Waiting for fresh G1 lowstate...", flush=True)
        receiver.wait(args.state_wait)
        if args.probe:
            sample = receiver.snapshot()
            print(json.dumps({"mode": "probe_no_motor_commands", "tick": sample.tick,
                              "mode_machine": sample.mode_machine,
                              "upper_joint_indices": UPPER_JOINTS,
                              "upper_q_deg": [round(math.degrees(q), 2) for q in sample.q]}),
                  flush=True)
            query_modes()
            return
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        initial = receiver.snapshot()  # Refresh after publisher discovery.
        motion = ArmMotion(publisher, unitree_hg_msg_dds__LowCmd_(), CRC(), receiver, initial)
        try:
            motion.run(args)
        except BaseException:
            motion.abort()
            raise
        print("Done: return trajectory completed; arm_sdk weight is zero.", flush=True)
    finally:
        try:
            if publisher is not None:
                publisher.Close()
        finally:
            subscriber.Close()


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps({
        "execute": args.execute, "probe": args.probe, "command_topic": "rt/arm_sdk",
        "state_topic": "rt/lowstate", "interface": args.interface, "domain": args.domain,
        "right_joint_indices": RIGHT_JOINTS,
        "right_target_rad": right_target(args),
        "sequence": "take control -> lift -> hold -> return -> release",
    }, indent=2), flush=True)
    if not args.execute and not args.probe:
        print("Preview only. Add --interface eth0 --execute on the robot-connected PC2.")
        return 0
    if args.execute:
        print("G1-29 must already be standing in a stable high-level balance mode. "
              "Stop other arm publishers; keep the arm workspace clear and the remote stop ready.",
              flush=True)
    execute(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; requested arm_sdk release.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print("ERROR: %s: %s" % (type(error).__name__, error), flush=True)
        raise SystemExit(2)
