import copy
from contextlib import contextmanager
import math
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import g1_right_arm_raise as arm


def state(tick=1, mode=5, q=0.0):
    return SimpleNamespace(tick=tick, mode_machine=mode,
                           motor_state=[SimpleNamespace(q=q) for _ in range(35)])


def command():
    return SimpleNamespace(mode_pr=0, mode_machine=0, crc=0,
                           motor_cmd=[SimpleNamespace(mode=0, q=0.0, dq=0.0, tau=0.0,
                                                      kp=0.0, kd=0.0) for _ in range(35)])


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.ok = True

    def Write(self, message):
        self.messages.append(copy.deepcopy(message))
        return self.ok


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch.object(arm.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.receiver = arm.StateReceiver()
        self.receiver.receive(state(q=0.1))
        self.publisher = FakePublisher()
        self.crc = SimpleNamespace(Crc=lambda msg: 12345)
        self.motion = arm.ArmMotion(self.publisher, command(), self.crc,
                                    self.receiver, self.receiver.snapshot())

    def advance(self, dt):
        self.now += dt
        self.receiver.receive(state(tick=self.receiver.state.tick + 1, q=0.1))

    def test_preview_does_not_import_sdk_or_execute(self):
        with patch.object(arm, "execute") as execute, patch("builtins.print"):
            self.assertEqual(arm.main([]), 0)
        execute.assert_not_called()

    def test_invalid_arguments_and_execute_without_interface(self):
        for argv in (["--execute"], ["--raise-deg", "nan"], ["--raise-deg", "91"],
                     ["--hold-seconds", "inf"], ["--move-seconds", "0"]):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                arm.parse_args(argv)

    def test_full_trajectory_holds_left_arm_and_waist_and_returns(self):
        args = arm.parse_args([])
        def follow_command(dt):
            self.now += dt
            sample = state(tick=self.receiver.state.tick + 1, q=0.1)
            for joint in arm.UPPER_JOINTS:
                sample.motor_state[joint].q = self.publisher.messages[-1].motor_cmd[joint].q
            self.receiver.receive(sample)
        with patch.object(arm.time, "sleep", side_effect=follow_command), patch("builtins.print"):
            self.motion.run(args)
        messages = self.publisher.messages
        target = arm.right_target(args)
        self.assertEqual(messages[0].motor_cmd[29].q, 0.0)
        self.assertEqual(messages[-1].motor_cmd[29].q, 0.0)
        self.assertAlmostEqual(min(m.motor_cmd[22].q for m in messages), -math.pi / 2)
        raised = next(m for m in messages if abs(m.motor_cmd[22].q - target[0]) < 1e-12)
        for joint, expected in zip(arm.RIGHT_JOINTS, target):
            self.assertAlmostEqual(raised.motor_cmd[joint].q, expected)
        for msg in messages:
            self.assertEqual(msg.crc, 12345)
            self.assertEqual(msg.mode_machine, 5)
            self.assertEqual(msg.mode_pr, 0)
            self.assertTrue(0 <= msg.motor_cmd[29].q <= 1)
            for joint in range(12):
                self.assertEqual(vars(msg.motor_cmd[joint]), vars(command().motor_cmd[joint]))
            for joint in range(12, 22):
                self.assertEqual(msg.motor_cmd[joint].q, 0.1)
        for joint in arm.UPPER_JOINTS:
            self.assertAlmostEqual(messages[-1].motor_cmd[joint].q, 0.1)
        for before, after in zip(messages, messages[1:]):
            for joint in arm.RIGHT_JOINTS:
                speed = abs(after.motor_cmd[joint].q - before.motor_cmd[joint].q) / arm.CONTROL_DT
                self.assertLessEqual(speed, arm.MAX_SPEED + 1e-9)

    def test_duplicate_or_old_tick_does_not_refresh_watchdog(self):
        for tick in (1, 0):
            self.now += 0.15
            self.receiver.receive(state(tick=tick))
        with self.assertRaisesRegex(RuntimeError, "stopped advancing"):
            self.receiver.snapshot()

    def test_tick_wrap_is_accepted(self):
        self.receiver = arm.StateReceiver()
        self.receiver.receive(state(tick=0xFFFFFFFF))
        self.now += 0.1
        self.receiver.receive(state(tick=0))
        self.assertEqual(self.receiver.snapshot().tick, 0)

    def test_bad_state_and_mode_change_are_rejected(self):
        self.receiver.receive(state(tick=2, mode=6))
        with self.assertRaisesRegex(RuntimeError, "mode_machine changed"):
            self.receiver.snapshot(mode_machine=5)
        self.receiver.receive(state(tick=3, q=float("nan")))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self.receiver.snapshot()

    def test_no_initial_state_sends_nothing(self):
        receiver = arm.StateReceiver()
        with patch.object(arm.time, "sleep", side_effect=lambda dt: setattr(self, "now", self.now + dt)):
            with self.assertRaisesRegex(RuntimeError, "no command was sent"):
                receiver.wait(0.1)
        self.assertEqual(self.publisher.messages, [])

    def test_stale_feedback_stops_trajectory_and_abort_releases(self):
        def lose_feedback(dt):
            self.now += dt
        with patch.object(arm.time, "sleep", side_effect=lose_feedback):
            with self.assertRaisesRegex(RuntimeError, "stopped advancing"):
                self.motion.run(arm.parse_args([]))
            count = len(self.publisher.messages)
            self.motion.abort()
        self.assertEqual(len(self.publisher.messages), count + 5)
        self.assertTrue(all(m.motor_cmd[29].q == 0 for m in self.publisher.messages[-5:]))

    def test_write_failure_does_not_continue_motion(self):
        self.publisher.ok = False
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            self.motion.write(self.motion.initial.q, 0.5)
        self.assertTrue(self.motion.attempted)
        self.assertEqual(self.motion.weight, 0)

    def test_abort_before_first_write_sends_nothing(self):
        self.motion.abort()
        self.assertEqual(self.publisher.messages, [])

    def test_fresh_but_motionless_feedback_is_not_success(self):
        with patch.object(arm.time, "sleep", side_effect=self.advance), patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "NO_ARM_RESPONSE"):
                self.motion.run(arm.parse_args([]))

    def test_probe_requires_interface_and_excludes_execute(self):
        for argv in (["--probe"], ["--probe", "--execute", "--interface", "eth0"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                arm.parse_args(argv)

    @contextmanager
    def fake_sdk(self, deliver_state=True):
        subscriber = SimpleNamespace(Close=unittest.mock.Mock())
        def init_subscriber(callback, queue):
            if deliver_state:
                callback(state(q=0.1))
        subscriber.Init = init_subscriber
        self.publisher.Init = lambda: None
        self.publisher.Close = unittest.mock.Mock()
        factory = unittest.mock.Mock()
        create_publisher = unittest.mock.Mock(return_value=self.publisher)
        modules = {}
        for name in ("unitree_sdk2py", "unitree_sdk2py.core", "unitree_sdk2py.core.channel",
                     "unitree_sdk2py.idl", "unitree_sdk2py.idl.default",
                     "unitree_sdk2py.idl.unitree_hg", "unitree_sdk2py.idl.unitree_hg.msg",
                     "unitree_sdk2py.idl.unitree_hg.msg.dds_",
                     "unitree_sdk2py.utils", "unitree_sdk2py.utils.crc"):
            modules[name] = ModuleType(name)
        channel = modules["unitree_sdk2py.core.channel"]
        channel.ChannelFactoryInitialize = factory
        channel.ChannelPublisher = create_publisher
        channel.ChannelSubscriber = lambda *args: subscriber
        modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__LowCmd_ = command
        idl = modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"]
        idl.LowCmd_, idl.LowState_ = object, object
        modules["unitree_sdk2py.utils.crc"].CRC = lambda: self.crc
        with patch.dict("sys.modules", modules):
            yield subscriber, create_publisher, factory

    def test_execute_timeout_closes_subscriber_without_creating_publisher(self):
        args = arm.parse_args(["--interface", "eth0", "--execute", "--state-wait", "1"])
        with self.fake_sdk(deliver_state=False) as (subscriber, create_publisher, _):
            with patch.object(arm.time, "sleep", side_effect=lambda dt: setattr(self, "now", self.now + dt)):
                with self.assertRaisesRegex(RuntimeError, "No G1 lowstate"):
                    arm.execute(args)
            create_publisher.assert_not_called()
            subscriber.Close.assert_called_once()
        self.assertEqual(self.publisher.messages, [])

    def test_probe_queries_modes_without_motor_publisher(self):
        args = arm.parse_args(["--probe", "--interface", "eth0"])
        with self.fake_sdk() as (subscriber, create_publisher, _):
            with patch.object(arm, "query_modes") as query, patch("builtins.print"):
                arm.execute(args)
            query.assert_called_once_with()
            create_publisher.assert_not_called()
            subscriber.Close.assert_called_once()
        self.assertEqual(self.publisher.messages, [])

    def test_execute_interrupt_releases_and_closes_both_channels(self):
        def interrupt(motion, args):
            motion.write(motion.initial.q, 1.0)
            raise KeyboardInterrupt()
        args = arm.parse_args(["--interface", "eth0", "--execute"])
        with self.fake_sdk() as (subscriber, create_publisher, factory):
            with patch.object(arm.ArmMotion, "run", new=interrupt), \
                    patch.object(arm.time, "sleep", side_effect=self.advance):
                with self.assertRaises(KeyboardInterrupt):
                    arm.execute(args)
            factory.assert_called_once_with(0, networkInterface="eth0")
            self.assertEqual(create_publisher.call_args.args[0], "rt/arm_sdk")
            self.publisher.Close.assert_called_once()
            subscriber.Close.assert_called_once()
        self.assertEqual(len(self.publisher.messages), 6)
        self.assertEqual(self.publisher.messages[0].motor_cmd[29].q, 1.0)
        self.assertTrue(all(msg.motor_cmd[29].q == 0 for msg in self.publisher.messages[1:]))


if __name__ == "__main__":
    unittest.main()
