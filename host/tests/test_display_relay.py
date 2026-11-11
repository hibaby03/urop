"""The Quest display relay hooked into the teleop's Vuer (third_party/.../display_relay.py).

No Vuer, Quest or xr_teleoperate here: Vuer's schemas and session are faked; everything else
(ZMQ transport, display_feed parsing, relay_to_vuer) is the real code.
"""

import asyncio
import importlib.util
import socket
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOST_DIR = ROOT / "third_party" / "unitree_g1_urop" / "host"
LAUNCHER = HOST_DIR / "run_teleop_with_neck.py"
sys.path.insert(0, str(HOST_DIR))

import display_feed          # noqa: E402  (bundled verbatim copy)
import display_relay         # noqa: E402


class FakeHands:
    def __init__(self, **kw):
        self.kw = kw


class FakeImage:
    def __init__(self, **kw):
        self.kw = kw


def fake_schemas():
    return FakeHands, FakeImage


class FakeSession:
    """Records upserts; raises AssertionError (how Vuer reports a gone client) after ``stop_after_images``."""

    def __init__(self, stop_after_images=1):
        self.calls, self.stop_after_images, self.images = [], stop_after_images, 0

    def upsert(self, element, to=None):
        elements = element if isinstance(element, list) else [element]
        self.calls.append((elements, to))
        if any(isinstance(e, FakeImage) for e in elements):
            self.images += 1
            if self.images >= self.stop_after_images:
                raise AssertionError("client disconnected")


class FakeTeleVuer:
    """Stand-in for televuer.TeleVuer: the scenes the teleop would show without the relay."""

    def __init__(self, use_hand_tracking=True):
        self.use_hand_tracking = use_hand_tracking

    async def main_image_binocular(self, session, fps=60):
        return "base-binocular"

    async def main_image_monocular(self, session, fps=60):
        return "base-monocular"


class FakeFeed:
    def __init__(self, ip, port):
        self.args, self.closed = (ip, port), False

    def close(self):
        self.closed = True


def make_class(relay_fn=None, **kw):
    feeds, relays = [], []

    async def default_relay(session, feed, image_background, **options):
        relays.append((feed, image_background, options))

    def make_feed(ip, port):
        feeds.append(FakeFeed(ip, port))
        return feeds[-1]

    cls = display_relay.build_relay_class(FakeTeleVuer, ip="192.168.123.164", port=55556, screen_y=-0.1,
                                          distance=1.0, receiver_cls=make_feed,
                                          relay_fn=relay_fn or default_relay, schemas=fake_schemas, **kw)
    return cls, feeds, relays


@pytest.mark.parametrize("scene", ["main_image_binocular", "main_image_monocular"])
def test_both_scenes_show_the_display_stream_and_mount_the_hand_stream_first(scene):
    cls, feeds, relays = make_class()
    session = FakeSession()
    asyncio.run(getattr(cls(), scene)(session))
    elements, target = session.calls[0]
    assert target == "bgChildren" and isinstance(elements[0], FakeHands)
    assert elements[0].kw == {"stream": True, "key": "hands", "hideLeft": True, "hideRight": True}
    assert feeds[0].args == ("192.168.123.164", 55556) and feeds[0].closed
    feed, image_background, options = relays[0]
    assert feed is feeds[0] and image_background is FakeImage
    assert options == {"screen_y": -0.1, "distance": 1.0, "tag": "display"}


def test_feed_is_closed_even_if_the_relay_fails():
    async def broken_relay(session, feed, image_background, **options):
        raise RuntimeError("boom")

    cls, feeds, _ = make_class(relay_fn=broken_relay)
    with pytest.raises(RuntimeError):
        asyncio.run(cls().main_image_binocular(FakeSession()))
    assert feeds[0].closed


def test_without_hand_tracking_the_original_scene_is_kept():
    cls, feeds, relays = make_class()
    assert asyncio.run(cls(use_hand_tracking=False).main_image_binocular(FakeSession())) == "base-binocular"
    assert asyncio.run(cls(use_hand_tracking=False).main_image_monocular(FakeSession())) == "base-monocular"
    assert not feeds and not relays


def test_install_replaces_the_class_the_wrapper_looks_up_and_needs_fork():
    module = SimpleNamespace(TeleVuer=FakeTeleVuer)
    installed = display_relay.install_display_relay(
        module, ip="1.2.3.4", port=55556, start_method=lambda: "fork", schemas=fake_schemas)
    assert module.TeleVuer is installed and issubclass(installed, FakeTeleVuer) and installed is not FakeTeleVuer
    assert installed.__name__ == "FakeTeleVuer"
    other = SimpleNamespace(TeleVuer=FakeTeleVuer)
    with pytest.raises(RuntimeError, match="fork"):
        display_relay.install_display_relay(other, ip="1.2.3.4", start_method=lambda: "spawn")
    assert other.TeleVuer is FakeTeleVuer                     # untouched when it cannot work


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_zmq_stream_reaches_vuer_as_the_same_jpeg_bytes():
    """PC2 message -> display_feed.DisplayFeedReceiver -> relay_to_vuer -> ImageBackground(src=bytes)."""
    import zmq

    left, right = b"\xff\xd8LEFT-JPEG-BYTES\xff\xd9", b"\xff\xd8RIGHT-JPEG-BYTES-LONGER\xff\xd9"
    header = display_feed.DISPLAY_STRUCT.pack(display_feed.DISPLAY_MAGIC, display_feed.DISPLAY_VERSION, 0, 0,
                                              7, 123456789, 0, len(left), len(right), 320, 240)
    message = header + left + right
    port = free_port()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://127.0.0.1:{port}")
    import threading
    stop = threading.Event()

    def publish():
        while not stop.is_set():
            publisher.send(message)
            time.sleep(0.01)

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        cls = display_relay.build_relay_class(FakeTeleVuer, ip="127.0.0.1", port=port, screen_y=-0.25,
                                              distance=2.0, schemas=fake_schemas)
        session = FakeSession(stop_after_images=1)
        asyncio.run(asyncio.wait_for(cls().main_image_monocular(session), timeout=15))
    finally:
        stop.set()
        thread.join(timeout=1)
        publisher.close(0)
        context.term()
    image_calls = [elements for elements, _ in session.calls if isinstance(elements[0], FakeImage)]
    left_el, right_el = image_calls[0]
    assert left_el.kw["src"] == left and right_el.kw["src"] == right          # bytes untouched: no decode/re-encode
    assert (left_el.kw["layers"], right_el.kw["layers"]) == (1, 2)
    assert left_el.kw["position"] == [0, -0.25, 0] and left_el.kw["distanceToCamera"] == 2.0
    assert left_el.kw["aspect"] == pytest.approx(320 / 240)


# --- launcher wiring ---------------------------------------------------------------------------------

def load_launcher():
    spec = importlib.util.spec_from_file_location("run_teleop_with_neck_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_consumes_the_display_flags_and_forwards_the_rest():
    launcher = load_launcher()
    args, teleop_args = launcher.parse_launcher_args(
        ["--neck-pose-ip", "10.0.0.5", "--display-port", "55560", "--screen-y", "-0.2", "--display-distance", "1.5",
         "--arm=G1_29", "--ee=dex3"])
    assert (args.display_port, args.screen_y, args.display_distance, args.display_ip) == (55560, -0.2, 1.5, None)
    assert args.no_display_relay is False and teleop_args == ["--arm=G1_29", "--ee=dex3"]
    assert launcher.parse_launcher_args(["--no-display-relay"])[0].no_display_relay is True
    default = launcher.parse_launcher_args([])[0]
    assert (default.display_port, default.screen_y, default.display_distance) == (55556, -0.1, 1.0)


@pytest.mark.parametrize("bad", [["--display-port", "0"], ["--display-port", "70000"], ["--display-distance", "0"]])
def test_launcher_rejects_bad_display_values(bad):
    launcher = load_launcher()
    with pytest.raises(SystemExit):
        launcher.parse_launcher_args(bad)


DRIVER = textwrap.dedent('''
    import multiprocessing, runpy, sys
    multiprocessing.set_start_method("fork", force=True)      # what Linux does by default
    sys.path.insert(0, {host_dir!r})
    sys.path.insert(0, {libs!r})
    namespace = runpy.run_path({launcher!r}, run_name="launcher_under_test")
    namespace["main"]({argv!r})
''')

FAKE_TELEOP = textwrap.dedent('''
    import televuer
    wrapper = televuer.TeleVuerWrapper(binocular=False, use_hand_tracking=True)
    vuer_class = type(wrapper.tvuer)
    print("RELAY_INSTALLED", hasattr(vuer_class, "_show_display_feed"))
    print("WRAPPER", type(wrapper).__name__)
    wrapper.close()
''')


def run_launcher_with_fake_xr_repo(tmp_path, extra_args):
    repo = tmp_path / "xr_teleoperate"
    (repo / "teleop").mkdir(parents=True)
    (repo / "teleop" / "teleop_hand_and_arm.py").write_text(FAKE_TELEOP)
    (repo / "televuer").mkdir()
    (repo / "televuer" / "__init__.py").write_text(textwrap.dedent('''
        class TeleVuer:
            def __init__(self, **kw): pass
        class TeleVuerWrapper:
            def __init__(self, **kw):
                self.tvuer = TeleVuer(**kw)        # looked up by name in this module, like the real wrapper
            def close(self): pass
    '''))
    libs = tmp_path / "libs"
    libs.mkdir()
    (libs / "logging_mp.py").write_text("INFO = 20\ndef basicConfig(**kw): pass\n")
    argv = ["--xr-repo", str(repo), "--neck-pose-ip", "127.0.0.1", "--neck-pose-port", "5599", *extra_args]
    code = DRIVER.format(host_dir=str(HOST_DIR), libs=str(libs), launcher=str(LAUNCHER), argv=argv)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_launcher_installs_the_relay_before_the_teleop_script_builds_the_wrapper(tmp_path):
    result = run_launcher_with_fake_xr_repo(tmp_path, [])
    assert result.returncode == 0, result.stderr[-600:]
    assert "RELAY_INSTALLED True" in result.stdout
    assert "WRAPPER NeckForwardingTeleVuerWrapper" in result.stdout       # the neck forwarding is still there


def test_no_display_relay_leaves_the_teleop_vuer_untouched(tmp_path):
    result = run_launcher_with_fake_xr_repo(tmp_path, ["--no-display-relay"])
    assert result.returncode == 0, result.stderr[-600:]
    assert "RELAY_INSTALLED False" in result.stdout and "WRAPPER NeckForwardingTeleVuerWrapper" in result.stdout
