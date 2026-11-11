"""Quest display from inside the teleop's own Vuer server: PC2's display stream, relayed unchanged.

ADDED BY urop (not part of unitree_g1_urop). It gives the installed teleop_hand_and_arm.py a Quest
picture without changing that script. The teleop script builds ``TeleVuerWrapper`` itself and the
wrapper builds the Vuer server (a forked child process) from the module-level ``TeleVuer`` class.
This module replaces that class by a subclass whose image scene is the same relay that
``zmq_stereo_vr.py`` and the old ``teleop_record_dex3_tactile.py`` (LegacyStereoXR) use:

  PC2 ZMQ 55556 (one message per frame: left + right eye JPEG)
    -> display_feed.DisplayFeedReceiver (thread, newest frame only)
    -> display_feed.relay_to_vuer -> ImageBackground(src=JPEG bytes) on the left/right eye layers

No JPEG decode, colour conversion, resize or re-encode on the Host. The hand-tracking element is
mounted exactly as LegacyStereoXR does, so the Quest keeps sending hand poses. Both image scenes
(binocular and monocular) are replaced, because which one TeleVuer runs depends on the
``binocular`` argument the teleop script chooses. Controller mode (no hand tracking) is left to the
original implementation.

The receiver is created inside the Vuer child after the fork, like LegacyStereoXR does; this needs
the ``fork`` start method (Linux default).
"""

from __future__ import annotations

import multiprocessing
from typing import Any, Callable, Optional

try:
    from .display_feed import DEFAULT_DISPLAY_PORT, DisplayFeedReceiver, relay_to_vuer
except ImportError:  # run as a script from this directory
    from display_feed import DEFAULT_DISPLAY_PORT, DisplayFeedReceiver, relay_to_vuer  # type: ignore[no-redef]


def _vuer_schemas() -> tuple[Any, Any]:
    from vuer.schemas import Hands, ImageBackground

    return Hands, ImageBackground


def build_relay_class(base: type, *, ip: str, port: int = DEFAULT_DISPLAY_PORT, screen_y: float = -0.1,
                      distance: float = 1.0,
                      receiver_cls: Optional[Callable[..., Any]] = None,
                      relay_fn: Optional[Callable[..., Any]] = None,
                      schemas: Optional[Callable[[], tuple[Any, Any]]] = None) -> type:
    """Subclass of ``base`` (TeleVuer) whose image scenes show PC2's display stream.

    ``receiver_cls``, ``relay_fn`` and ``schemas`` are injectable for tests without Vuer.
    """

    receiver = receiver_cls or DisplayFeedReceiver
    relay = relay_fn or relay_to_vuer
    load_schemas = schemas or _vuer_schemas

    class RelayTeleVuer(base):  # type: ignore[misc, valid-type]
        async def _show_display_feed(self, session: Any) -> None:
            hands, image_background = load_schemas()
            # The same element LegacyStereoXR mounts: it keeps the hand-tracking stream alive.
            session.upsert(hands(stream=True, key="hands", hideLeft=True, hideRight=True), to="bgChildren")
            feed = receiver(ip, port)
            try:
                await relay(session, feed, image_background, screen_y=screen_y, distance=distance,
                            tag="display")
            finally:
                feed.close()

        async def main_image_binocular(self, session: Any, fps: int = 60) -> None:
            if not getattr(self, "use_hand_tracking", True):
                return await super().main_image_binocular(session, fps)
            await self._show_display_feed(session)

        async def main_image_monocular(self, session: Any, fps: int = 60) -> None:
            if not getattr(self, "use_hand_tracking", True):
                return await super().main_image_monocular(session, fps)
            await self._show_display_feed(session)

    RelayTeleVuer.__name__ = base.__name__
    RelayTeleVuer.__qualname__ = base.__qualname__
    return RelayTeleVuer


def install_display_relay(wrapper_module: Any, *, ip: str, port: int = DEFAULT_DISPLAY_PORT,
                          screen_y: float = -0.1, distance: float = 1.0,
                          start_method: Callable[[], str] = multiprocessing.get_start_method,
                          **injected: Any) -> type:
    """Replace ``wrapper_module.TeleVuer`` (looked up by name when the wrapper builds its server)."""

    if start_method() != "fork":
        raise RuntimeError("the display relay needs the multiprocessing 'fork' start method (Linux)")
    base = wrapper_module.TeleVuer
    relay_class = build_relay_class(base, ip=ip, port=port, screen_y=screen_y, distance=distance, **injected)
    wrapper_module.TeleVuer = relay_class
    return relay_class
