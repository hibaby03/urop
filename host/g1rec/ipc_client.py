"""Client of the official xr_teleoperate IPC server (teleop/utils/ipc.py, ``--ipc`` flag).

The official teleop reads the keyboard itself (sshkeyboard) unless it is started with
``--ipc``. With ``--ipc`` it stops reading the keyboard and accepts the same three
commands over ZMQ, so the orchestrator can own the terminal without any conflict and
without touching the official code:

  CMD_START          -> the official 'r'  (robot starts following the operator)
  CMD_RECORD_TOGGLE  -> the official 's'  (never sent: recording is done by our recorder)
  CMD_STOP           -> the official 'q'  (exit)

Heartbeat (PUB, 10 Hz): {"START", "STOP", "RECORD_RUNNING", "RECORD_READY"}.
Sockets are abstract Unix sockets, so this works on Linux only (the Host).
"""

from __future__ import annotations

import itertools
import threading
from typing import Optional

DATA_ENDPOINT = "ipc://@xr_teleoperate_data.ipc"
HEARTBEAT_ENDPOINT = "ipc://@xr_teleoperate_hb.ipc"


class TeleopIpcClient:
    def __init__(self, data_endpoint: str = DATA_ENDPOINT, hb_endpoint: str = HEARTBEAT_ENDPOINT) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._data_endpoint, self._hb_endpoint = data_endpoint, hb_endpoint
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._req = None
        self._state: dict = {}
        self._stop = threading.Event()
        self._hb = self._ctx.socket(zmq.SUB)
        self._hb.setsockopt_string(zmq.SUBSCRIBE, "")
        self._hb.connect(hb_endpoint)
        self._thread = threading.Thread(target=self._hb_loop, name="teleop-heartbeat", daemon=True)
        self._thread.start()

    def _hb_loop(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._hb, self._zmq.POLLIN)
        while not self._stop.is_set():
            if dict(poller.poll(100)):
                try:
                    state = self._hb.recv_json(flags=self._zmq.NOBLOCK)
                except self._zmq.ZMQError:
                    continue
                with self._lock:
                    self._state = state

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def alive(self) -> bool:
        return bool(self.state())

    def _send(self, cmd: str, timeout_ms: int = 3000) -> Optional[dict]:
        zmq = self._zmq
        req = self._ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.setsockopt(zmq.RCVTIMEO, timeout_ms)
        req.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            req.connect(self._data_endpoint)
            req.send_json({"reqid": next(self._ids), "cmd": cmd})
            return req.recv_json()
        except zmq.ZMQError:
            return None
        finally:
            req.close()

    def start_following(self) -> bool:
        reply = self._send("CMD_START")
        return bool(reply and reply.get("status") == "ok")

    def stop_teleop(self) -> bool:
        reply = self._send("CMD_STOP")
        return bool(reply and reply.get("status") == "ok")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._hb.close(0)
