"""Configuration (config.yaml) for the orchestrator and the recorder.

Nothing here is guessed from the existing projects: every command, address and port is a
value the user can see and change in config.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from . import schema


@dataclass
class ProcessSpec:
    """One existing program that the orchestrator starts as an independent process."""

    name: str
    cmd: list[str] = field(default_factory=list)   # argv, started as-is (no shell)
    cwd: Optional[str] = None
    env: dict = field(default_factory=dict)
    ssh: Optional[str] = None                      # "user@pc2": run cmd on that machine over ssh
    mode: str = "launch"                           # launch | external (already running; not started)
    ready_text: Optional[str] = None               # substring of its output that means "ready"
    ready_timeout_s: float = 60.0
    stop_timeout_s: float = 10.0


@dataclass
class DdsConfig:
    domain: int = 0
    interface: Optional[str] = None                # e.g. "enp3s0"; None = SDK default
    motion_mode: bool = False                      # official teleop --motion => arm cmds on rt/arm_sdk
    arm_state_topic: str = "rt/lowstate"
    arm_cmd_topic_debug: str = "rt/lowcmd"
    arm_cmd_topic_motion: str = "rt/arm_sdk"
    hand_topics: dict = field(default_factory=lambda: {
        "left": {"state": "rt/dex3/left/state", "cmd": "rt/dex3/left/cmd"},
        "right": {"state": "rt/dex3/right/state", "cmd": "rt/dex3/right/cmd"}})


@dataclass
class ImageConfig:
    pc2_ip: str = "192.168.123.164"
    clock_sync_port: int = 5007
    fit: str = "letterbox"                         # letterbox | stretch (ZED 16:9 -> 480x640)


@dataclass
class NeckConfig:
    bind: str = "0.0.0.0"
    port: int = 5006                               # PC2 head_pose_receiver -> Host (neck state)


@dataclass
class Config:
    fps: int = schema.DEFAULT_FPS
    task: str = "pick up cube."
    task_desc: str = "task description"
    dataset_root: str = "recordings/master_dataset"
    raw_root: str = "recordings/raw"
    repo_id: str = "local/g1_master_dataset"
    cameras: list = field(default_factory=lambda: list(schema.REQUIRED_CAMERAS))
    max_age_ms: dict = field(default_factory=lambda: {
        "arm_state": 200.0, "hand_state": 200.0, "arm_cmd": 500.0, "hand_cmd": 500.0,
        "tactile": 500.0, "neck": 500.0, "image": 500.0})
    dds: DdsConfig = field(default_factory=DdsConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    g1_urop_host: Optional[str] = None             # null = the bundled copy in third_party/unitree_g1_urop/host
    pc2_sync: Optional[dict] = None                # {ssh, src, dest}: copy the bundled pc2/ to PC2 before launching
    xr_repo: Optional[str] = None                  # xr_teleoperate checkout (DDS message modules, teleimager)
    teleop_input: str = "ipc"                      # ipc: keys are sent to the official teleop over its IPC server
    action_alignment: str = "next"                 # next: first command issued at/after the tick (UAF-style); asof
    raw_flush_s: float = 2.0
    min_free_gb: float = 2.0
    processes: dict = field(default_factory=dict)  # name -> ProcessSpec

    def process(self, name: str) -> Optional[ProcessSpec]:
        return self.processes.get(name)


def _build(cls, data: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def from_dict(data: dict) -> Config:
    data = dict(data or {})
    for key, cls in (("dds", DdsConfig), ("images", ImageConfig), ("neck", NeckConfig)):
        if key in data:
            data[key] = _build(cls, data[key] or {})
    procs = {}
    for name, spec in (data.pop("processes", {}) or {}).items():
        procs[name] = _build(ProcessSpec, {"name": name, **(spec or {})})
    cfg = _build(Config, data)
    cfg.processes = procs
    merged_age = Config().max_age_ms
    merged_age.update(cfg.max_age_ms)
    cfg.max_age_ms = merged_age
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.fps < 1:
        raise ValueError("fps must be >= 1")
    for camera in schema.REQUIRED_CAMERAS:
        if camera not in cfg.cameras:
            raise ValueError(f"cameras must include the active stereo pair: missing {camera}")
    unknown = [c for c in cfg.cameras if c not in schema.CAMERAS]
    if unknown:
        raise ValueError(f"unknown cameras {unknown}")
    if cfg.images.fit not in ("letterbox", "stretch"):
        raise ValueError("images.fit must be letterbox or stretch")
    if cfg.pc2_sync is not None:
        missing = {"ssh", "src", "dest"} - set(cfg.pc2_sync)
        if missing or set(cfg.pc2_sync) - {"ssh", "src", "dest"}:
            raise ValueError("pc2_sync needs exactly the keys ssh, src, dest")
    if cfg.action_alignment not in ("next", "asof"):
        raise ValueError("action_alignment must be next or asof")
    if cfg.teleop_input not in ("ipc",):
        raise ValueError("teleop_input must be 'ipc'")
    for name, spec in cfg.processes.items():
        if spec.mode not in ("launch", "external"):
            raise ValueError(f"process {name}: mode must be launch or external")
        if spec.mode == "launch" and not spec.cmd:
            raise ValueError(f"process {name}: cmd is required in launch mode")


def load(path: Optional[str]) -> Config:
    """Read a YAML (or JSON) file; no file means defaults."""

    if not path:
        return from_dict({})
    text = Path(path).expanduser().read_text()
    if str(path).endswith(".json"):
        return from_dict(json.loads(text))
    import yaml

    return from_dict(yaml.safe_load(text) or {})
