# g1rec — master-dataset recorder for Unitree G1 + Dex3-1 + active stereo camera

`python main.py` starts the existing programs as independent processes, starts a read-only recorder,
and gives you `r` (start episode) / `s` (stop & save) / `q` (quit). Episodes are written as a
**LeRobotDataset** in the layout of the published UAF code, plus a raw, original-rate recording.

It contains no teleoperation, IK, retargeting or camera control. It never publishes a command.

## Self-contained, split into host/ and pc2/

```
urop/
  host/   runs on the Host: main.py, config.yaml, g1rec/ (orchestrator + recorder), tests/,
          third_party/unitree_g1_urop/ (verbatim Host-side files: run_teleop_with_neck.py + helpers,
          frame_timing.py, neck_state_receiver.py; see VENDORED.md)
  pc2/    runs on PC2: verbatim PC2 programs (zed_teleimager_server.py, head_pose_receiver.py and what they import)
```

Copy `host/` to the Host and `pc2/` to PC2. **Start the PC2 programs by hand on PC2** (commands in `pc2/README.md`);
`host/main.py` does not connect to PC2 by default. (Optional ssh auto-start: see the comments in `host/config.yaml`.)
Hashes in `MANIFEST.sha256` (and `tests/test_vendor.py`) guarantee the copied files are unmodified.

Not bundled, because they are separate installations and not code of this project: the official
`xr_teleoperate` checkout (`xr_repo`), the Python packages (`teleimager`, `unitree_sdk2py`, `televuer`,
`lerobot`, ...), the ZED SDK / pyzed and the Dynamixel hardware on PC2. The existing `unitree_g1_urop`
checkout is not needed at all.

## Layout (from the UAF code, not guessed)

Source: `kuehnrobin/UAF_unitree_g1` (`convert_unitree_json_to_lerobot.py`, `constants.py`) and
`kuehnrobin/UAF_lerobot` (`unified_ablation_framework.py`). `tests/test_schema.py` pins it to those files.

| feature | shape | content |
|---|---|---|
| `observation.state` | 110 | `left_arm[q,dq,tau]` 0-20, `right_arm[...]` 21-41, `left_hand[q,dq,tau,pressure12]` 42-74, `right_hand[...]` 75-107, `camera[pitch,yaw]` 108-109 |
| `action` | 30 | left arm q 7, right arm q 7, left hand q 7, right hand q 7, camera q 2 |
| `observation.images.cam_left_active` / `cam_right_active` | 3×480×640 video | ZED left / right eye (letterboxed) |

Names are the UAF ones (`left_arm_qpos_0` ...) so UAF's `feature_selection` ablation works on the same
dataset. `schema.state_mask({...})` builds the same masks; a masked ACT batch is tested.
`meta/g1_master_info.json` stores joint names, message indices, action semantics and conventions;
`meta/raw_episodes.json` maps each LeRobot episode to its raw directory.

## Where each value comes from (all read-only subscriptions)

| value | stream | notes |
|---|---|---|
| arm q / dq / tau_est | DDS `rt/lowstate`, motors 15-28 | measured |
| hand q / dq / tau_est, pressure | DDS `rt/dex3/{left,right}/state` | pressure = `press_sensor_state[0].pressure[0:12]` as in UAF; raw store keeps all sensors |
| arm action | DDS `rt/lowcmd` (debug) or `rt/arm_sdk` (`dds.motion_mode: true`) | what the official controller actually published (after IK and its velocity limiting), position targets |
| hand action | DDS `rt/dex3/{left,right}/cmd` | position targets |
| camera state / action | UDP :5006 from PC2 `head_pose_receiver` | measured / commanded neck angles, converted to rad, order **[pitch, yaw]**, relative to center (yaw right+, pitch up+). UAF used absolute encoder angles. |
| images | ZMQ via teleimager `ImageClient` (PC2 :55555) + PC2 clock sync | the PC2 capture stamp is used for image age |

## Time alignment

Tick rate = `fps` (30). For tick *t*: observations are the newest sample **received at or before t**
(never a later one); the action is the first command published **at or after t** (what the teleop did
next, as in UAF's recorder; `action_alignment: asof` uses the newest before). Ticks are processed one
period late so that command exists. A missing/stale required stream drops the tick (counted, never
zero-filled). Every message is also written untouched to `recordings/raw/episode_NNNNNN/*.parquet` with
`recv_ns`, `source_ns`, `seq`, `capture_ns` at its own rate (tactile is not downsampled). Raw rows are
flushed every 2 s; a crashed episode is repaired with `g1rec.raw_store.recover(dir)`.

## Requirements

* The Host must see the robot DDS network (wired to the robot LAN or DDS reachable): the official
  `xr_teleoperate` itself needs this. `dds.interface` selects the NIC. (The Orin-bridge path of
  `unitree_g1_urop` is not used.)
* Passwordless ssh (key login) from the Host to PC2 `192.168.123.164` if the orchestrator starts the PC2 programs.
* Official `xr_teleoperate` started with **`--ipc`**. This is how keys reach it without conflict: the
  orchestrator owns the terminal, children get no stdin, and the first `r` sends the official
  `CMD_START` over its IPC server. IPC uses abstract Unix sockets, so this works on Linux (Host).
  An older fork without `--ipc` is not supported.
* `lerobot[dataset]`, `pyarrow`, `pyzmq`, `pyyaml`, `opencv`; `unitree_sdk2py`, `teleimager` in the Host env.
* Only the recorder listens on UDP 5006 (neck state). Do not start another listener there.

## Run

```bash
python main.py --config config.yaml           # edit config.yaml first (paths, NIC, PC2 address)
python -m g1rec.verify recordings/master_dataset   # re-verify any dataset with the installed LeRobot
```

Logs of every child process are in `recordings/logs/`. After each saved episode the orchestrator runs
the verification and prints `LeRobotDataset: VALID|INVALID` and `ACT Training Compatibility: PASS|FAIL`
(`--no-act-check` prints `NOT RUN`; if lerobot cannot be imported it prints `NOT VERIFIED`).

## Tests

```bash
python -m pytest -q tests            # 36+ tests; the lerobot e2e test is skipped if lerobot is absent
python -m tests.e2e_synthetic        # prints the 10 verification checks on a synthetic dataset
```

## Verified here vs not verified

Verified (Mac, synthetic streams, lerobot 0.7.0): schema equals the UAF layout; as-of alignment has no
future leakage; stale/missing streams drop ticks; raw store keeps every message and is crash-recoverable;
the recorder side has no publisher; process start/ready/stop; two episodes of different length write a
real LeRobotDataset that LeRobot loads, decodes, with timestamps and action-chunk padding correct, and
a real batch (full and sensor-masked) passes ACT forward+backward.

**Not verified (needs the robot/Host, not run):** the DDS/ZMQ/UDP sources against live streams, the
official teleop's `--ipc` handshake, ssh launching of PC2 programs, Linux terminal handling, behaviour
with LeRobot 0.4.4 (the UAF fork; `verify` has a fallback for the moved import, nothing else was run on it),
and actual training. No robot was moved.
