# pc2/ — runs on PC2 (the Orin on the robot)

Verbatim copy of the PC2 programs of `unitree_g1_urop` (git `2485ea6`; `MANIFEST.sha256` lists the hashes and
`host/tests/test_vendor.py` checks them). Do not edit; they are used unchanged.

The Host's orchestrator (`host/main.py`) normally copies this directory to PC2 as `~/g1rec_pc2/pc2` and
starts the two programs over ssh. To do it by hand instead, copy this `pc2/` directory to PC2 and run, from
the directory that contains `pc2/` (these are the commands of the original README):

```bash
python3 pc2/zed_teleimager_server.py            # terminal 1: ZED stream (recording 55555, Quest display 55556)
python3 pc2/head_pose_receiver.py --yaw-limit-deg 15 --pitch-down-limit-deg 15 --pitch-up-limit-deg 15   # terminal 2: neck motors
```
then set `mode: external` for `active_camera_zed` and `active_camera_neck` in `host/config.yaml`.

PC2 needs the usual PC2 environment (not part of this copy): ZED SDK + `pyzed`, `teleimager`, the Dynamixel SDK
(`requirements.txt`) and the Dynamixel neck hardware.
