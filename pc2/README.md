# pc2/ — runs on PC2 (the Orin on the robot)

Verbatim copy of the PC2 programs of `unitree_g1_urop` (git `2485ea6`; `MANIFEST.sha256` lists the hashes and
`host/tests/test_vendor.py` checks them). Do not edit; they are used unchanged.

Copy this `pc2/` directory to PC2 and start the two programs **by hand on PC2** before running `python main.py`
on the Host (`host/main.py` does not connect to PC2 with the default `host/config.yaml`). Run these from the
directory that contains `pc2/` (the commands of the original README):

```bash
python3 pc2/zed_teleimager_server.py            # terminal 1: ZED stream (recording 55555, Quest display 55556)
python3 pc2/head_pose_receiver.py --yaw-limit-deg 15 --pitch-down-limit-deg 15 --pitch-up-limit-deg 15   # terminal 2: neck motors
```
`host/config.yaml` already has `mode: external` for `active_camera_zed` and `active_camera_neck`. (Optional: `main.py`
can copy `pc2/` and start them over ssh instead; see the comments in `host/config.yaml`; it needs key-based ssh login.)

PC2 needs the usual PC2 environment (not part of this copy): ZED SDK + `pyzed`, `teleimager`, the Dynamixel SDK
(`requirements.txt`) and the Dynamixel neck hardware.
