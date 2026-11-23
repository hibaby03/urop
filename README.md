# urop

```
host/   runs on the Host (Ubuntu): `cd host && python main.py`. See host/README.md
pc2/    runs on PC2 (Orin): ZED stream server + neck motor receiver. See pc2/README.md
```

Copy `host/` to the Host and `pc2/` to PC2, or keep both side by side and let `host/main.py` copy `pc2/` to PC2.

## Run the teleop with the neck (what is actually run today)

Order: PC2 first, then Host. Quit the teleop with **`q`**, never Ctrl+C (Ctrl+C causes IK exceptions and a hang).

**PC2** (`unitree@192.168.123.164`, repo at `~/active_camera/urop`):

```bash
cd ~/active_camera/urop && git pull
cd pc2 && python head_pose_receiver.py        # neck motors; starts at 2048/2048, no angle limits by default
# ZED stream server (only if you need the camera): python zed_teleimager_server.py  (see pc2/README.md)
```

**Host** (`idim@idim-l1`, conda env `tv`, repo at `~/hckang/jslee/urop`, `xr_teleoperate` at `~/hckang/xr_teleoperate`):

```bash
cd ~/hckang/jslee/urop && git pull && git log --oneline -1     # confirm you are on the latest commit
pgrep -af "teleop_hand_and_arm|run_teleop"                      # kill leftovers; port 8012 must be free

cd ~/hckang/xr_teleoperate
python ~/hckang/jslee/urop/host/run_teleop_fake_right_hand.py \
  --xr-repo ~/hckang/xr_teleoperate --neck-pose-ip 192.168.123.164 \
  --neck-pose-port 5005 --arm=G1_29 --ee=dex3
```

Then, before pressing `r`: put your arms in the robot's initial pose, clear the area around the robot and keep
the remote ready for damping (L2+B). Press `r` once to start following, `q` to quit.

Notes:

* `run_teleop_fake_right_hand.py` runs the official `teleop_hand_and_arm.py` unmodified. It fakes the right Dex3
  hand state (the right hand publishes none) so `Dex3_1_Controller` stops waiting for "Waiting to subscribe dds".
* `--headless` is added automatically (the Host has no GUI; without it `cv2.imshow` raises and the arms go home).
* `--arm=G1_29 --ee=dex3` are examples; give the same options you would give the official teleop.

### Diagnosing a stiff / jerky arm

`host/teleop_loop_probe.py` takes the same options plus `--fake-right-hand --headless` (the headless default is
not automatic here) and prints one `[loop]` line per second:

```bash
cd ~/hckang/xr_teleoperate
python ~/hckang/jslee/urop/host/teleop_loop_probe.py \
  --xr-repo ~/hckang/xr_teleoperate --neck-pose-ip 192.168.123.164 \
  --neck-pose-port 5005 --arm=G1_29 --ee=dex3 --fake-right-hand --headless 2>&1 \
  | grep --line-buffered "\[loop\]\|\[OK\]\|\[FAIL\]\|\[WARN\]"
```

* `cycles/s` should be tens (the loop targets 90 Hz). Around 1 means the loop is starved and the arms only get
  one target per second.
* The end of each line shows `slowest ms: motion_input IK ctrl loop_gap` (which stage is slow).
* In another terminal, `top -bn1 -o %CPU | head -12` while it runs: a process near 1800% CPU means the
  numeric libraries are oversubscribing the cores.

Experiment (**not yet confirmed to help**): limit the thread pools by putting this in front of the command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python ~/hckang/jslee/urop/host/teleop_loop_probe.py ...
```
