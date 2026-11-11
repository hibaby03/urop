# Files taken from unitree_g1_urop (with two urop changes, listed below)

Copy of the files this project needs from `unitree_g1_urop`, taken from git
`2485ea69068e4e926def6bd0e1f84c828f46e303` (origin/main, "Relay PC2-encoded per-eye JPEGs to the
Quest without Host pixel processing"), not from a working tree. `MANIFEST.sha256` lists the hashes of every
file in this directory; `tests/test_vendor.py` fails if one changes without the list being updated.

**Changed or added by urop (not verbatim):**

| file | what |
|---|---|
| `host/run_teleop_with_neck.py` | (1) configures `logging_mp` before importing televuer; (2) installs the Quest display relay (`display_relay.py`) in full-teleop mode, with the new flags `--display-ip`, `--display-port`, `--screen-y`, `--display-distance`, `--no-display-relay` (consumed by the launcher, not forwarded to the teleop) |
| `host/display_relay.py` | new: replaces `TeleVuer`'s image scenes by the relay of PC2's display stream (ZMQ 55556) so the teleop script, which does not receive the ZED stream itself, still shows video in the Quest |

Everything else is verbatim:

host/  (runs on the Host)
  display_feed.py          PC2 display stream receiver + Vuer relay (`DisplayFeedReceiver`, `relay_to_vuer`), used by display_relay.py
  frame_timing.py          ZED JPEG stamp + PC2 clock sync (used read-only by g1rec/sources.py)
  neck_state_receiver.py   neck-state packet decoder (used read-only by g1rec/sources.py)
  head_pose_udp.py, neck_trace.py, neck_clock.py   helpers of the launcher
pc2/   (a separate top-level directory of this repository)
  zed_teleimager_server.py, head_pose_receiver.py and what they import, cam_config_zed.yaml
Not copied (not needed): robot_bridge, g1_arm_control, dex3_control, teleop_record_dex3_tactile,
deploy_act, robot_link, hand_retarget, xr_legacy and the other host tools.
