# Vendored copy of unitree_g1_urop (unmodified)

Verbatim copy of the files this project needs from `unitree_g1_urop`, taken from git
`2485ea69068e4e926def6bd0e1f84c828f46e303` (origin/main, "Relay PC2-encoded per-eye JPEGs to the
Quest without Host pixel processing"), not from a working tree. `MANIFEST.sha256` lists the hashes;
`tests/test_vendor.py` fails if any file differs. Do not edit these files: update them by copying again.

host/  (runs on the Host)
  run_teleop_with_neck.py  launcher: runs the installed official teleop_hand_and_arm.py unchanged and
                           adds the head-pose UDP to the neck (needs head_pose_udp, neck_trace, neck_clock)
  frame_timing.py          ZED JPEG stamp + PC2 clock sync (used read-only by g1rec/sources.py)
  neck_state_receiver.py   neck-state packet decoder (used read-only by g1rec/sources.py)
pc2/   (copied to PC2 by the orchestrator, see `pc2_sync` in config.yaml)
  zed_teleimager_server.py, head_pose_receiver.py and what they import, cam_config_zed.yaml
Not copied (not needed): robot_bridge, g1_arm_control, dex3_control, teleop_record_dex3_tactile,
deploy_act, robot_link, hand_retarget, xr_legacy and the other host tools.
