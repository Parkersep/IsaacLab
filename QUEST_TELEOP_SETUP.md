# Meta Quest 3 Teleoperation with Isaac Lab

**Official docs:** https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html

## Overview

Control a simulated robot in Isaac Lab using Meta Quest 3 hand tracking via CloudXR streaming.
Uses [Isaac Teleop](https://nvidia.github.io/IsaacTeleop/main/index.html) which bundles CloudXR Runtime 6.1.0 and handles the runtime automatically.

**Architecture:**
- **Terminal 1:** Isaac Teleop CloudXR Server (runs CloudXR Runtime 6.1.0 with WebRTC)
- **Terminal 2:** Isaac Lab (simulates the robot, reads hand tracking via OpenXR)
- **Quest browser:** connects to hosted CloudXR web client, streams video and sends hand tracking

## Prerequisites

- Ubuntu 22.04 or 24.04
- NVIDIA GPU (RTX 5070 Ti or better)
- NVIDIA Driver 580.95.05+, CUDA 12.8+
- Python 3.12+
- Isaac Lab installed (`env_isaaclab`)
- Meta Quest 2/3/3S on the same LAN as the workstation
- WiFi 6 router recommended

## Step 1: Install System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y libvulkan1 libbsd0
```

## Step 2: Install Isaac Teleop

Install into your Isaac Lab environment:

```bash
pip install "isaacteleop[retargeters,ui,cloudxr]~=1.0.0" \
  --extra-index-url https://pypi.nvidia.com
```

> **Note:** If this causes dependency conflicts (e.g., numpy, lxml version changes), use `--no-deps` instead:
> ```bash
> uv pip install --python ./env_isaaclab/bin/python --no-deps 'isaacteleop~=1.0.0' --extra-index-url https://pypi.nvidia.com
> ```

## Step 3: Open Firewall Ports

```bash
sudo ufw allow 47998/udp
sudo ufw allow 49100,48322/tcp
```

## Step 4: Start CloudXR Server (Terminal 1)

**IMPORTANT:** For Quest optical hand tracking, you MUST set `NV_CXR_ENABLE_PUSH_DEVICES=0`. Without this, hand tracking data will not flow through and your virtual hands will not appear.

```bash
# Create the Quest hand tracking config (one time)
echo 'NV_CXR_ENABLE_PUSH_DEVICES=0' > ~/cloudxr_quest.env

# Start CloudXR
python -m isaacteleop.cloudxr --accept-eula --cloudxr-env-config=$HOME/cloudxr_quest.env
```

You should see:
```
Running Isaac Teleop 1.0.191, CloudXR Runtime 6.1.0
CloudXR runtime:   running
CloudXR WSS proxy: running
Activate CloudXR environment in another terminal: source /home/<user>/.cloudxr/run/cloudxr.env
Keep this terminal open, Ctrl+C to terminate.
```

**Keep this terminal open for the entire session.**

**Troubleshooting:**
- `Port 47998 is already in use` — Stop old CloudXR Docker containers:
  ```bash
  docker stop docker-cloudxr-runtime-1 wss-proxy 2>/dev/null
  docker rm docker-cloudxr-runtime-1 wss-proxy 2>/dev/null
  ```
- `CloudXR runtime failed to start` — Check logs: `cat ~/.cloudxr/logs/*.log`

## Step 5: Run Isaac Lab (Terminal 2)

**IMPORTANT:** You must source `cloudxr.env` in the same terminal before running Isaac Lab.

```bash
source ~/.cloudxr/run/cloudxr.env

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0 \
    --visualizer kit --xr
```

Wait for Isaac Sim to fully load and show "Teleoperation started."

### Alternative tasks

```bash
# Franka robot arm (simpler, single arm + gripper)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
    --teleop_device handtracking \
    --visualizer kit --xr

# GR1T2 humanoid (bimanual hand tracking)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-GR1T2-Abs-v0 \
    --teleop_device handtracking \
    --visualizer kit --xr

# OpenArm bimanual fixed-base pick-place (Quest controllers, Pink IK, tuned RPY offsets)
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-OpenArm-Bimanual-Abs-v0 \
    --visualizer kit --xr

# OpenArm bimanual sandbox — drive the wheeled base with joysticks while teleop-ing the arms
# Left stick = forward/back, right stick = yaw. Ground plane only, no objects.
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-Sandbox-OpenArm-Bimanual-Wheeled-v0 \
    --visualizer kit --xr
```

### Recording demonstrations

To record a dataset while teleoperating (instead of just running the agent), use `record_demos.py`:

```bash
# G1 locomanipulation pick-place — record 15 demos to HDF5
./isaaclab.sh -p scripts/tools/record_demos.py \
    --device cpu \
    --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
    --teleop_device handtracking \
    --dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
    --num_demos 15 \
    --visualizer kit
```

> **Note:** The `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` task's retargeting pipeline is hardcoded for **Oculus Touch controllers** (see `_build_g1_locomanipulation_pipeline` in `locomanipulation_g1_env_cfg.py`). The `--teleop_device handtracking` flag is effectively ignored for this task — hold the Quest controllers in your hands, don't use optical hand tracking.

### Annotating recorded demos for Mimic

After recording, annotate the dataset with Mimic subtask signals so it can be used as source data for dataset generation:

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --device cpu \
    --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
    --input_file ./datasets/dataset_g1_locomanip.hdf5 \
    --output_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
    --visualizer kit
```

Note the task ID differs from the recording step — annotation uses the `-Mimic-v0` variant.

## Step 6: Connect Meta Quest 3

1. On your Quest 3, open the **browser**
2. Navigate to: `https://nvidia.github.io/IsaacTeleop/client`
3. Enter your workstation IP address (find it with `hostname -I` on your PC)
4. Click the `https://<ip>:48322/` link to accept the self-signed SSL certificate
5. Click **Advanced** -> **Proceed to \<ip\> (unsafe)**
6. Navigate back to the client page
7. Click **Connect**

**IMPORTANT:** For hand tracking, put the Quest controllers down and away from you. The Quest auto-switches to optical hand tracking when controllers are not detected. Your virtual hands should appear in the VR view.

## Step 7: Start AR in Isaac Sim

In the Isaac Sim UI on your PC:

1. Open the **AR** panel
2. Set **Selected Output Plugin** to **OpenXR**
3. Set **OpenXR Runtime** to **System OpenXR Runtime**
4. Click **Start AR**

The viewport should show stereo rendering (two eyes). Your Quest should now display the simulation and hand movements should control the robot.

## Stopping

1. Press `Ctrl+C` in the Isaac Lab terminal (Terminal 2)
2. Press `Ctrl+C` in the CloudXR server terminal (Terminal 1)

## Code Changes to `session_lifecycle.py`

Three changes were made to `source/isaaclab_teleop/isaaclab_teleop/session_lifecycle.py` to fix standalone mode when Kit's XR teleop bridge extension (`isaacsim.kit.xr.teleop.bridge`) is not installed.

### Problem

`omni.kit.xr.system.openxr` ships with Kit and is always importable. However, `isaacsim.kit.xr.teleop.bridge` — which adds the handle-acquisition API (`get_instance_proc_addr`, etc.) — was not installed. The old code assumed "module importable = bridge available" and deferred session creation forever, never falling through to standalone mode where Isaac Teleop creates its own OpenXR session.

### Fix 1: `_acquire_kit_oxr_handles` — Catch `AttributeError`

The four `openxr.get_*()` calls are now wrapped in a `try/except AttributeError` block. Previously, `get_instance_proc_addr()` not existing on the module caused an unhandled crash:

```
AttributeError: module 'omni.kit.xr.system.openxr' has no attribute 'get_instance_proc_addr'
```

### Fix 2: `_try_start_session` — Standalone mode fallback

Previously, when `oxr_handles` was `None`, the method always deferred session creation (returned `False`). Now it distinguishes two cases:

- **Bridge available but handles incomplete** (e.g., user hasn't clicked "Start AR") — defer and retry on next `step()` call.
- **Bridge not available** — proceed with `oxr_handles=None`, letting Isaac Teleop create its own OpenXR session internally (standalone mode). `TeleopSession` supports this natively: when no handles are provided, it creates an `oxr.OpenXRSession` and manages the full lifecycle.

### Fix 3: `_kit_xr_bridge_available` — New helper method

Checks whether the Kit XR teleop bridge is fully functional, not just importable. It verifies that `omni.kit.xr.system.openxr` has the `get_instance_proc_addr` attribute — a function added by the bridge extension. Without this check, the base OpenXR module (always present in Kit) made the bridge appear "available", causing session creation to defer indefinitely and `advance()` to return `None` every frame.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Port 47998 already in use` | Stop old Docker containers (see Step 4) |
| `CloudXR runtime failed to start` | Check `cat ~/.cloudxr/logs/*.log` |
| `XR_ERROR_RUNTIME_FAILURE` | Forgot to `source ~/.cloudxr/run/cloudxr.env` before running Isaac Lab |
| `Failed to connect to socket ipc_cloudxr` | CloudXR server not running, or env not sourced |
| Quest "connection refused" | Check firewall ports (`sudo ufw status`) |
| No virtual hands in Quest | Set `NV_CXR_ENABLE_PUSH_DEVICES=0` (see Step 4) |
| Hands visible but robot not moving | Check "Teleoperation activated" in terminal; put controllers away |
| Session defers forever / `advance()` always returns `None` | `isaacsim.kit.xr.teleop.bridge` missing — apply the standalone mode fixes above |
| `AttributeError: get_instance_proc_addr` | Bridge extension not installed — the `_acquire_kit_oxr_handles` fix catches this |
| `XR_ERROR_INSTANCE_LOST` / `Broken pipe` | CloudXR server crashed; restart it (Step 4) |
| No video in headset | Click **Start AR** in Isaac Sim AR panel (Step 7) |
| Poor streaming quality | Use dedicated WiFi 6 router |

## References

- [Isaac Lab CloudXR Teleoperation (develop)](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html)
- [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [CloudXR.js Documentation](https://docs.nvidia.com/cloudxr-sdk/latest/usr_guide/cloudxr_js/index.html)
- [Isaac Teleop GitHub](https://github.com/NVIDIA/IsaacTeleop)
