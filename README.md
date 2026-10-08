# Pseudo-Thermal LWIR Simulator

Real-time pseudo-thermal simulator built on NVIDIA Isaac Sim 6.1.0.
Generates images that mimic a LWIR thermal camera by processing in 2D the
instance segmentation and viewport RGB.

Part of WP4 of GITCC/EMPLOI — multimodal simulator for a quadruped industrial
inspection robot.

## Requirements

- NVIDIA Isaac Sim 6.1.0 (`~/isaac_sim/`)
- Ubuntu 22.04, RTX 3090
- ROS2 Humble (only for topic publishing)
- ffmpeg (only for MP4 encoding)

No Python dependencies to install: NumPy, OpenCV, PyYAML, and pytest are
already bundled with Isaac's Python (`~/isaac_sim/python.sh`).

## Quick Setup

```bash
cd ~/multimodal_sim/thermal_sim

# Make the launcher executable (once only)
chmod +x launch_thermal.sh
```

## Structure

```
run_pseudo_thermal.py            entry: args → (ROS2 relaunch) → SimulationApp → build → loop
pseudo_thermal/
    config.py                    loads and validates YAMLs (dataclasses)
    cli.py                       argparse and precedence YAML < flags
    processing.py                PseudoThermalProcessor: ids + RGB → gray 0..255
    visualization.py             palettes, scale bar, HUD, compose_frame
    outputs.py                   cv2 window / PNG, frames, summary.json, MP4
    ros2_publisher.py            rclpy node and publishers
    scene_builder.py             USD scene from YAML                    (Isaac)
    sensors.py                   render product + annotators            (Isaac)
    robot.py                     robot, mounted camera, keyboard, auto-track  (Isaac)
configs/
    scenes/demo_industrial.yaml  objects, temperatures, lights, camera, robot
    thermal_camera.yaml          resolution, palette, luminance, blur, noise
tests/                           tests for pure modules (no Isaac)
scripts/
    compare_frames.py            compares two --save-frames runs
    run_legacy_seeded.py         runs the legacy script with a fixed seed (comparison only)
    live_pseudo_thermal_legacy.py  original single-file script, unmodified
launch_thermal.sh                launches run_pseudo_thermal.py with ROS2
```

**Import rule.** Modules marked `(Isaac)` import `pxr`/`omni`/`carb` and can
only be imported after `SimulationApp` is created; `run_pseudo_thermal.py`
handles that. The rest are pure (NumPy, cv2, PyYAML) and must not import
anything from Isaac — this is why they can be tested without opening the
simulator. `tests/test_purity.py` fails if anyone breaks the rule.

## Usage Modes

### 1. Record with moving camera (demo)

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py \
    --autoplay \
    --save-frames outputs/pseudo_thermal \
    --max-frames 90 \
    --auto-track linear
```

Available trajectories: `linear` (parallel to the wall) or `circular`
(90° arc around the scene). If ffmpeg is available, `outputs/pseudo_thermal_linear.mp4`
is encoded on exit.

### 2. Record static camera

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py \
    --autoplay \
    --save-frames outputs/pseudo_thermal \
    --max-frames 30
```

### 3. Publish via ROS2 (visualization in RViz)

**Terminal 1 — Isaac Sim** (do NOT `source /opt/ros/humble/setup.bash` here):

```bash
cd ~/multimodal_sim/thermal_sim
./launch_thermal.sh
# or with options:
./launch_thermal.sh --auto-track circular --save-frames outputs/demo --max-frames 90
```

**Terminal 2 — ROS2** (source here):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
rviz2
# or view a specific topic:
rqt_image_view /thermal_sim/ir/image_raw
```

Published topics:

| Topic | Encoding | Content |
|-------|----------|---------|
| `/thermal_sim/ir/image_raw` | bgr8 | Pseudo-thermal image with palette |
| `/thermal_sim/rgb/image_raw` | rgba8 | Viewport RGB image |

### 4. Interactive GUI mode (no recording)

```bash
~/isaac_sim/python.sh run_pseudo_thermal.py
```

Opens Isaac Sim, press Play manually, and move the camera with the gizmo.
The thermal image updates at `/tmp/live_pseudo_thermal.png`
(or in a window if OpenCV has GUI support; see Notes).

### 5. Inspector robot

```bash
# Keyboard: click the Isaac viewport and use WASD/arrows to move, Q/E to go up/down
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot

# Robot follows the path automatically
~/isaac_sim/python.sh run_pseudo_thermal.py --autoplay --robot --auto-track linear \
    --save-frames outputs/robot_demo --max-frames 90
```

With the `Pseudo-Thermal LWIR` window focused (only if OpenCV has GUI):
`q`/`Esc` closes, `c` cycles palette, `n` toggles noise.

## Flag Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--scene PATH` | `configs/scenes/demo_industrial.yaml` | Scene YAML |
| `--camera-config PATH` | `configs/thermal_camera.yaml` | Thermal camera YAML |
| `--seed N` | random | Noise seed (two runs with the same seed produce identical noise) |
| `--autoplay` | off | Automatically press Play on the timeline |
| `--save-frames DIR` | off | Save frames as `DIR/frame_NNNNN.png` + `summary.json` |
| `--max-frames N` | 0 (unlimited) | Stop and close after N frames |
| `--auto-track linear\|circular` | off | Move the camera (or robot) automatically |
| `--ros2` | off | Publish ROS2 topics (use with `launch_thermal.sh`) |
| `--robot` | off | Mount the camera on the inspector robot |
| `--speed M_S` | scene YAML (0.5) | `--robot`: linear and vertical speed, m/s |
| `--turn-speed DEG_S` | scene YAML (30) | `--robot`: turn speed, degrees/s |
| `--resolution WxH` | camera YAML (640x480) | Render product resolution |
| `--palette whitehot\|ironbow\|hot\|jet` | camera YAML (jet) | Initial palette |
| `--display auto\|cv2\|png` | auto | Visualization method |
| `--png-path PATH` | `/tmp/live_pseudo_thermal.png` | Output path in PNG mode |
| `--renderer` | RaytracedLighting | Isaac renderer |

**Precedence:** YAML values apply by default; a CLI flag, when passed,
overrides them for that run.

## Editing the Scene from YAML

Everything that used to be hardcoded constants in the script now lives in two
files, with each field commented (what it is and what units it uses):

- [configs/scenes/demo_industrial.yaml](configs/scenes/demo_industrial.yaml):
  thermal model, objects, lights, camera, and robot.
- [configs/thermal_camera.yaml](configs/thermal_camera.yaml): resolution,
  palette, luminance weight, blur, and noise.

For a custom scene, copy the demo YAML and pass it with `--scene`:

```bash
cp configs/scenes/demo_industrial.yaml configs/scenes/my_scene.yaml
~/isaac_sim/python.sh run_pseudo_thermal.py --scene configs/scenes/my_scene.yaml
```

### Objects and Temperatures

Each entry under `objects` creates the prim `/World/<name>`:

```yaml
objects:
  - name: Tank              # unique; prim /World/Tank
    type: cylinder          # box | cylinder | sphere
    position: [1.0, 1.0, 0.5]   # center, m
    radius: 0.4             # cylinder and sphere, m
    height: 1.0             # cylinder, m
    axis: Z                 # cylinder: X | Y | Z
    color: [0.2, 0.6, 0.3]  # RGB 0..1
    roughness: 0.5          # optional (0.6)
    metallic: 0.0           # optional (0.0)
    temperature_c: 60.0     # optional, °C
```

A `box` uses `size: [x, y, z]` instead of `radius`/`height`/`axis`.

- **Change a temperature:** edit `temperature_c` on the object.
- **Ambient-temperature object:** remove `temperature_c`. It receives
  `thermal.default_c` ± `thermal.default_spread_c` (fixed offset per prim
  path hash, so two objects without a temperature are visually distinct).
- **Visualization range:** `thermal.range_c: [min, max]` maps to gray 0..255
  and to the scale bar. The background is always clamped to the minimum.
- Child prims inherit the parent's temperature (`/World/Tank/Mesh` uses
  `/World/Tank`'s temperature).

Demo scene values:

| Object | USD Path | Temperature |
|--------|----------|-------------|
| Floor | `/World/Floor` | 20 °C |
| Wall | `/World/Wall` | 25 °C |
| Hot pipe | `/World/HotPipe` | 80 °C |
| Anomaly | `/World/Anomaly` | 150 °C |
| Robot (`--robot`) | `/World/Robot` | 35 °C |
| Crates and ball | `/World/Crate_01`, `/World/Crate_02`, `/World/Ball` | 22 °C ± 2 °C |

Visualization range: 15–160 °C.

### Configuration Errors

YAMLs are validated before Isaac Sim opens. An unknown key, a missing field,
an incorrect type, or a duplicate object name stops the program with the file
and field in the error message:

```
[pseudo-thermal] Invalid configuration: my_scene.yaml > objects[4] (Crate_01): unknown key 'colour' (valid: name, type, ...)
[pseudo-thermal] Invalid configuration: my_scene.yaml > objects: duplicate object name 'Crate_01'
```

The names `Looks`, `Lights`, `Robot`, and `ThermalCamera` are reserved.

## Tests

Cover the pure modules (configuration, thermal processing, CLI) and do not
open Isaac Sim; they run in a couple of seconds:

```bash
cd ~/multimodal_sim/thermal_sim
~/isaac_sim/python.sh -m pytest tests/
```

`pytest.ini` disables the ROS pytest plugins, which fail to load in Isaac's
Python 3.12 if the terminal has `source /opt/ros/humble/setup.bash` active.

## Comparing Against the Original Script

`scripts/live_pseudo_thermal_legacy.py` is the original single-file script,
unmodified. To verify that the package produces the same image, record the
same 20 frames with each and compare them. `--palette whitehot` makes
differences readable in gray levels, and the seed ensures identical noise
(the legacy script has no `--seed`; `run_legacy_seeded.py` runs it as-is
with the seed injected from outside).

```bash
cd ~/multimodal_sim/thermal_sim

# 1. Legacy
~/isaac_sim/python.sh scripts/run_legacy_seeded.py --seed 0 \
    --autoplay --auto-track linear --max-frames 20 --palette whitehot \
    --save-frames outputs/compare_legacy/frames

# 2. New package
~/isaac_sim/python.sh run_pseudo_thermal.py --seed 0 \
    --autoplay --auto-track linear --max-frames 20 --palette whitehot \
    --save-frames outputs/compare_refactor/frames

# 3. Comparison (mean and max absolute difference per frame)
~/isaac_sim/python.sh scripts/compare_frames.py \
    outputs/compare_legacy/frames outputs/compare_refactor/frames
```

The expected global mean difference is below 1 gray level: the only source of
variation between two runs is the RTX render luminance, which contributes 15%
to the image. `compare_frames.py` exits with code 0 if the threshold is met.

## Important Notes

- **ROS2 and Isaac do not share a terminal.** The system spdlog libs
  (`/opt/ros/humble/lib`) conflict with Isaac's. Use `launch_thermal.sh` or
  set manually:
```bash
  export LD_LIBRARY_PATH=/home/jumasaet/isaac_sim/exts/isaacsim.ros2.core/humble/lib
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export ROS_DOMAIN_ID=15
```
- **Headless OpenCV.** Isaac ships `opencv-python-headless`. For `cv2.imshow`
  window support:
```bash
  ~/isaac_sim/python.sh -m pip install --no-deps opencv-python
```
- Saved frames carry no text overlay (FPS, Tmax) so they can be used as clean
  data.
- `summary.json` is written on close with run statistics.
