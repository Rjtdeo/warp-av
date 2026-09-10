# Warp AV — Autonomous Vehicle Software Platform

Autonomous cargo vehicle software stack running in CARLA simulation with ROS 2.

## Quick Start

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with drivers installed
- Python 3.10+
- ROS 2 Humble

### 1. Install ROS 2 Humble
```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-pip
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. Install CARLA 0.9.15
```bash
# Download CARLA
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla focal main"
sudo apt update
sudo apt install -y carla-simulator

# Or download tar.gz from https://github.com/carla-simulator/carla/releases/tag/0.9.15
# Extract to ~/carla
```

### 3. Install Python Dependencies
```bash
cd warp-av
pip install -r requirements.txt
```

### 4. Build ROS 2 Workspace
```bash
cd warp-av
colcon build
source install/setup.bash
```

### 5. Launch Everything
```bash
# Terminal 1: Start CARLA
cd ~/carla && ./CarlaUE4.sh

# Terminal 2: Launch autonomy stack
ros2 launch warp_av full_stack.launch.py

# Terminal 3: Open operator console
cd src/console && python3 -m http.server 8080
# Open http://localhost:8080 in browser
```

### 6. Run a Mission
Open the operator console and click "Start Mission", or:
```bash
ros2 topic pub /mission/goal std_msgs/String '{"data": "destination_1"}' --once
```

## Architecture
See [architecture/README.md](architecture/README.md)

## Perception Stack

A classical multi-sensor perception stack, CARLA-validated, designed so each stage can be
replaced independently when real hardware arrives. `camera_lidar` is the boot default;
`ground_truth` remains an automatic fallback so the stack starts without a model file.

**Sensors**

| | |
|---|---|
| Front camera | 800×600, 90° FOV, 10 Hz, at (2.0, 0, 1.8) m, pitched 10° down |
| Left / right cameras | 480×360, 100° FOV, ~7 Hz, at (0, ∓1.1, 1.7) m, yawed ∓90°, pitched 15° down |
| Rear camera | 480×360, 90° FOV, ~7 Hz, at (−2.6, 0, 1.8) m, yawed 180° |
| Top-down camera | 420×420 at 22 m — **operator dashboard only, never used for perception** |
| Roof LiDAR | 32 channels, 150k points/s, 50 m range, 10 Hz, at (0, 0, 2.5) m |
| GNSS, IMU | attached and published, **not fused** — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) |

**Processing**

- **Four-view semantic camera coverage using a shared detector scheduler.** One YOLOX-S
  instance runs in a background thread and takes the four perception cameras in turn at
  4 detections/second in total — roughly **1 Hz per camera**. Detections older than 1 s are
  dropped rather than used stale.
- **LiDAR sweep accumulation** — per-frame wedges glued into a full 360° sweep, with returns
  off the van's own body removed.
- **Local ground filtering** — per-tile ground height from a neighbour median, with a
  road-plane fallback and kerb-line-aware point removal.
- **Grid flood-fill clustering** with oriented footprints: length, width, height and yaw.
- **Geometric camera–LiDAR fusion** — each cluster is projected through a calibrated camera
  model (mounts, tilt, yaw, focal length) into whichever cameras could see it, and takes its
  class from the detection box it lands in.
- **Multi-object tracking** — nearest-neighbour association with a constant-velocity Kalman
  filter: persistent IDs, world-frame velocity, and a moving/stationary decision that weighs
  distance actually travelled and path straightness, not just one speed reading.
- **Temporal occupancy / free-space grid** — 0.25 m cells, FREE / OCCUPIED / UNKNOWN plus a
  separate road-surface layer, with short-term memory and ego-motion compensation.
- **Blind-pocket detection** — finds unseen person-sized gaps ahead and caps speed by how
  near the nearest one is.
- **Kerb / road-edge estimation** — a fitted left and right kerb line, each with a confidence
  flag and a lateral-offset query.
- **Map-based traffic-light lookahead** — every light read once at start-up, matched to the
  route by `(road_id, lane_id)` **and** by actually reaching the stop line, with the colour
  queried at 4 Hz beyond 25 m and 10 Hz within it.
- **Map-guided camera traffic-light state** — the map says where the lamp housing is, the
  camera says what colour it is. An unreadable light reads `unknown`, which means stop.

**World model** (`world_model.py`) — tracked objects with position, world velocity, size,
heading, confidence and clearance radius; free / occupied / unknown space; kerb geometry;
traffic-control state; and sensor health.

Note that **occupancy and free space validate and slow the van; they do not generate
trajectories.** Planning still works from the route polyline and a corridor test.

## Scenarios (1000-scenario catalog + runner)
```bash
python3 scenarios/generate_catalog.py                  # regenerate catalog (deterministic)
python3 scenarios/run_scenario.py --status implemented  # run what should pass today (needs CARLA + stack running)
python3 scenarios/run_scenario.py WAV-0200 --dry-run    # print a plan without CARLA
python3 scenarios/report.py                             # aggregate -> scenarios/results/REPORT.md
```
See [scenarios/README.md](scenarios/README.md) and [docs/SCENARIO_STRATEGY.md](docs/SCENARIO_STRATEGY.md).
Catalog browser (static, deployed on Vercel from `web/public/`): see [web/README.md](web/README.md).

## Tests
```bash
pip install pytest
python3 -m pytest tests/ -q     # 700+ offline tests, no CARLA needed
python3 tools/full_check.py     # 28 live checks: needs a running CARLA + stack
```
Offline tests cover perception maths, ground filtering, clustering, tracking, occupancy,
prediction, planner geometry, behaviour priority, safety and the traffic-light lookahead.
`tools/full_check.py` places real objects in CARLA and reads the answer back through the
van's own interfaces — nothing is mocked. Scenario execution
(`scenarios/run_scenario.py`) also needs a live CARLA + stack and is not in CI.

## Operator / test API additions (port 5000)
| Endpoint | Purpose |
|---|---|
| `GET /api/route` | planned route waypoints (used by the scenario runner to place actors along the real route) |
| `POST /api/config/speed_limit {cruise_speed_mps}` | live speed-limit change |
| `POST /api/test/inject {component, action, ...}` | fault injection: perception/localization/camera/lidar/gnss/imu/controller/planner/vehicle_connection/tick_latency × disable/enable/freeze/stale/latency/low_confidence/noise/crash/nan_command/drop |
| `GET /api/state` | now also carries `timestamp, tick, autonomy_state, active_faults, last_tick_error, cruise_speed_mps, localization.confidence, destination` |


---

## Camera + LiDAR Perception Pipeline

Warp AV includes a Camera + LiDAR perception mode for more realistic autonomous driving and validation in CARLA.

### Perception and Driving Architecture

```text
CARLA Environment
        ↓
Front RGB Camera + LiDAR
        ↓
YOLOX Object Detection
        ↓
Camera + LiDAR Perception
        ↓
Behavior Planning
        ↓
Safety Supervisor
        ↓
Vehicle Controller
        ↓
CARLA Cargo Van
```

The goal is to keep sensing, perception, behavior, safety, control, and vehicle communication separated so each part can be tested and replaced independently.


### Front RGB Camera

A simulated front-facing RGB camera is attached to the CARLA cargo van. It is the sharpest
and fastest of the four perception cameras and is given first say when naming an object;
the left, right and rear views speak for what it cannot see. See
[Perception Stack](#perception-stack) for the full sensor layout.

The camera provides real image frames from the simulated environment.

The image is passed to:

```text
RGB Camera
     ↓
YOLOX-S
     ↓
OpenCV DNN
     ↓
Object Classification
```

The current perception system focuses on important road users such as:

- Pedestrians
- Cars
- Trucks
- Buses
- Motorcycles

The YOLOX model runs through OpenCV DNN using an ONNX model.


### LiDAR

The vehicle also uses a simulated 32-channel LiDAR.

LiDAR produces 3D point-cloud measurements around the vehicle.

Per-frame wedges are accumulated into a full 360° sweep, ground points are removed with a
local per-tile height estimate, and what remains is clustered into objects with a measured
footprint, height and heading.

LiDAR provides object geometry and position; the camera provides the class.

The simulated LiDAR has approximately a 50-meter sensing range, although the perception logic intentionally filters the data to focus on useful driving hazards.


### Camera + LiDAR Perception

The camera and LiDAR provide different information.

The camera helps answer:

```text
What is the object?
```

For example:

```text
PERSON
VEHICLE
```

LiDAR helps answer:

```text
How far away is the object?
```

For example:

```text
12 meters
```

The current perception system combines this information using a lightweight forward association.

Example:

```text
Camera → PERSON
              \
               → PEDESTRIAN approximately 12 m ahead
              /
LiDAR  → 12 m
```

For a vehicle:

```text
Camera → VEHICLE
               \
                → VEHICLE approximately 18 m ahead
               /
LiDAR  → 18 m
```

If LiDAR detects a relevant physical obstacle but the camera does not confidently classify it, the system can still report:

```text
OBSTACLE
```

This provides a safety fallback instead of completely ignoring an unknown object.


### Behavior Response

Perception output is passed to the behavior layer.

The behavior system decides whether the vehicle should:

```text
CONTINUE
SLOW DOWN
STOP
```

based on information such as:

- Object type
- Object distance
- Whether the object is relevant to the driving path
- Mission state
- Safety state

Example:

```text
Camera + LiDAR
      ↓
Vehicle detected ahead
      ↓
Distance decreases
      ↓
Behavior slows vehicle
      ↓
Hazard becomes too close
      ↓
Vehicle stops
```


### Safety Supervisor

Warp AV includes a dedicated Safety Supervisor separate from normal behavior planning.

The Safety Supervisor monitors conditions such as:

- Emergency stop state
- Vehicle connection health
- Camera health
- LiDAR health
- Sensor freshness
- Localization health
- Controller health
- Perception health

If an unsafe condition occurs, the Safety Supervisor can prevent normal driving commands and command the vehicle to stop.

This provides a separate safety layer instead of relying only on the behavior planner.


### Sensor Freshness Monitoring

Camera and LiDAR data are timestamped.

The system checks whether sensor information has become stale.

Examples include:

```text
CAMERA_STALE
LIDAR_STALE
```

This prevents the autonomy system from blindly trusting old sensor information if a sensor stops updating.


### LiDAR Self-Detection Filtering

During development, LiDAR initially detected parts of the cargo van itself as very close obstacles.

The perception pipeline includes filtering to remove these near-field vehicle-body returns so they are not treated as external hazards.


### Short-Term Hazard Persistence

Pedestrians and other narrow objects may produce inconsistent LiDAR returns between consecutive scans.

The perception system therefore keeps a recently confirmed hazard for a short period instead of immediately deleting it after a single missed scan.

This helps reduce unstable:

```text
DETECTED
NOT DETECTED
DETECTED
NOT DETECTED
```

behavior.


### Perception Modes

Warp AV supports two perception modes:

```text
ground_truth
camera_lidar
```

#### Ground Truth Mode

```text
ground_truth
```

uses CARLA actor information directly.

This provides a stable simulation fallback and is useful for validating the rest of the autonomy stack independently of computer vision.


#### Camera + LiDAR Mode

```text
camera_lidar
```

uses:

```text
CARLA RGB Camera
       +
CARLA LiDAR
       ↓
YOLOX + LiDAR Processing
       ↓
Perception Output
```

This mode provides a more realistic sensor-based perception path.


### Switching to Camera + LiDAR Mode

With Warp AV running:

```bash
curl -s -X POST \
-H "Content-Type: application/json" \
-d '{"mode":"camera_lidar"}' \
http://localhost:5000/api/perception/mode | python3 -m json.tool
```


### Live Front Camera

The same RGB camera used by the perception system can also be viewed live in the browser.

Open:

```text
http://localhost:5000/camera
```

This allows the operator to compare:

```text
What the camera sees
        ↓
What YOLOX detects
        ↓
What Camera + LiDAR perception reports
        ↓
How the vehicle responds
```

The camera viewer uses lightweight JPEG frame requests so it can run alongside Mission Control and the autonomy stack without creating a second CARLA camera sensor.


### Mission Control

The operator dashboard is available at:

```text
http://localhost:5000
```

Mission Control provides information about the autonomous system including:

- Mission state
- Vehicle speed
- Current behavior
- Route progress
- Perception information
- Safety state
- Vehicle state
- System health
- Live activity
- Scenario controls


### Controlled Traffic Validation

A CARLA traffic validation tool is included for testing the perception system with moving vehicles and pedestrians.

Start the autonomous mission first.

Then run:

```bash
python3 tools/dense_validation_traffic.py
```

The validation script creates controlled traffic around the Warp vehicle, including:

- Moving lead vehicles
- Nearby traffic
- Pedestrians near the route

This allows the Camera + LiDAR perception pipeline to be observed under more realistic dynamic conditions.

The traffic terminal also reports distances and vehicle speeds to help compare the simulation state with the perception system.

Press:

```text
Ctrl+C
```

in the traffic terminal to remove the validation actors created by the script.


### Complete Autonomy Flow

The current Warp AV software flow is:

```text
                    CARLA
                      │
          ┌───────────┴───────────┐
          │                       │
      RGB Camera                LiDAR
          │                       │
        YOLOX                  3D Points
          │                       │
          └───────────┬───────────┘
                      │
               Camera + LiDAR
                 Perception
                      │
                      ▼
                  Behavior
                      │
                      ▼
              Safety Supervisor
                      │
                      ▼
                 Controller
                      │
                      ▼
              Vehicle Interface
                      │
                      ▼
             CARLA Cargo Van
                      │
                      ▼
              Mission Telemetry
                      │
                      ▼
               Mission Control
```


### Current Scope

**Warp AV is a CARLA research prototype.** It has never driven a real vehicle, and it is
not a production autonomous-driving system.

The architecture is deliberately modular so each stage can be replaced when hardware
arrives. What follows is what the code actually does and does not do today — the full list
lives in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

#### Current limitations

- **Localization is simulator-provided.** CARLA's exact pose. GNSS and IMU are attached but
  not fused; there is no EKF and no pose uncertainty that planning reads.
- **One detector is shared across four cameras**, so semantic refresh is about 1 Hz per
  camera against 7–10 Hz frame rates.
- **Camera detections cannot create objects on their own.** Fusion labels LiDAR clusters;
  anything the LiDAR misses is invisible however clearly the camera sees it.
- **Camera and LiDAR are not tightly time-synchronized** — most-recent-frame fusion with a
  freshness cap, not matched timestamps.
- **Detector classes are generic COCO**, not a delivery-specific ontology. Cones, bollards,
  pallets and open doors are unnamed obstacles with measured geometry.
- **Traffic-light vision is map-guided.** The camera classifies the colour; the map supplies
  the lamp position. This is not autonomous signal discovery.
- **Occupancy / free space is not the motion-planning representation.** It caps speed and
  validates space; the planner still uses the route line and a corridor test.
- **Tracking uncertainty is measured but not consumed** by planning or behaviour.
- **Kerb sensing is geometric**, not road segmentation. No lane-marking perception.
- **Prediction is constant velocity**, not learned or multimodal.
- **Sensor health checks cover the front camera only** — a blinded side camera is not
  reported.
- **No radar, no calibration procedure, no DBW hardware integration, no real-sensor
  validation.**

#### Simulator-only dependencies

These are the paths that will not transfer, and what replaces each on a real van:

| Depends on CARLA today | Replaced on the real van by |
|---|---|
| Ego pose from `vehicle.get_transform()` | RTK GNSS + IMU + wheel-odometry fusion |
| Sensor pose used for LiDAR sweep de-skew | the same on-board localization estimate |
| Map topology for route planning | HD map or an on-board map service |
| Traffic-light stop lines and lane association | HD map signal layer |
| Exact lamp-housing geometry for the camera ROI | HD-map lamp prior, or an image-space signal-head detector |
| Traffic-light colour (`ground_truth` mode and `WARP_CAMERA_LIGHTS=0` only) | the camera classifier, which is already the default in `camera_lidar` mode |
| Per-sensor arrival timestamps | hardware-triggered / PTP-synchronized clocks |
| Collision sensor | test instrumentation only — no on-vehicle equivalent |
| Scenario ground-truth labels | offline annotation for scoring only |
