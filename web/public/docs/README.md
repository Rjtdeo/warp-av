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
python3 -m pytest tests/        # catalog integrity, evaluator, safety supervisor, behaviour priority, command validation, fault injector (no CARLA needed)
```

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

A simulated front-facing RGB camera is attached to the CARLA cargo van.

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

The current implementation filters the point cloud to focus mainly on hazards relevant to the vehicle's forward driving path.

LiDAR is primarily used to estimate the physical distance to potential obstacles.

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

This implementation is intentionally designed as a practical simulation-level autonomous vehicle stack.

The Camera + LiDAR system currently uses lightweight forward association between camera classifications and LiDAR hazard distance.

It is **not** intended to represent a production-grade 3D sensor-fusion system.

Current limitations include:

- No full camera-to-LiDAR 3D projection
- No production multi-object 3D tracker
- No unrestricted autonomous obstacle avoidance
- Camera perception focuses primarily on pedestrians and vehicle classes
- LiDAR hazard processing focuses mainly on the forward driving corridor

The architecture is modular so these components can be replaced with more advanced implementations later.

