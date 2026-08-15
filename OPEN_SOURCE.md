# Open Source & Third Party Components

## Major Third Party Projects

| Project | What It Provides | What We Use It For |
|---|---|---|
| CARLA 0.9.15 | Full driving simulator with roads, traffic, weather, sensors | Simulation environment, sensor simulation, traffic |
| CARLA PythonAPI | Python client to control CARLA | Vehicle control, sensor attachment, route planning |
| CARLA GlobalRoutePlanner | A* route planning on road network | Route from A to B |
| Flask | Python web framework | Operator console API |
| Flask-SocketIO | WebSocket support for Flask | Real-time telemetry to console |
| NumPy | Array math | Sensor data processing |
| OpenCV | Image processing | Camera frame handling |

## What We Wrote

All files in `src/warp_av/` are original code:

- `vehicle_interface.py` — Abstract vehicle interface + physical stub
- `adapters/carla_vehicle_adapter.py` — CARLA implementation of vehicle interface
- `adapters/carla_sensor_adapter.py` — CARLA sensor management
- `perception/perception.py` — Object detection system
- `localization/localization.py` — Position tracking
- `behavior/behavior.py` — Decision making with reason strings
- `planning/planner.py` — Route planning wrapper
- `control/controller.py` — PID + pure pursuit controller
- `safety/safety_supervisor.py` — Safety monitoring system
- `mission/mission_manager.py` — Mission lifecycle management
- `telemetry/logger.py` — JSONL structured logging
- `main.py` — Main autonomy loop + Flask API
- `console/index.html` — Operator dashboard

## What We Modified

- CARLA's `GlobalRoutePlanner` is used as-is (imported, not modified)

## Why Reuse vs Build

| Decision | Choice | Why |
|---|---|---|
| Simulator | Reuse CARLA | Building a driving simulator is years of work. CARLA provides roads, physics, traffic, weather, sensors. |
| Route planning | Reuse CARLA's GlobalRoutePlanner | It already knows the road network. Writing A* on the road graph from scratch adds no value. |
| Object detection | Shortcut: use CARLA ground truth | Getting YOLO running is a Day 4+ task. The architecture supports swapping in real ML detection later. |
| Steering controller | Build (pure pursuit) | Simple enough to write, and we need to understand it for tuning. |
| Speed controller | Build (PID) | Same — fundamental, needs to be understood. |
| Safety supervisor | Build from scratch | Core to the assignment. Must be ours. |
| Behavior/decisions | Build from scratch | Core to the assignment. Defines how the vehicle thinks. |
| Vehicle interface | Build from scratch | The key architectural abstraction. Must be clean and ours. |
| Operator console | Build from scratch | Simple HTML/JS, no framework needed. |
| Logging | Build from scratch | JSONL is simple and gives us exactly the format we want. |
