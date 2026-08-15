# Known Issues

## Perception Uses Ground Truth

Perception currently reads CARLA's actor list (world.get_actors()) instead of processing camera/lidar data with ML models. This means:
- Detection is perfect in simulation but wouldn't work with real sensors
- No camera-based object detection (YOLO etc.) is integrated yet
- This is an intentional shortcut to get the full pipeline working first
- Next step: run YOLOv8 on camera images for real perception

## No Lane Detection

The vehicle follows CARLA's route waypoints but does not detect lane markings visually. It relies on the map graph for staying in lane.

## Simple Steering Controller

Pure pursuit steering works but oscillates at higher speeds. A Stanley controller or MPC would be more stable. PID speed control also needs tuning.

## No Re-planning

If the route is blocked, the vehicle stops but does not re-plan around the obstacle. It waits until the path is clear.

## Single Camera Only

Only a front-facing camera is used. No rear, side, or surround view. Blind spots exist.

## No Sensor Fusion

Camera, lidar, GNSS, and IMU data are not fused. Each is used independently. Real system would need an EKF or similar.

## Console Is Polling-Based

Console polls the API every 200ms instead of using WebSocket push. Works but adds latency.

## Localization Is Perfect

In simulation, localization reads CARLA's ground truth position. With real sensors, this would need GNSS+IMU+odometry fusion with uncertainty estimation.

## No Automated Tests Yet

Scenario tests are manual scripts. Need to convert to automated pytest suite that runs scenarios and checks pass/fail criteria.

## Hardcoded CARLA Settings

CARLA host/port, vehicle type, sensor configurations are partially hardcoded. Should be fully configurable.

## No Map Boundaries

Vehicle doesn't know its allowed operating area. Safety supervisor should check geofencing.

## Technical Debt

- Error handling is minimal in some components
- No graceful recovery from component crashes (just stops)
- Logger doesn't handle disk-full scenarios
- No unit tests for individual components
