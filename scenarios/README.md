# Test Scenarios

Run each scenario with CARLA and the autonomy stack running.

## Scenario 1: Normal Mission
```bash
python3 scenarios/scenario_1_normal.py
```
Vehicle receives destination, drives there, completes mission.
Expected: mission_state → "completed"

## Scenario 2: Vehicle Ahead
```bash
python3 scenarios/scenario_2_vehicle.py
```
Spawns a stopped vehicle in the path.
Expected: behavior → "stopped_vehicle", reason explains why.

## Scenario 3: Pedestrian
```bash
python3 scenarios/scenario_3_pedestrian.py
```
Spawns a pedestrian crossing the road.
Expected: behavior → "stopped_pedestrian", vehicle waits.

## Scenario 4: Static Obstacle
```bash
python3 scenarios/scenario_4_obstacle.py
```
Places a static prop in the lane.
Expected: behavior → "stopped_obstacle"

## Scenario 5: Blocked Route
```bash
python3 scenarios/scenario_5_blocked.py
```
Multiple obstacles block the entire road.
Expected: vehicle stops, logs show blocked route.

## Scenario 6: Component Failure
```bash
# From console: click "Kill Perception" button
# Or:
curl -X POST http://localhost:5000/api/test/disable_perception
```
Expected: safety_state → "intervention", reason → "Perception system unhealthy"

## Scenario 7: Emergency Stop
```bash
# From console: click E-STOP button
# Or:
curl -X POST http://localhost:5000/api/estop
```
Expected: safety_state → "emergency_stop", vehicle full brake, requires manual clear.
