# Warp AV Architecture

## System Diagram

```
                          ┌─────────────────────┐
                          │   Operator Console   │
                          │    (HTML + JS)       │
                          └──────────┬───────────┘
                                     │ HTTP/WebSocket
                          ┌──────────▼───────────┐
                          │     Flask API         │
                          │  (mission control)    │
                          └──────────┬───────────┘
                                     │
    ┌────────────────────────────────────────────────────────┐
    │                    AUTONOMY LOOP (10 Hz)                │
    │                                                         │
    │  ┌─────────┐   ┌────────────┐   ┌──────────────┐      │
    │  │ Sensor   │──▶│ Perception │──▶│   Behavior   │      │
    │  │ Adapter  │   │ (objects)  │   │ (decisions)  │      │
    │  └─────────┘   └────────────┘   └──────┬───────┘      │
    │                                         │               │
    │  ┌─────────┐   ┌────────────┐   ┌──────▼───────┐      │
    │  │Localiz- │──▶│  Planner   │──▶│  Controller  │      │
    │  │ation    │   │ (route)    │   │ (steer/gas)  │      │
    │  └─────────┘   └────────────┘   └──────┬───────┘      │
    │                                         │               │
    │  ┌──────────────────┐           ┌──────▼───────┐      │
    │  │ Safety Supervisor │──────────▶│   Vehicle    │      │
    │  │ (can override)    │           │  Interface   │      │
    │  └──────────────────┘           └──────┬───────┘      │
    │                                         │               │
    │  ┌─────────┐   ┌────────────┐          │               │
    │  │ Mission  │   │  Logger    │          │               │
    │  │ Manager  │   │ (JSONL)    │          │               │
    │  └─────────┘   └────────────┘          │               │
    └─────────────────────────────────────────┼───────────────┘
                                              │
                          ┌───────────────────▼────────────────┐
                          │        Vehicle Adapter              │
                          │  ┌──────────┐  ┌────────────────┐  │
                          │  │  CARLA    │  │ Physical Van   │  │
                          │  │ (today)   │  │ (future stub)  │  │
                          │  └──────────┘  └────────────────┘  │
                          └────────────────────────────────────┘
```

## Key Design Decision: The Vehicle Interface

The most important boundary in the system. Everything above it is "the brain."
Everything below it is "the body."

- `VehicleInterface` (abstract class) defines: `send_command()`, `get_state()`, `engage_autonomy()`, `disengage_autonomy()`, `emergency_stop()`
- `CarlaVehicleAdapter` implements it for simulation
- `PhysicalVehicleAdapter` is a stub for the future real van

The autonomy stack NEVER imports `carla` directly. Only the adapters do.

## Component Responsibilities

| Component | Job | Rover Equivalent |
|---|---|---|
| Sensor Adapter | Read raw sensors, package for system | ESP32 + sensor_node.py |
| Perception | Detect objects (cars, people, obstacles) | decision_node checking front/left/right |
| Localization | Know where the vehicle is | GPS topic |
| Behavior | Decide what to do and WHY | decision_node.make_decision() |
| Planner | Plan route from A to B | (rover had none) |
| Controller | Convert decisions to steering/throttle/brake | motor_node sending serial commands |
| Safety Supervisor | Watch everything, stop if unsafe | front < 20 → STOP |
| Mission Manager | Track mission lifecycle | START/STOP from terminal |
| Logger | Record everything for replay | ROS logger.info() |
| Console | Operator dashboard | (rover had none) |

## Data Flow

Every tick (100ms):
1. Sensors produce data → Perception consumes it
2. Perception outputs detected objects → Behavior consumes them
3. Localization outputs position → Planner + Behavior consume it
4. Behavior outputs desired speed + stop/go → Controller consumes it
5. Planner outputs next waypoint → Controller consumes it
6. Controller outputs VehicleCommand → Vehicle Adapter consumes it
7. Safety Supervisor watches ALL of the above, can override command to STOP
8. Logger records ALL of the above to JSONL file
9. API serves ALL of the above to Console


## Test & scenario layer

```
scenarios/run_scenario.py ──▶ ScenarioRunner ──HTTP (same API as console)──▶ Flask API ──▶ WarpAV
        │                         │                                              │
        │                         └──carla PythonAPI──▶ CARLA world (spawn actors, weather, collision sensor)
        │                                                                        │
        └──▶ evaluator (trace → metrics → PASS/FAIL/GAP) ──▶ scenarios/results/   FaultInjector (testing/fault_injector.py)
                                                                                  ├─ perception.inject_fault / disable
                                                                                  ├─ localization.inject_fault / disable
                                                                                  ├─ sensor_adapter.<x>_enabled
                                                                                  ├─ controller.inject_fault / disable
                                                                                  ├─ planner.disable
                                                                                  ├─ vehicle_adapter.simulate_connection_loss
                                                                                  └─ main loop tick latency
```

The runner never reaches into the autonomy process: missions, e-stops, pauses, speed limits and fault injections all go through the operator API, so the same 1000 scenario definitions can later be pointed at a physical test-range backend (replace `carla_world.py`).
