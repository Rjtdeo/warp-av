"""
Warp AV Main Loop

This is the heart of the system. Every tick (~10Hz):

1. Read sensors (sensor adapter)
2. Perceive objects (perception)
3. Know where we are (localization)
4. Check safety (safety supervisor)
5. Decide what to do (behavior)
6. Find next waypoint (planner)
7. Compute steering/throttle/brake (controller)
8. Send command to vehicle (vehicle adapter)
9. Log everything (telemetry)
10. Send state to console (API)

YOUR ROVER did steps 1, 5, 8 in three nodes.
This does all 10 steps in a clean loop.
"""

import time
import sys
import signal
import threading
import math
import json
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO

# Our modules
from .adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from .adapters.carla_sensor_adapter import CarlaSensorAdapter
from .perception.perception import PerceptionSystem
from .localization.localization import LocalizationSystem
from .behavior.behavior import BehaviorSystem, DrivingBehavior
from .planning.planner import RoutePlanner
from .control.controller import VehicleController
from .safety.safety_supervisor import SafetySupervisor, SafetyState
from .mission.mission_manager import MissionManager, MissionState
from .telemetry.logger import TelemetryLogger
from .vehicle_interface import VehicleCommand


class WarpAV:
    """The complete autonomy system."""

    def __init__(self, carla_host="localhost", carla_port=2000):
        print("=" * 60)
        print("  WARP AV — Autonomous Vehicle Platform")
        print("=" * 60)

        # --- Initialize all subsystems ---
        print("[Init] Connecting to CARLA...")
        self.vehicle_adapter = CarlaVehicleAdapter(carla_host, carla_port)

        print("[Init] Setting up sensors...")
        self.sensor_adapter = CarlaSensorAdapter(
            self.vehicle_adapter.world, self.vehicle_adapter.vehicle
        )
        self.sensor_adapter.setup_sensors()

        print("[Init] Starting perception...")
        self.perception = PerceptionSystem(
            self.vehicle_adapter.world, self.vehicle_adapter.vehicle
        )

        print("[Init] Starting localization...")
        self.localization = LocalizationSystem(self.vehicle_adapter.vehicle)

        print("[Init] Starting behavior...")
        self.behavior = BehaviorSystem()

        print("[Init] Starting planner...")
        self.planner = RoutePlanner(self.vehicle_adapter.get_map())

        print("[Init] Starting controller...")
        self.controller = VehicleController()

        print("[Init] Starting safety supervisor...")
        self.safety = SafetySupervisor()

        print("[Init] Starting mission manager...")
        self.mission_manager = MissionManager()

        print("[Init] Starting logger...")
        self.logger = TelemetryLogger(log_dir="logs")

        # Current state for the API/console
        self._current_state = {}
        self._route = None
        self._running = False

        print("[Init] All systems ready!")
        print("=" * 60)

    def start_mission(self, dest_x: float, dest_y: float):
        """Begin a mission to the given destination."""
        pose = self.localization.update()

        # Create mission
        mission = self.mission_manager.start_mission(
            dest_x, dest_y, pose.x, pose.y
        )

        # Plan route
        self._route = self.planner.plan_route(pose.x, pose.y, dest_x, dest_y)
        if not self._route:
            self.mission_manager.fail_mission("Route planning failed")
            return False

        # Start logging
        self.logger.start_mission_log(mission.mission_id)
        self.logger.log_event("mission_started", f"Destination: ({dest_x}, {dest_y})")

        # Engage autonomy
        self.vehicle_adapter.engage_autonomy()
        self.behavior.set_mission()
        self.mission_manager.set_executing()

        return True

    def tick(self):
        """
        ONE CYCLE of the autonomy loop.
        This is called ~10 times per second.
        """

        # 1. Localize
        pose = self.localization.update()

        # 2. Perceive
        perception = self.perception.update()

        # 3. Safety check
        safety_output = self.safety.update(
            perception_healthy=perception.healthy,
            perception_timestamp=perception.timestamp,
            localization_healthy=pose.healthy,
            localization_confidence=pose.confidence,
            localization_timestamp=pose.timestamp,
            controller_healthy=self.controller._enabled,
            vehicle_alive=self.vehicle_adapter.is_alive(),
            current_speed=pose.speed,
        )

        # Keep the latest safety result for operator Resume checks.
        self._last_safety_output = safety_output

        # If safety stops an executing mission, pause the SAME mission.
        # After the fault is restored, the operator must press Resume.
        current_mission = self.mission_manager.current_mission

        if (
            current_mission
            and current_mission.state == MissionState.EXECUTING
            and not safety_output.driving_allowed
        ):
            self.mission_manager.pause_mission()
            self.vehicle_adapter.disengage_autonomy()
            self.logger.log_event(
                "mission_paused_safety",
                safety_output.reason
            )

        # 4. Behavior decision
        dest_dist = None
        if self._route and self.mission_manager.current_mission:
            m = self.mission_manager.current_mission
            dest_dist = self.planner.distance_to_destination(self._route, pose.x, pose.y)

        behavior_output = self.behavior.update(
            perception=perception,
            pose=pose,
            destination_distance=dest_dist,
            safety_ok=safety_output.driving_allowed,
        )

        # 5. Get next waypoint
        target_x, target_y = pose.x + math.cos(pose.yaw) * 10, pose.y + math.sin(pose.yaw) * 10
        if self._route:
            next_wp = self.planner.get_next_waypoint(self._route, pose.x, pose.y)
            if next_wp:
                target_x, target_y = next_wp.x, next_wp.y

        # 6. Compute vehicle command
        cmd = self.controller.compute_command(
            current_x=pose.x, current_y=pose.y,
            current_yaw=pose.yaw, current_speed=pose.speed,
            target_x=target_x, target_y=target_y,
            desired_speed=behavior_output.desired_speed_mps,
            should_stop=behavior_output.should_stop,
        )

        # 7. Send command to vehicle
        self.vehicle_adapter.send_command(cmd)

        # 8. Check mission completion
        if behavior_output.behavior == DrivingBehavior.MISSION_COMPLETE:
            self.vehicle_adapter.disengage_autonomy()
            self.mission_manager.complete_mission()
            self.logger.log_event("mission_completed", "Arrived at destination")
            self.logger.stop_mission_log()

        # 9. Log
        mission_state = "idle"
        if self.mission_manager.current_mission:
            mission_state = self.mission_manager.current_mission.state.value

        self.logger.log_tick(
            pose_x=pose.x, pose_y=pose.y, pose_yaw=pose.yaw, pose_speed=pose.speed,
            behavior=behavior_output.behavior.value,
            behavior_reason=behavior_output.reason,
            steering=cmd.steering, throttle=cmd.throttle, brake=cmd.brake,
            safety_state=safety_output.state.value,
            safety_reason=safety_output.reason,
            perception_objects=len(perception.objects),
            closest_obstacle=perception.closest_obstacle_distance,
            mission_state=mission_state,
        )

        # 10. Update state for console
        self._current_state = {
            "pose": {"x": round(pose.x, 1), "y": round(pose.y, 1),
                     "yaw": round(math.degrees(pose.yaw), 1), "speed": round(pose.speed, 1)},
            "behavior": behavior_output.behavior.value,
            "behavior_reason": behavior_output.reason,
            "command": {"steer": round(cmd.steering, 3), "throttle": round(cmd.throttle, 3),
                        "brake": round(cmd.brake, 3)},
            "safety": {"state": safety_output.state.value, "reason": safety_output.reason},
            "perception": {
                "object_count": len(perception.objects),
                "closest_distance": round(perception.closest_obstacle_distance, 1),
                "closest_type": perception.closest_obstacle_type.value,
                "path_blocked": perception.path_blocked,
            },
            "mission": self.mission_manager.get_status(),
            "warnings": self.safety.warnings,
            "errors": self.safety.errors,
        }

    def run(self, tick_rate=10):
        """Main loop."""
        self._running = True
        dt = 1.0 / tick_rate
        print(f"\n[WarpAV] Running at {tick_rate} Hz. Console at http://localhost:5000")

        while self._running:
            try:
                self.tick()
                time.sleep(dt)
            except KeyboardInterrupt:
                print("\n[WarpAV] Shutting down...")
                break
            except Exception as e:
                print(f"[WarpAV] TICK ERROR: {e}")
                time.sleep(dt)

        self.shutdown()

    def shutdown(self):
        self._running = False
        self.vehicle_adapter.disengage_autonomy()
        self.sensor_adapter.destroy()
        self.vehicle_adapter.destroy()
        self.logger.stop_mission_log()
        print("[WarpAV] Shutdown complete")

    # --- API methods (called by Flask console) ---

    def api_start_mission(self, dest_x, dest_y):
        return self.start_mission(dest_x, dest_y)

    def api_stop_mission(self):
        self.behavior.cancel_mission()
        self.vehicle_adapter.disengage_autonomy()
        if self.mission_manager.current_mission:
            self.mission_manager.cancel_mission()
        self.logger.stop_mission_log()

    def api_emergency_stop(self):
        self.safety.trigger_estop("Operator commanded emergency stop")
        self.vehicle_adapter.emergency_stop()
        self.logger.log_event("emergency_stop", "Operator triggered E-STOP")

    def api_clear_estop(self):
        self.safety.clear_estop()
        self.vehicle_adapter.clear_emergency_stop()

    def api_pause(self):
        self.mission_manager.pause_mission()
        self.vehicle_adapter.disengage_autonomy()

    def api_resume(self):
        # Never resume while a safety fault is still active.
        safety_output = getattr(self, "_last_safety_output", None)

        if safety_output is None or not safety_output.driving_allowed:
            print("[Mission] RESUME BLOCKED — safety is not healthy")
            return False

        mission = self.mission_manager.current_mission

        if not mission or mission.state != MissionState.PAUSED:
            print("[Mission] RESUME BLOCKED — no paused mission")
            return False

        self.mission_manager.resume_mission()
        self.vehicle_adapter.engage_autonomy()
        self.logger.log_event(
            "mission_resumed",
            "Operator resumed mission after safety recovery"
        )

        return True

    def api_get_state(self):
        return self._current_state

    def api_get_history(self):
        return self.mission_manager.get_history()

    def api_get_spawn_points(self):
        """Get available destinations."""
        points = self.vehicle_adapter.get_spawn_points()
        return [{"x": round(p.location.x, 1), "y": round(p.location.y, 1), "idx": i}
                for i, p in enumerate(points[:20])]  # first 20

    # --- Test controls (for Scenario 6) ---
    def api_disable_perception(self):
        self.perception.disable()
    def api_enable_perception(self):
        self.perception.enable()
    def api_disable_localization(self):
        self.localization.disable()
    def api_enable_localization(self):
        self.localization.enable()
    def api_disable_camera(self):
        self.sensor_adapter.camera_enabled = False
    def api_enable_camera(self):
        self.sensor_adapter.camera_enabled = True


# ============================================================
# Flask API Server (runs in background thread)
# ============================================================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
av_system: WarpAV = None


@app.route('/')
def operator_console():
    console_file = Path(__file__).resolve().parent / "console" / "index.html"
    return send_file(console_file)


@app.route('/api/state')
def get_state():
    return jsonify(av_system.api_get_state())

@app.route('/api/mission/start', methods=['POST'])
def start_mission():
    data = request.json
    ok = av_system.api_start_mission(data['x'], data['y'])
    return jsonify({"success": ok})

@app.route('/api/mission/stop', methods=['POST'])
def stop_mission():
    av_system.api_stop_mission()
    return jsonify({"success": True})

@app.route('/api/mission/pause', methods=['POST'])
def pause_mission():
    av_system.api_pause()
    return jsonify({"success": True})

@app.route('/api/mission/resume', methods=['POST'])
def resume_mission():
    ok = av_system.api_resume()
    return jsonify({"success": ok})

@app.route('/api/estop', methods=['POST'])
def estop():
    av_system.api_emergency_stop()
    return jsonify({"success": True})

@app.route('/api/estop/clear', methods=['POST'])
def clear_estop():
    av_system.api_clear_estop()
    return jsonify({"success": True})

@app.route('/api/history')
def get_history():
    return jsonify(av_system.api_get_history())

@app.route('/api/spawn_points')
def get_spawn_points():
    return jsonify(av_system.api_get_spawn_points())

# Test/debug controls
@app.route('/api/test/disable_perception', methods=['POST'])
def disable_perception():
    av_system.api_disable_perception()
    return jsonify({"success": True})

@app.route('/api/test/enable_perception', methods=['POST'])
def enable_perception():
    av_system.api_enable_perception()
    return jsonify({"success": True})

@app.route('/api/test/disable_localization', methods=['POST'])
def disable_localization():
    av_system.api_disable_localization()
    return jsonify({"success": True})

@app.route('/api/test/enable_localization', methods=['POST'])
def enable_localization():
    av_system.api_enable_localization()
    return jsonify({"success": True})

@app.route('/api/test/disable_camera', methods=['POST'])
def disable_camera():
    av_system.api_disable_camera()
    return jsonify({"success": True})

@app.route('/api/test/enable_camera', methods=['POST'])
def enable_camera():
    av_system.api_enable_camera()
    return jsonify({"success": True})


def run_api_server():
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)


# ============================================================
# Entry point
# ============================================================

def main():
    global av_system

    av_system = WarpAV()

    # Start API server in background
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()

    # Run main autonomy loop
    av_system.run(tick_rate=10)


if __name__ == "__main__":
    main()
