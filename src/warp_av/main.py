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
import carla
import cv2
from pathlib import Path
from flask import Flask, jsonify, request, send_file, Response
from flask_socketio import SocketIO

# Our modules
from .adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from .adapters.carla_sensor_adapter import CarlaSensorAdapter
from .perception.perception import PerceptionSystem
from .perception.camera_lidar_perception import CameraLidarPerception
from .localization.localization import LocalizationSystem
from .behavior.behavior import BehaviorSystem, DrivingBehavior
from .planning.planner import RoutePlanner
from .control.controller import VehicleController
from .safety.safety_supervisor import SafetySupervisor, SafetyState
from .mission.mission_manager import MissionManager, MissionState
from .telemetry.logger import TelemetryLogger
from .testing.fault_injector import FaultInjector
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

        # Stable original perception.
        self.ground_truth_perception = PerceptionSystem(
            self.vehicle_adapter.world,
            self.vehicle_adapter.vehicle
        )

        # Camera + LiDAR is loaded only when selected.
        self.camera_lidar_perception = None

        # Always start safely with the proven baseline.
        self.perception_mode = "ground_truth"
        self.perception = self.ground_truth_perception

        print("[Perception] Default mode: GROUND TRUTH")

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

        print("[Init] Starting fault injector (test hooks)...")
        self.fault_injector = FaultInjector(self)

        # Current state for the API/console
        self._current_state = {}
        self._route = None
        self._tick_count = 0
        self._last_tick_error = ""

        # Route selected on the dashboard before START is pressed.
        self._preview_route = None
        self._preview_destination = None

        # Temporary CARLA actors created from dashboard scenario tests.
        self._scenario_actors = []
        self._scenario_type = None
        self._scenario_lights_frozen = False

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

        # ----------------------------------------------------
        # Use the route already previewed on the dashboard
        # when it matches this destination.
        # ----------------------------------------------------

        use_preview = False

        if (
            self._preview_route
            and self._preview_destination
            and abs(self._preview_destination[0] - dest_x) < 0.1
            and abs(self._preview_destination[1] - dest_y) < 0.1
            and self._preview_route.waypoints
        ):
            first_wp = self._preview_route.waypoints[0]

            distance_from_preview_start = math.sqrt(
                (first_wp.x - pose.x) ** 2
                + (first_wp.y - pose.y) ** 2
            )

            # If the van has not moved significantly,
            # use EXACTLY the route we already showed.
            if distance_from_preview_start < 8.0:
                use_preview = True

        if use_preview:

            self._route = self._preview_route

            print(
                f"[Planner] Using previewed route: "
                f"{len(self._route.waypoints)} waypoints, "
                f"{self._route.total_distance:.0f}m"
            )

        else:

            self._route = self.planner.plan_route(
                pose.x,
                pose.y,
                dest_x,
                dest_y
            )

        # Preview has now become the active route.
        self._preview_route = None
        self._preview_destination = None

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

        self._tick_count += 1
        extra_delay = self.fault_injector.extra_tick_delay()
        if extra_delay > 0:
            time.sleep(extra_delay)

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

        # ----------------------------------------------------
        # SAFETY RESPONSE POLICY
        #
        # Short Camera/LiDAR stale events in CARLA are treated
        # as recoverable:
        #
        #     brake -> wait -> automatically continue
        #
        # Real component failures still:
        #
        #     pause mission -> disengage -> manual Resume
        # ----------------------------------------------------

        current_mission = self.mission_manager.current_mission

        perception_reason = getattr(
            perception,
            "reason",
            ""
        )

        transient_sensor_stale = (
            perception_reason.startswith("CAMERA_STALE_")
            or perception_reason.startswith("LIDAR_STALE_")
        )

        if (
            current_mission
            and current_mission.state == MissionState.EXECUTING
            and not safety_output.driving_allowed
        ):

            if transient_sensor_stale:

                # Do NOT pause or disengage.
                #
                # behavior.update(... safety_ok=False)
                # commands a safe stop.
                #
                # Once perception becomes healthy again,
                # safety_ok becomes True and the same mission
                # automatically continues.
                pass

            else:

                # Real failure.
                # Require explicit operator recovery.
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

        junction = self.planner.upcoming_turn(self._route, pose.x, pose.y) if self._route else None
        behavior_output = self.behavior.update(
            perception=perception,
            pose=pose,
            destination_distance=dest_dist,
            safety_ok=safety_output.driving_allowed,
            junction=junction,
        )

        # Curve-aware speed cap (Troy #2/#3): slow down BEFORE sharp bends.
        # Never overrides stops; only lowers a positive desired speed.
        curve_cap = None
        if self._route and not behavior_output.should_stop and behavior_output.desired_speed_mps > 0.5:
            curve_cap = self.planner.curve_speed_cap(
                self._route, pose.x, pose.y, cruise=self.behavior.cruise_speed)
            if curve_cap < behavior_output.desired_speed_mps - 0.2:
                behavior_output.desired_speed_mps = curve_cap
                behavior_output.reason += f" | curve ahead — slowing to {curve_cap:.1f} m/s"

        # 5. Get next waypoint — aim further ahead the faster we go (1.6 s of
        # travel, clamped 5–13 m). A fixed 5 m aim point caused weaving at speed.
        target_x, target_y = pose.x + math.cos(pose.yaw) * 10, pose.y + math.sin(pose.yaw) * 10
        cross_track = self.planner.signed_cross_track(self._route, pose.x, pose.y) if self._route else 0.0
        if self._route:
            lookahead = max(5.0, min(13.0, 1.6 * pose.speed))
            # In/near a bend, aim closer so the van follows the arc instead of
            # cutting across it (kerb/divider clipping fix).
            if curve_cap is not None and curve_cap < self.behavior.cruise_speed - 0.5:
                lookahead = min(lookahead, 5.5)
            # Off the lane centre by more than a metre (post-corner drift, lane
            # change): aim closer so it gets back into its lane NOW instead of
            # sliding diagonally between lanes for tens of metres.
            if abs(cross_track) > 1.0:
                lookahead = min(lookahead, 6.0)
            next_wp = self.planner.get_next_waypoint(self._route, pose.x, pose.y, lookahead=lookahead)
            if next_wp:
                target_x, target_y = next_wp.x, next_wp.y

        # 6. Compute vehicle command
        cmd = self.controller.compute_command(
            current_x=pose.x, current_y=pose.y,
            current_yaw=pose.yaw, current_speed=pose.speed,
            target_x=target_x, target_y=target_y,
            desired_speed=behavior_output.desired_speed_mps,
            should_stop=behavior_output.should_stop,
            cross_track_m=cross_track,
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

                # Objects shown on the operator map.
                # These come from our current CARLA ground-truth perception.
                "objects": [] if not pose.healthy else [
                    {
                        "id": obj.id,
                        "type": obj.object_type.value,
                        "distance": round(obj.distance, 1),

                        "x": round(
                            pose.x
                            + math.cos(pose.yaw) * obj.x
                            - math.sin(pose.yaw) * obj.y,
                            2
                        ),

                        "y": round(
                            pose.y
                            + math.sin(pose.yaw) * obj.x
                            + math.cos(pose.yaw) * obj.y,
                            2
                        ),
                    }
                    for obj in perception.objects
                ],
            },
            "perception_mode": self.perception_mode,

            "perception_runtime": {
                "source": (
                    "Camera + LiDAR"
                    if self.perception_mode == "camera_lidar"
                    else "CARLA Ground Truth"
                ),
                "yolox_inference_ms": round(
                    getattr(
                        self.camera_lidar_perception,
                        "last_inference_ms",
                        0.0
                    ),
                    1
                )
            },

            "mission": self.mission_manager.get_status(),

            # ------------------------------------------------
            # Sensor + system health for operator dashboard
            # ------------------------------------------------
            "health": {
                "camera": {
                    "healthy": self.sensor_adapter.is_camera_healthy(),
                    "enabled": self.sensor_adapter.camera_enabled,
                    "label": "Front Camera"
                },

                "lidar": {
                    "healthy": self.sensor_adapter.is_lidar_healthy(),
                    "enabled": self.sensor_adapter.lidar_enabled,
                    "label": "LiDAR"
                },

                "gps": {
                    "healthy": self.sensor_adapter.is_gnss_healthy(),
                    "enabled": self.sensor_adapter.gnss_enabled,
                    "label": "GPS"
                },

                "imu": {
                    "healthy": self.sensor_adapter.is_imu_healthy(),
                    "enabled": self.sensor_adapter.imu_enabled,
                    "label": "IMU"
                },

                "object_detection": {
                    "healthy": perception.healthy,
                    "enabled": getattr(self.perception, "_enabled", True),
                    "label": "Object Detection"
                },

                "vehicle_position": {
                    "healthy": pose.healthy,
                    "enabled": getattr(self.localization, "_enabled", True),
                    "label": "Vehicle Position"
                },

                "controller": {
                    "healthy": self.controller._enabled,
                    "enabled": self.controller._enabled,
                    "label": "Vehicle Controller"
                }
            },

            "warnings": self.safety.warnings,
            "errors": self.safety.errors + ([self.vehicle_adapter.last_command_rejected] if getattr(self.vehicle_adapter, "last_command_rejected", "") else []),
            "timestamp": time.time(),
            "tick": self._tick_count,
            "autonomy_state": self.vehicle_adapter._autonomy_state.value,
            "active_faults": dict(self.fault_injector.active),
            "last_tick_error": self._last_tick_error,
            "cruise_speed_mps": self.behavior.cruise_speed,
            "junction": junction,   # {"distance_m", "direction"} when a turn at a junction is within 20 m, else null

            "localization": {"confidence": round(pose.confidence, 2), "quality": pose.quality.value, "healthy": pose.healthy},
            "destination": ({"x": self.mission_manager.current_mission.destination_x, "y": self.mission_manager.current_mission.destination_y}
                            if self.mission_manager.current_mission else None),
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
                # A software fault must never leave the last throttle command applied.
                self._last_tick_error = f"{type(e).__name__}: {e}"
                print(f"[WarpAV] TICK ERROR: {e} -> commanding brake")
                try:
                    self.vehicle_adapter.send_command(self.controller.emergency_brake())
                    self.logger.log_event("tick_error", self._last_tick_error)
                except Exception:
                    pass
                time.sleep(dt)

        self.shutdown()

    def shutdown(self):
        self._running = False

        # Remove temporary scenario actors before destroying vehicle.
        self.clear_scenario()

        self.vehicle_adapter.disengage_autonomy()
        self.sensor_adapter.destroy()
        self.vehicle_adapter.destroy()
        self.logger.stop_mission_log()
        print("[WarpAV] Shutdown complete")

    # ========================================================
    # Dashboard road-scenario tests
    # ========================================================

    def clear_scenario(self):
        """Remove temporary pedestrian / vehicle / barrier actors."""

        for actor in self._scenario_actors:

            try:
                if actor and actor.is_alive:
                    actor.destroy()
            except Exception as e:
                print(f"[Scenario] Could not destroy actor: {e}")

        self._scenario_actors.clear()
        self._scenario_type = None

        if getattr(self, "_scenario_lights_frozen", False):
            try:
                for tl in self.vehicle_adapter.world.get_actors().filter("traffic.traffic_light"):
                    tl.freeze(False)
                print("[Scenario] Traffic lights released to automatic cycling")
            except Exception as e:
                print(f"[Scenario] Could not release traffic lights: {e}")
            self._scenario_lights_frozen = False

        print("[Scenario] Test objects cleared")


    def _get_scenario_waypoint(self, distance_m=20.0):
        """
        Find a driving-lane waypoint roughly distance_m ahead.

        If a mission route exists, prefer that route so the object
        appears on the road the van is actually following.
        """

        pose = self.localization.update()

        if not pose.healthy:
            return None

        carla_map = self.vehicle_adapter.get_map()

        # ----------------------------------------------------
        # Prefer the ACTIVE planned route.
        # ----------------------------------------------------

        if self._route:

            route_wp = self.planner.get_next_waypoint(
                self._route,
                pose.x,
                pose.y,
                lookahead=distance_m
            )

            if route_wp:

                target = carla_map.get_waypoint(
                    carla.Location(
                        x=route_wp.x,
                        y=route_wp.y,
                        z=0.0
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                if target:
                    return target


        # ----------------------------------------------------
        # Fallback: use the vehicle's current driving lane.
        # ----------------------------------------------------

        current_location = (
            self.vehicle_adapter.vehicle.get_location()
        )

        current_wp = carla_map.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if not current_wp:
            return None

        candidates = current_wp.next(distance_m)

        if not candidates:
            return None

        return candidates[0]


    def api_spawn_scenario(self, scenario_type):
        """
        Spawn one controlled test object ahead of the moving van.

        Supported:
        pedestrian
        vehicle
        barrier
        red_light   (freezes every traffic light red; cleared by clear_scenario)
        """

        allowed = {
            "pedestrian",
            "vehicle",
            "barrier",
            "red_light",
        }

        if scenario_type not in allowed:
            return {
                "success": False,
                "reason": "Unknown scenario type"
            }


        # Require an active trip so the demo is unambiguous.
        mission = self.mission_manager.current_mission

        if (
            not mission
            or mission.state != MissionState.EXECUTING
        ):
            return {
                "success": False,
                "reason": "Start a trip before creating a road scenario"
            }


        # Remove any previous test object first.
        self.clear_scenario()


        # Red light test: no object to spawn — freeze every light red.
        if scenario_type == "red_light":
            lights = list(self.vehicle_adapter.world.get_actors().filter("traffic.traffic_light"))
            if not lights:
                return {"success": False, "reason": "This map has no traffic lights"}
            for tl in lights:
                tl.set_state(carla.TrafficLightState.Red)
                tl.freeze(True)
            self._scenario_lights_frozen = True
            self._scenario_type = scenario_type
            try:
                self.logger.log_event("scenario_spawned", f"red_light: {len(lights)} lights frozen RED")
            except Exception:
                pass
            return {"success": True,
                    "reason": f"{len(lights)} traffic lights frozen RED — the van must stop at the next junction. CLEAR releases them."}


        target_wp = self._get_scenario_waypoint(
            distance_m=20.0
        )

        if not target_wp:
            return {
                "success": False,
                "reason": "Could not find a road position ahead"
            }


        world = self.vehicle_adapter.world
        blueprints = world.get_blueprint_library()

        location = target_wp.transform.location
        rotation = target_wp.transform.rotation

        actor = None


        # ----------------------------------------------------
        # PEDESTRIAN
        # ----------------------------------------------------

        if scenario_type == "pedestrian":

            walker_blueprints = blueprints.filter(
                "walker.pedestrian.*"
            )

            if not walker_blueprints:
                return {
                    "success": False,
                    "reason": "No pedestrian blueprint available"
                }

            bp = walker_blueprints[0]

            transform = carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.5
                ),
                carla.Rotation(
                    yaw=rotation.yaw + 90.0
                )
            )

            actor = world.try_spawn_actor(
                bp,
                transform
            )


        # ----------------------------------------------------
        # VEHICLE
        # ----------------------------------------------------

        elif scenario_type == "vehicle":

            vehicle_blueprints = blueprints.filter(
                "vehicle.audi.*"
            )

            if not vehicle_blueprints:
                vehicle_blueprints = blueprints.filter(
                    "vehicle.*"
                )

            if not vehicle_blueprints:
                return {
                    "success": False,
                    "reason": "No vehicle blueprint available"
                }

            bp = vehicle_blueprints[0]

            transform = carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.5
                ),
                rotation
            )

            actor = world.try_spawn_actor(
                bp,
                transform
            )

            if actor:
                try:
                    actor.apply_control(
                        carla.VehicleControl(
                            throttle=0.0,
                            brake=1.0,
                            hand_brake=True
                        )
                    )
                except Exception:
                    pass


        # ----------------------------------------------------
        # ROAD BARRIER
        # ----------------------------------------------------

        elif scenario_type == "barrier":

            try:
                bp = blueprints.find(
                    "static.prop.streetbarrier"
                )
            except Exception:
                return {
                    "success": False,
                    "reason": "Road barrier blueprint unavailable"
                }

            transform = carla.Transform(
                carla.Location(
                    x=location.x,
                    y=location.y,
                    z=location.z + 0.2
                ),
                carla.Rotation(
                    yaw=rotation.yaw + 90.0
                )
            )

            actor = world.try_spawn_actor(
                bp,
                transform
            )


        if not actor:

            return {
                "success": False,
                "reason": (
                    "Spawn location was occupied. "
                    "Try again after the van moves a little."
                )
            }


        self._scenario_actors.append(actor)
        self._scenario_type = scenario_type


        try:
            self.logger.log_event(
                "scenario_spawned",
                f"{scenario_type} placed ahead of vehicle"
            )
        except Exception:
            pass


        print(
            f"[Scenario] {scenario_type.upper()} "
            f"spawned ahead at "
            f"({location.x:.1f}, {location.y:.1f})"
        )


        return {
            "success": True,
            "type": scenario_type,
            "x": round(location.x, 1),
            "y": round(location.y, 1),
            "message": (
                f"{scenario_type.capitalize()} "
                f"placed about 20 m ahead"
            )
        }


    def api_clear_scenario(self):

        self.clear_scenario()

        try:
            self.logger.log_event(
                "scenario_cleared",
                "Operator cleared road test object"
            )
        except Exception:
            pass

        return {
            "success": True
        }


    # --- API methods (called by Flask console) ---

    def api_preview_route(self, dest_x, dest_y):
        """
        Calculate a route WITHOUT moving the vehicle.

        The dashboard can show this route before the operator
        presses START.
        """

        pose = self.localization.update()

        if not pose.healthy:
            return {
                "success": False,
                "reason": "Vehicle position is unavailable"
            }

        route = self.planner.plan_route(
            pose.x,
            pose.y,
            dest_x,
            dest_y
        )

        if not route:
            return {
                "success": False,
                "reason": "No road route found"
            }

        self._preview_route = route
        self._preview_destination = (
            float(dest_x),
            float(dest_y)
        )

        return {
            "success": True,

            "distance_m": round(
                route.total_distance,
                1
            ),

            "waypoint_count": len(
                route.waypoints
            ),

            "route": [
                {
                    "x": round(wp.x, 2),
                    "y": round(wp.y, 2)
                }
                for wp in route.waypoints
            ]
        }

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

    def api_get_route(self):
        if not self._route:
            return []
        return [{"x": round(w.x, 2), "y": round(w.y, 2)} for w in self._route.waypoints]

    def api_set_speed_limit(self, speed_mps):
        self.behavior.set_cruise_speed(speed_mps)
        self.logger.log_event("speed_limit_changed", f"cruise={self.behavior.cruise_speed}")
        return self.behavior.cruise_speed

    def api_inject(self, component, action, **params):
        return self.fault_injector.inject(component, action, **params)

    def api_get_state(self):
        return self._current_state

    def api_get_history(self):
        return self.mission_manager.get_history()

    def api_get_map_data(self):
        """
        Return CARLA road geometry plus the current planned route.

        Road geometry is generated once and cached because the
        Town10HD road network does not change during a mission.
        """

        # ----------------------------------------------------
        # Build road geometry once
        # ----------------------------------------------------

        if not hasattr(self, "_map_geometry_cache"):

            carla_map = self.vehicle_adapter.get_map()

            # Group sampled driving waypoints by lane.
            lanes = {}

            for wp in carla_map.generate_waypoints(4.0):

                if wp.lane_type != carla.LaneType.Driving:
                    continue

                key = (
                    wp.road_id,
                    wp.section_id,
                    wp.lane_id,
                )

                loc = wp.transform.location

                lanes.setdefault(key, []).append(
                    (
                        float(wp.s),
                        round(loc.x, 2),
                        round(loc.y, 2),
                    )
                )

            roads = []

            for points in lanes.values():

                # OpenDRIVE 's' gives us the order along the lane.
                points.sort(key=lambda item: item[0])

                if len(points) < 2:
                    continue

                roads.append([
                    {
                        "x": x,
                        "y": y,
                    }
                    for _, x, y in points
                ])

            # Buildings are MAP context for the dashboard.
            # We are not claiming perception detected these.
            buildings = []

            env_buildings = self.vehicle_adapter.world.get_environment_objects(
                carla.CityObjectLabel.Buildings
            )

            for building in env_buildings:

                bb = building.bounding_box

                buildings.append({
                    "id": int(building.id),
                    "x": round(float(bb.location.x), 2),
                    "y": round(float(bb.location.y), 2),
                    "extent_x": round(float(bb.extent.x), 2),
                    "extent_y": round(float(bb.extent.y), 2),
                    "yaw": round(float(bb.rotation.yaw), 2),
                })

            self._map_geometry_cache = {
                "name": carla_map.name,
                "roads": roads,
                "buildings": buildings,
            }

            print(
                f"[Map] Dashboard buildings ready: "
                f"{len(buildings)} building objects"
            )

            print(
                f"[Map] Dashboard road geometry ready: "
                f"{len(roads)} lane segments"
            )

        # ----------------------------------------------------
        # Current planned route
        # ----------------------------------------------------

        route_points = []

        if self._route:

            route_points = [
                {
                    "x": round(wp.x, 2),
                    "y": round(wp.y, 2),
                }
                for wp in self._route.waypoints
            ]

        return {
            "name": self._map_geometry_cache["name"],
            "roads": self._map_geometry_cache["roads"],
            "buildings": self._map_geometry_cache["buildings"],
            "route": route_points,
        }

    def api_get_spawn_points(self):
        """Get available destinations."""
        points = self.vehicle_adapter.get_spawn_points()
        return [{"x": round(p.location.x, 1), "y": round(p.location.y, 1), "idx": i}
                for i, p in enumerate(points[:20])]  # first 20

    # --- Test controls (for Scenario 6) ---
    def api_set_perception_mode(self, mode):
        """
        Switch between:

        ground_truth:
            CARLA actor information.
            Stable fallback.

        camera_lidar:
            Actual CARLA camera pixels -> YOLOX
            Actual CARLA LiDAR -> forward distance.
        """

        mode = str(mode).strip().lower()

        if mode not in ("ground_truth", "camera_lidar"):
            return {
                "success": False,
                "reason": "Mode must be ground_truth or camera_lidar",
                "mode": self.perception_mode
            }

        # Do not change perception while the van is actively driving.
        mission = self.mission_manager.current_mission

        if (
            mission
            and mission.state in (
                MissionState.EXECUTING,
                MissionState.PAUSED
            )
        ):
            return {
                "success": False,
                "reason": "Stop the current trip before changing perception mode",
                "mode": self.perception_mode
            }

        if mode == "ground_truth":

            self.perception = self.ground_truth_perception
            self.perception_mode = "ground_truth"

            # Restore normal simulation speeds.
            self.behavior.cruise_speed = 8.0
            self.behavior.slow_speed = 3.0

            print(
                "[Perception] Mode switched -> "
                "CARLA GROUND TRUTH"
            )

        else:

            # Load YOLOX only the first time this mode is selected.
            if self.camera_lidar_perception is None:

                print(
                    "[Perception] Loading "
                    "Camera + LiDAR perception..."
                )

                self.camera_lidar_perception = (
                    CameraLidarPerception(
                        self.sensor_adapter
                    )
                )

            self.perception = self.camera_lidar_perception
            self.perception_mode = "camera_lidar"

            # Lower speed for the first sensor-based driving demo.
            self.behavior.cruise_speed = 4.0
            self.behavior.slow_speed = 2.0

            print(
                "[Perception] Mode switched -> "
                "CAMERA + LIDAR"
            )

        return {
            "success": True,
            "mode": self.perception_mode,
            "source": (
                "Camera + LiDAR"
                if self.perception_mode == "camera_lidar"
                else "CARLA Ground Truth"
            ),
            "cruise_speed_mps": self.behavior.cruise_speed
        }


    def api_get_perception_mode(self):

        return {
            "mode": self.perception_mode,
            "source": (
                "Camera + LiDAR"
                if self.perception_mode == "camera_lidar"
                else "CARLA Ground Truth"
            )
        }


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
        print("[Fault Test] CAMERA feed disabled")

    def api_enable_camera(self):
        self.sensor_adapter.camera_enabled = True
        print("[Fault Test] CAMERA feed restored")

    def api_disable_lidar(self):
        self.sensor_adapter.lidar_enabled = False
        print("[Fault Test] LIDAR feed disabled")

    def api_enable_lidar(self):
        self.sensor_adapter.lidar_enabled = True
        print("[Fault Test] LIDAR feed restored")

    def api_disable_gnss(self):
        self.sensor_adapter.gnss_enabled = False
        print("[Fault Test] GPS/GNSS signal disabled")

    def api_enable_gnss(self):
        self.sensor_adapter.gnss_enabled = True
        print("[Fault Test] GPS/GNSS signal restored")

    def api_disable_imu(self):
        self.sensor_adapter.imu_enabled = False
        print("[Fault Test] IMU signal disabled")

    def api_enable_imu(self):
        self.sensor_adapter.imu_enabled = True
        print("[Fault Test] IMU signal restored")

    def api_disable_controller(self):
        self.controller.disable()
        print("[Fault Test] VEHICLE CONTROLLER disabled")

    def api_enable_controller(self):
        self.controller.enable()
        print("[Fault Test] VEHICLE CONTROLLER restored")


# ============================================================
# Flask API Server (runs in background thread)
# ============================================================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.after_request
def _add_cors_headers(resp):
    # The operator console may be served from another origin (e.g. warp-av.vercel.app).
    # Without these headers the browser blocks its fetches to this API.
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp
av_system: WarpAV = None


@app.route('/')
def operator_console():
    console_file = Path(__file__).resolve().parent / "console" / "index.html"
    return send_file(console_file)



# ============================================================
# Live front RGB camera preview
#
# IMPORTANT:
# This uses ONE JPEG PER REQUEST instead of an infinite MJPEG
# stream. That keeps Mission Control responsive while YOLOX
# and CARLA are also running.
# ============================================================

@app.route('/api/camera/frame')
def camera_frame():

    system = globals().get("av_system")

    if system is None:
        return Response(
            "AV system not ready",
            status=503
        )

    sensor_adapter = getattr(
        system,
        "sensor_adapter",
        None
    )

    if sensor_adapter is None:
        return Response(
            "Sensor adapter not ready",
            status=503
        )

    frame = sensor_adapter.latest_camera

    if frame is None:
        return Response(
            "Camera frame not available yet",
            status=503
        )

    try:

        # CARLA gives BGRA.
        # OpenCV uses BGR.
        image_bgr = frame.image[:, :, :3]

        # Dashboard preview does not need full 800x600.
        # Smaller image = less CPU + less network work.
        image_bgr = cv2.resize(
            image_bgr,
            (640, 480),
            interpolation=cv2.INTER_AREA
        )

        success, jpeg = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )

        if not success:
            return Response(
                "JPEG encoding failed",
                status=500
            )

        return Response(
            jpeg.tobytes(),
            mimetype="image/jpeg",
            headers={
                "Cache-Control":
                    "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except Exception as exc:

        print(
            "[CameraFrame] Encode error:",
            exc
        )

        return Response(
            f"Camera error: {exc}",
            status=500
        )


@app.route('/camera')
def camera_viewer():

    return Response(
        r"""
<!doctype html>
<html>
<head>
    <meta charset="utf-8">

    <title>Warp AV Front Camera</title>

    <style>

        body {
            margin: 0;
            background: #05070a;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .viewer {
            width: min(95vw, 900px);
        }

        h2 {
            margin-bottom: 6px;
        }

        .status {
            color: #9aa4b2;
            margin-bottom: 12px;
        }

        img {
            width: 100%;
            display: block;
            background: #111;
            border-radius: 12px;
        }

    </style>
</head>

<body>

<div class="viewer">

    <h2>Warp AV — Front RGB Camera</h2>

    <div class="status" id="status">
        Waiting for camera...
    </div>

    <img id="camera">

</div>

<script>

const camera =
    document.getElementById("camera");

const status =
    document.getElementById("status");


function requestFrame() {

    camera.src =
        "/api/camera/frame?t=" +
        Date.now();
}


camera.onload = function() {

    status.textContent =
        "LIVE • Front RGB Sensor • ~2 FPS";

    /*
     * Request the NEXT frame only after
     * the current one completely loaded.
     *
     * This prevents overlapping HTTP requests.
     */

    setTimeout(
        requestFrame,
        500
    );
};


camera.onerror = function() {

    status.textContent =
        "Waiting for camera frame...";

    setTimeout(
        requestFrame,
        1000
    );
};


requestFrame();

</script>

</body>
</html>
""",
        mimetype="text/html"
    )


@app.route('/api/state')
def get_state():
    return jsonify(av_system.api_get_state())

@app.route('/api/route/preview', methods=['POST'])
def preview_route():
    data = request.get_json(silent=True) or {}

    if "x" not in data or "y" not in data:
        return jsonify({
            "success": False,
            "reason": "Destination x/y required"
        }), 400

    result = av_system.api_preview_route(
        float(data["x"]),
        float(data["y"])
    )

    return jsonify(result)


@app.route('/api/scenario/spawn', methods=['POST'])
def spawn_scenario():
    data = request.get_json(silent=True) or {}

    scenario_type = data.get("type")

    result = av_system.api_spawn_scenario(
        scenario_type
    )

    return jsonify(result)


@app.route('/api/scenario/clear', methods=['POST'])
def clear_scenario():
    return jsonify(
        av_system.api_clear_scenario()
    )


@app.route('/api/mission/start', methods=['POST'])
def start_mission():
    data = request.json or {}
    try:
        x, y = float(data['x']), float(data['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify({"success": False, "reason": "x and y (numbers) required"}), 400
    if not (abs(x) < 1e5 and abs(y) < 1e5):
        return jsonify({"success": False, "reason": "destination out of range"}), 400
    ok = av_system.api_start_mission(x, y)
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

@app.route('/api/route')
def get_route():
    return jsonify(av_system.api_get_route())

@app.route('/api/config/speed_limit', methods=['POST'])
def set_speed_limit():
    data = request.json or {}
    if 'cruise_speed_mps' not in data:
        return jsonify({"success": False, "reason": "cruise_speed_mps required"}), 400
    return jsonify({"success": True, "cruise_speed_mps": av_system.api_set_speed_limit(data['cruise_speed_mps'])})

@app.route('/api/test/inject', methods=['POST'])
def inject_fault():
    """Generic fault injection. Body: {"component": "...", "action": "...", ...params}"""
    data = dict(request.json or {})
    component = data.pop('component', None)
    action = data.pop('action', None)
    if not component or not action:
        return jsonify({"success": False, "reason": "component and action required"}), 400
    result = av_system.api_inject(component, action, **data)
    return jsonify(result), (200 if result.get("success") else 422)

@app.route('/api/spawn_points')
def get_spawn_points():
    return jsonify(av_system.api_get_spawn_points())


@app.route('/api/map')
def get_map_data():
    return jsonify(av_system.api_get_map_data())

# Perception source
@app.route('/api/perception/mode', methods=['GET', 'POST'])
def perception_mode():

    if request.method == 'GET':
        return jsonify(
            av_system.api_get_perception_mode()
        )

    data = request.get_json(silent=True) or {}

    result = av_system.api_set_perception_mode(
        data.get("mode", "")
    )

    return jsonify(result)


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


@app.route('/api/test/disable_lidar', methods=['POST'])
def disable_lidar():
    av_system.api_disable_lidar()
    return jsonify({"success": True})


@app.route('/api/test/enable_lidar', methods=['POST'])
def enable_lidar():
    av_system.api_enable_lidar()
    return jsonify({"success": True})


@app.route('/api/test/disable_gnss', methods=['POST'])
def disable_gnss():
    av_system.api_disable_gnss()
    return jsonify({"success": True})


@app.route('/api/test/enable_gnss', methods=['POST'])
def enable_gnss():
    av_system.api_enable_gnss()
    return jsonify({"success": True})


@app.route('/api/test/disable_imu', methods=['POST'])
def disable_imu():
    av_system.api_disable_imu()
    return jsonify({"success": True})


@app.route('/api/test/enable_imu', methods=['POST'])
def enable_imu():
    av_system.api_enable_imu()
    return jsonify({"success": True})


@app.route('/api/test/disable_controller', methods=['POST'])
def disable_controller():
    av_system.api_disable_controller()
    return jsonify({"success": True})


@app.route('/api/test/enable_controller', methods=['POST'])
def enable_controller():
    av_system.api_enable_controller()
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
