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

import os
import time
import sys
import signal
import threading
import math
from collections import Counter
import json
import carla
import cv2
from pathlib import Path
from flask import Flask, jsonify, request, send_file, Response
from flask_socketio import SocketIO

# Our modules
from .adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from .adapters.carla_sensor_adapter import CarlaSensorAdapter
from .perception.perception import (PerceptionSystem, DetectedObject, ObjectType,
                                    VULNERABLE_TYPES)
from .perception.camera_lidar_perception import CameraLidarPerception
from .pacing import sleep_remainder
from .localization.localization import LocalizationSystem
from .behavior.behavior import (BehaviorSystem, DrivingBehavior, EASE_OFF_REASONS,
                               EASE_OFF_MPS)
from .planning.planner import (RoutePlanner, Route, WaitingIsPointless, overtake_blocker,
                              nothing_is_standing_there, pass_refused, pass_options,
                              what_the_ground_says, GROUND_LOOK_M, GROUND_KEEP_M,
                              lane_change_blocker)
from .behavior.transitions import (GO_AROUND_START, GO_AROUND_WAIT, GO_AROUND_DONE,
                                   SPOT_CHOSEN, SPOT_CONFIRMED, SPOT_RECHOSEN, SPOT_GIVEN_UP,
                                   GROUND_SEEN_FREE, GROUND_BLOCKED, REROUTED, NO_WAY_ROUND,
                                   VEHICLE_IN_PATH, OBSTACLE_IN_PATH, ROUTE_BLOCKED_TOO_LONG,
                                   JUNCTION_KEEP_CLEAR, LANE_CHANGE_WAIT, LANE_CHANGE_WAITING,
                                   LANE_CHANGE_GO)
from .planning.prediction import predict_route_conflict
from .control.controller import VehicleController
from .safety.safety_supervisor import SafetySupervisor, SafetyState
from .mission.mission_manager import MissionManager, MissionState
from .telemetry.logger import TelemetryLogger
from .testing.fault_injector import FaultInjector
from .vehicle_interface import VehicleCommand, GearState
from .planning.sensed_slots import sensed_parking_slots, nearest_free_slot, consistent_with, hold_short_point
from .planning.rl_parker import RLParker, box_outline_points, stop_overrides_brain
from .planning.instrumentation import (PhaseTimer, PlannerDecision, BLOCKED_OCCUPANCY,
                                       UNKNOWN_SPACE)
from .planning.footprint_config import FootprintBlockingConfig
from .planning.parking_check import spot_view, spot_counts, SPOT_DEFAULT_LEN_M, SPOT_DEFAULT_WID_M
from .planning.footprint_debug import FootprintDebugConfig, FootprintDebugDrawer, build_frame
from .perception.bay_finder import why_no_kerb


from .world_model import build_world_model
from .perception.traffic_lights import SignalMap, TrafficLightLookahead, carla_state_source
from .perception.road_signs import read_signs, signs_on_route, next_sign
from .perception.light_camera import CameraLightReader, LampMap, camera_lights_wanted
from .sensor_health import HealthMonitor, read_sensors

#: Confirming a parking spot with the van's own LiDAR (2026-09-11) -- see WarpAV._confirm_parking_spot.
#: The second opinion on a thing standing in the way (_unblock_if_the_ground_is_seen_free):
#: how far ahead of the bumper the laser's free-space map is read. What the reading has to
#: say before the van drives on is planner.nothing_is_standing_there.
SEEN_FREE_LOOK_M = 7.0
SEEN_FREE_MIN_LOOK_M = 2.0

#: While getting past something standing in the lane: how close a body may come before the
#: van stops mid-manoeuvre, and how fast it may go. A squeeze inside the lane passes within
#: arm's reach of the thing by design, so the lane change's 1.6 m would freeze it beside what
#: it is passing; 1.0 m from the van's middle is 1 cm from its side.
PASS_ABORT_M, SQUEEZE_ABORT_M = 1.6, 1.0
PASS_SPEED_MPS, SQUEEZE_SPEED_MPS = 3.0, 2.0

SPOT_CONFIRM_FROM_M = 28.0   # start looking this far from the spot (the LiDAR's map reaches 30 m)
SPOT_FREE_LOOKS = 2          # seen free on this many looks in a row before the van turns in
SPOT_WAIT_S = 3.0            # still unseen at the start of the pull-in: wait this long, then re-choose
SPOT_BLOCKED_LOOKS = 4       # a thing in the way of the pull-in must be there on this many looks in a
                             # row (half a second) before the spot is given up: once, a lamp post
                             # beyond the strip was judged in the way and a good strip refused


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

        # Ground-truth contact sensor: the final referee for every test run.
        # Consecutive events against the same actor within 1 s count once.
        self._collision_count = 0
        self._last_collision = None
        self._collision_sensor = None
        try:
            col_bp = self.vehicle_adapter.world.get_blueprint_library().find("sensor.other.collision")
            self._collision_sensor = self.vehicle_adapter.world.spawn_actor(
                col_bp, carla.Transform(), attach_to=self.vehicle_adapter.vehicle)
            self._collision_sensor.listen(self._on_collision)
            print("[Init] Collision sensor attached")
        except Exception as e:
            print(f"[Init] Collision sensor unavailable: {e}")

        # Which code is running (shown in /api/state so remote testing can
        # verify a deploy actually took effect).
        self._start_time = time.time()
        try:
            import os as _os, subprocess as _sp
            self._git_rev = _sp.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                text=True, stderr=_sp.DEVNULL).strip()
        except Exception:
            self._git_rev = "unknown"

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
        # Where parking slots come from: "map" (CARLA's lane data, the
        # baseline) or "lidar" (bay_finder on the live sweep, map as fallback).
        self.parking_source = "map"
        # Who drives the last 16 m into a slot: "rules" (the hand-written
        # parker) or "rl" (round 6's learned parker, rl/models/parking_ppo_round6.zip).
        self.parker = "rules"
        self.rl_parker = RLParker()
        self.perception = self.ground_truth_perception

        print("[Perception] Default mode: GROUND TRUTH")

        print("[Init] Starting localization...")
        self.localization = LocalizationSystem(self.vehicle_adapter.vehicle)

        print("[Init] Starting behavior...")
        self.behavior = BehaviorSystem()

        self._health = None               # the day-8 sensor report, rebuilt every tick
        self._blind_spot_m = None         # the day-12 distance to the nearest unseen pocket
        self._signal_lookahead = None     # traffic lights, known about before we reach them
        self._light_reader = None         # phase 15b: their colour, read off the camera
        self._signal_ahead = None
        self.health_monitor = HealthMonitor()
        print("[Init] Starting planner...")
        self.planner = RoutePlanner(self.vehicle_adapter.get_map())
        # Every traffic light on the map, and the lanes each one governs. Read ONCE: the
        # geometry never moves, and finding it was what cost 65 ms a tick before.
        try:
            smap = SignalMap.from_world(self.vehicle_adapter.world)
            # Phase 15b: in camera mode the COLOUR comes off the camera, not out of the
            # simulator. The map still says WHERE each lamp head is -- a real van's HD map
            # records the same thing -- and the camera says what colour it is.
            #
            # Which source is used is decided per read, not here, because the perception mode
            # is switched over the API long after start-up and can be switched back. In
            # ground-truth mode there is no picture to read and asking the simulator is the
            # honest answer; in camera mode the camera answers even when it cannot tell, and
            # "cannot tell" already means stop.
            self._light_reader = None
            if camera_lights_wanted():
                lamps = LampMap.from_world(self.vehicle_adapter.world)
                if len(lamps):
                    self._light_reader = CameraLightReader(lamps)
                    print(f"[Signals] lamp heads for {len(lamps)} lights read in "
                          f"{lamps.build_ms:.0f} ms -- in camera mode the COLOUR comes "
                          f"from the camera")
                else:
                    print("[Signals] no lamp heads found; the simulator will give the colour")
            from_sim = carla_state_source(smap)

            def light_colour(light_id, _reader=lambda: self._light_reader, _sim=from_sim):
                reader = _reader()
                if reader is not None and self.perception_mode == "camera_lidar":
                    return reader.read(light_id)
                return _sim(light_id)

            self._signal_lookahead = TrafficLightLookahead(smap, state_source=light_colour)
            # ...and the signs on the same map: stop and give-way (perception/road_signs.py)
            self._signs = read_signs(self.vehicle_adapter.get_map())
            kinds = Counter(sign.kind for sign in self._signs)
            print(f"[Signs] {kinds.get('stop', 0)} stop and {kinds.get('give_way', 0)} give-way "
                  f"lanes read from the map")
            lanes = sum(len(s.lanes) for s in smap.signals.values())
            lined = sum(len(s.lines) for s in smap.signals.values())
            print(f"[Signals] {len(smap)} traffic lights read from the map "
                  f"in {smap.build_ms:.0f} ms; stop lines worked out for {lined} of {lanes} lanes")
        except Exception as e:
            self._signal_lookahead = None
            self._light_reader = None
            self._signs = []
            print(f"[Signals] could not read the map's traffic lights: {e}")
        # Planning V2: swept-path blocking, OFF by default. The van's real size
        # is read from the CARLA bounding box (fallback 2.96 x 0.99 m).
        self.footprint_blocking = FootprintBlockingConfig.from_vehicle(self.vehicle_adapter.vehicle)
        # Where the front bumper is, ahead of the position localization reports: the van's own
        # size, not knowledge of the world. The red-light stop is measured from it.
        try:
            bb = self.vehicle_adapter.vehicle.bounding_box
            self.behavior.front_offset_m = float(bb.location.x) + float(bb.extent.x)
        except Exception:
            pass
        fb = self.footprint_blocking.state()
        print(f"[Planner] footprint {fb['vehicle_half_length_m']} x {fb['vehicle_half_width_m']} m "
              f"({fb['dimensions_source']}), margin {fb['safety_margin_m']} m, "
              f"swept-path blocking {'ON' if fb['footprint_blocking_enabled'] else 'OFF'}")
        # Debug drawing of footprint / swept path in the CARLA world. OFF by
        # default, independent of the blocking flag, never touches decisions.
        self.footprint_debug = FootprintDebugConfig()
        self._footprint_drawer = FootprintDebugDrawer(self.vehicle_adapter.world)

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
        self._loop_hz = None          # measured decisions per second (EMA), exported to /api/state
        self._tick_ms = 0.0           # measured work per tick (EMA)
        self._phase_ms = {}           # ... and which part of it took how long
        # Planning V2 phase 0. The old timer kept ONE exponentially-smoothed number per
        # phase, so its "worst" was the worst SMOOTHED value: a single 300 ms tick moved
        # the average by 30 ms and then vanished. This keeps a rolling window instead, so
        # avg, p95 and a genuine worst case are all available -- and the phases below are
        # split finely enough to name a culprit rather than a third of the tick.
        self._phases = PhaseTimer()
        self._last_tick_error = ""

        # Route selected on the dashboard before START is pressed.
        self._preview_route = None
        self._preview_destination = None

        # Temporary CARLA actors created from dashboard scenario tests.
        self._scenario_actors = []
        self._scenario_type = None
        self._scenario_lights_frozen = False
        self._parking_spot = None
        self._lidar_rescan_done = False
        self._lidar_rescan_tries = 0
        self._hold_short_done = False
        self.rl_parker.reset()
        self._traffic_vehicles = []
        self._traffic_walkers = []      # (walker, controller) pairs
        self._scenario_jaywalkers = []  # (walker, start_location)
        self._cutin = None              # state machine for the cut-in car
        self._parked_cars = []          # cars parked in bays via /api/test/park_cars
        self._weather_preset = "default"

        # Operator decision (2026-08-30): CAMERA VISION is the boot default.
        # Ground truth remains the automatic fallback so the stack always
        # starts (fresh clone without models/yolox_s.onnx, load errors, ...).
        try:
            result = self.api_set_perception_mode("camera_lidar")
            if isinstance(result, dict) and not result.get("success", True):
                raise RuntimeError(result.get("reason", "switch refused"))
            print("[Perception] BOOT DEFAULT: Camera + LiDAR")
        except Exception as e:
            print(f"[Perception] camera default unavailable ({e}) — staying on ground truth")
        self._blocked_since = None      # when STOPPED_VEHICLE began (overtake timer)
        self._overtake_point = None     # rejoin Waypoint while a pass is active
        self._overtake_retry_at = 0.0

        self._running = False

        print("[Init] All systems ready!")
        print("=" * 60)

    def _signs_ahead(self, pose):
        """(bumper to the next sign's line, its kind, which one it is) or (None, None, None).

        The signs on this route are matched once, when the route is made; only "how far to the
        next one" is worked out each tick."""
        if not self._route or not getattr(self, "_signs", None):
            return (None, None, None)
        if getattr(self, "_signs_route", None) is not self._route.waypoints:
            self._signs_route = self._route.waypoints
            self._signs_on_route = signs_on_route(self._signs, self._route.waypoints)
        if not self._signs_on_route:
            return (None, None, None)
        total = sum(math.hypot(b.x - a.x, b.y - a.y)
                    for a, b in zip(self._route.waypoints, self._route.waypoints[1:]))
        along_now = total - self.planner.route_left_m(self._route, pose.x, pose.y)
        found = next_sign(self._signs_on_route, along_now)
        if found is None:
            return (None, None, None)
        gap, sign = found
        return (gap - self.behavior.front_offset_m, sign.kind,
                (sign.road_id, sign.lane_id, round(sign.x, 1), round(sign.y, 1)))

    def _dress_route_for_parking(self):
        """Bend the end of the route to a parking spot, and remember the route it was drawn on.

        Done when a mission starts, and again whenever the route itself is replaced -- a way
        round a blocked street -- because a spot drawn on a route the van is no longer taking
        is not a spot."""
        self._parking_slots = None
        self._parking_rechecked = False
        # Bend the end of the route to a kerbside parking spot (Troy #7):
        # finish pulled over on the right, not dead-centre on the road. The route is carried
        # on past the pin first, so that where there is no room to pull in gently before it,
        # the spot can be a little past it instead.
        try:
            pin_index = len(self._route.waypoints) - 1
            self.planner.extend_past_pin(self._route, self.planner.PARK_FAR_PAST_PIN_M)
            # The route as planned, before any pull-in is drawn on it: every later choice of
            # spot is drawn afresh from this, not on top of the last one.
            self._route_base = list(self._route.waypoints)
            self._pin_index = pin_index
            self._parking_rejected = []
            self._parking_wait_since = None
            chosen = self._choose_spot()
            if chosen is not None:
                self._route.waypoints, self._parking_spot = chosen
            else:
                # nowhere at all: the pin itself, but still judged on being straight there
                self._route.waypoints = list(self._route_base[:pin_index + 1])
                end = self._route.waypoints[-1]
                self._parking_spot = {"x": end.x, "y": end.y, "yaw": end.yaw, "kind": "lane",
                                      "offset_m": 0.0, "moved_back_m": 0, "confirmed": True,
                                      "note": "no workable spot near the pin"}
        except Exception as e:
            print(f"[Mission] pull-over computation failed ({e}) — parking on the lane")
            self._parking_spot = None
            self._lidar_rescan_done = False
            self._lidar_rescan_tries = 0
            self._hold_short_done = False
            self.rl_parker.reset()
        if self._parking_spot:
            moved = self._parking_spot.get("moved_back_m", 0)
            kind = self._parking_spot.get("kind", "kerb")
            what = {"bay": "PARKING BAY off the driving lane",
                    "kerb": "kerb-hug inside the lane (no bay to pull into)",
                    "lane": "straight in the lane (no bay or kerb to pull into)"}.get(kind, kind)
            past = self._parking_spot.get("past_pin_m") or 0
            note = (f", {moved} m before the pin" if moved > 1 else
                    f", {past} m past the pin" if past > 1 else "")
            self._parking_note = (f"{what}: ({self._parking_spot['x']}, {self._parking_spot['y']}), "
                                  f"{self._parking_spot['offset_m']} m right of lane centre{note}")
            print(f"[Mission] parking spot: {self._parking_spot}")
        else:
            self._parking_note = None
            print("[Mission] no kerbside spot found near the pin — will park on the lane")

    #: A road that stays blocked: how long the van watches it before looking for another way
    #: round, how far ahead a blockage counts, and how often it may ask the map.
    REROUTE_AFTER_S = 6.0
    REROUTE_WITHIN_M = 45.0
    REROUTE_EVERY_S = 15.0
    #: what counts as "the road is blocked" for this: something in the way, a road that has
    #: stayed blocked, and a junction whose far side is blocked (the van is held before it)
    BLOCKED_REASONS = (VEHICLE_IN_PATH, OBSTACLE_IN_PATH, ROUTE_BLOCKED_TOO_LONG,
                       JUNCTION_KEEP_CLEAR)

    def _maybe_reroute(self, pose, perception, behavior_output):
        """A street that stays blocked is a street to go round: ask the map for another way to
        the same destination, and take it if there is one.

        Only worth asking while a junction still lies between the van and the blockage -- the
        van has no reverse gear, so a way round that starts behind it is no way round at all.
        planner.plan_route_avoiding makes that piece of road expensive and searches again; it
        answers None when every route to the destination still goes through it.
        """
        mission = self.mission_manager.current_mission
        if not self._route or mission is None or self._overtake_point is not None:
            return
        blocked = behavior_output.why in self.BLOCKED_REASONS or perception.path_blocked
        if not blocked:
            self._blocked_road_since = None
            return
        now = time.time()
        if getattr(self, "_blocked_road_since", None) is None:
            self._blocked_road_since = now
            return
        if now - self._blocked_road_since < self.REROUTE_AFTER_S \
                or now - getattr(self, "_reroute_asked_at", 0.0) < self.REROUTE_EVERY_S:
            return
        self._reroute_asked_at = now
        at_m = float(getattr(perception, "closest_obstacle_distance", 0.0) or 0.0)
        if at_m <= 0.0 or at_m > self.REROUTE_WITHIN_M:
            return
        span = self.planner.junction_span(self._route, pose.x, pose.y)
        if span is None or span[0] > at_m:
            self._note_move(NO_WAY_ROUND,
                            f"the way is blocked {at_m:.0f} m ahead and there is no junction "
                            f"between here and it — nothing to turn off at")
            return
        c, s_ = math.cos(pose.yaw), math.sin(pose.yaw)
        bx, by = pose.x + c * at_m, pose.y + s_ * at_m
        dest = (mission.destination_x, mission.destination_y)
        other = self.planner.plan_route_avoiding(pose.x, pose.y, dest[0], dest[1], bx, by)
        if other is None:
            self._note_move(NO_WAY_ROUND,
                            f"every way to the destination goes through the blockage "
                            f"{at_m:.0f} m ahead — waiting")
            return
        self._route = other
        self._dress_route_for_parking()
        if self._signal_lookahead is not None:
            self._signal_lookahead.set_route(self._route)
        self._blocked_road_since = None
        self._note_move(REROUTED,
                        f"the road is blocked {at_m:.0f} m ahead — going round: a new route of "
                        f"{other.total_distance:.0f} m to the same destination")
        self.logger.log_event("rerouted",
                              f"blocked {at_m:.1f} m ahead at ({bx:.1f}, {by:.1f}); new route "
                              f"{other.total_distance:.0f} m, {len(other.waypoints)} points")
        print(f"[Route] blocked {at_m:.1f} m ahead — another way round: {other.total_distance:.0f} m")

    #: Changing lane: how far ahead the van starts looking into the lane it will move into,
    #: and how close to the move it may still be rolling while that lane is not clear.
    LANE_CHANGE_LOOK_M = 25.0
    LANE_CHANGE_HOLD_M = 2.0
    #: ...and how long it waits for a gap that is not coming before it goes anyway. Only ever
    #: for something STANDING there -- which is not going to move, so there is nothing to wait
    #: for beyond being sure it is standing. Traffic MOVING in that lane is waited for however
    #: long it takes. At 20 s the waiting cost a 510 m mission more than a minute.
    LANE_CHANGE_GIVE_UP_S = 3.0
    LANE_CHANGE_GIVE_UP_HOLDS_S = 30.0

    def _wait_for_a_gap(self, pose, perception, behavior_output):
        """Look into the lane the route moves into before moving into it, and wait for a gap.

        The map's route changes lane where it likes; until 2026-09-11 the van simply swerved
        across, having looked at nothing. Now it asks the same question the go-around asks of
        the lane it borrows (planner.lane_change_blocker) and, while the answer is no, holds
        its own lane: slowing as it comes up to the move, and stopping short of it rather than
        crossing into what is there.
        """
        if not self._route or self._overtake_point is not None or behavior_output.should_stop:
            return
        if time.time() < getattr(self, "_gap_given_up_until", 0.0):
            return                       # already decided to take this one
        nxt = self.planner.next_lane_change(self._route, pose.x, pose.y, pose.yaw,
                                            within_m=self.LANE_CHANGE_LOOK_M)
        if nxt is None:
            self._gap_wait_since = None
            return
        start_m, side = nxt
        why = lane_change_blocker(perception.objects, side, pose.yaw)
        if why is None:
            if getattr(self, "_gap_wait_since", None) is not None:
                self._note_move(LANE_CHANGE_GO, f"the lane is clear now — moving over")
                self._gap_wait_since = None
            return
        if getattr(self, "_gap_wait_since", None) is None:
            self._gap_wait_since = time.time()
            self._note_move(LANE_CHANGE_WAITING,
                            f"the route moves {'right' if side > 0 else 'left'} in "
                            f"{start_m:.0f} m, but {why} — waiting for a gap")
        if "standing" in why and time.time() - self._gap_wait_since > self.LANE_CHANGE_GIVE_UP_S:
            # ...and it stays given up: reconsidering every tick is how the van spent a whole
            # mission stopped in 20-second cycles at one kerbside blob (2026-09-11).
            self._note_move(LANE_CHANGE_GO,
                            f"{why}, and it is not going to move — taking the lane anyway and "
                            f"letting the usual rules deal with what is in front")
            self._gap_wait_since = None
            self._gap_given_up_until = time.time() + self.LANE_CHANGE_GIVE_UP_HOLDS_S
            return
        crawl = max(0.0, min(behavior_output.desired_speed_mps,
                             0.6 * max(0.0, start_m - self.LANE_CHANGE_HOLD_M)))
        behavior_output.desired_speed_mps = crawl
        behavior_output.should_stop = crawl <= 0.05
        behavior_output.reason += f" | waiting for a gap to move over: {why}"
        behavior_output.why = LANE_CHANGE_WAIT

    def _forget_the_manoeuvre(self):
        """A pass belongs to the mission it was begun in. A mission cancelled mid-pass used to
        leave the rejoin point set for ever, and everything that asks "am I mid-pass?" kept
        saying yes: the go-around would not start, and the laser's own second opinion on the
        ground was switched off for the rest of the stack's life (found live 2026-09-11)."""
        self._overtake_point = None
        self._overtake_retry_at = 0.0
        self._gap_wait_since = None
        self._gap_given_up_until = 0.0
        self._blocked_since = None
        self._ground_block = None

    def start_mission(self, dest_x: float, dest_y: float):
        """Begin a mission to the given destination."""
        pose = self.localization.update()

        # Fresh mission — the parking approach re-scan may fire again.
        self._parking_rechecked = False
        # One-shot parking flags and the learned parker reset for EVERY mission,
        # not only on the branch that finds a kerbside spot (arm C: a parker
        # left engaged by a stopped mission judged the next mission's slot
        # 183 m away and gave up on the spot).
        self._lidar_rescan_done = False
        self._lidar_rescan_tries = 0
        self._hold_short_done = False
        self._give_up = WaitingIsPointless()     # fix 3: a fresh clock for every mission
        self._forget_the_manoeuvre()             # a pass belongs to the mission it began in
        self.rl_parker.reset()

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

        # Ease out of the previous parking bay instead of full-lock swinging
        # onto the lane (the "between lanes at the start" observation).
        try:
            if self.planner.blend_departure(self._route, pose.x, pose.y):
                print("[Planner] departure blend: easing out of the bay onto the lane")
        except Exception as e:
            print(f"[Planner] departure blend failed ({e}) — starting as planned")

        self._dress_route_for_parking()

        # Start logging
        self.logger.start_mission_log(mission.mission_id)
        self.logger.log_event("mission_started", f"Destination: ({dest_x}, {dest_y})")
        if getattr(self, "_parking_note", None):
            self.logger.log_event("parking_spot", self._parking_note)
            self._note_move(SPOT_CHOSEN, self._parking_note)

        # Slot parking is the DEFAULT: find the boxes near the destination now,
        # skip occupied ones, and aim the mission into the best free box. The
        # dashboard shows them from the first metre. Falls back to the kerbside
        # spot when the street has no usable slots (or all are taken).
        try:
            sp0 = self._parking_spot or {}
            px, py = self._pin_xy()
            nearest = (math.hypot(sp0["x"] - px, sp0["y"] - py) + self.planner.SLOT_LEN_M
                       if sp0.get("kind") == "bay" else None)
            auto = self.api_find_parking(not_further_than_m=nearest)
            if not auto.get("success"):
                print(f"[Parking] no slot targeted ({auto.get('reason')}) — using the kerbside spot")
        except Exception as e:
            print(f"[Parking] auto slot search failed ({e}) — using the kerbside spot")

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
        self._step_scenario_hazards()
        if self._traffic_walkers and self._tick_count % 300 == 0:
            try:
                world = self.vehicle_adapter.world
                for _, c in self._traffic_walkers:
                    dest = world.get_random_location_from_navigation()
                    if dest:
                        c.go_to_location(dest)
            except Exception:
                pass
        extra_delay = self.fault_injector.extra_tick_delay()
        if extra_delay > 0:
            time.sleep(extra_delay)

        # Where the tick's time actually goes. Two thirds of it was unaccounted for -- the
        # only timings on show were the ground filter's and the free-space map's, both
        # inside perception, so a slow tick could not be pinned on anything.
        self._phases.start()
        _phases = {}

        def _phase(name):
            _phases[name] = self._phases.mark(name)

        # 1. Localize
        pose = self.localization.update()
        _phase("where am i")

        # 2. Perceive
        perception = self.perception.update()
        _phase("perception")
        # Raw surroundings for the learned parker's feelers (van frame), taken
        # BEFORE the corridor filter throws away everything beside us.
        try:
            self._obstacle_points_xy = self._collect_obstacle_points(perception, pose)
        except Exception:
            self._obstacle_points_xy = []
        # A route the van is not driving is not a plan, and must not be judged against.
        #
        # api_stop_mission never cleared self._route, so after a mission ended the corridor
        # kept measuring against the dead one. Seen on the operator console 2026-09-10:
        # parked with no mission, the van reported blocked_swept_path on a post 7.8 m out to
        # its side, calling it 2.08 m from "the line" -- the line being a ten-point stub left
        # over from before. During a real mission the same check is well behaved: 34 of 35
        # blocking objects were on the road, 0.5 to 2.4 m off the centre.
        m_now = self.mission_manager.current_mission
        driving_a_plan = m_now is not None and m_now.state == MissionState.EXECUTING

        # Route-aware in-path check: judge objects against the corridor we will
        # actually drive, not the direction the nose points (mid-turn the nose
        # sweeps neighbouring lanes -> false "vehicle ahead" stops).
        # Camera mode: objects come from the sensors, but SIGNALS come from
        # the map/V2I feed (like production AVs) — without this the camera
        # stack would sail through red lights.
        # Which signal is next on OUR route, how far off, and what colour. The old way asked
        # CARLA "is a light affecting me right now", which only answers inside a small box at
        # the stop line -- measured, RED first appeared at 2 m, and the van needs 14 m to
        # stop. This knows about the signal from 50 m out.
        # Hand the light reader the newest picture and where we were when it was taken,
        # BEFORE the lookahead asks it for a colour.
        if self._light_reader is not None:
            try:
                frame = getattr(self.sensor_adapter, "latest_camera", None)
                if frame is not None and frame.image is not None:
                    self._light_reader.update(frame.image[:, :, :3], pose.x, pose.y,
                                              math.degrees(pose.yaw), pose.z,
                                              now=frame.timestamp)
            except Exception:
                pass
        try:
            signal = self._signal_lookahead.update(self._route, pose.x, pose.y) \
                if self._signal_lookahead is not None else None
        except Exception:
            signal = None
        if (self.perception_mode == "camera_lidar" and perception.healthy
                and (signal is None or signal.light_id is None)
                and self._signal_lookahead is not None):
            # No signal on the route. The MAP says whether a light governs the lane we are
            # actually in; the CAMERA says its colour. (This used to ask the simulator for the
            # colour -- CARLA is now used for where lights and lines are, never what they show.)
            try:
                wp = self.vehicle_adapter.get_map().get_waypoint(
                    carla.Location(x=pose.x, y=pose.y, z=pose.z))
                lane = (int(wp.road_id), int(wp.lane_id)) if wp is not None else None
                signal = self._signal_lookahead.on_lane(lane, pose.x, pose.y, pose.yaw)
            except Exception:
                pass
        self._signal_ahead = signal
        self._stop_line_m = None
        if signal is not None and signal.light_id is not None:
            perception.traffic_light = signal.state
            perception.traffic_light_distance_m = signal.distance_m
            if signal.distance_m is not None:
                self._stop_line_m = signal.distance_m - self.behavior.front_offset_m
        _phase("traffic light")

        if self._route and driving_a_plan and perception.healthy and pose.healthy:
            # The map's DECORATIVE parked cars are not actors, so ground-truth
            # perception cannot see them — the van slammed one at full
            # parking-approach speed (sweep: impulse 9281). Feed them in as
            # stationary pseudo-vehicles. Camera mode skips this: the LiDAR
            # physically sees those cars already.
            if self.perception_mode == "ground_truth":
                try:
                    perception.objects = list(perception.objects) + \
                        self._static_vehicle_objects(pose)
                except Exception:
                    pass
            perception = self.planner.filter_to_route_corridor(
                perception, self._route, pose.x, pose.y, pose.yaw,
                danger_m=getattr(self.perception, "danger_distance", 8.0),
                footprint=self.footprint_blocking.active_footprint(),
            )
            # ...and the laser's own ground as a second opinion on it, both ways round
            try:
                self._second_opinion_on_the_ground(perception, pose,
                                                   self.behavior.current_behavior)
            except Exception as e:
                print(f"[FreeSpace] second opinion failed: {e}")
        _phase("route corridor")

        self._last_perception = perception      # the learned parker's stop-override rule reads this
        # One sheet saying what the van knows, built once and read by everyone else
        # (Perception V2 day 7). It is a view of the numbers above, never a second opinion.
        try:
            grid = getattr(self.perception, "grid", None)
            free_space = grid.summary().as_dict() if (grid is not None and grid.updated) else None
            edges = getattr(self.perception, "road_edges", None)
            if free_space is not None and edges is not None:
                free_space["kerbs"] = edges.as_dict()
            # The nearest place beside our lane the van cannot see into (day 12). Perception
            # only reports the distance; what speed that is worth is the behaviour's call.
            self._blind_spot_m = grid.blind_spot_ahead() if (grid is not None and grid.updated) else None
            if free_space is not None:
                free_space["blind_spot_m"] = self._blind_spot_m
            self._world = build_world_model(perception, pose, source=self.perception_mode,
                                            free_space=free_space)
        except Exception as e:
            self._world = None
            self._blind_spot_m = None
            self._world_error = repr(e)
        _phase("occupancy and world model")
        if self.footprint_debug.enabled and self._route:
            # Visualisation only: draws what the swept-path rule sees and what
            # the planner decided. Any failure is counted, never raised.
            try:
                frame = build_frame((pose.x, pose.y, pose.yaw), self.footprint_blocking.footprint,
                                    self._route.waypoints, perception.objects,
                                    blocking_enabled=self.footprint_blocking.enabled,
                                    planner_blocked=perception.path_blocked,
                                    planner_distance=perception.closest_obstacle_distance)
                self._footprint_drawer.draw(frame, z=pose.z + 0.15)
            except Exception as e:
                self._footprint_drawer.note_failure(e)
        # 3. Safety check
        # Which senses are working, and what losing one means (Perception V2 day 8)
        try:
            self._health = self.health_monitor.update(read_sensors(
                getattr(self, "sensor_adapter", None),
                perception_healthy=perception.healthy and not perception.degraded,
                perception_reason=perception.reason,
                pose=pose, controller_healthy=self.controller._enabled,
                vehicle_alive=self.vehicle_adapter.is_alive()))
        except Exception:
            self._health = None
        _phase("health monitor")
        safety_output = self.safety.update(
            perception_healthy=perception.healthy,
            perception_timestamp=perception.timestamp,
            localization_healthy=pose.healthy,
            localization_confidence=pose.confidence,
            localization_timestamp=pose.timestamp,
            controller_healthy=self.controller._enabled,
            vehicle_alive=self.vehicle_adapter.is_alive(),
            current_speed=pose.speed,
            health=self._health,
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

        _phase("safety supervisor")

        # 4. Behavior decision
        dest_dist = None
        if self._route and self.mission_manager.current_mission:
            m = self.mission_manager.current_mission
            dest_dist = self.planner.distance_to_destination(self._route, pose.x, pose.y)

        junction = self.planner.upcoming_turn(self._route, pose.x, pose.y) if self._route else None
        junction_ahead = self.planner.distance_to_next_junction(self._route, pose.x, pose.y) if self._route else None
        park_heading_ok = True
        park_position_ok = True
        if getattr(self, "_parking_spot", None):
            herr = abs((pose.yaw - self._parking_spot["yaw"] + math.pi) % (2 * math.pi) - math.pi)
            park_heading_ok = herr < math.radians(6)   # parallel to the lane line, visibly straight
            sp = self._parking_spot
            _slots_now = getattr(self, "_parking_slots", None)
            if (sp.get("kind") == "slot" and _slots_now
                    and sp.get("slot_index", 1 << 30) < len(_slots_now)):
                # slot parking is only done when the WHOLE van is inside the box
                slot = _slots_now[sp["slot_index"]]
                try:
                    ext = self.vehicle_adapter.vehicle.bounding_box.extent
                    half_len, half_wid = float(ext.x), float(ext.y)
                except Exception:
                    half_len, half_wid = 2.9, 1.0
                park_position_ok = self.planner.parked_in_slot(
                    pose.x, pose.y, pose.yaw, half_len, half_wid, slot)
        _phase("route context")

        # Prediction: yield to crossers/cut-ins BEFORE they are in the path.
        predicted = None
        if self._route and pose.healthy and pose.speed > 0.5:
            try:
                predicted = predict_route_conflict(
                    perception.objects, self._route.waypoints,
                    pose.x, pose.y, pose.yaw, pose.speed)
            except Exception:
                predicted = None
        self._predicted_conflict = predicted

        _phase("prediction")

        try:
            sign_m, sign_kind, sign_at = self._signs_ahead(pose)
        except Exception:
            sign_m, sign_kind, sign_at = (None, None, None)
        behavior_output = self.behavior.update(
            perception=perception,
            world=self._world,                 # day 7: the one sheet of what the van knows
            speed_cap_mps=safety_output.speed_cap_mps,   # day 8: slow while a sense is missing
            blind_spot_m=getattr(self, "_blind_spot_m", None),   # day 12: how near the unseen is
            pose=pose,
            destination_distance=dest_dist,
            safety_ok=safety_output.driving_allowed,
            junction=junction,
            park_heading_ok=park_heading_ok,
            park_position_ok=park_position_ok,
            predicted_conflict=predicted,
            stop_line_m=getattr(self, "_stop_line_m", None),
            light_id=(signal.light_id if signal is not None else None),
            speed_limit_mps=self._speed_limit_mps(pose),
            seen_ahead_m=self._seen_ahead_m(),
            sign_m=sign_m, sign_kind=sign_kind, sign_at=sign_at,
            junction_span=(self.planner.junction_span(self._route, pose.x, pose.y)
                           if self._route else None),
        )

        # Every change of what the van is doing goes in the mission log, so a drive can be
        # read afterwards as a list of changes and reasons (P3).
        try:
            seen = getattr(self, "_changes_logged", 0)
            if self.behavior.transitions.total > seen:
                for change in self.behavior.transitions.recent(
                        self.behavior.transitions.total - seen):
                    self.logger.log_event("behaviour", str(change))
                self._changes_logged = self.behavior.transitions.total
        except Exception:
            pass

        # Waiting is pointless when what blocks the way into the spot cannot move (fix 3).
        try:
            self._maybe_give_up_on_the_spot(perception, behavior_output, dest_dist, pose)
        except Exception as e:
            print(f"[Parking] give-up check failed: {e}")

        # Re-check slot occupancy once the destination is within 30 m, not
        # only when the parking behaviour begins: by then the van was already
        # too close to a stolen slot to stop short of it (arm A, 4 Sep).
        rescan_now = (not getattr(self, "_parking_rechecked", False)
                      and (behavior_output.behavior == DrivingBehavior.PARKING
                           or (dest_dist is not None and dest_dist < 30.0)))
        # A car that grabbed OUR slot after selection can also BLOCK the
        # approach before the parking phase ever begins (sweep finding:
        # van held 7 m behind it until timeout). While blocked close to
        # the destination, re-scan periodically so the van retargets.
        if (not rescan_now
                and behavior_output.behavior in (DrivingBehavior.STOPPED_VEHICLE,
                                                 DrivingBehavior.STOPPED_OBSTACLE)
                and dest_dist is not None and dest_dist < 45.0
                and time.time() - getattr(self, "_last_blocked_rescan", 0.0) > 5.0):
            self._last_blocked_rescan = time.time()
            rescan_now = True
        # With the lidar as the slot source, look for the real bay once the map's
        # slot is within reach (the finder sees ~30 m ahead) and swap onto it.
        # Keeps trying every 2 s while closing in (the kerb may only fit
        # once the bay is well inside the sweep), and while blocked near the
        # destination, where real occupancy from the lidar is the way out.
        if (self.parking_source == "lidar" and not getattr(self, "_lidar_rescan_done", False)
                and dest_dist is not None and 6.0 < dest_dist < 22.0
                and time.time() - getattr(self, "_last_lidar_rescan", 0.0) > 2.0):
            self._last_lidar_rescan = time.time()
            try:
                self._lidar_rescan_on_approach(pose)
            except Exception as e:
                print(f"[Parking] lidar approach re-scan failed: {e}")
        if rescan_now:
            self._parking_rechecked = True
            if self.perception_mode != "camera_lidar":       # camera mode: _confirm_parking_spot
                try:
                    self._recheck_parking_on_approach(pose)
                except Exception as e:
                    print(f"[Parking] approach re-scan failed: {e}")
        if self.perception_mode == "camera_lidar":
            try:
                self._confirm_parking_spot(pose, behavior_output, dest_dist)
            except Exception as e:
                print(f"[Parking] could not check the spot: {e}")

        # Changing lane: look into the lane the route moves into, and wait for a gap.
        try:
            self._wait_for_a_gap(pose, perception, behavior_output)
        except Exception as e:
            print(f"[Lane] gap check failed: {e}")

        # A street that stays blocked: is there another way to the same destination?
        try:
            self._maybe_reroute(pose, perception, behavior_output)
        except Exception as e:
            print(f"[Route] re-route check failed: {e}")

        # Go-around: pass a vehicle that is genuinely dead in our lane.
        try:
            self._maybe_overtake(pose, perception, behavior_output, junction_ahead)
        except Exception as e:
            print(f"[Overtake] check failed: {e}")
        if self._overtake_point is not None:
            d_rejoin = math.hypot(self._overtake_point.x - pose.x,
                                  self._overtake_point.y - pose.y)
            if d_rejoin < 4.0:
                self._overtake_point = None
                try:
                    self.logger.log_event("overtake", "pass complete — back in lane")
                except Exception:
                    pass
                self._note_move(GO_AROUND_DONE, "pass complete — back in lane")
                print("[Overtake] pass complete — back in lane")
            else:
                # Belt and braces: any body within 1.6 m while passing —
                # tracking lag, mis-judged widths, anything — pauses the
                # maneuver. A stall is acceptable; a scrape is not.
                too_tight = any(o.distance < getattr(self, "_overtake_tight_m", PASS_ABORT_M)
                                for o in perception.objects)
                if too_tight and pose.speed > 0.3:
                    behavior_output.desired_speed_mps = 0.0
                    behavior_output.should_stop = True
                    behavior_output.reason += " | overtake paused — clearance tight"
                else:
                    cap = getattr(self, "_overtake_cap_mps", PASS_SPEED_MPS)
                    if behavior_output.desired_speed_mps > cap:
                        behavior_output.desired_speed_mps = cap
                    behavior_output.reason += " | getting past something standing in the lane"

        # Comfort: ease off rather than step down, and only where the slowing is for comfort
        # (behavior.EASE_OFF_REASONS). Live on 2026-09-11 one object crossing the 20 m line
        # stepped the van 4.0 -> 2.0 -> 4.0 m/s. A light, a junction, a yield, the run-in to a
        # parking spot and every stop are still obeyed the moment they are decided.
        if (not behavior_output.should_stop
                and behavior_output.why in EASE_OFF_REASONS
                and behavior_output.desired_speed_mps < getattr(self, "_eased_speed", 0.0)):
            behavior_output.desired_speed_mps = max(behavior_output.desired_speed_mps,
                                                    self._eased_speed - EASE_OFF_MPS)
            behavior_output.reason += " | easing off"
        self._eased_speed = (0.0 if behavior_output.should_stop
                             else behavior_output.desired_speed_mps)

        # Curve-aware speed cap (Troy #2/#3): slow down BEFORE sharp bends.
        _phase("behaviour")

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
            # Terminal parking precision: with the 5 m aim floor the van aims
            # past the spot for the whole straight-in and carries ~0.2 m of
            # lateral error into the box (sweep finding: parked 0.14-0.22 m
            # over the side line). Aim short for the last metres of pull-in.
            if (behavior_output.behavior == DrivingBehavior.PARKING
                    and dest_dist is not None and dest_dist < 12.0):
                lookahead = max(2.5, min(lookahead, 0.8 * dest_dist))
            # Passing a dead car: the swerve must be TRACKED, not smoothed
            # away — a 5 m aim floor made v1 clip the car's corner.
            if self._overtake_point is not None:
                lookahead = min(lookahead, 3.2)
            next_wp = self.planner.get_next_waypoint(self._route, pose.x, pose.y, lookahead=lookahead)
            if next_wp:
                target_x, target_y = next_wp.x, next_wp.y

        _phase("control target")

        # 6. Compute vehicle command
        cmd = self.controller.compute_command(
            current_x=pose.x, current_y=pose.y,
            current_yaw=pose.yaw, current_speed=pose.speed,
            target_x=target_x, target_y=target_y,
            desired_speed=behavior_output.desired_speed_mps,
            should_stop=behavior_output.should_stop,
            cross_track_m=cross_track,
        )

        # 6b. The learned parker takes the wheel for the last 16 m of a slot
        # parking when selected; the behaviour layer's stops still win.
        cmd, behavior_output = self._maybe_learned_parker(cmd, behavior_output, pose)

        _phase("controller")

        # 7. Send command to vehicle
        self.vehicle_adapter.send_command(cmd)

        # 8. Check mission completion
        if behavior_output.behavior == DrivingBehavior.MISSION_COMPLETE:
            self.vehicle_adapter.disengage_autonomy()
            self.mission_manager.complete_mission()
            detail = "Arrived at destination"
            if getattr(self, "_parking_spot", None):
                sp = self._parking_spot
                d = math.hypot(pose.x - sp["x"], pose.y - sp["y"])
                herr = abs((pose.yaw - sp["yaw"] + math.pi) % (2 * math.pi) - math.pi)
                detail = f"Parked {d:.2f} m from the kerbside spot, heading off {math.degrees(herr):.0f} deg"
                if sp.get("kind") == "hold":
                    detail = (f"Held short of the taken bay, {d:.2f} m from the hold point "
                              f"(slot #{sp.get('slot_index')} was occupied, none free ahead)")
                if sp.get("kind") == "lane":
                    detail = ("No free parking spot near the destination: stopped straight in the "
                              f"lane, {d:.2f} m from the stop point")
                _sl_list = getattr(self, "_parking_slots", None)
                if (sp.get("kind") == "slot" and _sl_list
                        and sp.get("slot_index", 1 << 30) < len(_sl_list)):
                    slot = _sl_list[sp["slot_index"]]
                    try:
                        ext = self.vehicle_adapter.vehicle.bounding_box.extent
                        half_len, half_wid = float(ext.x), float(ext.y)
                    except Exception:
                        half_len, half_wid = 2.9, 1.0
                    inside, m_along, m_side = self.planner.van_in_slot(
                        pose.x, pose.y, pose.yaw, half_len, half_wid, slot)
                    detail += (f" | INSIDE slot #{sp['slot_index']}: {'YES' if inside else 'NO'}"
                               f" (margins {m_along} m front/back, {m_side} m side)")
                    if slot.get("kerb_offset_m") is not None:
                        # US rule: parallel-parked within 18 in (0.46 m) of the kerb.
                        # The kerb is on the slot's right; right unit vector = (-sin yaw, cos yaw).
                        right = (-(pose.x - slot["x"]) * math.sin(slot["yaw"])
                                 + (pose.y - slot["y"]) * math.cos(slot["yaw"]))
                        flank = slot["kerb_offset_m"] - right - half_wid
                        detail += f" | kerb {flank:.2f} m ({'OK' if flank <= 0.46 else 'too far'}, US rule <= 0.46 m)"
            self.logger.log_event("mission_completed", detail)
            print(f"[Mission] {detail}")
            self.logger.stop_mission_log()

        # 9. Log
        mission_state = "idle"
        if self.mission_manager.current_mission:
            mission_state = self.mission_manager.current_mission.state.value

        _phase("drive and the rest")
        self._phases.end()
        # The smoothed picture is kept as it was, because the console draws it and one odd
        # tick should not make the bars jump. It is no longer the ONLY picture: the window
        # beside it holds the worst case, which is what a smoothed number cannot show.
        for _k, _v in _phases.items():
            self._phase_ms[_k] = 0.85 * self._phase_ms.get(_k, _v) + 0.15 * _v

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
            # ...and the same thing as one code from a fixed list, with the changes so far
            # (Planning V2 P3: behavior/transitions.py)
            "behavior_why": behavior_output.why,
            "behavior_changes": self.behavior.transitions.as_dict(),
            "command": {"steer": round(cmd.steering, 3), "throttle": round(cmd.throttle, 3),
                        "brake": round(cmd.brake, 3)},
            "safety": {"state": safety_output.state.value, "reason": safety_output.reason,
                       "driving_allowed": safety_output.driving_allowed,
                       "speed_cap_mps": safety_output.speed_cap_mps,
                       "failed_sensors": list(safety_output.failed_sensors)},
            # every sense and what losing it means (day 8)
            "sensors": (self._health.as_dict() if getattr(self, "_health", None) else None),
            "perception": {
                "object_count": len(perception.objects),
                "closest_distance": round(perception.closest_obstacle_distance, 1),
                "closest_type": perception.closest_obstacle_type.value,
                "path_blocked": perception.path_blocked,
                # when the free-space map overruled a body the corridor check drew (P2)
                "ground_seen_free": getattr(self, "_ground_seen_free", None),
                # ...when it STOPPED the van instead (P2), and what it says this tick
                "ground_blocked": getattr(self, "_ground_block", None),
                "ground_says": getattr(self, "_ground_says", None),
                "closest_lateral_m": getattr(perception, "closest_obstacle_lateral_m", None),

                # Objects shown on the operator map.
                # These come from our current CARLA ground-truth perception.
                # from the world model (day 7) when it is available, so the page and any
                # other reader see exactly what the rest of the stack sees
                "objects": ([o.as_dict() for o in self._world.objects]
                            if (pose.healthy and getattr(self, "_world", None) is not None)
                            else [] if not pose.healthy else [
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

                        # the footprint the LiDAR measured (Perception V2 day 4);
                        # 0.0 means not measured, as in camera-only and ground-truth modes
                        "length_m": round(getattr(obj, "length_m", 0.0), 2),
                        "width_m": round(getattr(obj, "width_m", 0.0), 2),
                        "height_m": round(getattr(obj, "height_m", 0.0), 2),
                        "yaw_deg": round(getattr(obj, "yaw_deg", 0.0), 1),
                        # is it parked, or is it going somewhere? (Perception V2 day 6)
                        "stationary": bool(getattr(obj, "stationary", True)),
                        "speed": round(getattr(obj, "speed", 0.0), 2),
                        # can it move at all? (Planning V2, perception/motion_class.py)
                        "motion_class": getattr(obj, "motion_class", "dynamic"),
                    }
                    for obj in perception.objects
                ]),
            },
            "perception_mode": self.perception_mode,
            "parking_source": self.parking_source,
            "parker": self.parker,
            "planning": {**self.footprint_blocking.state(), **self.footprint_debug.state(),
                         "debug_draw_failures": self._footprint_drawer.failures},
            "rl_parker": ({"engaged": self.rl_parker.engaged, "done": self.rl_parker.done,
                           "result": self.rl_parker.result,
                           "brain": os.path.basename(self.rl_parker.model_path),
                           "inputs": self.rl_parker.n_obs, "controls": self.rl_parker.n_act,
                           "handover_m": self.rl_parker.handover_m}),

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
                ),
                # Perception V2 day 1: where the detector runs and how old its result is
                "yolox_mode": (("inline" if getattr(self.camera_lidar_perception, "yolox_inline", True) else "thread")
                               if (self.camera_lidar_perception is not None and self.perception_mode == "camera_lidar") else "n/a"),
                "detection_age_s": (round(float(getattr(self.camera_lidar_perception, "last_detection_age_s", 0.0) or 0.0), 2)
                                    if self.perception_mode == "camera_lidar" else "n/a"),
                "detector_errors": int(getattr(getattr(self.camera_lidar_perception, "_worker", None), "errors", 0) or 0),
                # perception fix 2: road removal by local patches
                "ground_filter": getattr(self.camera_lidar_perception, "ground_filter_mode", "n/a"),
                "ground_filter_ms": round(float(getattr(self.camera_lidar_perception, "last_ground_ms", 0.0) or 0.0), 1),
                # the free-space map's own cost, so a slow tick can be pinned on the right part
                "grid_ms": round(float(getattr(self.camera_lidar_perception, "last_grid_ms", 0.0) or 0.0), 1),
                "ground_tiles": int(getattr(self.camera_lidar_perception, "last_ground_tiles", 0) or 0),
                "borrowed_tiles": int(getattr(self.camera_lidar_perception, "last_borrowed_tiles", 0) or 0),
                "points_kept": int(getattr(self.camera_lidar_perception, "last_points_kept", 0) or 0),
                "road_edges_dropped": int(getattr(self.camera_lidar_perception, "last_road_edges_dropped", 0) or 0),
                # perception fix 3: thinning step (1 = every point) and how many blobs came out
                "lidar_thin_step": (int(getattr(self.camera_lidar_perception, "thin_step", 0) or 0)
                                    if self.camera_lidar_perception is not None else "n/a"),
                "clusters": int(len(getattr(self.camera_lidar_perception, "last_clusters", []) or [])),
                "clusters_before_cap": int(getattr(self.camera_lidar_perception, "last_clusters_before_cap", 0) or 0),
                # Planning V2: static or dynamic -- how many things earned "static", by which
                # rule, how many blobs this sweep had a static SHAPE, and the map lookups spent
                "static_dynamic": ({
                    "on": bool(getattr(self.camera_lidar_perception, "static_dynamic", False)),
                    "static": int(getattr(self.camera_lidar_perception, "last_static_count", 0) or 0),
                    "by_rule": dict(getattr(self.camera_lidar_perception, "last_static_by_rule", {}) or {}),
                    "shape_candidates": int(getattr(self.camera_lidar_perception, "last_static_candidates", 0) or 0),
                    "map_lookups": int(getattr(getattr(self.camera_lidar_perception, "_road_gap", None),
                                               "lookups", 0) or 0),
                } if self.camera_lidar_perception is not None else None),
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
                    "label": "LiDAR",
                    **self._lidar_sweep_telemetry(),
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
            "loop_hz": round(self._loop_hz, 1) if self._loop_hz else None,
            "tick_ms": round(self._tick_ms, 1),
            # where the tick's time goes, so a slow one can be pinned on something
            "tick_phases_ms": {k: round(v, 1) for k, v in sorted(self._phase_ms.items())},
            # Planning V2 phase 0. avg / p95 / worst over a rolling window, per phase, so a
            # single slow tick can be found instead of being smoothed into the average above.
            "tick_timing": self._phases.as_dict(),
            "slowest_phase": self._phases.worst_phase(),
            # ...and WHY the path was judged the way it was. "blocked = true" cannot tell a
            # parked lorry from the same kerb sliver reported forty times.
            "planner": (self.planner.last_decision.as_dict()
                        if getattr(self, "planner", None) is not None
                        and getattr(self.planner, "last_decision", None) is not None else None),
            "autonomy_state": self.vehicle_adapter._autonomy_state.value,
            "active_faults": dict(self.fault_injector.active),
            "last_tick_error": self._last_tick_error,
            "cruise_speed_mps": self.behavior.cruise_speed,
            "speed_limit_mps": getattr(self, "_limit_mps", None),
            "seen_ahead_m": (round(self._seen_ahead_m(), 1) if self._seen_ahead_m() is not None
                             else None),
            "junction": junction,   # {"distance_m", "direction"} when a turn at a junction is within 20 m, else null
            "junction_ahead_m": junction_ahead,
            "parking_spot": getattr(self, "_parking_spot", None),
            "parking_slots": getattr(self, "_parking_slots", None),
            "traffic": {"vehicles": len(self._traffic_vehicles), "walkers": len(self._traffic_walkers),
                        "parked_cars": len(getattr(self, "_parked_cars", []))},
            "signal_ahead": (self._signal_ahead.as_dict() if self._signal_ahead is not None else None),
            "road_sign": ({"kind": sign_kind, "to_line_m": round(sign_m, 1), "at": sign_at}
                          if sign_kind is not None and sign_m is not None else None),
            "signal_lookahead": (self._signal_lookahead.as_dict()
                                 if self._signal_lookahead is not None else None),
            "light_colour_from": ("camera" if self._light_reader is not None else "simulator"),
            "light_camera_raw": (self._light_reader.last_raw
                                 if self._light_reader is not None else None),
            "light_camera_confidence": (round(self._light_reader.last_confidence, 2)
                                        if self._light_reader is not None else None),
            "light_camera_ms": (round(self._light_reader.read_ms, 3)
                                if self._light_reader is not None else None),
            # stop_line_m: FRONT BUMPER to the stop line (negative once over it). choice: what the
            # van decided when the light stopped being green -- "stop", or "go" and why.
            "traffic_light": {"state": perception.traffic_light,
                              "light_id": (self._signal_ahead.light_id
                                           if self._signal_ahead is not None else None),
                              "stop_line_m": (None if getattr(self, "_stop_line_m", None) is None
                                              else round(self._stop_line_m, 2)),
                              "choice": self.behavior.light_status},
            "collision": {"count": self._collision_count, "last": self._last_collision},
            "overtaking": self._overtake_point is not None,
            "predicted_conflict": getattr(self, "_predicted_conflict", None),
            "weather": getattr(self, "_weather_preset", "default"),
            "version": getattr(self, "_git_rev", "unknown"),
            "uptime_s": round(time.time() - self._start_time, 1),

            "localization": {"confidence": round(pose.confidence, 2), "quality": pose.quality.value, "healthy": pose.healthy},
            "destination": ({"x": self.mission_manager.current_mission.destination_x, "y": self.mission_manager.current_mission.destination_y}
                            if self.mission_manager.current_mission else None),
        }

    def _lidar_sweep_telemetry(self) -> dict:
        """Perception fix 1: is the LiDAR handing over whole sweeps? One read
        of the latest scan so every field describes the same scan."""
        scan = getattr(self.sensor_adapter, "latest_lidar", None)
        pts = getattr(scan, "points", None)
        return {
            "full_sweep": bool(getattr(self.sensor_adapter, "lidar_full_sweep", False)),
            "frames_per_scan": int(getattr(scan, "frames", 0) or 0),
            "span_s": round(float(getattr(scan, "span_s", 0.0) or 0.0), 3),
            "points_per_scan": int(len(pts)) if pts is not None else 0,
            "sweep_errors": int(getattr(self.sensor_adapter, "lidar_sweep_errors", 0) or 0),
        }

    def run(self, tick_rate=10):
        """Main loop."""
        self._running = True
        dt = 1.0 / tick_rate
        print(f"\n[WarpAV] Running at {tick_rate} Hz. Console at http://localhost:5000")
        last_start = None

        while self._running:
            try:
                started = time.perf_counter()
                if last_start is not None:
                    period = started - last_start
                    if period > 0:
                        hz = 1.0 / period
                        self._loop_hz = hz if self._loop_hz is None else 0.9 * self._loop_hz + 0.1 * hz
                last_start = started
                self.tick()
                self._tick_ms = 0.9 * self._tick_ms + 0.1 * (time.perf_counter() - started) * 1000.0
                sleep_remainder(started, dt)
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
        if getattr(self, "camera_lidar_perception", None) is not None:
            try:
                self.camera_lidar_perception.close()
            except Exception:
                pass

        # Remove temporary scenario actors before destroying vehicle.
        self.clear_scenario()
        self.api_clear_traffic()

        self.vehicle_adapter.disengage_autonomy()
        self.sensor_adapter.destroy()
        self.vehicle_adapter.destroy()
        self.logger.stop_mission_log()
        print("[WarpAV] Shutdown complete")

    # ========================================================
    # Dashboard road-scenario tests
    # ========================================================

    def _step_scenario_hazards(self):
        """Advance the jaywalker / cut-in mini state machines each tick."""
        for walker, start in list(self._scenario_jaywalkers):
            try:
                loc = walker.get_location()
                if math.hypot(loc.x - start.x, loc.y - start.y) > 12.0:
                    walker.apply_control(carla.WalkerControl(speed=0.0))
                    self._scenario_jaywalkers.remove((walker, start))
            except Exception:
                self._scenario_jaywalkers.remove((walker, start))
        if self._cutin is not None:
            c = self._cutin
            try:
                car = c["actor"]
                age = time.time() - c["t0"]
                ego = self.vehicle_adapter.vehicle.get_location()
                gap = math.hypot(car.get_location().x - ego.x, car.get_location().y - ego.y)
                if c["phase"] == 0 and (gap < 16.0 or age > 3.0):
                    try:
                        car.disable_constant_velocity()
                    except Exception:
                        pass
                    car.apply_control(carla.VehicleControl(throttle=0.55, steer=0.35 * c["steer"]))
                    c["phase"], c["t0"] = 1, time.time()
                elif c["phase"] == 1 and age > 0.8:
                    car.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.35 * c["steer"]))
                    c["phase"], c["t0"] = 2, time.time()
                elif c["phase"] == 2 and age > 0.8:
                    car.apply_control(carla.VehicleControl(brake=1.0))
                    self._cutin = None      # done; car stays until CLEAR
            except Exception:
                self._cutin = None

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
        self._scenario_jaywalkers = []
        self._cutin = None

        # ALWAYS release the lights, not only when this process froze them:
        # CARLA keeps frozen lights across a stack restart, but the flag dies
        # with the old process — a whole town stuck red (sweep run 67).
        try:
            for tl in self.vehicle_adapter.world.get_actors().filter("traffic.traffic_light"):
                tl.freeze(False)
            if getattr(self, "_scenario_lights_frozen", False):
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
        jaywalker   (pedestrian crosses mid-block right in front of the van)
        cutin       (car in the next lane swerves into ours and brakes)
        """

        allowed = {
            "pedestrian",
            "vehicle",
            "barrier",
            "red_light",
            "jaywalker",
            "cutin",
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


        # Jaywalker: pedestrian steps off the right sidewalk ~18 m ahead and
        # crosses mid-block. Managed in tick(); stops on the far side.
        if scenario_type == "jaywalker":
            wp = self._get_scenario_waypoint(distance_m=18.0)
            if not wp:
                return {"success": False, "reason": "Could not find a road position ahead"}
            world = self.vehicle_adapter.world
            t = wp.transform
            rv = t.get_right_vector()
            side = wp.lane_width / 2.0 + 2.0
            spawn = carla.Transform(
                carla.Location(x=t.location.x + rv.x * side,
                               y=t.location.y + rv.y * side,
                               z=t.location.z + 1.0),
                carla.Rotation(yaw=t.rotation.yaw + 180.0))
            bp = world.get_blueprint_library().filter("walker.pedestrian.0001")[0]
            walker = world.try_spawn_actor(bp, spawn)
            if walker is None:
                return {"success": False, "reason": "Sidewalk spawn was blocked — try again in a second"}
            walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(-rv.x, -rv.y, 0.0), speed=2.2))
            self._scenario_actors.append(walker)
            self._scenario_jaywalkers.append((walker, walker.get_location()))
            self._scenario_type = scenario_type
            self.logger.log_event("scenario_spawned", "jaywalker crossing mid-block 18 m ahead")
            return {"success": True, "reason": "Jaywalker crossing 18 m ahead — the van must stop for them"}


        # Cut-in: car in the adjacent lane ahead swerves into ours and brakes.
        if scenario_type == "cutin":
            wp = self._get_scenario_waypoint(distance_m=14.0)
            if not wp:
                return {"success": False, "reason": "Could not find a road position ahead"}
            lane = wp.get_left_lane()
            if (lane is None or lane.lane_type != carla.LaneType.Driving
                    or abs((lane.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180) > 60):
                lane = wp.get_right_lane()
                steer_sign = -1.0     # merging leftwards into us
            else:
                steer_sign = 1.0      # merging rightwards into us
            if (lane is None or lane.lane_type != carla.LaneType.Driving):
                return {"success": False, "reason": "No adjacent same-direction lane here for a cut-in"}
            world = self.vehicle_adapter.world
            t = lane.transform
            bp = world.get_blueprint_library().filter("vehicle.audi.tt")[0]
            car = world.try_spawn_actor(
                bp, carla.Transform(carla.Location(x=t.location.x, y=t.location.y,
                                                   z=t.location.z + 0.4), t.rotation))
            if car is None:
                return {"success": False, "reason": "Adjacent lane occupied — try again in a second"}
            try:
                fwd = t.get_forward_vector()
                car.enable_constant_velocity(carla.Vector3D(6.5, 0.0, 0.0))
            except Exception:
                pass
            self._scenario_actors.append(car)
            self._cutin = {"actor": car, "phase": 0, "t0": time.time(), "steer": steer_sign}
            self._scenario_type = scenario_type
            self.logger.log_event("scenario_spawned", "cut-in car launched in the adjacent lane")
            return {"success": True, "reason": "Cut-in car launched — it will swerve into your lane and brake"}


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
        self._forget_the_manoeuvre()
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

    def api_spawn_traffic(self, cars=15, walkers=12, cyclists=4, near=120.0):
        """Dashboard traffic: autopilot cars + cyclists and road-crossing
        walkers concentrated around the van. Managed by this process, so it
        lives as long as the stack does."""
        import random as _r
        if self._traffic_vehicles or self._traffic_walkers:
            return {"success": False, "reason": "Traffic already active — clear it first"}
        try:
            world = self.vehicle_adapter.world
            tm = self.vehicle_adapter.client.get_trafficmanager()
            tm.set_global_distance_to_leading_vehicle(2.5)
            bp_lib = world.get_blueprint_library()
            pose = self.localization.get_last_pose()

            points = world.get_map().get_spawn_points()
            _r.shuffle(points)
            if pose.healthy:
                points.sort(key=lambda p: math.hypot(p.location.x - pose.x, p.location.y - pose.y) > near)

            car_bps = [bp for bp in bp_lib.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels").as_int()) == 4]
            bike_bps = (list(bp_lib.filter("vehicle.bh.crossbike"))
                        + list(bp_lib.filter("vehicle.diamondback.century"))
                        + list(bp_lib.filter("vehicle.gazelle.omafiets")))
            n_cars = n_bikes = 0
            for sp in points:
                if n_cars >= cars and n_bikes >= cyclists:
                    break
                if n_cars < cars:
                    v = world.try_spawn_actor(_r.choice(car_bps), sp)
                    if v is not None:
                        v.set_autopilot(True, tm.get_port())
                        self._traffic_vehicles.append(v)
                        n_cars += 1
                        continue
                if n_bikes < cyclists and bike_bps:
                    b = world.try_spawn_actor(_r.choice(bike_bps), sp)
                    if b is not None:
                        b.set_autopilot(True, tm.get_port())
                        tm.vehicle_percentage_speed_difference(b, 55)
                        self._traffic_vehicles.append(b)
                        n_bikes += 1

            world.set_pedestrians_cross_factor(0.35)
            walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
            ctrl_bp = bp_lib.find("controller.ai.walker")
            tries = 0
            while len(self._traffic_walkers) < walkers and tries < walkers * 6:
                tries += 1
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                if pose.healthy and math.hypot(loc.x - pose.x, loc.y - pose.y) > near:
                    continue
                w = world.try_spawn_actor(_r.choice(walker_bps), carla.Transform(loc))
                if w is None:
                    continue
                c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
                c.start()
                dest = world.get_random_location_from_navigation()
                if dest:
                    c.go_to_location(dest)
                c.set_max_speed(_r.uniform(1.0, 1.8))
                self._traffic_walkers.append((w, c))

            msg = (f"traffic ON: {n_cars} cars, {n_bikes} cyclists, "
                   f"{len(self._traffic_walkers)} walkers around the van")
            self.logger.log_event("traffic_spawned", msg)
            print(f"[Traffic] {msg}")
            return {"success": True, "message": msg,
                    "vehicles": len(self._traffic_vehicles), "walkers": len(self._traffic_walkers)}
        except Exception as e:
            self.api_clear_traffic()
            return {"success": False, "reason": f"traffic spawn failed: {e}"}

    def api_clear_traffic(self, all_actors=False):
        # Destroy through ONE batched RPC: destroying dozens of TM-driven
        # actors with individual destroy() calls can hard-crash the CARLA
        # client mid-storm (observed: stack died with no traceback).
        n = 0
        client = self.vehicle_adapter.client
        batch = []
        for w, c in self._traffic_walkers:
            try:
                c.stop()
            except Exception:
                pass
            batch.append(carla.command.DestroyActor(c))
            batch.append(carla.command.DestroyActor(w))
        for v in self._traffic_vehicles:
            batch.append(carla.command.DestroyActor(v))
        self._traffic_vehicles = []
        self._traffic_walkers = []
        if all_actors:
            # Sweep-mode reset: remove EVERY vehicle/walker that is not the
            # van, whoever spawned it (leftover tools, dead runners, ...).
            try:
                ego_id = self.vehicle_adapter.vehicle.id
                world = self.vehicle_adapter.world
                queued = {getattr(cmd, "actor_id", None) for cmd in batch}
                for c in world.get_actors().filter("controller.ai.walker"):
                    if c.id in queued:
                        continue
                    try:
                        c.stop()
                    except Exception:
                        pass
                    batch.append(carla.command.DestroyActor(c))
                for a in list(world.get_actors().filter("walker.pedestrian.*")) + \
                         list(world.get_actors().filter("vehicle.*")):
                    if a.id == ego_id or a.id in queued:
                        continue
                    batch.append(carla.command.DestroyActor(a))
            except Exception as e:
                print(f"[Traffic] full clear enumeration failed: {e}")
            self._parked_cars = []
            self._scenario_jaywalkers = []
            self._cutin = None
        try:
            if batch:
                results = client.apply_batch_sync(batch, False)
                n = sum(1 for r in results if not r.error)
        except Exception as e:
            print(f"[Traffic] batch destroy failed: {e}")
        try:
            self.logger.log_event("traffic_cleared", f"removed {n} traffic actors")
        except Exception:
            pass
        print(f"[Traffic] cleared {n} actors")
        return {"success": True, "removed": n}

    def _on_collision(self, event):
        """CARLA collision sensor callback (fires from the sensor thread)."""
        try:
            other = event.other_actor.type_id if event.other_actor else "unknown"
        except Exception:
            other = "unknown"
        try:
            imp = event.normal_impulse
            intensity = round(math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2), 1)
        except Exception:
            intensity = 0.0
        now = time.time()
        last = self._last_collision
        # A scrape produces a burst of events — count contact once per second.
        if not (last and last["with"] == other and now - last["time"] < 1.0):
            self._collision_count += 1
            try:
                self.logger.log_event("collision", f"COLLISION with {other} (impulse {intensity})")
            except Exception:
                pass
            print(f"[COLLISION] with {other} (impulse {intensity})")
        self._last_collision = {"with": other, "intensity": intensity,
                                "tick": getattr(self, "_tick_count", 0), "time": now}

    def api_set_weather(self, preset):
        """Set a CARLA weather preset by name (e.g. HardRainNight)."""
        if not preset or not isinstance(preset, str) or not hasattr(carla.WeatherParameters, preset):
            return {"success": False, "reason": f"Unknown weather preset '{preset}'"}
        try:
            self.vehicle_adapter.world.set_weather(getattr(carla.WeatherParameters, preset))
            self._weather_preset = preset
            try:
                ls = carla.VehicleLightState
                if "Night" in preset or "Rain" in preset or "Storm" in preset or "Sunset" in preset:
                    self.vehicle_adapter.vehicle.set_light_state(ls(ls.LowBeam | ls.Position))
                else:
                    self.vehicle_adapter.vehicle.set_light_state(ls.NONE)
            except Exception:
                pass
            try:
                self.logger.log_event("weather", f"weather set to {preset}")
            except Exception:
                pass
            print(f"[Weather] {preset}")
            return {"success": True, "preset": preset}
        except Exception as e:
            return {"success": False, "reason": f"weather set failed: {e}"}

    def api_park_cars(self, count=4, spacing=14.0, fill_all=False, clear=False,
                      take_chosen=False):
        """Park stationary cars in the bay along the final stretch of the
        active route, so occupied-slot handling can be tested remotely.
        take_chosen additionally drops a car into the CHOSEN slot to force
        the approach re-scan to retarget."""
        world = self.vehicle_adapter.world
        if clear:
            n = 0
            for v in list(world.get_actors().filter("vehicle.*")):
                if v.attributes.get("role_name") == "warp_parked":
                    try:
                        v.destroy(); n += 1
                    except Exception:
                        pass
            self._parked_cars = []
            print(f"[Parking test] removed {n} parked cars")
            return {"success": True, "removed": n}

        route = self._route
        if not route or len(route.waypoints) < 5:
            return {"success": False, "reason": "Start a mission first — cars are parked near its destination"}

        cmap = self.vehicle_adapter.get_map()

        def right_bay(x, y):
            wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.3),
                                   project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                return None
            for _ in range(3):
                r = wp.get_right_lane()
                if (r is not None and r.lane_type == carla.LaneType.Driving
                        and abs((r.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180) < 60):
                    wp = r
                else:
                    break
            bay = wp.get_right_lane()
            if (bay is not None and bay.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder)
                    and bay.lane_width >= 1.8):
                t = bay.transform
                return (t.location.x, t.location.y, t.rotation.yaw)
            return None

        wps = route.waypoints
        pts = [(w.x, w.y) for w in wps]
        arc_back, tail = 0.0, [pts[-1]]
        for p, q in zip(reversed(pts[:-1]), reversed(pts)):
            arc_back += math.hypot(q[0] - p[0], q[1] - p[1])
            tail.append(p)
            if arc_back > 70.0:
                break
        tail.reverse()

        bays = []
        for x, y in tail:
            b = right_bay(x, y)
            if b is not None:
                bays.append(b)
        if len(bays) < 3 and not take_chosen:
            return {"success": False, "reason": "No parking bay on the final stretch of this route"}

        bp_lib = world.get_blueprint_library()
        models = ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.nissan.patrol", "vehicle.mini.cooper_s"]

        def park_at(x, y, yaw, i):
            bp = bp_lib.filter(models[i % len(models)])[0]
            bp.set_attribute("role_name", "warp_parked")
            car = world.try_spawn_actor(
                bp, carla.Transform(carla.Location(x=x, y=y, z=0.3), carla.Rotation(yaw=yaw)))
            if car is not None:
                car.apply_control(carla.VehicleControl(hand_brake=True))
                self._parked_cars.append(car)
            return car is not None

        spawned = 0
        if take_chosen:
            sp = getattr(self, "_parking_spot", None)
            if sp and sp.get("kind") in ("slot", "bay"):
                if park_at(sp["x"], sp["y"], math.degrees(sp["yaw"]), 0):
                    spawned += 1

        use_spacing = 8.0 if fill_all else spacing
        want = 999 if fill_all else count
        usable = bays if fill_all else bays[:max(1, len(bays) - 5)]
        next_at, arc = 0.0, 0.0
        prev = usable[0] if usable else None
        for b in usable:
            arc += math.hypot(b[0] - prev[0], b[1] - prev[1])
            prev = b
            if arc < next_at or spawned >= want:
                continue
            if park_at(b[0], b[1], b[2], spawned):
                spawned += 1
                next_at = arc + use_spacing

        try:
            self.logger.log_event("parked_cars", f"parked {spawned} test cars in the destination bay")
        except Exception:
            pass
        print(f"[Parking test] parked {spawned} cars near the destination")
        return {"success": spawned > 0, "parked": spawned,
                "take_chosen": bool(take_chosen)}

    def _static_vehicle_points(self):
        """2D outline points (centre + box corners) of DECORATIVE parked cars
        baked into the map's static layer. They are not actors, so neither
        occupancy nor perception saw them — discovered when the collision
        sensor caught the van parking into one. Cached: the layer never
        changes."""
        pts = getattr(self, "_static_vehicle_pts", None)
        if pts is not None:
            return pts
        pts = []
        try:
            world = self.vehicle_adapter.world
            seen = set()
            for name in ("Vehicles", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"):
                lbl = getattr(carla.CityObjectLabel, name, None)
                if lbl is None:
                    continue
                for obj in world.get_environment_objects(lbl):
                    if obj.id in seen:
                        continue
                    seen.add(obj.id)
                    bb = obj.bounding_box
                    cx, cy = bb.location.x, bb.location.y
                    ext = bb.extent
                    vyaw = math.radians(bb.rotation.yaw)
                    c_, s_ = math.cos(vyaw), math.sin(vyaw)
                    p = [(cx, cy)]
                    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                        p.append((cx + sx * c_ * ext.x - sy * s_ * ext.y,
                                  cy + sx * s_ * ext.x + sy * c_ * ext.y))
                    pts.append(p)
        except Exception as e:
            print(f"[Parking] static vehicle scan failed: {e}")
        self._static_vehicle_pts = pts
        print(f"[Parking] static-layer parked vehicles known: {len(pts)}")
        return pts

    def _speed_limit_mps(self, pose):
        """The limit on this piece of road, from the MAP -- the same place the stop lines come
        from. A speed-limit sign in OpenDRIVE is landmark type 274; where a map carries none,
        the town's own default stands (Town10HD has no speed-limit signs at all: 30 km/h)."""
        try:
            wp = self.vehicle_adapter.get_map().get_waypoint(
                carla.Location(x=pose.x, y=pose.y, z=0.3), lane_type=carla.LaneType.Driving)
            key = (wp.road_id, wp.lane_id)
            if key != getattr(self, "_limit_key", None):
                self._limit_key = key
                found = [lm for lm in wp.get_landmarks_of_type(150.0, "274")]
                if found:
                    value = float(found[0].value)
                    self._limit_mps = value / 3.6 if found[0].unit in ("km/h", "") else value * 0.44704
                else:
                    kmh = float(self.vehicle_adapter.vehicle.get_speed_limit() or 0.0)
                    self._limit_mps = kmh / 3.6 if kmh > 0 else None
            return getattr(self, "_limit_mps", None)
        except Exception:
            return None

    def _seen_ahead_m(self):
        """How far ahead the laser has actually seen the road FREE, over a strip as wide as the
        van. Unseen ground is not free ground, so this caps the speed (behaviour)."""
        grid = getattr(self.perception, "grid", None)
        if grid is None or not getattr(grid, "updated", False):
            return None
        try:
            return float(grid.free_distance(0.0, self.footprint_blocking.footprint.swept_half_width))
        except Exception:
            return None

    def _lane_ok(self, x, y):
        """Is this position on a real driving lane? (overtake feasibility)"""
        try:
            wp = self.vehicle_adapter.get_map().get_waypoint(
                carla.Location(x=x, y=y, z=0.3), project_to_road=False,
                lane_type=carla.LaneType.Driving)
            return wp is not None
        except Exception:
            return False

    def _lane_width(self, pose, fallback_m=3.5):
        """How wide the lane under the van is -- what says whether there is room to squeeze
        past something inside it (planner.pass_options)."""
        try:
            wp = self.vehicle_adapter.get_map().get_waypoint(
                carla.Location(x=pose.x, y=pose.y, z=0.3), lane_type=carla.LaneType.Driving)
            width = float(getattr(wp, "lane_width", 0.0) or 0.0)
            return width if 1.5 < width < 6.0 else fallback_m
        except Exception:
            return fallback_m

    #: What the van will go round, once it has stood still for OVERTAKE_AFTER_S: a dead car,
    #: and anything else standing in the lane that is not a person. Never a person or someone
    #: riding -- they may step aside, and the van waits for them however long it takes.
    #: A thing is only passed while the camera is working: dead ahead in its view, a thing it
    #: has NOT called a person is a thing it looked at and did not call a person. With the
    #: camera stale or missing that is not a judgement, it is ignorance (perception.degraded).
    OVERTAKE_STATES = (DrivingBehavior.STOPPED_VEHICLE, DrivingBehavior.STOPPED_OBSTACLE,
                       DrivingBehavior.STOPPED_BLOCKED)
    OVERTAKE_AFTER_S = 10.0

    def _maybe_overtake(self, pose, perception, behavior_output, junction_ahead):
        """Anything that stays dead in our lane for 10 s on an open straight gets passed:
        swing one lane left, by, and back. Conservative by design — any doubt (lights,
        junctions, other traffic, bends, a body the way round would touch) means keep waiting.

        Until 2026-09-11 this was for a stopped CAR only, and everything else -- a barrel, a
        box, a cone -- stopped the van until a person came. Measured that day: a 0.45 m barrel
        in the lane, stopped 8.7 m short, still there when the test ended."""
        if self._overtake_point is not None:
            return
        if behavior_output.behavior not in self.OVERTAKE_STATES:
            self._blocked_since = None
            return
        now = time.time()
        if self._blocked_since is None:
            self._blocked_since = now
            return
        if now - self._blocked_since < self.OVERTAKE_AFTER_S or now < self._overtake_retry_at:
            return
        def waiting(why):
            if now - getattr(self, "_overtake_why_at", 0.0) > 20.0:
                self._overtake_why_at = now
                print(f"[Overtake] waiting: {why}")
            self._overtake_retry_at = now + 10.0
            self._note_move(GO_AROUND_WAIT, why)

        if perception.traffic_light in ("red", "yellow"):
            return                       # that's a queue, not a dead car
        if junction_ahead is not None and junction_ahead < 25.0:
            waiting(f"junction only {junction_ahead:.0f} m ahead")
            return
        lead_d = perception.closest_obstacle_distance
        if lead_d is None or lead_d > 14.0:
            return
        # WHAT it is decides whether a pass may even be considered (planner.pass_refused);
        # whether the way round is clear is the geometry below.
        what = getattr(perception.closest_obstacle_type, "value", "thing")
        why_not = pass_refused(what, getattr(perception, "closest_obstacle_speed", 0.0),
                               bool(getattr(perception, "degraded", False)))
        if why_not is not None:
            waiting(why_not)
            return
        if getattr(self, "_ground_block", None):
            # the laser's squares, with nothing tracked on them: there is no measured body to
            # slide the van's own past, so there is no way to say a way round is clear (P2)
            waiting("solid ground squares ahead that nothing is tracked on — not going round "
                    "what cannot be measured")
            return
        # Clearance: traffic moving where the pass would go (planner.overtake_blocker)...
        back_in_m = lead_d + self.planner.OVERTAKE_REJOIN_M + 8.0
        why = overtake_blocker(perception.objects, lead_d, back_in_m, pose.yaw)
        if why is not None:
            waiting(why)
            return
        # ...then the way round itself: the SMALLEST one that works. A nudge inside our own
        # lane first, either way round, and only then a whole lane (planner.pass_options).
        # Each is planned on a copy, because plan_overtake rewrites the route it is given.
        taken, refused = None, "geometry refused (bend/junction/no lane/route end)"
        for over_m, in_lane in pass_options(self._lane_width(pose),
                                            self.footprint_blocking.footprint.half_width,
                                            self.planner.OVERTAKE_SHIFT_M):
            trial = Route(waypoints=list(self._route.waypoints),
                          total_distance=self._route.total_distance)
            rejoin = self.planner.plan_overtake(trial, pose.x, pose.y, lead_d,
                                                lane_ok=self._lane_ok, shift_m=over_m)
            if rejoin is None:
                continue
            # ...and what stands on it: the van's body slid along that path, as before a pull-in
            in_way = self.planner.pull_in_blocker(perception, trial, pose.x, pose.y, pose.yaw,
                                                  self.footprint_blocking.footprint,
                                                  horizon_m=back_in_m)
            if in_way is None:
                taken = (over_m, in_lane, trial, rejoin)
                break
            obj, dist, hit, where, box = in_way
            stands = getattr(getattr(obj, "object_type", None), "value", "thing")
            refused = (f"{abs(over_m):.2f} m over is not enough: {stands} (id "
                       f"{getattr(obj, 'id', None)}) {dist:.0f} m away would be touched "
                       f"{hit.along_m:.0f} m on, {hit.lateral_m:+.1f} m off the path")
        if taken is None:
            waiting(refused)
            return
        over_m, in_lane, trial, rejoin = taken
        way = (f"squeezing past inside our own lane, {abs(over_m):.2f} m over to the "
               f"{'left' if over_m > 0 else 'right'}" if in_lane else "passing on the left")
        self._route.waypoints = trial.waypoints          # one swap: the tick may be reading it
        self._overtake_point = rejoin
        # While squeezing, the thing IS close: the abort line has to be the body's, not the
        # lane change's 1.6 m, or the van would freeze beside what it is passing.
        self._overtake_tight_m = SQUEEZE_ABORT_M if in_lane else PASS_ABORT_M
        self._overtake_cap_mps = SQUEEZE_SPEED_MPS if in_lane else PASS_SPEED_MPS
        self._blocked_since = None
        try:
            self.logger.log_event(
                "overtake",
                f"a {what} has been in the way for {self.OVERTAKE_AFTER_S:.0f} s — {way}, "
                f"rejoining {self.planner.OVERTAKE_REJOIN_M:.0f} m beyond it")
        except Exception:
            pass
        self._note_move(GO_AROUND_START,
                        f"{what} standing at {lead_d:.1f} m — {way}, rejoining "
                        f"{self.planner.OVERTAKE_REJOIN_M:.0f} m beyond it")
        print(f"[Overtake] {what} standing at {lead_d:.1f} m — {way}")

    def _static_vehicle_objects(self, pose):
        """Nearby static-layer parked cars as pseudo-detections (VEHICLE,
        speed 0) in the ego frame, for the route-corridor check."""
        out = []
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for i, pts in enumerate(self._static_vehicle_points()):
            cx, cy = pts[0]
            dx, dy = cx - pose.x, cy - pose.y
            d2 = dx * dx + dy * dy
            if d2 > 45.0 ** 2:
                continue
            ex = dx * c + dy * s          # forward
            ey = -dx * s + dy * c         # matches the corridor's inverse
            if ex < -5.0:
                continue
            out.append(DetectedObject(object_type=ObjectType.VEHICLE,
                                      x=ex, y=ey, distance=math.sqrt(d2),
                                      speed=0.0, id=900000 + i))
        return out

    def _mark_slot_occupancy(self, slots):
        """Which slots are taken. Camera mode: what the van's own LiDAR sees (_spot_view) --
        a slot it has not seen is not taken, but it is not confirmed free either (slot["seen"]
        says which). Ground-truth mode, a test mode where everything comes from the simulator:
        the simulator's list of cars, as before."""
        if self.perception_mode == "camera_lidar":
            pose = self.localization.get_last_pose()
            for sl in slots:
                sl["seen"] = self._spot_view(sl, pose)
                sl["occupied"] = sl["seen"] == "taken"
            return
        self._slots_taken_in_simulator(slots)

    def _slots_taken_in_simulator(self, slots):
        """GROUND-TRUTH MODE ONLY. A slot is taken if ANY PART of another vehicle overlaps it
        (centre + four bounding-box corners: straddlers claim every slot
        they touch). Covers live actors AND the map's baked-in parked cars."""
        try:
            ego_id = self.vehicle_adapter.vehicle.id
            others = []
            for a in self.vehicle_adapter.world.get_actors().filter("vehicle.*"):
                if a.id == ego_id:
                    continue
                loc = a.get_location()
                pts = [(loc.x, loc.y)]
                try:
                    ext = a.bounding_box.extent
                    vyaw = math.radians(a.get_transform().rotation.yaw)
                    c_, s_ = math.cos(vyaw), math.sin(vyaw)
                    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                        pts.append((loc.x + sx * c_ * ext.x - sy * s_ * ext.y,
                                    loc.y + sx * s_ * ext.x + sy * c_ * ext.y))
                except Exception:
                    pass
                others.append(pts)
        except Exception:
            others = []
        others = others + self._static_vehicle_points()
        for sl in slots:
            sl["occupied"] = any(self.planner.point_in_slot(px, py, sl, inflate=0.25)
                                 for pts in others for px, py in pts)

    def _maybe_give_up_on_the_spot(self, perception, behavior_output, dest_dist, pose):
        """Blocked on the way into the parking spot by something that will not move
        (planner.WaitingIsPointless). Before the pull-in has begun: choose another spot. Once
        it has begun there is no way forward and no reverse gear -- the mission FAILS, saying
        why, and the van stays where it is.

        This used to finish the mission right there and call it parked: measured against
        CARLA on 2026-09-11, that left the van 39 degrees across the driving lane, 4.1 m of
        it inside the lane, 11.6 m from the spot, with the mission reported complete."""
        sp = getattr(self, "_parking_spot", None)
        if not sp:
            return
        if getattr(self, "_give_up", None) is None:
            self._give_up = WaitingIsPointless()
        blocked = behavior_output.behavior in (DrivingBehavior.STOPPED_OBSTACLE,
                                               DrivingBehavior.STOPPED_BLOCKED)
        bid = getattr(self.planner.last_decision, "blocker_id", None)
        obj = next((o for o in perception.objects if int(getattr(o, "id", 0) or 0) == bid), None) \
            if bid is not None else None
        what = self._give_up.update(blocked, dest_dist, obj, time.time())
        if what is None:
            return
        self._give_up = WaitingIsPointless()
        turning_in = dest_dist is not None and dest_dist <= (sp.get("approach_m") or 0.0) + 1.0
        if not turning_in:
            self._rechoose_parking(pose, f"the way in is blocked by a {what} that will not move")
            return
        why = f"Could not get into the parking spot: a {what} blocks the way in (no reverse gear)"
        self.mission_manager.fail_mission(why)
        self.behavior.has_mission = False
        behavior_output.behavior = DrivingBehavior.STOPPED_BLOCKED
        behavior_output.reason = why
        behavior_output.should_stop = True
        behavior_output.desired_speed_mps = 0.0
        self.vehicle_adapter.disengage_autonomy()
        self._note_move(SPOT_GIVEN_UP, why)
        self.logger.log_event("parking_failed",
                              f"a {what} {getattr(obj, 'distance', 0.0):.1f} m ahead blocked the way into the "
                              f"spot for {WaitingIsPointless.AFTER_S:.0f} s, {dest_dist:.1f} m from it, "
                              f"already turning in")
        self.logger.stop_mission_log()
        print(f"[Parking] {why}")

    def _note_move(self, what, said):
        """Put a move the van decided to make into the drive's story, beside the changes of
        state (behavior/transitions.py). The tick loop copies new entries into the mission
        log, so this is also how a move reaches the log."""
        try:
            self.behavior.transitions.note_move(what, said,
                                                state=self.behavior.current_behavior.value)
        except Exception:
            pass

    def _second_opinion_on_the_ground(self, perception, pose, behaviour=None):
        """The laser's own free-space map, read over the ground the van's body is about to
        cover, as a second opinion on the corridor check -- both ways round (Planning V2, P2).

        It can say DRIVE ON: when every square of that ground has been SEEN empty, a body the
        corridor check drew there is not there (planner.nothing_is_standing_there). Never for
        a person or a rider, never for anything moving, and unseen is never free.

        And it can say STOP: when solid squares sit on that ground and nothing is tracked on
        them. Perception builds objects by clustering laser points and a thing can fall
        between the clusters; the squares themselves cannot. That is `blocked_occupancy`, a
        reason the planner has had a name for since P0 and has never been able to produce.

        Not while parking or mid-pass: both go close to things on purpose, and a kerb is 12 cm
        of solid squares.
        """
        self._ground_says, self._ground_block = None, None
        grid = getattr(self.perception, "grid", None)
        if grid is None or not pose.healthy:
            return
        front = self.behavior.front_offset_m
        if perception.path_blocked:
            self._drive_on_if_the_ground_is_seen_free(perception, pose, grid, front)
            return
        if behaviour == DrivingBehavior.PARKING or self._overtake_point is not None:
            return
        half = self.footprint_blocking.footprint.half_width + GROUND_KEEP_M
        counts = grid.strip_ahead(front, front + GROUND_LOOK_M, half)
        self._ground_says = what_the_ground_says(counts)
        if self._ground_says != BLOCKED_OCCUPANCY:
            return
        at = grid.nearest_block_ahead(front, front + GROUND_LOOK_M, half)
        if at is None:
            return
        free, blocked, unseen = counts
        perception.path_blocked = True
        perception.closest_obstacle_distance = min(perception.closest_obstacle_distance, at)
        perception.closest_obstacle_type = ObjectType.UNKNOWN     # solid squares, not a thing
        self._ground_block = {"at_m": round(at, 1), "squares": blocked, "looked_m": GROUND_LOOK_M}
        try:
            self.planner.last_decision.reason = BLOCKED_OCCUPANCY
            self.planner.last_decision.blocker_distance_m = at
            self.planner.last_decision.blocker_kind = "solid squares"
        except Exception:
            pass
        if time.time() - getattr(self, "_ground_block_at", 0.0) > 2.0:
            self._ground_block_at = time.time()
            self._note_move(GROUND_BLOCKED,
                            f"{blocked} solid squares on the ground the van would cover, the "
                            f"nearest {at:.1f} m ahead, and nothing tracked on them")
            self.logger.log_event(
                "ground_blocked",
                f"nothing is tracked in the way, but the laser has {blocked} solid squares on "
                f"the ground the van would cover, the nearest {at:.1f} m ahead -- stopping")
            print(f"[FreeSpace] {blocked} solid squares {at:.1f} m ahead, nothing tracked "
                  f"there -- stopping")

    def _drive_on_if_the_ground_is_seen_free(self, perception, pose, grid, front):
        """The half of the second opinion that lets the van drive on (see above)."""
        blocker_m = float(getattr(perception, "closest_obstacle_distance", 0.0) or 0.0)
        look = min(SEEN_FREE_LOOK_M, max(SEEN_FREE_MIN_LOOK_M, blocker_m - front + 1.0))
        counts = grid.strip_ahead(front, front + look,
                                  self.footprint_blocking.footprint.swept_half_width)
        self._ground_says = what_the_ground_says(counts)
        what = perception.closest_obstacle_type.value
        if not nothing_is_standing_there(what, getattr(perception, "closest_obstacle_speed", 0.0),
                                         counts):
            return
        free, _, unseen = counts
        perception.path_blocked = False
        self._ground_seen_free = {"what": what, "at_m": round(blocker_m, 1),
                                  "looked_m": round(look, 1), "free": free, "unseen": unseen}
        if time.time() - getattr(self, "_ground_seen_free_at", 0.0) > 2.0:
            self._ground_seen_free_at = time.time()
            self._note_move(GROUND_SEEN_FREE,
                            f"a {what} said to be in the way {blocker_m:.1f} m ahead, but all "
                            f"{free} squares of the next {look:.1f} m of ground are seen empty")
            self.logger.log_event(
                "ground_seen_free",
                f"the corridor check says a {what} is in the way {blocker_m:.1f} m ahead, but the "
                f"laser has seen all {free} squares of the next {look:.1f} m of ground empty "
                f"-- driving on")
            print(f"[FreeSpace] a {what} at {blocker_m:.1f} m, but the next {look:.1f} m of ground "
                  f"is seen empty -- driving on")

    # ---------------- parking: is the spot free? (the LiDAR, 2026-09-11) ----------------
    def _spot_view(self, spot, pose):
        """What the van's own LiDAR says about a parking spot: "taken", "free" or "unseen"
        (planning/parking_check.spot_view). Ground-truth perception mode is a test mode with no
        LiDAR map, where everything comes from the simulator; there the simulator's list of
        cars stands in, as before."""
        if self.perception_mode != "camera_lidar":
            probe = [dict(spot, length=spot.get("length") or SPOT_DEFAULT_LEN_M,
                          width=spot.get("width") or SPOT_DEFAULT_WID_M)]
            self._slots_taken_in_simulator(probe)
            return "taken" if probe[0].get("occupied") else "free"
        if pose is None:
            return "unseen"
        return spot_view(getattr(self.perception, "grid", None), spot, pose.x, pose.y, pose.yaw)

    def _pin_xy(self):
        base = getattr(self, "_route_base", None)
        pin = getattr(self, "_pin_index", None)
        if base and pin is not None and pin < len(base):
            return base[pin].x, base[pin].y
        m = self.mission_manager.current_mission
        return (m.destination_x, m.destination_y) if m else (0.0, 0.0)

    def _retarget_from_base(self, slot):
        """Point the route into `slot`, drawn afresh from the route as planned before any
        pull-in (a slot further on than the current end of the route stays reachable)."""
        base = getattr(self, "_route_base", None) or list(self._route.waypoints)
        route = Route(waypoints=list(base), timestamp=self._route.timestamp)
        got = self.planner.retarget_to_slot(route, slot)
        if got:
            self._route.waypoints = route.waypoints      # atomic swap, same route object
        return got

    def _confirm_parking_spot(self, pose, behavior_output, dest_dist):
        """Camera mode. Before turning in, the van must SEE the spot free with its LiDAR.

        Looked at from SPOT_CONFIRM_FROM_M out (by road). Seen taken: another spot at once.
        Free on SPOT_FREE_LOOKS looks in a row, with nothing in the way of the pull-in by the
        planner's own test: confirmed. Something in the way on SPOT_BLOCKED_LOOKS looks in a
        row: another spot. Not confirmed when the van reaches the start of its pull-in: it
        waits there SPOT_WAIT_S, then chooses another -- not being able to confirm a spot is
        not permission to drive into it."""
        sp = getattr(self, "_parking_spot", None)
        if (not sp or sp.get("kind") not in ("slot", "bay") or sp.get("confirmed")
                or dest_dist is None or dest_dist > SPOT_CONFIRM_FROM_M):
            return
        area = self._pull_in_area(sp)
        view = self._spot_view(area, pose)
        sp["seen"] = view
        if view != "free" and self.perception_mode == "camera_lidar" \
                and time.time() - getattr(self, "_last_view_log", 0.0) > 1.0:
            self._last_view_log = time.time()
            counts = spot_counts(getattr(self.perception, "grid", None), area, pose.x, pose.y, pose.yaw)
            self.logger.log_event("parking_view",
                                  f"{view}: squares free/blocked/unseen {counts} over "
                                  f"{area['length']:.1f} x {area['width']:.1f} m centred "
                                  f"({area['x']:.1f}, {area['y']:.1f}); van at ({pose.x:.1f}, {pose.y:.1f}), "
                                  f"{dest_dist:.1f} m from the spot")
            print(f"[Parking] spot {view}: free/blocked/unseen {counts}, {dest_dist:.1f} m out")
        if view == "taken":
            self._rechoose_parking(pose, "the LiDAR sees something in it")
            return
        if view == "free":
            # ...and the planner's own test finds nothing standing still that the van's body
            # would touch on the way in: a shelter at the kerb beside the spot is not IN the
            # strip, but the van cannot pass it (2026-09-11)
            in_way = self.planner.pull_in_blocker(getattr(self, "_last_perception", None), self._route,
                                                  pose.x, pose.y, pose.yaw, self.footprint_blocking.footprint)
            if in_way is None:
                sp["blocked_looks"] = 0
                sp["free_looks"] = sp.get("free_looks", 0) + 1
                if sp["free_looks"] >= SPOT_FREE_LOOKS:
                    sp["confirmed"] = True
                    self._parking_wait_since = None
                    self.logger.log_event("parking_confirmed",
                                          f"the LiDAR sees the {sp['kind']} free, {dest_dist:.1f} m before it")
                    self._note_move(SPOT_CONFIRMED,
                                    f"the laser sees the {sp['kind']} free, {dest_dist:.1f} m "
                                    f"before it — turning in")
                    print(f"[Parking] the LiDAR sees the spot free, {dest_dist:.1f} m out — turning in")
                    return
            else:
                sp["free_looks"] = 0
                sp["blocked_looks"] = sp.get("blocked_looks", 0) + 1
                obj, dist, hit, where, box = in_way
                what = getattr(getattr(obj, "object_type", None), "value", "thing")
                if sp["blocked_looks"] == 1 or sp["blocked_looks"] >= SPOT_BLOCKED_LOOKS:
                    self.logger.log_event(
                        "parking_way_in_blocked",
                        f"look {sp['blocked_looks']}: {what} id {getattr(obj, 'id', None)} centred "
                        f"({where[0]:.2f}, {where[1]:.2f}), measured {getattr(obj, 'box_length_m', 0):.2f} x "
                        f"{getattr(obj, 'box_width_m', 0):.2f} x {getattr(obj, 'height_m', 0):.2f} m"
                        + (f", taken as {2 * box.half_length:.2f} x {2 * box.half_width:.2f} m" if box is not None
                           else f", no box: radius {getattr(obj, 'clearance_radius_m', None)}")
                        + f"; touched with the van at ({hit.station_x:.2f}, {hit.station_y:.2f}) heading "
                        f"{math.degrees(hit.heading):.1f} deg, {hit.along_m:.1f} m on, {hit.lateral_m:+.2f} m "
                        f"off the line; {getattr(obj, 'motion_class', '?')} ({getattr(obj, 'motion_why', '')})")
                if sp["blocked_looks"] >= SPOT_BLOCKED_LOOKS:
                    self._rechoose_parking(pose, f"the way in passes too close to a {what} "
                                                 f"{dist:.0f} m ahead")
                    return
        else:
            sp["free_looks"] = 0
        # Not confirmed yet. Fine while there is road left before the pull-in starts; at its
        # start, wait there SPOT_WAIT_S -- not being able to confirm a spot is not permission
        # to drive into it -- then choose another.
        if dest_dist > (sp.get("approach_m") or 0.0) + 2.0:
            return
        now = time.time()
        if self._parking_wait_since is None:
            self._parking_wait_since = now
        if now - self._parking_wait_since >= SPOT_WAIT_S:
            self._rechoose_parking(pose, f"the spot could not be confirmed in {SPOT_WAIT_S:.0f} s")
            return
        behavior_output.behavior = DrivingBehavior.PARKING
        behavior_output.reason = "Parking — checking the spot is free before turning in"
        behavior_output.should_stop = True
        behavior_output.desired_speed_mps = 0.0

    def _pull_in_area(self, sp):
        """The spot AND the stretch of strip the van sweeps while pulling in -- from where its
        body first leaves the lane (planner.bay_needed_behind_m) to the spot's far end. A car
        parked just behind the spot is in the way as surely as one in it."""
        length = sp.get("length") or SPOT_DEFAULT_LEN_M
        approach = sp.get("approach_m") or 0.0
        straight = self.planner.PULL_IN_STRAIGHT_M
        back = 0.7 * max(0.0, approach - straight) + straight if approach > 0 else length / 2.0
        s0, s1 = -max(back, length / 2.0), length / 2.0
        mid = (s0 + s1) / 2.0
        return dict(sp, x=sp["x"] + mid * math.cos(sp["yaw"]), y=sp["y"] + mid * math.sin(sp["yaw"]),
                    length=s1 - s0, width=sp.get("width") or SPOT_DEFAULT_WID_M)

    def _choose_spot(self, ahead_of=None):
        """Where to park, drawn afresh from the route as planned: (route points, spot) or None.

        In order -- a van stopped in the driving lane makes everything behind it wait, so the
        lane is only ever the last resort (Rajat, 2026-09-11):
          1. a strip spot within PARK_PAST_PIN_M past the pin (or 40 m before it)
          2. a strip spot up to PARK_FAR_PAST_PIN_M past it
          3. the kerb edge of the lane, or straight in the lane, near the pin
        Spots already turned down, and pull-ins that would start behind the van, are skipped."""
        base = getattr(self, "_route_base", None)
        if not base:
            return None
        tries = ((self.planner.PARK_PAST_PIN_M, False), (self.planner.PARK_FAR_PAST_PIN_M, False),
                 (self.planner.PARK_PAST_PIN_M, True))
        for past, in_lane in tries:
            route = Route(waypoints=list(base), timestamp=self._route.timestamp)
            spot = self.planner.apply_pullover(route, side="right", pin_index=self._pin_index,
                                               avoid=self._parking_rejected, ahead_of=ahead_of,
                                               past_pin_m=past, in_lane_ok=in_lane)
            if spot is not None:
                if in_lane:
                    spot["note"] = "no strip spot will do within %.0f m -- stopping in the lane" % (
                        self.planner.PARK_FAR_PAST_PIN_M)
                return route.waypoints, spot
        return None

    def _rechoose_parking(self, pose, why):
        """The spot we were heading for will not do. Choose again, from the route as planned
        before any pull-in, skipping every spot already turned down, and only where the
        pull-in starts ahead of the van. Nothing left: stop at the pin, straight, in the lane
        -- or just ahead, if the pin is already behind."""
        sp = self._parking_spot or {}
        self._parking_rejected.append((sp.get("x", 0.0), sp.get("y", 0.0)))
        self._parking_wait_since = None
        self.behavior._park_best_d = None
        base = getattr(self, "_route_base", None)
        if not base:
            return
        chosen = self._choose_spot(ahead_of=(pose.x, pose.y))
        if chosen is not None:
            self._route.waypoints, new = chosen
            self._parking_spot = new
            msg = f"{why} — now heading for a {new['kind']} {new.get('from_pin_m')} m from the pin"
            if new.get("note"):
                msg += f" ({new['note']})"
        else:
            wps = list(base[:self._pin_index + 1])      # cut back to the pin
            here = min(range(len(base)), key=lambda k: (base[k].x - pose.x) ** 2 + (base[k].y - pose.y) ** 2)
            if here + 2 >= len(wps):                    # the pin is behind us: stop just ahead
                wps = list(base[:min(len(base), here + 4)])
            self._route.waypoints = wps
            last = wps[-1]
            self._parking_spot = {"x": last.x, "y": last.y, "yaw": last.yaw, "kind": "lane",
                                  "offset_m": 0.0, "moved_back_m": 0, "confirmed": True}
            msg = f"{why} — no other spot near the destination: stopping straight in the lane"
        if self._signal_lookahead is not None:
            self._signal_lookahead.set_route(self._route)    # a new tail can cross a junction
        self.logger.log_event("parking_rechosen", msg)
        self._note_move(SPOT_RECHOSEN, msg)
        print(f"[Parking] {msg}")

    def _recheck_parking_on_approach(self, pose):
        """Entering the parking phase: occupancy may be stale (cars parked
        after the mission started). Re-scan; if the chosen slot got taken,
        re-choose and retarget."""
        sp = getattr(self, "_parking_spot", None)
        slots = getattr(self, "_parking_slots", None)
        if not sp or sp.get("kind") != "slot" or not slots:
            return
        self._mark_slot_occupancy(slots)
        idx = sp.get("slot_index")
        if idx is None or idx >= len(slots) or not slots[idx]["occupied"]:
            return
        # Only slots the van can still reach driving FORWARD count (no
        # reverse gear): a slot behind the van caused an instant overshoot
        # "parked" at 156° in sweep testing.
        fwd = (math.cos(pose.yaw), math.sin(pose.yaw))
        ahead = [i for i, s in enumerate(slots)
                 if (s["x"] - pose.x) * fwd[0] + (s["y"] - pose.y) * fwd[1] > 4.0]
        new_idx = None
        for i in reversed(ahead):               # prefer nearest the destination
            if self.planner.slot_reachable(slots, i):
                new_idx = i
                break
        if new_idx is None:
            # only slots that do not say which bay they are in keep the old last resort
            for i in reversed(ahead):
                if "bay" not in slots[i] and not slots[i]["occupied"]:
                    new_idx = i
                    break
        if new_idx is None:
            # Do not keep aiming at a slot we KNOW is taken and hope the
            # obstacle logic stops us: in camera mode it did not, twice (4 Sep,
            # third and fourth park-checks - collisions with the parked car).
            # Stop short of it, in the lane, like a driver would.
            if getattr(self, "_hold_short_done", False):
                return
            hold = hold_short_point(slots[idx], pose.x, pose.y, back_m=5.0)
            ahead = (hold["x"] - pose.x) * fwd[0] + (hold["y"] - pose.y) * fwd[1]
            self._hold_short_done = True
            if ahead > 3.0 and self._retarget_from_base(hold):
                self._parking_spot = {"x": hold["x"], "y": hold["y"], "yaw": hold["yaw"],
                                      "offset_m": None, "moved_back_m": 0, "kind": "hold",
                                      "slot_index": idx}
                self.behavior._park_best_d = None
                self.logger.log_event("parking_rescan",
                                      f"chosen slot #{idx} now occupied and NO free slot remains AHEAD — "
                                      f"holding {ahead:.1f} m ahead, short of the taken bay")
                print(f"[Parking] slot #{idx} taken, none free ahead — stopping short of it")
            else:
                self.logger.log_event("parking_rescan",
                                      f"chosen slot #{idx} now occupied and NO free slot remains AHEAD "
                                      f"(too close to stop short — obstacle logic must hold)")
            return
        slots[idx]["chosen"] = False
        slots[new_idx]["chosen"] = True
        sl = slots[new_idx]
        if self._retarget_from_base(sl):
            self._parking_spot = {"x": sl["x"], "y": sl["y"], "yaw": sl["yaw"],
                                  "offset_m": None, "moved_back_m": 0, "kind": "slot",
                                  "slot_index": new_idx}
            # Fresh spot — forget the old approach or the overshoot escape
            # fires instantly against the previous slot's closest-distance.
            self.behavior._park_best_d = None
            self.logger.log_event("parking_rescan", f"slot #{idx} was taken — re-targeted to slot #{new_idx}")
            print(f"[Parking] slot #{idx} taken — switching to slot #{new_idx}")

    def _lidar_rescan_on_approach(self, pose):
        """Near the bay, with parking_source == "lidar": look with the lidar,
        mark occupancy on what it found, and swap the map's slot for the free
        real one nearest the current target. Runs once per mission. FIND
        PARKING itself runs at mission start, out of lidar reach, so this is
        the moment the lidar can actually see the bay."""
        if self.parking_source != "lidar" or getattr(self, "_lidar_rescan_done", False):
            return
        sp = getattr(self, "_parking_spot", None)
        if not sp or sp.get("kind") != "slot":
            return
        scan = getattr(self.sensor_adapter, "latest_lidar", None)
        if scan is None:
            self.logger.log_event("parking_lidar", "approach re-scan: no lidar data")
            return
        try:
            slots = sensed_parking_slots(scan.points, pose.x, pose.y, pose.yaw)
        except Exception as e:
            self.logger.log_event("parking_lidar", f"approach re-scan failed: {e}")
            return
        if not slots:
            # not done: try again as the van closes in (the trigger re-fires)
            n = getattr(self, "_lidar_rescan_tries", 0) + 1
            self._lidar_rescan_tries = n
            if n in (1, 3, 6, 10):
                self.logger.log_event("parking_lidar",
                                      f"approach re-scan #{n}: lidar saw no bay ({why_no_kerb(scan.points)}) - keeping the map slot for now")
            return
        self._lidar_rescan_done = True
        self._mark_slot_occupancy(slots)
        new_idx = nearest_free_slot(slots, sp["x"], sp["y"], max_dist_m=12.0)
        if new_idx is None:
            self.logger.log_event("parking_lidar",
                                  f"approach re-scan: {len(slots)} lidar slots, none free within 12 m of the target - keeping the map slot")
            return
        sl = slots[new_idx]
        # The lidar sharpens the bay; it does not move it to the next kerb along.
        # A free map target may only be sharpened (about one slot along); a
        # TAKEN map target may be swapped for a free lidar slot farther along.
        map_slots = getattr(self, "_parking_slots", None) or []
        target_taken = bool(map_slots and sp.get("slot_index") is not None
                            and sp["slot_index"] < len(map_slots)
                            and map_slots[sp["slot_index"]].get("occupied"))
        why = consistent_with(sp, sl, max_along_m=12.0 if target_taken else 4.0)
        if why:
            n = getattr(self, "_lidar_rescan_tries", 0) + 1
            self._lidar_rescan_tries = n
            if n in (1, 3, 6, 10):
                self.logger.log_event("parking_lidar",
                                      f"approach re-scan #{n}: lidar slot rejected - {why} - keeping the map slot for now")
            return
        shift = math.hypot(sl["x"] - sp["x"], sl["y"] - sp["y"])
        if not self._retarget_from_base(sl):
            self.logger.log_event("parking_lidar", "approach re-scan: could not retarget to the lidar slot")
            return
        for s_ in slots:
            s_["chosen"] = False
        sl["chosen"] = True
        self._parking_slots = slots
        self._parking_spot = {"x": sl["x"], "y": sl["y"], "yaw": sl["yaw"],
                              "offset_m": None, "moved_back_m": 0, "kind": "slot",
                              "slot_index": new_idx}
        self.behavior._park_best_d = None
        self.logger.log_event("parking_lidar",
                              f"approach re-scan: {len(slots)} lidar slots, re-targeted to lidar slot #{new_idx} "
                              f"({shift:.2f} m from the map slot)")
        print(f"[Parking] lidar re-scan: target moved {shift:.2f} m onto a lidar-found slot")

    def _collect_obstacle_points(self, perception, pose, reach_m=15.0):
        """Outline points of everything solid near the van, in the van's own
        frame (x forward, y right, metres) - what the round-8 brain's four
        feelers read. Camera/lidar mode: the lidar clusters (centre and
        extent), so the brain sees what the sensors see. Ground-truth mode:
        live actors' boxes plus the map's decorative parked cars."""
        pts = []
        if self.perception_mode == "camera_lidar":
            for c in (getattr(self.perception, "last_clusters", None) or []):
                if c["distance"] > reach_m:
                    continue
                # perception fix 2 makes low things visible (kerb fragments,
                # planters); the brain was trained on car outlines, so keep
                # its feelers to blobs at least 0.30 m tall
                h = c.get("height")
                if h is not None and h < 0.30:
                    continue
                e = min(float(c["extent"]), 3.0)
                x, y = float(c["x"]), float(c["y"])
                pts += [(x, y), (x + e, y), (x - e, y), (x, y + e), (x, y - e)]
            return pts
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)

        def to_van(wx, wy):
            dx, dy = wx - pose.x, wy - pose.y
            return (dx * c + dy * s, -dx * s + dy * c)

        try:
            ego_id = self.vehicle_adapter.vehicle.id
            for a in self.vehicle_adapter.world.get_actors().filter("vehicle.*"):
                if a.id == ego_id:
                    continue
                loc = a.get_location()
                if math.hypot(loc.x - pose.x, loc.y - pose.y) > reach_m:
                    continue
                tf = a.get_transform()
                ext = a.bounding_box.extent
                pts += [to_van(px, py) for px, py in box_outline_points(
                    tf.location.x, tf.location.y, math.radians(tf.rotation.yaw), ext.x, ext.y)]
        except Exception:
            pass
        try:
            for vpts in self._static_vehicle_points():
                cx, cy = vpts[0]
                if math.hypot(cx - pose.x, cy - pose.y) > reach_m:
                    continue
                pts += [to_van(px, py) for px, py in vpts]
        except Exception:
            pass
        return pts

    def _maybe_learned_parker(self, cmd, behavior_output, pose):
        """Swap the command for the learned parker's while it is at the wheel."""
        if self.parker != "rl" or not self.mission_manager.current_mission:
            return cmd, behavior_output
        sp = getattr(self, "_parking_spot", None)
        if not sp or sp.get("kind") != "slot":
            return cmd, behavior_output
        rl = self.rl_parker
        if rl.done:
            return cmd, behavior_output
        if not rl.engaged:
            if not rl.should_take_over(sp, pose.x, pose.y, pose.yaw):
                return cmd, behavior_output
            self.logger.log_event("parking_rl", f"learned parker took the wheel "
                                  f"{rl.distance_to(sp, pose.x, pose.y):.1f} m from the slot ({rl.describe()})")
            print("[Parking] learned parker at the wheel")
            rl.engaged = True      # even if a stop wins this tick, the hand-over is logged once
        # A stop demanded by the behaviour layer (obstacle, pedestrian, safety)
        # is honoured: the learned parker waits with the brakes on.
        if behavior_output.should_stop and behavior_output.behavior != DrivingBehavior.PARKING:
            per = getattr(self, "_last_perception", None)
            otype = getattr(getattr(per, "closest_obstacle_type", None), "value", "unknown") if per else "unknown"
            if per is None or stop_overrides_brain(otype, getattr(per, "closest_obstacle_speed", 0.0),
                                                   getattr(per, "closest_obstacle_distance", None)):
                return cmd, behavior_output
            # a stationary vehicle farther than 2 m: the brain sees it through its
            # feelers and drives on (the stop would freeze us beside a parked car)
            behavior_output.reason = f"learned parker driving past a stationary {otype} ({per.closest_obstacle_distance:.1f} m)"
        out = rl.act(pose.x, pose.y, pose.yaw, pose.speed, sp,
                     obstacle_points=getattr(self, "_obstacle_points_xy", None))
        cmd = VehicleCommand(steering=out["steering"], throttle=out["throttle"], brake=out["brake"],
                             gear=GearState.REVERSE if out.get("reverse") else GearState.DRIVE)
        if out["parked"]:
            self.logger.log_event("parking_rl", out["reason"])
            behavior_output.behavior = DrivingBehavior.MISSION_COMPLETE
            behavior_output.reason = "Parked - learned parker"
            behavior_output.should_stop = True
            self.behavior.mission_complete = True
            self.behavior.has_mission = False
        elif out["gave_up"]:
            self.logger.log_event("parking_rl", out["reason"])
            print(f"[Parking] {out['reason']}")
        else:
            behavior_output.reason = out["reason"]
        return cmd, behavior_output

    def api_set_parker(self, who, brain=None, handover_m=None):
        """Who drives the last metres into a slot: "rules" or "rl". `brain`
        picks the .zip (path relative to the repo or absolute; default round
        6); `handover_m` how far out the brain takes the wheel (default 16.5)."""
        who = str(who).strip().lower()
        if who not in ("rules", "rl"):
            return {"success": False, "reason": "Parker must be rules or rl", "parker": self.parker}
        if who == "rl" and brain:
            path = str(brain).strip()
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), "..", "..", path)
            path = os.path.abspath(path)
            if not os.path.exists(path):
                return {"success": False, "reason": f"no brain at {path}", "parker": self.parker}
            if path != self.rl_parker.model_path:
                self.rl_parker = RLParker(model_path=path, handover_m=self.rl_parker.handover_m)
        if who == "rl" and handover_m is not None:
            try:
                h = float(handover_m)
            except (TypeError, ValueError):
                return {"success": False, "reason": "handover_m must be a number", "parker": self.parker}
            if not 5.0 <= h <= 40.0:
                return {"success": False, "reason": "handover_m must be 5-40 m", "parker": self.parker}
            self.rl_parker.handover_m = h
        if who == "rl" and not os.path.exists(self.rl_parker.model_path):
            return {"success": False, "reason": f"no brain at {self.rl_parker.model_path}", "parker": self.parker}
        if who == "rl":
            # Loading the brain's libraries takes ~10 s the first time: do it
            # now, between missions, never at the hand-over on the road.
            t0 = time.time()
            try:
                self.rl_parker._load()
            except Exception as e:
                return {"success": False, "reason": f"could not load the brain: {e}", "parker": self.parker}
            print(f"[Parking] learned parker loaded in {time.time() - t0:.1f} s ({self.rl_parker.describe()})")
        self.parker = who
        self.logger.log_event("parker", f"the last 16 m are now driven by the {'learned' if who == 'rl' else 'hand-written'} parker"
                              + (f" ({self.rl_parker.describe()})" if who == "rl" else ""))
        return {"success": True, "parker": who,
                "brain": os.path.basename(self.rl_parker.model_path) if who == "rl" else None,
                "handover_m": self.rl_parker.handover_m}

    def api_set_footprint_blocking(self, enabled=None, safety_margin_m=None, debug_enabled=None):
        """Planning V2 runtime switches: swept-path blocking on/off, its safety
        margin, and (independently) the debug drawing of footprint and swept path."""
        result = self.footprint_blocking.set(enabled=enabled, safety_margin_m=safety_margin_m)
        if result.get("success") and debug_enabled is not None:
            dbg = self.footprint_debug.set(enabled=debug_enabled)
            if not dbg.get("success"):
                return {**dbg, **self.footprint_blocking.state()}
        result = {**result, **self.footprint_debug.state()}
        if result.get("success"):
            self.logger.log_event("planning", f"swept-path blocking "
                                  f"{'ON' if result['footprint_blocking_enabled'] else 'OFF'}, "
                                  f"margin {result['safety_margin_m']} m, debug drawing "
                                  f"{'ON' if result['footprint_debug_enabled'] else 'OFF'}")
        return result

    def api_set_parking_source(self, source):
        """Where FIND PARKING gets its slots: "map" (CARLA lane data) or
        "lidar" (the bay finder on the live sweep, map as fallback)."""
        source = str(source).strip().lower()
        if source not in ("map", "lidar"):
            return {"success": False, "reason": "Source must be map or lidar",
                    "source": self.parking_source}
        self.parking_source = source
        self.logger.log_event("parking_source", f"parking slots now come from the {source}")
        return {"success": True, "source": source}

    def _parking_slots_from_source(self):
        """(slots, where_from). Lidar first when selected; the map when the
        lidar sees no kerb, when it is disabled, or when the source is map."""
        if self.parking_source == "lidar":
            scan = getattr(self.sensor_adapter, "latest_lidar", None)
            pose = self.localization.get_last_pose()
            if scan is not None and pose is not None and getattr(pose, "healthy", True):
                try:
                    slots = sensed_parking_slots(scan.points, pose.x, pose.y, pose.yaw)
                except Exception as e:
                    self.logger.log_event("parking_slots", f"lidar bay finder failed: {e}")
                    slots = []
                if slots and self._route and self._route.waypoints:
                    dest = self._route.waypoints[-1]
                    near = [sl for sl in slots if math.hypot(sl["x"] - dest.x, sl["y"] - dest.y) <= 40.0]
                    if near:
                        self.logger.log_event("parking_slots", f"{len(near)} slots from the LIDAR near the destination")
                        return near, "lidar"
                    self.logger.log_event("parking_slots",
                                          f"lidar saw {len(slots)} bays beside the START, none within 40 m of the destination - using the map")
                    slots = []
            self.logger.log_event("parking_slots", "lidar saw no bay - falling back to the map")
        base = getattr(self, "_route_base", None)
        back = 70.0
        pin = getattr(self, "_pin_index", None)
        if base and pin is not None and pin < len(base):
            past = sum(math.hypot(base[i].x - base[i - 1].x, base[i].y - base[i - 1].y)
                       for i in range(pin + 1, len(base)))
            back = past + self.planner.PARK_MAX_PULLBACK_M
        slots = self.planner.find_parking_slots(Route(waypoints=list(base)) if base else self._route,
                                                search_back_m=back)
        for sl in slots:
            sl.setdefault("source", "map")
        return slots, "map"

    def api_find_parking(self, not_further_than_m=None):
        """FIND PARKING: slice the bays near the destination into van-sized
        slots, skip occupied ones, retarget the mission to the best free slot."""
        if not self._route or not self.mission_manager.current_mission:
            return {"success": False, "reason": "Start a mission first — parking is searched near its destination"}
        slots, where_from = self._parking_slots_from_source()
        if not slots:
            return {"success": False, "reason": "No parking bays on the final stretch of this route"}

        self._mark_slot_occupancy(slots)
        for sl in slots:
            sl["chosen"] = False

        # Nearest the PIN first (the route may run on past it), and the first one with a gentle
        # way in from our lane. Occupancy here is what the LiDAR can see from the start, which
        # is usually nothing: a slot not yet seen is not taken, but not free either -- it is
        # confirmed as the van arrives (_confirm_parking_spot) before the van turns in.
        pin = self._pin_xy()
        chosen_idx, approach = None, None
        for i in self.planner.free_slots_by_distance(slots, pin):
            if (not_further_than_m is not None
                    and math.hypot(slots[i]["x"] - pin[0], slots[i]["y"] - pin[1]) > not_further_than_m):
                break                            # the pull-over spot already chosen is nearer
            approach = self._retarget_from_base(slots[i])
            if approach:
                chosen_idx = i
                break
        if chosen_idx is None:
            self._parking_slots = slots
            free = sum(1 for x in slots if not x["occupied"])
            if free:
                why = (f"{len(slots)} slots found, {free} not seen taken, but none the van can drive "
                       f"into forwards (it needs {self.planner.APPROACH_BAY_BEHIND_M:.0f} m of free bay "
                       f"behind the slot and a gentle way in from its own lane)")
            else:
                why = f"All {len(slots)} slots are occupied"
            self.logger.log_event("parking_slots", why)
            print(f"[Parking] {why} — parking at the kerb in the lane")
            return {"success": False, "reason": why, "slots": slots}

        sl = slots[chosen_idx]
        sl["chosen"] = True
        self._parking_slots = slots
        self._parking_spot = {"x": sl["x"], "y": sl["y"], "yaw": sl["yaw"],
                              "length": sl.get("length"), "width": sl.get("width"),
                              "offset_m": None, "moved_back_m": 0, "kind": "slot",
                              "slot_index": chosen_idx, "approach_m": approach["approach_m"]}
        occ = sum(1 for x in slots if x["occupied"])
        msg = f"{len(slots)} slots on the bay, {occ} occupied — parking in slot #{chosen_idx}"
        self.logger.log_event("parking_slots", msg)
        print(f"[Parking] {msg}")
        return {"success": True, "slots": slots, "chosen": chosen_idx, "message": msg}

    def api_get_route(self):
        if not self._route:
            return []
        # road_id / lane_id go out too: a traffic light is matched to the route BY LANE, so
        # when the van says it can see no signal ahead this is the only way to tell whether
        # the route simply never travels the lane the light governs.
        return [{"x": round(w.x, 2), "y": round(w.y, 2),
                 "road_id": w.road_id, "lane_id": w.lane_id} for w in self._route.waypoints]

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
                for i, p in enumerate(points)]  # all spawn points — numbering matches tools/find_parking_bays.py

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
            if self.camera_lidar_perception is not None:
                try:
                    self.camera_lidar_perception.close()     # no detector thread while on ground truth
                except Exception:
                    pass

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

    view = request.args.get("view", "front")
    if view == "front":
        frame = sensor_adapter.latest_camera
    else:
        frame = getattr(sensor_adapter, "latest_frames", {}).get(view)

    if frame is None:
        return Response(
            f"Camera frame not available yet ({view})",
            status=503
        )

    try:

        # CARLA gives BGRA.
        # OpenCV uses BGR.
        image_bgr = frame.image[:, :, :3]

        # Dashboard preview does not need full 800x600.
        # Smaller image = less CPU + less network work.
        # Front gets the big preview; surround views are already small.
        if view == "front":
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

        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-top: 8px;
        }
        .grid .cell { position: relative; }
        .grid img { border-radius: 8px; }
        .grid .tag {
            position: absolute; top: 6px; left: 8px;
            font-size: 11px; letter-spacing: 1px;
            color: #cfd6df; background: rgba(0,0,0,.55);
            padding: 2px 7px; border-radius: 4px;
        }

    </style>
</head>

<body>

<div class="viewer">

    <h2>Warp AV — Surround View</h2>

    <div class="status" id="status">
        Waiting for camera...
    </div>

    <img id="camera">

    <div class="grid">
        <div class="cell"><span class="tag">LEFT</span><img id="cam_left"></div>
        <div class="cell"><span class="tag">RIGHT</span><img id="cam_right"></div>
        <div class="cell"><span class="tag">REAR</span><img id="cam_rear"></div>
        <div class="cell"><span class="tag">TOP</span><img id="cam_top"></div>
    </div>

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

const sideViews = ["left", "right", "rear", "top"];
sideViews.forEach(function(v, i) {
    const el = document.getElementById("cam_" + v);
    function tick() {
        el.src = "/api/camera/frame?view=" + v + "&t=" + Date.now();
    }
    el.onload = function() { setTimeout(tick, 500); };
    el.onerror = function() { setTimeout(tick, 1500); };
    setTimeout(tick, 300 + i * 150);
});


camera.onload = function() {

    status.textContent =
        "LIVE • Front + Surround • ~2 FPS";

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

@app.route('/api/grid')
def api_grid():
    """The free-space map as a small picture, for an operator (Perception V2 day 9)."""
    grid = getattr(getattr(av_system, "perception", None), "grid", None)
    if grid is None or not grid.updated:
        return jsonify({"available": False, "reason": "no free-space map yet"}), 503
    span = float(request.args.get("span_m", 14.0))
    edges = getattr(getattr(av_system, "perception", None), "road_edges", None)
    return jsonify({**grid.summary().as_dict(),
                    "kerbs": edges.as_dict() if edges is not None else None,
                    "legend": {".": "free", "#": "blocked", " ": "not seen"},
                    "picture": grid.as_text(span_m=span).split("\n")})


@app.route('/api/world')
def api_world():
    """What the van knows right now, in one sheet (Perception V2 day 7)."""
    wm = getattr(av_system, "_world", None)
    if wm is None:
        return jsonify({"available": False,
                        "reason": getattr(av_system, "_world_error", None) or "not built yet"}), 503
    return jsonify(wm.as_dict())


@app.route('/api/history')
def get_history():
    return jsonify(av_system.api_get_history())

@app.route('/api/traffic/spawn', methods=['POST'])
def spawn_traffic_api():
    data = request.get_json(silent=True) or {}   # body is optional (bare button press)
    return jsonify(av_system.api_spawn_traffic(
        cars=int(data.get('cars', 15)), walkers=int(data.get('walkers', 12)),
        cyclists=int(data.get('cyclists', 4))))

@app.route('/api/traffic/clear', methods=['POST'])
def clear_traffic_api():
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_clear_traffic(all_actors=bool(data.get('all', False))))

@app.route('/api/weather', methods=['GET', 'POST'])
def weather_api():
    if request.method == 'GET':
        return jsonify({"preset": getattr(av_system, "_weather_preset", "default")})
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_set_weather(data.get('preset')))

@app.route('/api/test/park_cars', methods=['POST'])
def park_cars_api():
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_park_cars(
        count=int(data.get('count', 4)),
        spacing=float(data.get('spacing', 14.0)),
        fill_all=bool(data.get('fill_all', False)),
        clear=bool(data.get('clear', False)),
        take_chosen=bool(data.get('take_chosen', False))))

@app.route('/api/planning/footprint_blocking', methods=['GET', 'POST'])
def planning_footprint_blocking():
    if request.method == 'GET':
        return jsonify({**av_system.footprint_blocking.state(), **av_system.footprint_debug.state()})
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_set_footprint_blocking(enabled=data.get("enabled"),
                                                        safety_margin_m=data.get("safety_margin_m"),
                                                        debug_enabled=data.get("debug_enabled")))


@app.route('/api/parking/parker', methods=['GET', 'POST'])
def parking_parker():
    if request.method == 'GET':
        return jsonify({"parker": av_system.parker})
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_set_parker(data.get("parker", ""), brain=data.get("brain"),
                                            handover_m=data.get("handover_m")))


@app.route('/api/parking/source', methods=['GET', 'POST'])
def parking_source():
    if request.method == 'GET':
        return jsonify({"source": av_system.parking_source})
    data = request.get_json(silent=True) or {}
    return jsonify(av_system.api_set_parking_source(data.get("source", "")))


@app.route('/api/parking/find', methods=['POST'])
def find_parking():
    return jsonify(av_system.api_find_parking())

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
