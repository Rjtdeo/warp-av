"""
Behavior / Decision Module

YOUR ROVER equivalent:
    decision_node.py checks front < 50, right < 20, etc.
    If front is blocked, turn right. If all clear, go forward.

THIS VERSION:
    Same idea but with named states and REASONS for every decision.
    This is the most important thing Warp is testing:
    "If the vehicle stops, we should be able to determine why."

    Every behavior change publishes a REASON STRING.
    That single feature answers half the observability questions.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..perception.perception import PerceptionOutput, ObjectType
from ..localization.localization import Pose, LocalizationQuality


class DrivingBehavior(Enum):
    IDLE = "idle"
    FOLLOWING_ROUTE = "following_route"
    FOLLOWING_VEHICLE = "following_vehicle"
    APPROACHING_DESTINATION = "approaching_destination"
    STOPPED_OBSTACLE = "stopped_obstacle"
    STOPPED_PEDESTRIAN = "stopped_pedestrian"
    STOPPED_VEHICLE = "stopped_vehicle"
    STOPPED_BLOCKED = "stopped_blocked"
    STOPPED_SAFETY = "stopped_safety"
    STOPPED_ESTOP = "stopped_estop"
    STOPPED_RED_LIGHT = "stopped_red_light"
    WAITING_AT_JUNCTION = "waiting_at_junction"
    PARKING = "parking"
    MISSION_COMPLETE = "mission_complete"
    NO_MISSION = "no_mission"


@dataclass
class BehaviorOutput:
    """What the behavior layer decided to do and WHY."""
    behavior: DrivingBehavior = DrivingBehavior.IDLE
    reason: str = ""                    # THE KEY FIELD — human-readable explanation
    desired_speed_mps: float = 0.0
    should_stop: bool = False
    timestamp: float = field(default_factory=time.time)


class BehaviorSystem:
    """
    Decides what the vehicle should do based on perception + localization + mission.

    Think of it as your decision_node.make_decision() but it outputs
    a named behavior + reason instead of just "FORWARD" or "TURN_LEFT".
    """

    def __init__(self):
        self.current_behavior = DrivingBehavior.NO_MISSION
        self.current_reason = "No mission assigned"
        self.has_mission = False
        self.mission_complete = False

        # Tuning
        self.cruise_speed = 8.0          # m/s (~18 mph, good for cargo van)
        self.slow_speed = 3.0            # m/s when approaching obstacle
        self.stop_distance = 8.0         # meters — informational; the actual stop trigger is perception.danger_distance
        self.slow_distance = 20.0        # meters — slow down (Troy #4: was 15.0)

        # Car-following (Troy #6). Engages only for a MOVING vehicle ahead;
        # a stopped one still uses slow-zone + stop. Camera mode reports lead
        # speed 0 (no tracking yet), so it safely falls back to slow/stop.
        self.follow_engage_m = 30.0      # start following when lead within this
        self.follow_time_gap_s = 1.5     # keep this many seconds behind the lead
        self.follow_standstill_m = 8.0   # plus this fixed gap (matches stop buffer)
        self.follow_gain = 0.3           # how hard to close/open the gap (1/s)
        self.follow_min_lead_mps = 0.7   # below this the lead counts as stopped

        # Junction give-way (Troy request): before turning at a junction, stop,
        # look for a moment, and only go when no moving vehicle is nearby.
        self.junction_stop_within_m = 12.0   # start handling the turn this close
        self.hold_line_m = 3.0               # give-way hold: centre this far from the crossing
        self.light_hold_m = 2.6              # red light fallback: centre 2.6 m from the JUNCTION ENTRY (stop-bar junctions without zebra data) -> bumper ~0.25 m from the boundary
        self.light_hold_line_m = 2.6         # centre 2.6 m from the PAINTED line -> bumper ~0.25 m before the paint (operator-tuned 2026-08-30)
        self.junction_dwell_s = 1.5          # mandatory look time even if clear
        self.junction_conflict_radius_m = 25.0
        self.junction_wait_timeout_s = 12.0  # then creep instead of deadlocking
        self.junction_creep_mps = 2.0
        self._junction_wait_started = None
        self._junction_done = False          # cleared for the junction we're in
        self._park_best_d = None             # closest approach to the parking spot
        self.block_release_s = 2.0           # blocked verdicts must stay clear this long before moving again
        self._block_memory = None            # (t_last_blocked, kind, distance)
        self.destination_threshold = 1.5 # meters — parked when this close to the SPOT (was 5.0 anywhere on the road)
        self.parked_max_speed = 0.8      # ...and slower than this
        self.park_zone_m = 15.0          # final approach: taper to walking pace

        # If the path stays blocked for this long,
        # treat it as a blocked route instead of a temporary obstacle.
        self.blocked_timeout = 3.0
        self._blocked_since = None

    def update(
        self,
        perception: PerceptionOutput,
        pose: Pose,
        destination_distance: Optional[float],
        safety_ok: bool,
        junction: Optional[dict] = None,
        junction_ahead_m: Optional[float] = None,
        park_heading_ok: bool = True,
        park_position_ok: bool = True,
        white_line_m: Optional[float] = None
    ) -> BehaviorOutput:
        """
        One decision cycle.

        Your rover did:
            if front < 20: STOP
            elif front < 50: TURN
            else: FORWARD

        This does the same thing but with richer states and always a reason.
        """

        # --- Safety override (highest priority) ---
        if not safety_ok:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                "Safety supervisor commanded stop",
                speed=0.0, stop=True
            )

        # --- No mission ---
        if not self.has_mission:
            return self._decide(
                DrivingBehavior.NO_MISSION,
                "No mission assigned — waiting for destination",
                speed=0.0, stop=True
            )

        # --- Localization lost ---
        if not pose.healthy or pose.quality == LocalizationQuality.LOST:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Localization unhealthy: {pose.reason}",
                speed=0.0, stop=True
            )

        # --- Perception unhealthy ---
        if not perception.healthy:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Perception unhealthy: {perception.reason}",
                speed=0.0, stop=True
            )

        # Track the closest we ever got to the spot: if we start moving AWAY
        # again at parking speed, we overshot — stop there rather than creep
        # off down the road hunting perfection.
        if destination_distance is not None and destination_distance < self.park_zone_m:
            if self._park_best_d is None or destination_distance < self._park_best_d:
                self._park_best_d = destination_distance
        overshot = (self._park_best_d is not None
                    and self._park_best_d < 2.5
                    and destination_distance is not None
                    and destination_distance > self._park_best_d + 0.8)
        if overshot and pose.speed < self.parked_max_speed and self.has_mission:
            self.mission_complete = True
            self.has_mission = False
            return self._decide(
                DrivingBehavior.MISSION_COMPLETE,
                f"Parked (overshot the spot by {destination_distance - self._park_best_d:.1f} m)",
                speed=0.0, stop=True
            )

        # --- Parked at the spot (close, nearly stopped, straight, IN the box) ---
        if (destination_distance is not None
                and destination_distance < self.destination_threshold
                and pose.speed < self.parked_max_speed
                and (park_heading_ok or destination_distance < 0.5)
                and (park_position_ok or destination_distance < 0.30)):
            self.mission_complete = True
            self.has_mission = False
            return self._decide(
                DrivingBehavior.MISSION_COMPLETE,
                f"Parked — {destination_distance:.1f} m from the spot",
                speed=0.0, stop=True
            )

        # --- Persistent blocked route ---
        #
        # A pedestrian or stopped vehicle is a temporary road situation,
        # not automatically a "blocked route".
        #
        # Only static/other obstacles can become a persistent blocked road.
        if (
            perception.path_blocked
            and perception.closest_obstacle_type
            not in (
                ObjectType.PEDESTRIAN,
                ObjectType.VEHICLE,
            )
        ):
            if self._blocked_since is None:
                self._blocked_since = time.time()

            blocked_duration = time.time() - self._blocked_since

            if blocked_duration >= self.blocked_timeout:
                return self._decide(
                    DrivingBehavior.STOPPED_BLOCKED,
                    (
                        f"Route blocked for {blocked_duration:.1f}s by "
                        f"{perception.closest_obstacle_type.value} "
                        f"at {perception.closest_obstacle_distance:.1f}m "
                        f"— replan or operator action required"
                    ),
                    speed=0.0,
                    stop=True
                )
        else:
            self._blocked_since = None

        # (helper for the release latch below)
        # --- Path blocked by pedestrian (ALWAYS stop for pedestrians) ---
        if perception.path_blocked and perception.closest_obstacle_type == ObjectType.PEDESTRIAN:
            self._note_block(DrivingBehavior.STOPPED_PEDESTRIAN,
                             perception.closest_obstacle_distance)
            return self._decide(
                DrivingBehavior.STOPPED_PEDESTRIAN,
                f"PEDESTRIAN in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Path blocked by vehicle ---
        if perception.path_blocked and perception.closest_obstacle_type == ObjectType.VEHICLE:
            self._note_block(DrivingBehavior.STOPPED_VEHICLE,
                             perception.closest_obstacle_distance)
            return self._decide(
                DrivingBehavior.STOPPED_VEHICLE,
                f"VEHICLE blocking path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Path blocked by obstacle ---
        if perception.path_blocked:
            self._note_block(DrivingBehavior.STOPPED_OBSTACLE,
                             perception.closest_obstacle_distance)
            return self._decide(
                DrivingBehavior.STOPPED_OBSTACLE,
                f"OBSTACLE in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Release latch: a close blocker that BLINKS out of detection for
        # a moment must not release the van instantly. In a dense-traffic
        # brawl the verdict flapped every 1-3 s and the van crept half a
        # metre per blink into a shrinking gap (two contacts). Stay stopped
        # until the path has been continuously clear for block_release_s.
        if (self._block_memory is not None
                and pose.speed < 1.2
                and time.time() - self._block_memory[0] < self.block_release_s):
            kind, dist = self._block_memory[1], self._block_memory[2]
            return self._decide(
                kind,
                f"Path just cleared (was blocked {dist:.1f}m ahead) — confirming for "
                f"{self.block_release_s:.0f}s before moving",
                speed=0.0, stop=True
            )
        self._block_memory = None

        # --- Traffic light (Troy #1): roll up to the stop line, hold there.
        # Ranked below pedestrian/vehicle/obstacle stops (a closer physical
        # hazard always wins) and above following/cruising. Green releases it
        # automatically on the next tick.
        if perception.traffic_light in ("red", "yellow"):
            # Committed: already entering/inside the junction when the light
            # changed — clear it, never freeze inside the box.
            if junction_ahead_m is not None and junction_ahead_m < 1.0:
                pass
            else:
                # Best reference first: the PAINTED white line (crosswalk
                # polygon on our route) — hold with the bumper just before
                # the paint. Then the junction entry (edge polygons sit
                # metres before the paint at some junctions — operator/Troy
                # complaint), then CARLA's early stop waypoints.
                if white_line_m is not None:
                    d, line, hold = white_line_m, "white line", self.light_hold_line_m
                elif junction_ahead_m is not None:
                    d, line, hold = junction_ahead_m, "junction edge", self.light_hold_m
                else:
                    d, line, hold = perception.traffic_light_distance_m, "stop line", self.light_hold_m
                if d is not None and d > hold + 0.5:
                    creep = max(0.6, min(3.0, 0.45 * (d - hold)))
                    return self._decide(
                        DrivingBehavior.FOLLOWING_ROUTE,
                        f"{perception.traffic_light.upper()} light ahead ({d:.0f} m to {line}) — rolling up",
                        speed=creep, stop=False
                    )
                return self._decide(
                    DrivingBehavior.STOPPED_RED_LIGHT,
                    f"{perception.traffic_light.upper()} traffic light — holding at the {line}, waiting for green",
                    speed=0.0, stop=True
                )

        # --- Give way before turning at a junction ---
        if junction is None or junction.get("distance_m", 99) > 15.0:
            self._junction_done = False      # next junction is a fresh decision
        if (junction is not None
                and not self._junction_done
                and junction.get("distance_m", 99) <= self.junction_stop_within_m):
            direction = junction.get("direction", "?")
            jdist = junction.get("distance_m", 99)
            # Phase 1: roll up to the crossing first (like a driver), THEN wait.
            if jdist > self.hold_line_m + 0.5:
                self._junction_wait_started = None
                creep = max(0.8, min(3.0, 0.5 * (jdist - self.hold_line_m)))
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Approaching {direction} turn — rolling up to the crossing ({jdist:.0f} m)",
                    speed=creep, stop=False
                )
            now = time.time()
            if self._junction_wait_started is None:
                self._junction_wait_started = now
            waited = now - self._junction_wait_started
            conflict = self._junction_conflict(perception)
            if waited >= self.junction_wait_timeout_s:
                self._junction_done = True
                self._junction_wait_started = None
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Give-way timeout at {direction} turn ({waited:.0f}s) — proceeding carefully",
                    speed=self.junction_creep_mps, stop=False
                )
            if waited < self.junction_dwell_s:
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Approaching {direction} turn — pausing to check for traffic",
                    speed=0.0, stop=True
                )
            if conflict is not None:
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Giving way at {direction} turn — moving vehicle {conflict:.0f} m away",
                    speed=0.0, stop=True
                )
            self._junction_done = True
            self._junction_wait_started = None
            print(f"[Behavior] Junction clear after {waited:.1f}s — taking the {direction} turn")
        elif self._junction_wait_started is not None and (
                junction is None or junction.get("distance_m", 99) > self.junction_stop_within_m):
            self._junction_wait_started = None

        # --- Moving vehicle ahead: follow at a time gap instead of stop-and-go ---
        if (perception.closest_obstacle_type == ObjectType.VEHICLE
                and perception.closest_obstacle_speed > self.follow_min_lead_mps
                and perception.closest_obstacle_distance < self.follow_engage_m):
            gap = perception.closest_obstacle_distance
            lead = perception.closest_obstacle_speed
            desired_gap = self.follow_standstill_m + self.follow_time_gap_s * lead
            target = lead + self.follow_gain * (gap - desired_gap)
            target = max(0.0, min(self.cruise_speed, target))
            return self._decide(
                DrivingBehavior.FOLLOWING_VEHICLE,
                f"Following vehicle: gap {gap:.1f}m (want {desired_gap:.1f}m), "
                f"lead {lead:.1f} m/s — target {target:.1f} m/s",
                speed=target, stop=False
            )

        # --- Object ahead, slow down ---
        if perception.closest_obstacle_distance < self.slow_distance:
            return self._decide(
                DrivingBehavior.FOLLOWING_ROUTE,
                f"Object detected at {perception.closest_obstacle_distance:.1f}m — slowing to {self.slow_speed:.1f} m/s",
                speed=self.slow_speed, stop=False
            )

        # --- Final approach: park at the kerb ---
        if destination_distance is not None and destination_distance < self.park_zone_m:
            creep = max(0.5, min(2.5, 0.35 * destination_distance))
            return self._decide(
                DrivingBehavior.PARKING,
                f"Parking — pulling over, {destination_distance:.1f} m to the spot",
                speed=creep, stop=False
            )

        # --- Approaching destination ---
        if destination_distance is not None and destination_distance < 25.0:
            return self._decide(
                DrivingBehavior.APPROACHING_DESTINATION,
                f"Approaching destination ({destination_distance:.1f}m) — slowing",
                speed=self.slow_speed, stop=False
            )

        # --- All clear, drive normally ---
        return self._decide(
            DrivingBehavior.FOLLOWING_ROUTE,
            f"Route clear — cruising at {self.cruise_speed:.1f} m/s",
            speed=self.cruise_speed, stop=False
        )

    def _note_block(self, kind, distance):
        """Remember a CLOSE physical blocker so a one-tick detection blink
        cannot release the van instantly (release latch above)."""
        if distance is not None and distance < 12.0:
            self._block_memory = (time.time(), kind, distance)

    def _decide(self, behavior, reason, speed, stop) -> BehaviorOutput:
        # Log when behavior CHANGES (important for debugging)
        if behavior != self.current_behavior:
            print(f"[Behavior] {self.current_behavior.value} → {behavior.value}: {reason}")
        self.current_behavior = behavior
        self.current_reason = reason
        return BehaviorOutput(
            behavior=behavior,
            reason=reason,
            desired_speed_mps=speed,
            should_stop=stop
        )

    def _junction_conflict(self, perception: PerceptionOutput):
        """Distance of the nearest moving vehicle that could cross our turn,
        or None. Vehicles directly ahead in our own lane are the car-following
        problem, not a junction conflict; far/parked/behind vehicles ignored."""
        nearest = None
        for obj in perception.objects:
            if obj.object_type != ObjectType.VEHICLE:
                continue
            if obj.speed < 1.0:
                continue
            if obj.distance > self.junction_conflict_radius_m:
                continue
            if obj.x < -3.0:
                continue                      # well behind us
            if obj.x > 0 and abs(obj.y) < 1.75:
                continue                      # our own lane: following handles it
            if nearest is None or obj.distance < nearest:
                nearest = obj.distance
        return nearest

    def set_cruise_speed(self, speed_mps: float):
        """Runtime speed-limit change from operator/API. Clamped to [0, 15] m/s."""
        self.cruise_speed = max(0.0, min(15.0, float(speed_mps)))
        print(f"[Behavior] cruise speed set to {self.cruise_speed:.1f} m/s")

    def set_mission(self):
        self.has_mission = True
        self.mission_complete = False
        self._park_best_d = None
        self.current_behavior = DrivingBehavior.IDLE

    def cancel_mission(self):
        self.has_mission = False
