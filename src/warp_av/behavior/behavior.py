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

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..perception.perception import PerceptionOutput, ObjectType, VULNERABLE_TYPES
from ..localization.localization import Pose, LocalizationQuality
from ..perception.road_signs import STOP, GIVE_WAY
from .transitions import (TransitionLog, SAFETY_HOLD, LOCALIZATION_LOST, PERCEPTION_LOST,
                          NO_MISSION, VRU_IN_PATH, VEHICLE_IN_PATH, OBSTACLE_IN_PATH,
                          ROUTE_BLOCKED_TOO_LONG, CONFIRMING_CLEAR, FOLLOWING_LEAD,
                          OBJECT_AHEAD_SLOW, ROUTE_CLEAR, PREDICTED_CROSSER_STOP,
                          PREDICTED_CROSSER_SLOW, LIGHT_ROLL_UP, LIGHT_HOLD,
                          SIGN_ROLL_UP, SIGN_STOP_HOLD, SIGN_GIVE_WAY, JUNCTION_KEEP_CLEAR,
                          JUNCTION_ROLL_UP, JUNCTION_PAUSE, JUNCTION_GIVE_WAY,
                          JUNCTION_TIMEOUT, DESTINATION_NEAR, PARKING_PULL_IN, PARKED,
                          PARKED_OVERSHOT)


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
    YIELDING_PREDICTED = "yielding_predicted"   # crosser/cut-in WILL be in our path
    PARKING = "parking"
    MISSION_COMPLETE = "mission_complete"
    NO_MISSION = "no_mission"


#: The wanted speeds that may be EASED into rather than stepped to (main.py applies it, where
#: the decision has been made and the command is about to be formed): slowing for something in
#: sight, keeping back from the car ahead, and going back up to cruising speed. Everything
#: else -- a light, a junction, a yield, the run-in to a parking spot, every stop -- takes
#: effect the moment it is decided. The decision itself is never shaped: what the van wanted
#: is what the drive's story records.
EASE_OFF_REASONS = (OBJECT_AHEAD_SLOW, FOLLOWING_LEAD, ROUTE_CLEAR)
#: how much the wanted speed may fall per decision: 0.35 m/s at ~9 a second is 3 m/s2
EASE_OFF_MPS = 0.35


#: Colours that mean "do not go". UNKNOWN is deliberately one of them: at a stop line we
#: know about, not being able to read the light is never permission.
LIGHT_MEANS_STOP = ("red", "yellow", "unknown")


#: How fast the van may go while its own body is heading over a kerb line.
KERB_CRAWL_MPS = 1.5


#: The front bumper stops this far short of a sign's line, rolls up to it below this much
#: more, and counts as stopped below this speed, for this long. A stop sign means stopped.
JUNCTION_ENTER_WITHIN_M = 6.0
SIGN_STOP_GAP_M = 0.5
SIGN_STOP_ROLL_M = 0.2
SIGN_STOPPED_MPS = 0.2
SIGN_DWELL_S = 1.0


#: The front bumper stops this far short of the stop line.
LIGHT_STOP_GAP_M = 0.5

#: From the 0.6 m/s the last metre is crawled at, the van rolls about this much further once
#: told to stop, so "stopped" is declared this much before the gap is reached.
LIGHT_STOP_ROLL_M = 0.2

#: The yellow-light choice, made once when a light stops being green: stop if the van can do
#: it before the line braking at this rate, otherwise keep going. Braking hard and ending up
#: IN the junction as it turns red is worse than going through on the yellow.
COMFORT_DECEL_MPS2 = 2.5

#: Where the front bumper is, ahead of the van's reported position, when nobody says.
#: Measured on the Sprinter in CARLA: 2.947 m.
DEFAULT_FRONT_OFFSET_M = 2.95


def _light_words(state: str) -> str:
    """What to say about it, so an unreadable light does not read as a red one."""
    if state == "unknown":
        return "TRAFFIC LIGHT AHEAD, COLOUR UNREADABLE"
    return f"{state.upper()} light"


BLIND_REACTION_S = 0.4      # noticing, deciding and the brakes taking hold, at ~9 decisions a second
BLIND_DECEL_MPS2 = 3.0      # comfortable braking for a laden van, not an emergency stop


def stopping_speed_for(distance_m: float) -> float:
    """The fastest the van may go and still stop within `distance_m`.

    Solving distance = v * reaction + v^2 / (2 * decel) for v. This is the whole rule behind
    slowing for a blind pocket: never travel faster than you could stop in the distance to
    the nearest place you cannot see into. It needs no threshold and no tuning table -- at
    5.5 m it gives 4.7 m/s, at 8 m it gives 5.8 m/s, and beyond about 14 m it is above the
    van's cruising speed and stops mattering by itself.
    """
    d = max(0.0, float(distance_m))
    a, t_r = BLIND_DECEL_MPS2, BLIND_REACTION_S
    return max(0.0, -a * t_r + math.sqrt((a * t_r) ** 2 + 2.0 * a * d))


@dataclass
class Situation:
    """Everything one decision is made from, in one place, so a rule reads the situation
    instead of being handed fourteen arguments."""
    perception: PerceptionOutput
    pose: Pose
    destination_distance: Optional[float] = None
    safety_ok: bool = True
    junction: Optional[dict] = None
    park_heading_ok: bool = True
    park_position_ok: bool = True
    predicted_conflict: Optional[dict] = None
    stop_line_m: Optional[float] = None
    light_id: Optional[int] = None
    world: object = None
    #: the next stop or give-way sign on the route: how far the bumper is from its line, what
    #: kind it is, and which one it is -- so a sign already stopped for is not stopped for
    #: twice (perception/road_signs.py)
    sign_m: Optional[float] = None
    sign_kind: Optional[str] = None
    sign_at: Optional[tuple] = None
    #: where the next junction on the route begins and ends (planner.junction_span)
    junction_span: Optional[tuple] = None
    #: what the light asks: worked out by the crosser rule, read by the light rule below it
    light: Optional[tuple] = None


@dataclass
class BehaviorOutput:
    """What the behavior layer decided to do and WHY."""
    behavior: DrivingBehavior = DrivingBehavior.IDLE
    reason: str = ""                    # THE KEY FIELD — human-readable explanation
    why: str = ""                       # ...and the same thing as one of transitions.ALL_WHY
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
        self.current_why = NO_MISSION
        self._rule_now = (0, "")         # which rule is being asked (RULES), for the record
        #: every change of state, in order, each with one reason code (P3)
        self.transitions = TransitionLog()
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
        # Traffic lights: the stop line comes from the map, measured from the FRONT BUMPER
        # (see traffic_lights.stop_line_for_lane). The choice to stop or go is made once each
        # time a light stops being green, and kept until it is green again.
        self.front_offset_m = DEFAULT_FRONT_OFFSET_M
        self._light_key = None               # which light the choice below is about
        self._light_choice = None            # ("stop"|"go", why), or None while green / no light
        self.light_status = None             # for the telemetry
        self.junction_dwell_s = 1.5          # mandatory look time even if clear
        self.junction_conflict_radius_m = 25.0
        self.junction_wait_timeout_s = 12.0  # then creep instead of deadlocking
        self.junction_creep_mps = 2.0
        self._junction_wait_started = None
        self._junction_done = False          # cleared for the junction we're in
        #: room the van wants beyond a junction before it enters: its own length and a bit
        self.keep_clear_m = 8.0
        self._sign_done = None               # the sign whose line we have already stopped at
        self._sign_still_since = None        # when the van came to rest at that line
        self._park_best_d = None             # closest approach to the parking spot
        self.block_release_s = 2.0           # blocked verdicts must stay clear this long before moving again
        self._block_memory = None            # (t_last_blocked, kind, distance)
        # ...but the latch only ARMS once a blocker has been there a moment. It exists for a
        # blocker that BLINKS OUT of detection, and it was arming on one that blinked IN.
        #
        # Measured live in Town10HD on 2026-09-10, empty road: a phantom block lasted FOUR
        # frames; a real cone in the lane lasted 385, which is every frame it was in front of
        # the van. Four frames of blocking bought two seconds of standing still, so one
        # flicker cost about twenty-two frames -- and 62% of the van's obstacle-stopped time
        # was spent in this latch with nothing blocking at all.
        #
        # This costs nothing in safety. The van STOPS on the very first blocked frame either
        # way; all this decides is whether it keeps standing after the path is clear again.
        # A real blocker earns that within two thirds of a second, while it is already stopped.
        self.block_latch_after_s = 0.6
        self._block_run_since = None         # when the current run of blocked frames began
        # Slowing for something in sight is steadied: once slowing, keep slowing until it is
        # well beyond the line, or a moment has passed. Live on 2026-09-11 one object crossing
        # the 20 m line made the van change its mind nine times in a minute.
        #
        # Nothing steadies a BLOCK the same way, on purpose. Holding a stop after the path is
        # clear was measured on 2026-09-10 and cost 62% of the van's obstacle-stopped time
        # with nothing in front of it (see the release latch above, which only arms for a
        # blocker that has been there 0.6 s). A brief stop for a brief block is honest: when
        # one thing is called in the way, then in sight, then in the way -- three times in the
        # last 15 m of a pull-in on 2026-09-11 -- the fault is in the corridor check that
        # keeps changing its mind, not in the behaviour that believes it.
        self.slow_release_m = 4.0
        self.slow_hold_s = 1.5
        self._slowing_since = 0.0
        self.destination_threshold = 1.5 # meters — parked when this close to the SPOT (was 5.0 anywhere on the road)
        self.parked_max_speed = 0.8      # ...and slower than this
        self.park_zone_m = 15.0          # final approach: taper to walking pace

        # If the path stays blocked for this long,
        # treat it as a blocked route instead of a temporary obstacle.
        self.blocked_timeout = 3.0
        self._blocked_since = None
        self._speed_cap_mps = None       # set each tick by safety (day 8)

    def update(
        self,
        perception: PerceptionOutput,
        pose: Pose,
        destination_distance: Optional[float],
        safety_ok: bool,
        junction: Optional[dict] = None,
        park_heading_ok: bool = True,
        park_position_ok: bool = True,
        predicted_conflict: Optional[dict] = None,
        stop_line_m: Optional[float] = None,     # front bumper to the light's stop line, along the route
        light_id: Optional[int] = None,          # which light that is
        world=None,                      # the day-7 world model, when the caller has one
        speed_cap_mps: Optional[float] = None,   # safety's cap while a sense is missing (day 8)
        blind_spot_m: Optional[float] = None,    # how near the nearest unseen pocket is (day 12)
        speed_limit_mps: Optional[float] = None,  # the limit on this piece of road, from the map
        seen_ahead_m: Optional[float] = None,     # how far the laser has seen the road ahead FREE
        over_the_kerb: bool = False,              # the fitted kerb line is under the van's path
        sign_m: Optional[float] = None,           # front bumper to the next sign's line, by road
        sign_kind: Optional[str] = None,          # "stop" or "give_way" (perception/road_signs)
        sign_at: Optional[tuple] = None,          # which sign that is: (road, lane, x, y)
        junction_span: Optional[tuple] = None,    # (metres to the next junction, to its far side)
    ) -> BehaviorOutput:
        """One decision cycle: ask the rules in RULES, in order, until one answers.

        Your rover did:
            if front < 20: STOP
            elif front < 50: TURN
            else: FORWARD

        This does the same thing but with richer states, always a reason, and the order of
        the questions written down (RULES) instead of left to the order the lines happen to
        sit in the file.
        """
        self._speed_cap_mps = speed_cap_mps
        self._blind_spot_m = blind_spot_m
        self._speed_limit_mps = speed_limit_mps
        self._seen_ahead_m = seen_ahead_m
        self._over_the_kerb = bool(over_the_kerb)
        now = Situation(sign_m=sign_m, sign_kind=sign_kind, sign_at=sign_at,
                        junction_span=junction_span,
                        perception=perception, pose=pose,
                        destination_distance=destination_distance, safety_ok=safety_ok,
                        junction=junction, park_heading_ok=park_heading_ok,
                        park_position_ok=park_position_ok, predicted_conflict=predicted_conflict,
                        stop_line_m=stop_line_m, light_id=light_id, world=world)
        for rank, (name, rule, _place) in enumerate(self.RULES, start=1):
            self._rule_now = (rank, name)
            answer = rule(self, now)
            if answer is not None:
                return answer
        raise AssertionError("the last rule always answers")   # pragma: no cover

    # ---------------------------------------------------------------- the rules, in order

    def _rule_safety(self, now):
        if not now.safety_ok:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                "Safety supervisor commanded stop",
                speed=0.0, stop=True, why=SAFETY_HOLD
            )
        return None

    def _rule_no_mission(self, now):
        if not self.has_mission:
            return self._decide(
                DrivingBehavior.NO_MISSION,
                "No mission assigned — waiting for destination",
                speed=0.0, stop=True, why=NO_MISSION
            )
        return None

    def _rule_localization(self, now):
        pose = now.pose
        if not pose.healthy or pose.quality == LocalizationQuality.LOST:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Localization unhealthy: {pose.reason}",
                speed=0.0, stop=True, why=LOCALIZATION_LOST
            )
        return None

    def _rule_perception(self, now):
        if not now.perception.healthy:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Perception unhealthy: {now.perception.reason}",
                speed=0.0, stop=True, why=PERCEPTION_LOST
            )
        return None

    def _rule_parked(self, now):
        """Is the mission over? Asked before anything about the road: a van that is parked is
        parked, whatever is standing beside the spot."""
        destination_distance, pose = now.destination_distance, now.pose
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
                speed=0.0, stop=True, why=PARKED_OVERSHOT
            )

        # --- Parked at the spot (close, nearly stopped, straight, IN the box) ---
        if (destination_distance is not None
                and destination_distance < self.destination_threshold
                and pose.speed < self.parked_max_speed
                and (now.park_heading_ok or destination_distance < 0.5)
                and (now.park_position_ok or destination_distance < 0.30)):
            self.mission_complete = True
            self.has_mission = False
            return self._decide(
                DrivingBehavior.MISSION_COMPLETE,
                f"Parked — {destination_distance:.1f} m from the spot",
                speed=0.0, stop=True, why=PARKED
            )
        return None

    def _rule_blocked_too_long(self, now):
        """A pedestrian or stopped vehicle is a temporary road situation, not automatically a
        blocked route. Only static/other obstacles can become a persistent blocked road."""
        perception = now.perception
        if (
            perception.path_blocked
            and perception.closest_obstacle_type
            not in (
                ObjectType.PEDESTRIAN,
                ObjectType.CYCLIST,     # a cyclist will move on; the road is not blocked
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
                    stop=True, why=ROUTE_BLOCKED_TOO_LONG
                )
        else:
            self._blocked_since = None
        return None

    def _rule_vru_in_path(self, now):
        """A person on foot or on a bike: ALWAYS stop."""
        perception = now.perception
        if perception.path_blocked and perception.closest_obstacle_type in VULNERABLE_TYPES:
            self._note_block(DrivingBehavior.STOPPED_PEDESTRIAN,
                             perception.closest_obstacle_distance)
            who = perception.closest_obstacle_type.value.upper()
            return self._decide(
                DrivingBehavior.STOPPED_PEDESTRIAN,
                f"{who} in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True, why=VRU_IN_PATH
            )
        return None

    def _rule_vehicle_in_path(self, now):
        perception = now.perception
        if perception.path_blocked and perception.closest_obstacle_type == ObjectType.VEHICLE:
            self._note_block(DrivingBehavior.STOPPED_VEHICLE,
                             perception.closest_obstacle_distance)
            return self._decide(
                DrivingBehavior.STOPPED_VEHICLE,
                f"VEHICLE blocking path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True, why=VEHICLE_IN_PATH
            )
        return None

    def _rule_obstacle_in_path(self, now):
        perception = now.perception
        if perception.path_blocked:
            self._note_block(DrivingBehavior.STOPPED_OBSTACLE,
                             perception.closest_obstacle_distance)
            return self._decide(
                DrivingBehavior.STOPPED_OBSTACLE,
                f"OBSTACLE in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True, why=OBSTACLE_IN_PATH
            )
        return None

    def _rule_confirming_clear(self, now):
        """A close blocker that BLINKS out of detection for a moment must not release the van
        instantly. In a dense-traffic brawl the verdict flapped every 1-3 s and the van crept
        half a metre per blink into a shrinking gap (two contacts). Stay stopped until the
        path has been continuously clear for block_release_s."""
        # nothing is blocking this frame, so the run of blocked frames is over
        self._block_run_since = None
        if (self._block_memory is not None
                and now.pose.speed < 1.2
                and time.time() - self._block_memory[0] < self.block_release_s):
            kind, dist = self._block_memory[1], self._block_memory[2]
            return self._decide(
                kind,
                f"Path just cleared (was blocked {dist:.1f}m ahead) — confirming for "
                f"{self.block_release_s:.0f}s before moving",
                speed=0.0, stop=True, why=CONFIRMING_CLEAR
            )
        self._block_memory = None
        return None

    def _rule_predicted_crosser(self, now):
        """Someone OUTSIDE our lane is about to be IN it (crosser at a junction, cut-in from
        the side). Yield before the danger exists instead of braking when it does. Ranked
        below physical blocks (a real thing in the path always wins) and above the light.

        What the light asks is worked out HERE, before the yield, and handed to the light rule
        below: this branch used to return "slowing to 2.5 m/s" without ever reaching the light,
        so a crosser predicted at a junction let the van roll through a red one.
        """
        now.light = self._traffic_light(now.perception, now.pose, now.stop_line_m, now.light_id)
        predicted_conflict, light = now.predicted_conflict, now.light
        if predicted_conflict is not None:
            p_t = predicted_conflict.get("t", 0.0)
            p_along = predicted_conflict.get("along_m", 0.0)
            p_what = predicted_conflict.get("type", "vehicle")
            if p_along < 14.0 or p_t < 1.1:
                return self._decide(
                    DrivingBehavior.YIELDING_PREDICTED,
                    f"Yielding — {p_what} will cross our path {p_along:.0f}m ahead in {p_t:.1f}s",
                    speed=0.0, stop=True, why=PREDICTED_CROSSER_STOP
                )
            if light is not None and (light[3] or light[2] < 2.5):
                return self._decide(*light)       # the stricter of the two wins
            return self._decide(
                DrivingBehavior.YIELDING_PREDICTED,
                f"Slowing — {p_what} predicted in our path {p_along:.0f}m ahead in {p_t:.1f}s",
                speed=2.5, stop=False, why=PREDICTED_CROSSER_SLOW
            )
        return None

    def _rule_traffic_light(self, now):
        """Roll up to the stop line, hold there (Troy #1). Ranked below pedestrian/vehicle/
        obstacle stops (a closer physical hazard always wins) and above following/cruising.
        See _traffic_light, which the rule above has already asked."""
        if now.light is not None:
            return self._decide(*now.light)
        return None

    def _rule_road_sign(self, now):
        """A stop or give-way sign from the map (perception/road_signs.py).

        A stop sign asks for a FULL stop at its line -- not a slow roll -- and then the
        junction rule below gives way as usual. A give-way asks for a crawl at the line and
        the same looking. A sign is stopped for ONCE: `sign_at` says which one it is, so the
        van does not stop again for the same sign as it creeps over the line.
        """
        if now.sign_m is None or now.sign_kind is None:
            self._sign_done = None if now.sign_at is None else self._sign_done
            return None
        if now.sign_at is not None and self._sign_done == now.sign_at:
            return None                               # already stopped for this one
        d, speed = float(now.sign_m), float(getattr(now.pose, "speed", 0.0) or 0.0)
        if now.sign_kind == GIVE_WAY:
            if d > SIGN_STOP_GAP_M:
                creep = max(1.0, min(self.slow_speed, 0.45 * d))
                return self._decide(DrivingBehavior.WAITING_AT_JUNCTION,
                                    f"Give-way line in {d:.1f} m — slowing to look",
                                    speed=creep, stop=False, why=SIGN_GIVE_WAY)
            self._sign_done = now.sign_at             # the junction rule looks from here on
            return None
        if d > SIGN_STOP_GAP_M + SIGN_STOP_ROLL_M:
            creep = max(0.6, min(3.0, 0.45 * (d - SIGN_STOP_GAP_M)))
            return self._decide(DrivingBehavior.WAITING_AT_JUNCTION,
                                f"STOP sign in {d:.1f} m — rolling up to the line",
                                speed=creep, stop=False, why=SIGN_ROLL_UP)
        # at the line: a full stop, held long enough to be a stop and not a hesitation
        if speed > SIGN_STOPPED_MPS:
            self._sign_still_since = None
        elif self._sign_still_since is None:
            self._sign_still_since = time.time()
        waited = 0.0 if self._sign_still_since is None else time.time() - self._sign_still_since
        if waited >= SIGN_DWELL_S:
            self._sign_done = now.sign_at
            return None                               # stopped properly: the junction decides
        return self._decide(DrivingBehavior.WAITING_AT_JUNCTION,
                            f"STOP sign — stopped at the line ({waited:.1f} s of {SIGN_DWELL_S:.0f})",
                            speed=0.0, stop=True, why=SIGN_STOP_HOLD)

    def _rule_junction_box(self, now):
        """Never enter a junction the van cannot clear.

        A van that stops inside a junction blocks everyone crossing it, and there is no way out
        but forward. So: about to enter, and the first thing in the way sits BEYOND the far
        side but nearer than the van's own length past it -- wait at the line instead.
        """
        span = now.junction_span
        if span is None:
            return None
        entry_m, exit_m = float(span[0]), float(span[1])
        if entry_m > JUNCTION_ENTER_WITHIN_M:
            return None                           # not about to enter one
        blocker = getattr(now.perception, "closest_obstacle_distance", None)
        if blocker is None or blocker > exit_m + self.keep_clear_m or blocker <= entry_m:
            return None                           # nothing beyond it, or it is on this side
        return self._decide(
            DrivingBehavior.WAITING_AT_JUNCTION,
            f"Junction ahead is {exit_m - entry_m:.0f} m across and the way out is blocked "
            f"{blocker:.1f} m on — waiting on this side of it",
            speed=0.0, stop=True, why=JUNCTION_KEEP_CLEAR)

    def _rule_junction(self, now):
        """Give way before turning at a junction."""
        perception, junction = now.perception, now.junction
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
                    speed=creep, stop=False, why=JUNCTION_ROLL_UP
                )
            moment = time.time()
            if self._junction_wait_started is None:
                self._junction_wait_started = moment
            waited = moment - self._junction_wait_started
            conflict = self._junction_conflict(perception, now.world,
                                               ego_yaw_rad=getattr(now.pose, "yaw", None))
            if waited >= self.junction_wait_timeout_s:
                self._junction_done = True
                self._junction_wait_started = None
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Give-way timeout at {direction} turn ({waited:.0f}s) — proceeding carefully",
                    speed=self.junction_creep_mps, stop=False, why=JUNCTION_TIMEOUT
                )
            if waited < self.junction_dwell_s:
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Approaching {direction} turn — pausing to check for traffic",
                    speed=0.0, stop=True, why=JUNCTION_PAUSE
                )
            if conflict is not None:
                return self._decide(
                    DrivingBehavior.WAITING_AT_JUNCTION,
                    f"Giving way at {direction} turn — moving vehicle {conflict:.0f} m away",
                    speed=0.0, stop=True, why=JUNCTION_GIVE_WAY
                )
            self._junction_done = True
            self._junction_wait_started = None
            print(f"[Behavior] Junction clear after {waited:.1f}s — taking the {direction} turn")
        elif self._junction_wait_started is not None and (
                junction is None or junction.get("distance_m", 99) > self.junction_stop_within_m):
            self._junction_wait_started = None
        return None

    def _rule_following_lead(self, now):
        """A moving vehicle ahead: follow at a time gap instead of stop-and-go."""
        perception = now.perception
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
                speed=target, stop=False, why=FOLLOWING_LEAD
            )
        return None

    def _rule_object_ahead(self, now):
        """Something in sight but not in the way: slow down.

        Once slowing, keep slowing until it is well past the line (slow_release_m) or a moment
        has gone by (slow_hold_s). One object crossing the 20 m line back and forth made the
        van change its mind nine times in a minute, 4.0 -> 2.0 -> 4.0 m/s each time."""
        seen = now.perception.closest_obstacle_distance
        near = seen < self.slow_distance
        if near:
            self._slowing_since = time.time()
        elif not (seen < self.slow_distance + self.slow_release_m
                  and time.time() - self._slowing_since < self.slow_hold_s):
            return None
        return self._decide(
            DrivingBehavior.FOLLOWING_ROUTE,
            f"Object detected at {seen:.1f}m — slowing to {self.slow_speed:.1f} m/s",
            speed=self.slow_speed, stop=False, why=OBJECT_AHEAD_SLOW
        )

    def _rule_parking(self, now):
        """The final approach: park at the kerb.

        Asked BEFORE following a lead car and before slowing for something in sight, because
        the van in the last metres of a pull-in is parking, whatever else it can see. Live on
        2026-09-11 those two rules kept taking the state away from it: through the last 15 m
        the van flapped parking -> object_ahead_slow -> stopped_obstacle -> parking five
        times, and the log of the drive said nothing true about what it was doing. It still
        never goes FASTER for that: anything in sight holds it to the slowing speed, and
        anything in the WAY is three rules above this one and stops it."""
        destination_distance = now.destination_distance
        if destination_distance is not None and destination_distance < self.park_zone_m:
            creep = max(0.5, min(2.5, 0.35 * destination_distance))
            seen = now.perception.closest_obstacle_distance
            if seen is not None and seen < self.slow_distance:
                creep = min(creep, self.slow_speed)
                return self._decide(
                    DrivingBehavior.PARKING,
                    f"Parking — pulling over, {destination_distance:.1f} m to the spot "
                    f"(something in sight at {seen:.1f} m: {creep:.1f} m/s)",
                    speed=creep, stop=False, why=PARKING_PULL_IN
                )
            return self._decide(
                DrivingBehavior.PARKING,
                f"Parking — pulling over, {destination_distance:.1f} m to the spot",
                speed=creep, stop=False, why=PARKING_PULL_IN
            )
        return None

    def _rule_approaching(self, now):
        destination_distance = now.destination_distance
        if destination_distance is not None and destination_distance < 25.0:
            return self._decide(
                DrivingBehavior.APPROACHING_DESTINATION,
                f"Approaching destination ({destination_distance:.1f}m) — slowing",
                speed=self.slow_speed, stop=False, why=DESTINATION_NEAR
            )
        return None

    def _rule_cruise(self, now):
        """Nothing to report: drive. The last rule, and the only one that always answers."""
        return self._decide(
            DrivingBehavior.FOLLOWING_ROUTE,
            f"Route clear — cruising at {self.cruise_speed:.1f} m/s",
            speed=self.cruise_speed, stop=False, why=ROUTE_CLEAR
        )

    #: What the van may be doing, in the order the questions are asked. The first rule with
    #: an answer wins and the rest are never asked, so this list IS the priority order --
    #: written down here instead of left to the order the lines happen to sit in the file.
    #: (name, the rule, why it sits where it does)
    RULES = (
        ("safety", _rule_safety, "the supervisor's stop beats every rule of the road"),
        ("no_mission", _rule_no_mission, "nowhere to go: stay put"),
        ("localization", _rule_localization, "not knowing WHERE it is beats knowing what it sees"),
        ("perception", _rule_perception, "blind is stopped"),
        ("parked", _rule_parked, "a van at its spot is parked, whatever stands beside it"),
        ("blocked_too_long", _rule_blocked_too_long, "a road blocked this long is an operator's problem"),
        ("vru_in_path", _rule_vru_in_path, "a person or a rider in the way beats everything below"),
        ("vehicle_in_path", _rule_vehicle_in_path, "then a vehicle in the way"),
        ("obstacle_in_path", _rule_obstacle_in_path, "then anything else in the way"),
        ("confirming_clear", _rule_confirming_clear, "a blocker that blinked out of sight is still there"),
        ("predicted_crosser", _rule_predicted_crosser,
         "what is ABOUT to be in the way, before it is -- and the light is read here, so a "
         "crosser cannot hide a red one"),
        ("traffic_light", _rule_traffic_light, "the law, once nothing physical is in the way"),
        ("road_sign", _rule_road_sign, "...and the signs painted on it, which say to stop or "
         "to give way before the junction rule below looks at all"),
        ("junction_box", _rule_junction_box, "never enter a junction the van cannot clear"),
        ("junction", _rule_junction, "give way before turning across traffic"),
        ("parking", _rule_parking, "the last few metres to the spot: nothing in sight takes "
         "the state away from a van that is parking (it still slows for it)"),
        ("following_lead", _rule_following_lead, "a moving car ahead is followed, not stopped for"),
        ("object_ahead", _rule_object_ahead, "something in sight but not in the way: slow down"),
        ("approaching", _rule_approaching, "slow down near the destination"),
        ("cruise", _rule_cruise, "nothing to report: drive"),
    )

    def _traffic_light(self, perception, pose, stop_line_m, light_id):
        """What the light ahead asks of the van: (behaviour, reason, speed, stop, why), or None when
        it asks nothing -- no light, a green one, or the van is committed to going through.

        Where to stop. `stop_line_m` is from the FRONT BUMPER to the lane's stop line in the
        map. It used to be the zebra crossing or the junction edge, measured from the middle
        of the van: the zebra starts up to 3.9 m past the paint, and the middle of the van
        held 2.6 m short of it put the bumper 0.35 m past even that. Measured
        against CARLA on 2026-09-11 the van crossed the stop line on red at three lights out
        of four, stopping up to 3.8 m over it.

        When to go on through. Decided ONCE, the first moment a light is not green, and kept
        until it is green again:
          * already over the line   -> go on and clear the junction
          * yellow, and too close to stop braking at COMFORT_DECEL_MPS2 -> go on through
          * anything else           -> stop, and stay stopped
        Deciding once is what keeps a creep a few centimetres over the line at a red from
        ever turning into "committed, carry on". The old rule -- carry on through any colour
        once within 1 m of the junction edge -- let the van do exactly that.
        """
        state = perception.traffic_light
        d = stop_line_m
        if d is None and perception.traffic_light_distance_m is not None:
            d = perception.traffic_light_distance_m - self.front_offset_m
        if light_id != self._light_key:
            self._light_key, self._light_choice = light_id, None
        if state not in LIGHT_MEANS_STOP:
            self._light_choice = None            # green, or no light: the next change is new
            self.light_status = None if state in (None, "none") else {
                "light_id": light_id, "state": state, "stop_line_m": d, "choice": "go",
                "why": "green"}
            return None
        speed = max(0.0, float(getattr(pose, "speed", 0.0) or 0.0))
        too_close_to_stop = (state == "yellow" and d is not None
                             and speed * speed / (2.0 * COMFORT_DECEL_MPS2) > d)
        if self._light_choice is None:
            if d is not None and d < 0.0:
                self._light_choice = ("go", f"already {-d:.1f} m over the stop line when it changed")
            elif too_close_to_stop:
                self._light_choice = ("go", f"turned yellow {d:.1f} m from the line at "
                                            f"{speed:.1f} m/s, too close to stop")
            else:
                self._light_choice = ("stop", "can stop before the line")
        elif self._light_choice[0] == "go" and d is not None and d > 0.0 and not too_close_to_stop:
            # Going is kept only while it is still TRUE, and only until the line: measured live
            # on 2026-09-11, the van decided to go on a yellow at 4 m/s, then slowed to 1.9 m/s
            # behind traffic and crossed the paint of light 10 on RED with 3 m of stopping
            # distance in hand, because the decision was made once and never looked at again.
            # A "stop" is still never re-opened -- that is what stops a creep a few centimetres
            # over the line from becoming "committed, carry on" -- and neither is a "go" once
            # the van is actually past the line, where the only safe way out is forward.
            self._light_choice = ("stop", f"slowed to {speed:.1f} m/s: it can stop before the "
                                          f"line after all")
        choice, why = self._light_choice
        self.light_status = {"light_id": light_id, "state": state, "stop_line_m": d,
                             "choice": choice, "why": why}
        if choice == "go":
            return None
        words = _light_words(state)
        if d is not None and d > LIGHT_STOP_GAP_M + LIGHT_STOP_ROLL_M:
            creep = max(0.6, min(3.0, 0.45 * (d - LIGHT_STOP_GAP_M)))
            return (DrivingBehavior.FOLLOWING_ROUTE,
                    f"{words} ahead ({d:.1f} m to the stop line) — rolling up", creep, False,
                    LIGHT_ROLL_UP)
        return (DrivingBehavior.STOPPED_RED_LIGHT,
                f"{words} — holding short of the stop line, waiting for green", 0.0, True,
                LIGHT_HOLD)

    def _note_block(self, kind, distance):
        """Remember a CLOSE physical blocker so a one-tick detection blink
        cannot release the van instantly (release latch above).

        Only once it has been there for block_latch_after_s: see the note on that setting.
        A thing that appears for four frames and vanishes has not earned a two-second hold.
        """
        if distance is None or distance >= 12.0:
            self._block_run_since = None
            return
        now = time.time()
        if self._block_run_since is None:
            self._block_run_since = now
        if now - self._block_run_since >= self.block_latch_after_s:
            self._block_memory = (now, kind, distance)

    def _decide(self, behavior, reason, speed, stop, why) -> BehaviorOutput:
        # Every cap can only ever slow the van down, never speed it up, and none of them can
        # turn a stop into driving. The tightest one wins.
        caps = []
        # Safety's cap while a sense is missing (Perception V2 day 8).
        sensor_cap = getattr(self, "_speed_cap_mps", None)
        if sensor_cap is not None:
            caps.append((float(sensor_cap), "a sensor is missing"))
        # Never go faster than you could stop in the distance to the nearest place you
        # cannot see into (Perception V2 day 12).
        blind = getattr(self, "_blind_spot_m", None)
        if blind is not None:
            caps.append((stopping_speed_for(float(blind)),
                         f"cannot see past {float(blind):.1f} m beside the lane"))
        # ...and the same rule for the road AHEAD: unseen ground is not free ground, so never
        # travel faster than you could stop inside what the laser has actually seen empty
        # (Planning V2, P2: the half of unknown_space that does something).
        seen = getattr(self, "_seen_ahead_m", None)
        if seen is not None:
            caps.append((stopping_speed_for(float(seen)),
                         f"the road is only seen clear for {float(seen):.1f} m"))
        # About to put a wheel over the kerb the laser fitted: down to a crawl, so the
        # steering has time to bring the van back before its body is over it.
        if getattr(self, "_over_the_kerb", False):
            caps.append((KERB_CRAWL_MPS, "the kerb is under where the van would be"))
        # The limit on this piece of road, from the map. Not a cap that can be argued with.
        limit = getattr(self, "_speed_limit_mps", None)
        if limit is not None:
            caps.append((float(limit), f"the limit here is {float(limit) * 3.6:.0f} km/h"))
        if caps:
            cap, capped_by = min(caps, key=lambda cw: cw[0])
            if speed > cap:
                speed = max(0.0, cap)
                reason = f"{reason} (held to {speed:.1f} m/s: {capped_by})"
                if speed == 0.0:
                    stop = True
        # Every change of state, with its reason code, kept in order (P3): one line of a
        # drive's story. Same state and same reason next tick is not a change.
        rank, rule = getattr(self, "_rule_now", (0, ""))
        change = self.transitions.note(was=self.current_behavior.value, now=behavior.value,
                                       why=why, said=reason, speed_mps=speed, stopping=stop,
                                       rule=rule, rank=rank)
        if change is not None and change.was != change.now:
            print(f"[Behavior] {change.was} → {change.now} ({why}): {reason}")
        self.current_behavior = behavior
        self.current_reason = reason
        self.current_why = why
        return BehaviorOutput(
            behavior=behavior,
            reason=reason,
            why=why,
            desired_speed_mps=speed,
            should_stop=stop
        )

    def _junction_conflict(self, perception: PerceptionOutput, world=None, ego_yaw_rad=None):
        """Distance of the nearest moving vehicle that is going to CROSS our turn, or None.

        Vehicles directly ahead in our own lane are the car-following problem, not a junction
        conflict; far, parked and well-behind ones are ignored -- and, given the van's heading,
        so are the ones driving AWAY (world_model.crossing_vehicles).

        Asked of the world model (Perception V2 day 7) when one is supplied; the loop below
        stays for callers that still hand over a bare PerceptionOutput, and it is the old
        radius answer: it has no velocities to judge a crossing with.
        """
        if world is not None:
            crossing = world.crossing_vehicles(self.junction_conflict_radius_m,
                                               ego_yaw_rad=ego_yaw_rad)
            return crossing[0].distance_m if crossing else None
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
        self._light_key, self._light_choice, self.light_status = None, None, None
        self.current_behavior = DrivingBehavior.IDLE

    def cancel_mission(self):
        self.has_mission = False
