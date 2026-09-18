"""V2B: three failures V1 recorded live, replayed through the REAL decision chain offline on
the V2A harness (tests/test_v2a_end_to_end.py): the same WarpAV, built by its own __init__,
the same stubbed outside world, the recorded numbers as input.

  RC-1  a shoulder-parked car whose fitted box came out 2.56 m wide (truth 2.00 m) blocked
        the van on the swept path (V1.8 clear-road run, WAV-0001, t+60.1 s: "VEHICLE blocking
        path at 1.7 m", 0.86 m of real clearance).
  RC-4  "a go-around never ends" (13 of 56 V1 runs "ACTIVE_PASS" for minutes).
  RC-5  one 1.17 s tick made the pose look 1.2 s old and the supervisor paused the mission
        (V1 WAV-0272 run a: perception phase 1167 ms, "Localization stale (1.2 s)")."""
import math
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_v2a_end_to_end import (van, start_mission, place, see, chain, straight_road,  # noqa: F401
                                 StubMap)
from warp_av.localization.localization import Pose
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.planner import Route, Waypoint
from warp_av.planning.instrumentation import PATH_BLOCKED, BLOCKED_SWEPT_PATH
from warp_av.behavior.behavior import DrivingBehavior
from warp_av.behavior.transitions import GO_AROUND_START, GO_AROUND_DONE, SAFETY_HOLD
from warp_av.mission.mission_manager import MissionState
from warp_av.vehicle_interface import AutonomyState


# ================================================================ RC-1: the oversized parked-car box

def left_arc_road(radius_m=39.5, length_m=240.0, step_m=2.0):
    """The road the live van was on: a gentle left bend (the recorded intended path turned
    -1.45 deg/m), in the van's frame at the moment of the block."""
    pts = []
    s = -20.0
    while s <= length_m:
        a = s / radius_m
        pts.append(Waypoint(x=radius_m * math.sin(a), y=-radius_m * (1.0 - math.cos(a)),
                            yaw=-a, speed=8.0))
        s += step_m
    return Route(waypoints=pts, total_distance=length_m + 20.0)


def recorded_mercedes():
    """WAV-0001 (run v18_after) at t+60.1 s, exactly as perception handed it to the planner:
    the point centroid 1.88 m ahead and 2.54 m to the right, the points' spread 4.29 x 1.36 m,
    the fitted rectangle 4.50 x 2.56 m centred (0.32, 0.82) m from the centroid, heading 172 deg."""
    return DetectedObject(object_type=ObjectType.VEHICLE, x=1.88, y=2.54, distance=3.2, id=5573,
                          stationary=True, speed=0.0, length_m=4.29, width_m=1.36, height_m=1.14,
                          yaw_deg=-13.0, box_dx=0.32, box_dy=0.82,
                          box_length_m=4.5, box_width_m=2.56, box_yaw_deg=172.0,
                          motion_class="dynamic", size_uncertain=False, confidence=0.93)


def recorded_mercedes_fit_made_consistent():
    """The same sighting, the same 4.50 x 2.56 m rectangle at the same fitted centre -- but
    with the points' spread said to match it, so the planner believes the fit."""
    o = recorded_mercedes()
    o.length_m, o.width_m = 4.5, 2.56
    return o


def true_mercedes():
    """CARLA's Mercedes as it really stood (world.get_environment_objects, V3A): centre 2.57 m
    ahead, 2.98 m to the right, 4.67 x 2.00 m, 5.7 deg off the van's heading -- 0.8 m from
    the van's body."""
    return DetectedObject(object_type=ObjectType.VEHICLE, x=2.57, y=2.98, distance=3.94, id=5573,
                          stationary=True, speed=0.0, length_m=4.67, width_m=2.00, height_m=1.45,
                          yaw_deg=-5.7, box_dx=0.0, box_dy=0.0,
                          box_length_m=4.67, box_width_m=2.00, box_yaw_deg=-5.7,
                          motion_class="static", size_uncertain=False, confidence=0.93)


def rc1_chain(van, obj):
    start_mission(van, route=left_arc_road())
    place(van, x=0.0, y=0.0, yaw=0.0, speed=0.0)
    see(van, [obj])
    van.tick()
    c = chain(van)
    c["lateral_m"] = van._path.closest_lateral_m
    c["blocker"] = van._path.blocker_kind
    return c


def test_rc1_the_recorded_oversized_box_blocks_exactly_as_live(van):
    c = rc1_chain(van, recorded_mercedes())
    print("RC-1 recorded box:", c)
    assert c["planner_level"] == PATH_BLOCKED and c["planner_reason"] == BLOCKED_SWEPT_PATH, c
    assert c["blocker"] == "vehicle" and 1.0 <= c["closest_m"] <= 3.0, c        # live: 1.7 m
    assert abs(c["lateral_m"] - 2.55) < 0.15, c                                  # live: 2.55 m
    assert c["behaviour"] == DrivingBehavior.STOPPED_VEHICLE.value, c
    assert c["stop"] is True and c["throttle"] == 0.0 and c["brake"] == 1.0, c


def test_rc1_the_true_car_lets_the_van_go_on(van):
    c = rc1_chain(van, true_mercedes())
    print("RC-1 true car:", c)
    assert c["planner_level"] != PATH_BLOCKED, c
    assert c["stop"] is False and c["throttle"] > 0.0 and c["brake"] == 0.0, c


def test_rc1_the_block_comes_from_the_rejected_fit_not_from_the_width(van):
    """Where in the sighting the block comes from. The planner refuses a rectangle that claims
    much more ground than the points it was fitted to (fit_is_believable) and falls back to
    the points' own spread, centred on the centroid at the covariance heading. That spread box
    (4.29 x 1.36 m, heading -13 deg, 2.54 m off the line, widened by the vehicle floor and the
    heading tolerance) reaches 0.7 m from the line: inside the swept path. The very same
    oversized rectangle, believed, sits 0.8 m further out at its fitted centre and clears."""
    from warp_av.planning.planner import fit_is_believable, obstacle_box_for
    rec = recorded_mercedes()
    assert fit_is_believable(rec) is False
    box = obstacle_box_for(rec, 0.0)
    assert (round(box.dx, 2), round(box.dy, 2)) == (0.0, 0.0)          # on the centroid, not the fit
    assert abs(math.degrees(box.heading) - (-13.0)) < 0.1              # the points' axis, 17 deg off the truth
    assert box.half_width >= 1.25 and box.half_length >= 2.2            # the spread, floored and widened
    c_believed = rc1_chain(van, recorded_mercedes_fit_made_consistent())
    print("RC-1 same rectangle, believed:", c_believed)
    assert c_believed["planner_level"] != PATH_BLOCKED and c_believed["throttle"] > 0.0, c_believed


# ================================================================ RC-4: does a pass end?

class TwoLaneMap(StubMap):
    """A straight road in the van's frame: our lane |y| <= 1.75, a lane running OUR way on the
    left (y in [-5.25, -1.75]; left is -y here), kerb on the right. Answers the three map
    questions the go-around asks and nothing else."""
    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        y = float(loc.y)
        if -5.25 <= y <= 1.75:
            lane = -1 if y >= -1.75 else -2
            return SimpleNamespace(road_id=1, lane_id=lane, lane_width=3.5, is_junction=False,
                                   transform=SimpleNamespace(location=SimpleNamespace(x=loc.x, y=(0.0 if lane == -1 else -3.5), z=0.0),
                                                             rotation=SimpleNamespace(yaw=0.0)))
        return None


def dead_car_world(x=10.3, y=0.3):
    return (x, y)


def car_seen_from(van_pose, world_xy):
    """The dead car as perception hands it over, in the van's frame at this pose."""
    px, py = world_xy
    dx, dy = px - van_pose.x, py - van_pose.y
    c, s = math.cos(van_pose.yaw), math.sin(van_pose.yaw)
    ex, ey = dx * c + dy * s, -dx * s + dy * c
    return DetectedObject(object_type=ObjectType.VEHICLE, x=ex, y=ey, distance=math.hypot(ex, ey),
                          id=30414, stationary=True, speed=0.0, length_m=4.5, width_m=1.9, height_m=1.5,
                          yaw_deg=-math.degrees(van_pose.yaw), box_length_m=4.5, box_width_m=1.9,
                          box_yaw_deg=-math.degrees(van_pose.yaw), motion_class="static")


def route_point_at(route, arc_m):
    """(x, y, yaw) of the route `arc_m` metres along from its first point."""
    wps = route.waypoints
    s = 0.0
    for a, b in zip(wps, wps[1:]):
        seg = math.hypot(b.x - a.x, b.y - a.y)
        if s + seg >= arc_m:
            t = (arc_m - s) / seg if seg else 0.0
            return a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), math.atan2(b.y - a.y, b.x - a.x)
        s += seg
    return wps[-1].x, wps[-1].y, wps[-1].yaw


def test_rc4_a_lane_pass_starts_ends_at_the_rejoin_point_and_the_diagnostic_never_says_so(van, monkeypatch):
    import carla
    monkeypatch.setattr(carla, "LaneType", SimpleNamespace(Driving="driving"), raising=False)
    van.vehicle_adapter.get_map = lambda: TwoLaneMap()
    start_mission(van)
    car = dead_car_world()
    # T0: a dead car in the lane, 10 m ahead -> the swept path is blocked, the van stops
    place(van, x=0.0, speed=3.0)
    see(van, [car_seen_from(van.pose_source.current, car)])
    van.tick()
    c0 = chain(van)
    assert c0["planner_reason"] == BLOCKED_SWEPT_PATH and c0["behaviour"] == DrivingBehavior.STOPPED_VEHICLE.value, c0
    assert van._overtake_point is None
    # standing at it for the 10 s the go-around waits before weighing the ways round
    place(van, x=0.0, speed=0.0)
    for _ in range(3):
        van.clock.advance(3.6)
        see(van, [car_seen_from(van.pose_source.current, car)])
        van.tick()
    # T1: the REAL go-around accepted a way round and rewrote the route
    assert van._overtake_point is not None, van._go_around
    assert van._go_around["taken"] == "LANE_LEFT", van._go_around
    rejoin = van._overtake_point
    lead_d = van._go_around["blocker_distance_m"]
    recent = van._current_state["behavior_changes"]["recent"]
    assert any(n["why"] == GO_AROUND_START for n in recent), recent
    assert abs(rejoin.y) < 0.05 and rejoin.x == pytest.approx(lead_d + 16.0, abs=2.5), (rejoin, lead_d)
    passing = [wp for wp in van._route.waypoints if wp.y < -3.0]
    assert passing, "the rewritten route moves a lane over to the left"
    # T2..T4: drive the rewritten route past the car and back, one tick per 2 m
    ended_at, states = None, []
    for arc in range(2, 40, 2):
        x, y, yaw = route_point_at(van._route, 20.0 + arc)       # the route starts 20 m behind the van
        place(van, x=x, y=y, yaw=yaw, speed=3.0)
        van.clock.advance(0.7)
        see(van, [car_seen_from(van.pose_source.current, car)])
        van.tick()
        c = chain(van)
        states.append((arc, round(y, 2), c["planner_reason"], c["behaviour"], van._overtake_point is not None))
        if van._overtake_point is None and ended_at is None:
            ended_at = arc
    print("RC-4 pass:", states)
    assert ended_at is not None, states                                  # the pass ENDS
    assert abs(ended_at - rejoin.x) <= 4.0, (ended_at, rejoin.x)          # ...within 4 m of the rejoin point
    recent = van._current_state["behavior_changes"]["recent"]
    assert any(n["why"] == GO_AROUND_DONE for n in recent), recent
    assert van._current_state["overtaking"] is False
    # the recorded artefact: the diagnostic record still says the pass is active, and will
    # until something weighs a way round again -- V1 counted THIS as "a go-around never ends"
    assert van._go_around["taken"] == "LANE_LEFT" and van._go_around["gate"]["reason_code"] == "ACTIVE_PASS", van._go_around
    for arc in (40, 60, 80):
        x, y, yaw = route_point_at(van._route, 20.0 + arc)
        place(van, x=x, y=y, yaw=yaw, speed=5.0)
        van.clock.advance(0.5)
        see(van)
        van.tick()
    assert van._current_state["overtaking"] is False and chain(van)["behaviour"] == DrivingBehavior.FOLLOWING_ROUTE.value
    assert van._go_around["gate"]["reason_code"] == "ACTIVE_PASS", van._go_around   # 60 m on, still "active"


# ================================================================ RC-5: one long tick

def slow_perception_tick(van, delay_s=1.17):
    """One tick whose perception phase takes `delay_s`: the pose is read at the top, then the
    clock moves, then perception hands over a FRESH picture, then the safety check runs."""
    out = van.perception.output

    def slow_update():
        van.clock.advance(delay_s)
        out.timestamp = time.time()
        return out
    van.perception.update = slow_update
    tick_start = time.time()
    van.tick()
    van.perception.update = lambda: van.perception.output
    return tick_start


def test_rc5_one_slow_perception_phase_reads_as_stale_localization(van):
    """The recorded failure, reproduced: the age the supervisor measures is the tick's own
    length. Before the V2B fix this paused the mission and disengaged autonomy for an
    operator; now it is a stop for this tick, and the same mission goes on when the next
    tick's pose is fresh."""
    start_mission(van)
    for i in range(3):                                                    # a normal 4 Hz run-up
        place(van, x=2.0 * i, speed=6.0)
        van.clock.advance(0.25)
        see(van)
        van.tick()
    assert chain(van)["safety_state"] == "ok"
    place(van, x=8.0, speed=6.0)
    van.clock.advance(0.25)
    see(van)
    tick_start = slow_perception_tick(van, 1.17)
    c = chain(van)
    pose = van.localization._last_pose
    print("RC-5:", c, "pose.timestamp - tick start =", round(pose.timestamp - tick_start, 3),
          "age at the safety check =", round(time.time() - pose.timestamp, 3))
    assert pose.healthy is True and pose.reason == "OK"                    # localization itself was fine
    assert pose.timestamp == pytest.approx(tick_start, abs=1e-6)          # stamped at the top of the tick
    assert time.time() - pose.timestamp == pytest.approx(1.17, abs=0.01)  # the age IS the tick's own length
    assert van._health.by_name("position").healthy is True                # the health monitor: fine
    # the supervisor still refuses this tick, and the van still stops for it
    assert c["safety_state"] == "intervention" and "Localization stale" in c["safety_reason"], c
    assert van._last_safety_output.transient is True
    assert c["behaviour"] == DrivingBehavior.STOPPED_SAFETY.value and c["why"] == SAFETY_HOLD, c
    assert c["brake"] == 1.0 and c["throttle"] == 0.0, c
    # the fix: a stale READING is not a failed PART -- the mission is not paused for it
    assert c["mission"] == MissionState.EXECUTING.value and c["autonomy"] == AutonomyState.AUTONOMOUS.value, c
    # ...and the next ordinary tick, with a fresh pose, drives on in the same mission
    place(van, x=8.5, speed=0.5)
    van.clock.advance(0.25)
    see(van)
    van.tick()
    c2 = chain(van)
    assert c2["safety_state"] == "ok" and c2["driving_allowed"] is True, c2
    assert c2["mission"] == MissionState.EXECUTING.value and c2["autonomy"] == AutonomyState.AUTONOMOUS.value, c2
    assert c2["behaviour"] == DrivingBehavior.FOLLOWING_ROUTE.value and c2["throttle"] > 0.0, c2


def test_rc5_a_pose_source_that_fails_still_pauses_the_mission(van):
    """Not weakened: a source that says it has failed is a failed part, not a stale reading."""
    from warp_av.localization.localization import LocalizationQuality
    start_mission(van)
    place(van, x=0.0, speed=6.0)
    see(van)
    van.tick()
    place(van, x=1.5, speed=6.0, healthy=False, reason="LOCALIZATION_LOST",
          confidence=0.0, quality=LocalizationQuality.LOST)
    see(van)
    van.tick()
    c = chain(van)
    assert c["safety_state"] == "intervention" and van._last_safety_output.transient is False, c
    assert c["mission"] == MissionState.PAUSED.value and c["autonomy"] == AutonomyState.DISENGAGED.value, c
    assert c["brake"] == 1.0, c


def test_rc5_the_supervisor_marks_only_its_age_verdicts_transient():
    from warp_av.safety.safety_supervisor import SafetySupervisor
    now = time.time()
    fresh = dict(perception_healthy=True, perception_timestamp=now, localization_healthy=True,
                 localization_confidence=1.0, localization_timestamp=now, controller_healthy=True,
                 vehicle_alive=True, current_speed=5.0)
    assert SafetySupervisor().update(**fresh).transient is False
    stale_pose = SafetySupervisor().update(**{**fresh, "localization_timestamp": now - 1.2})
    assert stale_pose.driving_allowed is False and stale_pose.transient is True and "Localization stale" in stale_pose.reason
    stale_view = SafetySupervisor().update(**{**fresh, "perception_timestamp": now - 1.2})
    assert stale_view.driving_allowed is False and stale_view.transient is True and "Perception data stale" in stale_view.reason
    dead = SafetySupervisor().update(**{**fresh, "localization_healthy": False})
    assert dead.driving_allowed is False and dead.transient is False
    low = SafetySupervisor().update(**{**fresh, "localization_confidence": 0.1})
    assert low.driving_allowed is False and low.transient is False


def test_rc5_the_age_check_cannot_see_a_pose_source_that_has_stopped_updating(van):
    """The other side of the same coin: the timestamp is the tick's own clock, so a source
    that keeps returning the same old pose passes the staleness check for ever."""
    start_mission(van)
    frozen = Pose(x=5.0, y=0.0, yaw=0.0, speed=6.0)
    van.pose_source.pose = lambda: frozen                                 # never moves again
    for _ in range(8):
        van.clock.advance(0.5)
        see(van)
        van.tick()
    c = chain(van)
    print("RC-5 frozen source after 4 s:", c)
    assert c["safety_state"] == "ok" and c["driving_allowed"] is True, c
    assert time.time() - van.localization._last_pose.timestamp < 0.01
