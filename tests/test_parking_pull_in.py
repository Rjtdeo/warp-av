"""The way into a parking spot: gentle, from the van's own lane, and never "parked" across it.

Measured against CARLA on 2026-09-11 (route 78,139 -> -67.3,28.0): CARLA's route changed
from the left lane to the right one just before the pin, the pull-in ramp's window reached
back across that change, and one ramp had to move the van 6.5 m sideways in about 6 m of road
-- up to 57 degrees. The van cut in, its path met the pavement kerb, and the old give-up rule
finished the mission there: 39 degrees across the driving lane, 4.1 m of the van inside it,
11.6 m from the spot, reported as parked.
"""
import math

import numpy as np
import pytest

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.parking_check import spot_view, TAKEN, FREE, UNSEEN
from warp_av.perception.occupancy import OccupancyGrid
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput
from warp_av.localization.localization import Pose
from warp_av.control.controller import VehicleController
from test_controller_stability import SimVan, DT

LANE_1, LANE_2, BAY = 0.0, 3.5, 6.55      # centres across the road, as on Town10HD (CARLA: + is right)
BAY_WIDTH = 2.6


def planner(bay_from=60.0, bay_to=160.0):
    p = RoutePlanner.__new__(RoutePlanner)          # no CARLA: geometric fallbacks

    def right_bay(x, y, z):
        return (x, BAY, 0.0, BAY_WIDTH) if bay_from <= x <= bay_to else None
    p._right_bay = right_bay
    return p


def lane_changing_route(change_at=86.0, end=150.0):
    """Lane 1 until change_at, then lane 2 -- CARLA's route changes lane in one 2 m step."""
    wps = []
    x = 0.0
    while x <= end + 1e-6:
        wps.append(Waypoint(x=x, y=LANE_1 if x <= change_at else LANE_2, yaw=0.0))
        x += 2.0
    return Route(waypoints=wps)


def pin_at(route, x):
    return min(range(len(route.waypoints)), key=lambda i: abs(route.waypoints[i].x - x))


def steepest_deg(wps):
    return max(math.degrees(math.atan2(abs(b.y - a.y), b.x - a.x))
               for a, b in zip(wps, wps[1:]) if b.x > a.x)


# ---------------------------------------------------------------- the geometry of 2026-09-11

def test_the_pull_in_never_starts_before_the_lane_change():
    p, r = planner(), lane_changing_route()
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    assert spot is not None and spot["kind"] == "bay"
    assert spot["y"] == pytest.approx(BAY)
    assert spot["ramp_start"][1] == pytest.approx(LANE_2, abs=0.75), "the ramp began in the other lane"
    assert spot["offset_m"] == pytest.approx(BAY - LANE_2, abs=0.05)


def test_the_pull_in_is_gentle():
    p, r = planner(), lane_changing_route()
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    tail = [w for w in r.waypoints if w.x >= spot["ramp_start"][0]]
    assert steepest_deg(tail) <= p.PULL_IN_MAX_DEG + 0.5, "the old ramp reached 57 degrees here"


def test_with_no_room_before_the_pin_the_spot_is_a_little_past_it():
    """12 m of the right lane before the pin; a gentle 3 m move needs about 22 m."""
    p, r = planner(), lane_changing_route()
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    assert 0.0 < spot["past_pin_m"] <= p.PARK_PAST_PIN_M
    assert r.waypoints[-1].x == pytest.approx(spot["x"])


def test_with_room_before_the_pin_it_parks_at_the_pin():
    p, r = planner(), lane_changing_route(change_at=40.0)
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    assert spot["past_pin_m"] == 0.0 and abs(spot["x"] - 100.0) <= 2.0


def test_a_bay_two_lanes_over_is_never_cut_across_to():
    """The route never leaves lane 1: the strip is 6.55 m away, across a lane of traffic."""
    p = planner()
    r = Route(waypoints=[Waypoint(x=i * 2.0, y=LANE_1, yaw=0.0) for i in range(76)])
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    assert spot is None or spot["kind"] != "bay"
    assert spot is None or spot["y"] < LANE_1 + p.LANE_TO_SPOT_MAX_M


def test_a_stop_in_the_lane_is_straight_and_short_of_the_junction():
    """No strip, no kerb: the van stops in the lane -- on a straight stretch, never on a stop
    line, and never part-way through a lane change."""
    p = planner(bay_from=999.0, bay_to=999.0)
    wps = [Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(52)]
    for i in range(1, 30):                       # the road bends away through a junction after the pin
        a = i * 0.08
        wps.append(Waypoint(x=102 + 20 * math.sin(a), y=20 - 20 * math.cos(a), yaw=a, is_junction=True))
    r = Route(waypoints=wps)
    p._pullover_target = lambda last: (last.x, last.y, last.yaw, 0.0)      # no kerb either
    spot = p.apply_pullover(r, pin_index=51)
    assert spot is not None and spot["kind"] == "lane"
    first_junction = min(w.x for w in wps if w.is_junction)
    assert first_junction - spot["x"] >= p.LANE_STOP_CLEAR_M
    assert all(abs(w.y) < 1e-6 and w.yaw == 0.0 for w in r.waypoints[-6:]), "not straight at the end"


def test_a_pin_straight_after_a_lane_change_is_never_the_stop():
    """2026-09-11: the route changed lane 8 m before the pin and the van stopped there, 22
    degrees across the line. The stop moves to where the lane has straightened out."""
    p = planner(bay_from=999.0, bay_to=999.0)
    p._pullover_target = lambda last: (last.x, last.y, last.yaw, 0.0)
    r = lane_changing_route(change_at=94.0)      # pin at 100: 6 m into the new lane
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    assert spot is not None and spot["kind"] == "lane"
    assert abs(spot["x"] - 100.0) > 1.0, "stopped on the pin, part-way through the lane change"
    end = r.waypoints[-1]
    run_in = [w for w in r.waypoints if end.x - (p.LANE_STOP_STRAIGHT_M + p.PULL_IN_SETTLE_M) <= w.x <= end.x]
    assert len({w.y for w in run_in}) == 1, "the last stretch before the stop is not all one lane"


def test_nothing_workable_cuts_the_route_back_to_the_pin():
    p = planner(bay_from=999.0, bay_to=999.0)
    p._pullover_target = lambda last: (last.x, last.y, last.yaw, 0.0)
    wps = [Waypoint(x=30 * math.sin(t / 40), y=30 - 30 * math.cos(t / 40), yaw=t / 40)
           for t in range(0, 120)]                # a road that never stops bending
    r = Route(waypoints=wps)
    assert p.apply_pullover(r, pin_index=80) is None
    assert len(r.waypoints) == 81, "the part past the pin must be dropped again"


# ---------------------------------------------------------------- choosing again

def test_a_spot_turned_down_is_not_chosen_again():
    p = planner()
    first = p.apply_pullover(lane_changing_route(), pin_index=pin_at(lane_changing_route(), 100.0))
    r = lane_changing_route()
    again = p.apply_pullover(r, pin_index=pin_at(r, 100.0), avoid=[(first["x"], first["y"])])
    assert again is not None
    assert math.hypot(again["x"] - first["x"], again["y"] - first["y"]) >= 6.0


def test_a_pull_in_that_would_start_behind_the_van_is_not_chosen():
    p = planner()
    r = lane_changing_route()
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0), ahead_of=(104.0, LANE_2))
    assert spot is not None and spot["ramp_start"][0] > 104.0


def test_a_slot_is_entered_the_same_gentle_way():
    p = planner()
    r = lane_changing_route(change_at=40.0)
    got = p.retarget_to_slot(r, {"x": 100.0, "y": BAY, "yaw": 0.0, "length": 7.0, "width": BAY_WIDTH})
    assert got and got["approach_m"] > 20.0
    assert steepest_deg([w for w in r.waypoints if w.x >= 70.0]) <= p.PULL_IN_MAX_DEG + 0.5


def test_a_slot_with_no_gentle_way_in_is_refused():
    p = planner()
    r = lane_changing_route(change_at=94.0)      # only 6 m of the right lane before the slot
    assert not p.retarget_to_slot(r, {"x": 100.0, "y": BAY, "yaw": 0.0, "length": 7.0, "width": BAY_WIDTH})


def test_a_hold_short_point_in_the_lane_is_driven_to_straight():
    p = planner()
    r = lane_changing_route(change_at=40.0)
    got = p.retarget_to_slot(r, {"x": 90.0, "y": LANE_2, "yaw": 0.0})
    assert got is not None and got["approach_m"] == 0.0
    assert (r.waypoints[-1].x, r.waypoints[-1].y) == (90.0, LANE_2)


# ---------------------------------------------------------------- driving it

def test_the_van_drives_the_pull_in_and_ends_straight_inside_the_strip():
    p, r = planner(), lane_changing_route()
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    van = SimVan(x=40.0, y=LANE_1, yaw=0.0, speed=6.0)
    ctrl = VehicleController()
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    steepest, widest = 0.0, -99.0
    done = False
    for _ in range(int(90 / DT)):
        last = r.waypoints[-1]
        dest_d = math.hypot(van.x - last.x, van.y - last.y)
        pose.speed = van.speed
        herr = abs((van.yaw - spot["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        out = b.update(PerceptionOutput(), pose, dest_d, True, park_heading_ok=herr < math.radians(6))
        if out.behavior == DrivingBehavior.MISSION_COMPLETE:
            done = True
            break
        la = max(5.0, min(13.0, 1.6 * van.speed))
        ct = p.signed_cross_track(r, van.x, van.y)
        if abs(ct) > 1.0:
            la = min(la, 6.0)
        wp = p.get_next_waypoint(r, van.x, van.y, lookahead=la)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed, wp.x, wp.y,
                                   out.desired_speed_mps, out.should_stop, cross_track_m=ct)
        van.step(cmd)
        if van.x >= spot["ramp_start"][0]:       # the pull-in itself, not CARLA's lane change
            steepest = max(steepest, abs(math.degrees(van.yaw)))
            widest = max(widest, van.y)
    assert done, "never finished parking"
    assert abs(van.y - BAY) < 0.4, f"ended {van.y - BAY:+.2f} m off the strip's middle"
    assert abs(math.degrees(van.yaw)) < 6.0, "ended crooked"
    assert steepest <= 20.0, f"turned in at {steepest:.0f} degrees"
    assert widest + 1.0 <= BAY + BAY_WIDTH / 2.0, "swung out over the far kerb"


# ---------------------------------------------------------------- is the spot free? (LiDAR)

def swept_grid(extra=None, reach=25.0):
    """A LiDAR map around a van at the origin facing +x, every bearing seen to `reach` metres
    of open road, plus any `extra` solid returns (van frame: x forward, y right)."""
    ang = np.radians(np.arange(0.0, 360.0, 0.25))
    ground = np.stack([reach * np.cos(ang), reach * np.sin(ang)], axis=1)
    pts, occ = [ground], [np.zeros(len(ground), dtype=bool)]
    if extra is not None:
        pts.append(np.asarray(extra, dtype=float))
        occ.append(np.ones(len(extra), dtype=bool))
    return OccupancyGrid().update(np.concatenate(pts), np.concatenate(occ))


SPOT = {"x": 10.0, "y": 3.0, "yaw": 0.0, "length": 7.0, "width": 2.2}


def test_an_empty_spot_the_laser_swept_is_free():
    assert spot_view(swept_grid(), SPOT, 0.0, 0.0, 0.0) == FREE


def test_a_parked_car_in_it_is_taken():
    side = [(x, 2.5) for x in np.arange(7.5, 12.5, 0.1)]         # its near side, seen from the lane
    assert spot_view(swept_grid(side), SPOT, 0.0, 0.0, 0.0) == TAKEN


def test_the_kerb_along_its_far_edge_does_not_make_it_taken():
    kerb = [(x, 4.05) for x in np.arange(0.0, 20.0, 0.1)]         # 1.05 m out from its middle
    assert spot_view(swept_grid(kerb), SPOT, 0.0, 0.0, 0.0) == FREE


def test_a_spot_hidden_behind_something_is_unseen_not_free():
    wall = [(3.0, y) for y in np.arange(0.5, 6.0, 0.05)]           # blocks every beam towards it
    assert spot_view(swept_grid(wall), SPOT, 0.0, 0.0, 0.0) == TAKEN or \
        spot_view(swept_grid(wall), dict(SPOT, x=14.0), 0.0, 0.0, 0.0) == UNSEEN
    assert spot_view(swept_grid(wall), dict(SPOT, x=14.0), 0.0, 0.0, 0.0) != FREE


def test_a_spot_beyond_the_maps_reach_is_unseen():
    assert spot_view(swept_grid(), dict(SPOT, x=40.0), 0.0, 0.0, 0.0) == UNSEEN


def test_no_map_yet_is_unseen():
    assert spot_view(None, SPOT, 0.0, 0.0, 0.0) == UNSEEN
    assert spot_view(OccupancyGrid(), SPOT, 0.0, 0.0, 0.0) == UNSEEN


def test_the_spot_is_placed_by_the_vans_heading():
    """Van facing +y (CARLA yaw 90): a spot 10 m ahead and 3 m to its right is at (-3, 10)."""
    side = [(x, 2.5) for x in np.arange(7.5, 12.5, 0.1)]
    grid = swept_grid(side)
    turned = dict(SPOT, x=-3.0, y=10.0, yaw=math.pi / 2)
    assert spot_view(grid, turned, 0.0, 0.0, math.pi / 2) == TAKEN


# ---------------------------------------------------------------- the bus shelter (2026-09-11)

from warp_av.perception.motion_class import (vehicle_name_implausible, NOT_A_VEHICLE_MIN_HEIGHT_M,
                                             NOT_A_VEHICLE_MIN_LENGTH_M, NOT_A_VEHICLE_CLEAR_M)
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.footprint import VehicleFootprint


def test_a_shelter_on_the_pavement_is_no_vehicle():
    """Measured live: 6.7 m long, 2.9 m tall, about 0.4 m beyond the parking strip."""
    assert vehicle_name_implausible(2.87, 6.69, 0.4)


@pytest.mark.parametrize("height,longest,gap,why", [
    (2.6, 5.9, -1.0, "a van parked IN the strip"),
    (1.0, 1.7, 1.1, "a motorbike parked on the pavement"),
    (3.3, 1.9, 0.54, "a car glued to a pole"),
    (4.0, 3.9, 0.26, "a car glued to a pole and a sign, 0.26 m clear"),
    (2.9, 6.7, None, "no lane found near it"),
    (2.9, 6.7, 15.0, "too far from any road to matter"),
])
def test_what_a_vehicle_name_is_still_believed_for(height, longest, gap, why):
    assert not vehicle_name_implausible(height, longest, gap), why


@pytest.mark.parametrize("path", ["tests/fixtures/static_truth/town10_seed7.npz",
                                  "tests/fixtures/static_truth/town10_seed23_fresh.npz"])
def test_no_real_vehicle_in_either_recording_would_lose_its_name(path):
    """CARLA's answer key: not one real vehicle is that tall, that long and that far off the
    road -- and not with every threshold one step looser either."""
    import pathlib
    d = np.load(pathlib.Path(__file__).parent.parent / path, allow_pickle=True)
    real = np.isin(d["tag_name"], ["car", "truck", "bus", "motorcycle", "bicycle"])
    longest = np.maximum(d["length"], d["width"])
    for h, L, g in ((NOT_A_VEHICLE_MIN_HEIGHT_M, NOT_A_VEHICLE_MIN_LENGTH_M, NOT_A_VEHICLE_CLEAR_M),
                    (2.2, 2.8, 0.27)):
        gap = d["road_gap"]
        hit = real & np.isfinite(gap) & (gap >= g) & (gap <= 12.0) & (d["height"] >= h) & (longest >= L) & (d["n"] >= 3)
        assert int(hit.sum()) == 0, f"{int(hit.sum())} real vehicles at {h}/{L}/{g}"


def van():
    return VehicleFootprint(half_length=2.95, half_width=0.99, safety_margin=0.30)


def thing(ego_x, ego_y_right, length, width, height=2.9, kind=ObjectType.OBSTACLE, speed=0.0):
    return DetectedObject(object_type=kind, x=ego_x, y=ego_y_right, distance=math.hypot(ego_x, ego_y_right),
                          speed=speed, length_m=length, width_m=width, height_m=height,
                          box_length_m=length, box_width_m=width, box_yaw_deg=0.0, stationary=speed < 0.5)


class _Seen:
    def __init__(self, objects):
        self.objects = objects


def bay_route():
    """Straight in lane 2 (y = 3.5), then a gentle pull-in to the strip's middle (y = 6.55)."""
    p = planner()
    r = lane_changing_route(change_at=20.0)
    spot = p.apply_pullover(r, pin_index=pin_at(r, 100.0))
    return p, r, spot


def test_a_shelter_beside_the_spot_is_found_before_turning_in():
    """The van at the start of the pull-in; the shelter on the pavement beside the spot, its
    near side 0.63 m from where the van's side will stop (as measured live)."""
    p, r, spot = bay_route()
    ex, ey = spot["ramp_start"]
    shelter_y = BAY + 0.99 + 0.63 + 0.79            # world y (CARLA: + is right)
    # the van's frame: x forward, y to the RIGHT (as camera_lidar_perception reports objects)
    obj = thing(spot["x"] - ex, shelter_y - ey, 6.69, 1.58)
    got = p.pull_in_blocker(_Seen([obj]), r, ex, ey, 0.0, van())
    assert got is not None and got[0] is obj


def test_a_clear_pull_in_has_no_blocker():
    p, r, spot = bay_route()
    ex, ey = spot["ramp_start"]
    far_back = thing(spot["x"] - ex, BAY + 4.0 - ey, 6.69, 1.58)        # well back on the pavement
    assert p.pull_in_blocker(_Seen([far_back]), r, ex, ey, 0.0, van()) is None
    assert p.pull_in_blocker(_Seen([]), r, ex, ey, 0.0, van()) is None


def test_something_moving_is_left_to_the_running_check():
    p, r, spot = bay_route()
    ex, ey = spot["ramp_start"]
    walker = thing(spot["x"] - ex, BAY - ey, 0.5, 0.5, height=1.7, kind=ObjectType.PEDESTRIAN, speed=1.2)
    assert p.pull_in_blocker(_Seen([walker]), r, ex, ey, 0.0, van()) is None


# ---------------------------------------------------------------- by road, not as the crow flies

def test_a_route_round_the_block_is_far_from_its_end_until_it_gets_there():
    """2026-09-11: the parking check fired on the wrong street, 22.7 m from the spot in a
    straight line with about 250 m still to drive, and threw a good spot away."""
    p = planner()
    out_leg = [Waypoint(x=float(x), y=0.0, yaw=0.0) for x in range(0, 101, 2)]         # east 100 m
    up = [Waypoint(x=100.0, y=float(y), yaw=math.pi / 2) for y in range(2, 21, 2)]      # north 20 m
    back = [Waypoint(x=float(x), y=20.0, yaw=math.pi) for x in range(98, 39, -2)]       # west 60 m
    r = Route(waypoints=out_leg + up + back)                                              # ends at (40, 20)
    at_start_of_block = (40.0, 0.0)                                                       # 20 m from the end
    assert math.hypot(40.0 - 40.0, 20.0 - 0.0) == 20.0
    d = p.distance_to_destination(r, *at_start_of_block)
    assert d == pytest.approx(60.0 + 20.0 + 60.0, abs=1.0), "measured as the crow flies"


def test_at_the_end_and_past_it_the_straight_line_counts():
    p = planner()
    r = Route(waypoints=[Waypoint(x=float(x), y=0.0, yaw=0.0) for x in range(0, 41, 2)])
    assert p.distance_to_destination(r, 39.0, 0.3) == pytest.approx(1.04, abs=0.05)
    assert p.distance_to_destination(r, 43.0, 0.0) == pytest.approx(3.0, abs=0.01), "an overshoot must grow"
