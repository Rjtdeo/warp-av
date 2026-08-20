"""
Troy #7: park at the kerb, inside a box, aligned — not mid-road on the pin.
"""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput
from warp_av.localization.localization import Pose
from warp_av.control.controller import VehicleController
from test_controller_stability import SimVan, DT


def planner():
    return RoutePlanner.__new__(RoutePlanner)   # no CARLA: geometric fallback path


def straight_route(n=60):
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(n)])


def test_pullover_bends_tail_to_the_right():
    p = planner()
    r = straight_route()
    end_before = (r.waypoints[-1].x, r.waypoints[-1].y)
    spot = p.apply_pullover(r)
    assert spot is not None and spot["offset_m"] > 0.5
    last = r.waypoints[-1]
    # ends exactly at the spot, offset to the RIGHT (CARLA frame: +y at yaw 0)
    assert math.isclose(last.x, spot["x"], abs_tol=0.01) and math.isclose(last.y, spot["y"], abs_tol=0.01)
    assert last.y > 0.5 and abs(last.x - end_before[0]) < 3.0
    # ramp is monotone (no swing left before pulling right)
    ys = [w.y for w in r.waypoints[-8:]]
    assert all(b >= a - 0.05 for a, b in zip(ys, ys[1:]))
    # heading stays the road heading
    assert abs(last.yaw - 0.0) < 0.01


def test_twisty_tail_parks_before_the_bend_not_across_it():
    p = planner()
    wps = [Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(10)]
    for i in range(1, 8):   # sharp turn right up to the destination
        a = (math.pi / 2) * i / 7
        wps.append(Waypoint(x=20 + 6 * math.sin(a), y=6 - 6 * math.cos(a), yaw=a))
    r = Route(waypoints=wps)
    spot = p.apply_pullover(r)
    assert spot is not None and spot["moved_back_m"] > 0
    last = r.waypoints[-1]
    assert last.x <= 20.5, "route must end on the straight, before the corner"
    assert last.y > 0.5, "pulled to the right at the pre-corner spot"
    # the ramp must not contain the corner's curvature
    assert all(abs(w.yaw) < 0.05 for w in r.waypoints[-5:])


def test_behavior_parking_taper_and_completion():
    b = BehaviorSystem(); b.set_mission()
    moving = Pose(healthy=True); moving.speed = 2.0
    out = b.update(PerceptionOutput(), moving, 12.0, True)
    assert out.behavior == DrivingBehavior.PARKING
    assert 0.7 <= out.desired_speed_mps <= 2.5 and "Parking" in out.reason
    # close but still rolling: NOT complete yet
    out = b.update(PerceptionOutput(), moving, 1.2, True)
    assert out.behavior != DrivingBehavior.MISSION_COMPLETE
    # close and crawling: parked
    slow = Pose(healthy=True); slow.speed = 0.3
    out = b.update(PerceptionOutput(), slow, 1.2, True)
    assert out.behavior == DrivingBehavior.MISSION_COMPLETE and "Parked" in out.reason


def test_closed_loop_parks_in_the_box():
    p = planner()
    route = straight_route()
    spot = p.apply_pullover(route)
    assert spot is not None
    van = SimVan(speed=6.0)
    ctrl = VehicleController()
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    done_reason = ""
    for _ in range(int(60 / DT)):
        last = route.waypoints[-1]
        dest_d = math.hypot(van.x - last.x, van.y - last.y)
        pose.speed = van.speed
        out = b.update(PerceptionOutput(), pose, dest_d, True)
        if out.behavior == DrivingBehavior.MISSION_COMPLETE:
            done_reason = out.reason
            break
        la = max(5.0, min(13.0, 1.6 * van.speed))
        ct = p.signed_cross_track(route, van.x, van.y)
        if abs(ct) > 1.0:
            la = min(la, 6.0)
        wp = p.get_next_waypoint(route, van.x, van.y, lookahead=la)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed,
                                   wp.x, wp.y, out.desired_speed_mps, out.should_stop,
                                   cross_track_m=ct)
        van.step(cmd)
    assert done_reason, "never completed the parking approach"
    # in the box: within 1.5 m of the kerbside spot, roughly road-aligned
    err = math.hypot(van.x - spot["x"], van.y - spot["y"])
    herr = abs((van.yaw - spot["yaw"] + math.pi) % (2 * math.pi) - math.pi)
    assert err < 1.5, f"stopped {err:.2f} m from the spot"
    assert herr < math.radians(15), f"parked {math.degrees(herr):.0f} deg off the road direction"
    assert van.y > 0.4, "did not actually pull over to the right"


def test_pin_in_a_bend_parks_before_it():
    # straight road, then the pin sits INSIDE a junction curve at the end
    p = planner()
    wps = [Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(25)]      # straight to x=48
    for i in range(1, 9):                                                # then a bend to the pin
        a = (math.pi / 2) * i / 8
        wps.append(Waypoint(x=48 + 8 * math.sin(a), y=8 - 8 * math.cos(a), yaw=a, is_junction=True))
    r = Route(waypoints=wps)
    spot = p.apply_pullover(r)
    assert spot is not None, "must park before the bend, not give up"
    assert spot["moved_back_m"] > 5.0
    last = r.waypoints[-1]
    assert last.x < 50.0, "route must be truncated before the bend"
    assert last.y > 0.5, "still pulled over to the right at the new spot"


def test_endless_bend_still_refuses():
    p = planner()
    wps = [Waypoint(x=30 * math.sin(t / 40), y=30 - 30 * math.cos(t / 40),
                    yaw=t / 40, is_junction=True) for t in range(0, 80)]
    assert p.apply_pullover(Route(waypoints=wps)) is None
