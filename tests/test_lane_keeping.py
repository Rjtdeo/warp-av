"""
Lane-keeping regression for the two dashboard observations:
  - van riding between two lanes after corners
  - drifting into the left lane on the approach to a right turn
Simulates the full pipeline over a lane-change route and asserts the van
settles into the new lane quickly and stays centred.
"""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.control.controller import VehicleController
from test_controller_stability import SimVan, DT
from test_curve_speed import _dist_to_polyline


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def lane_change_route():
    """Straight in lane A, shift 3.5 m to lane B over 15 m, continue straight."""
    wps = [Waypoint(x=i * 2.0, y=0.0) for i in range(0, 20)]                    # to x=38
    for i in range(1, 9):
        wps.append(Waypoint(x=38 + i * 2.0, y=-3.5 * i / 8.0))                  # shift to y=-3.5 by x=54
    for i in range(1, 40):
        wps.append(Waypoint(x=54 + i * 2.0, y=-3.5))
    return Route(waypoints=wps)


def drive(route, start_y=0.0, speed=8.0, seconds=30):
    p = planner()
    van = SimVan(y=start_y, speed=speed)
    ctrl = VehicleController()
    traj = []
    for _ in range(int(seconds / DT)):
        cap = p.curve_speed_cap(route, van.x, van.y, cruise=8.0)
        la = max(5.0, min(13.0, 1.6 * van.speed))
        if cap < 7.5:
            la = min(la, 5.5)
        ct = p.signed_cross_track(route, van.x, van.y)
        if abs(ct) > 1.0:
            la = min(la, 6.0)
        wp = p.get_next_waypoint(route, van.x, van.y, lookahead=la)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed,
                                   wp.x, wp.y, min(8.0, cap), False, cross_track_m=ct)
        van.step(cmd)
        traj.append((van.x, van.y, cmd.steering))
        if van.x > 185.0:      # stop before the route ends (this sim has no arrival logic)
            break
    return traj


def test_completes_lane_change_promptly():
    route = lane_change_route()
    traj = drive(route)
    # by 20 m after the shift ends (x=74) the van must be centred in the NEW lane
    late = [(x, y) for x, y, _ in traj if x > 74.0]
    assert late, "van never reached the post-change section"
    worst = max(abs(y + 3.5) for x, y in late)
    assert worst < 0.5, f"still {worst:.2f} m off the new lane centre after the change (riding between lanes)"


def test_recovers_from_between_lanes_quickly():
    # van starts dead on the divider (1.75 m off centre) at cruise speed
    route = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(120)])
    traj = drive(route, start_y=1.75)
    back = next((x for x, y, _ in traj if abs(y) < 0.3), None)
    assert back is not None and back < 40.0, f"took {back} m to get back into the lane (limit 40 m)"
    # and stays there without weaving
    tail = [s for x, y, s in traj if x > 60.0]
    flips = sum(1 for a, b in zip(tail, tail[1:]) if a * b < 0 and abs(a - b) > 0.05)
    assert flips <= 4, f"weaving after recovery: {flips} steering flips"
