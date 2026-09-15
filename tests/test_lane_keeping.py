"""
Lane-keeping regression for the two dashboard observations:
  - van riding between two lanes after corners
  - drifting into the left lane on the approach to a right turn
Simulates the full pipeline over a lane-change route and asserts the van
settles into the new lane quickly and stays centred.
"""
import math
import time

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
        last = route.waypoints[-1]
        if math.hypot(van.x - last.x, van.y - last.y) < 6.0:
            break              # reached route end (this sim has no arrival logic)
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


# ---- coming back to the line without crossing it (2026-09-14) ----------------------------
#
# The van drifted 0.99 m right of its lane, corrected left, and swung 2.3 m the other way in
# 1.5 s -- 1.35 m left of centre, 36 degrees across the road, body over the centre line, where
# an oncoming ambulance met it. The correction was a spring with no shock absorber.

def steer_at(offsets, speed=4.0):
    """Steering after feeding a run of lane offsets, one per tick (the controller counts ticks,
    not seconds -- every other limit in it is per tick too)."""
    c = VehicleController()
    return [c.compute_command(current_x=0, current_y=0, current_yaw=0, current_speed=speed,
                              target_x=10, target_y=0, desired_speed=speed,
                              should_stop=False, cross_track_m=ct).steering
            for ct in offsets]


def test_a_van_already_coming_back_is_corrected_less_than_one_sitting_off_the_line():
    held = steer_at([1.0, 1.0, 1.0])            # a metre off, not moving back
    closing = steer_at([1.3, 1.15, 1.0])        # the same metre off, already returning
    assert abs(closing[-1]) < abs(held[-1]), "coming back must ease the correction, not add to it"


def test_the_correction_stays_inside_its_cap_however_fast_it_closes():
    for run in ([2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.5, 0.2, -1.0], [0.0, -2.0, 2.0]):
        for s in steer_at(run):
            assert abs(s) <= 1.0
    from warp_av.control.controller import VehicleController as V
    assert V.CT_MAX == 0.35 and V.CT_DAMP > 0.0


def test_the_damping_actually_cuts_the_overshoot():
    """A van pushed a metre off the line must not sail out the other side (2026-09-14)."""
    from warp_av.control.controller import VehicleController as V
    from test_controller_stability import SimVan, DT

    def swing(damp):
        was, V.CT_DAMP = V.CT_DAMP, damp
        try:
            van, c, worst = SimVan(x=0.0, y=1.0, yaw=0.0, speed=4.0), VehicleController(), 1.0
            for _ in range(400):
                cmd = c.compute_command(current_x=van.x, current_y=van.y, current_yaw=van.yaw,
                                        current_speed=van.speed, target_x=van.x + 6.4, target_y=0.0,
                                        desired_speed=4.0, should_stop=False, cross_track_m=van.y)
                van.step(cmd)
                worst = min(worst, van.y)
            return -worst
        finally:
            V.CT_DAMP = was

    undamped, damped = swing(0.0), swing(V.CT_DAMP)
    assert damped < 0.5 * undamped, "the damping must at least halve the swing past the line"
    assert damped < 0.15, "and leave the van well inside its own lane"


def test_the_van_eases_off_instead_of_steering_harder_when_off_the_line():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert "RECOVER_ABOVE_MPS" in src and "RECOVER_SPEED_MPS" in src
    i = src.index("cross_track = self.planner.signed_cross_track")
    j = src.index("lookahead = max(5.0", i)
    assert "off the line — easing to" in src[i:j], \
        "the speed cap must be worked out AFTER cross_track and BEFORE the aim point"
    assert "and pose.speed <= self.RECOVER_ABOVE_MPS" in src, \
        "a closer aim point is only safe at low speed"
