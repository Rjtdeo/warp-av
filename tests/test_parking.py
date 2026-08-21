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
        herr_now = abs((van.yaw - spot["yaw"] + math.pi) % (2 * math.pi) - math.pi)
        out = b.update(PerceptionOutput(), pose, dest_d, True,
                       park_heading_ok=herr_now < math.radians(15))
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
    assert herr < math.radians(10), f"parked {math.degrees(herr):.0f} deg off the road direction"
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


def test_bay_preferred_over_kerb_hug(monkeypatch):
    """When a stopping bay exists to the right, park fully in it (kind=bay)."""
    p = planner()
    r = straight_route()
    bay_at = {"x": None}

    def fake_bay(x, y, z):
        # bay exists only in the stretch x in [90, 110]: centre 2.8 m right
        if 90 <= x <= 110:
            bay_at["x"] = x
            return (x, 2.8, 0.0, 2.5)
        return None

    monkeypatch.setattr(p, "_right_bay", fake_bay)
    spot = p.apply_pullover(r)
    assert spot is not None and spot["kind"] == "bay"
    assert spot["offset_m"] > 2.0, "must move fully off the driving lane into the bay"
    last = r.waypoints[-1]
    assert abs(last.y - 2.8) < 0.05 and 90 <= last.x <= 112
    # ramp is monotone toward the bay
    ys = [w.y for w in r.waypoints[-6:]]
    assert all(b >= a - 0.05 for a, b in zip(ys, ys[1:]))


def test_no_bay_falls_back_to_kerb_hug(monkeypatch):
    p = planner()
    r = straight_route()
    monkeypatch.setattr(p, "_right_bay", lambda x, y, z: None)
    spot = p.apply_pullover(r)
    assert spot is not None and spot["kind"] == "kerb"
    assert 0.5 <= spot["offset_m"] <= 1.5


def test_ramp_ends_with_a_straight_section():
    p = planner()
    r = straight_route()
    spot = p.apply_pullover(r)
    assert spot is not None
    tail = r.waypoints[-4:]
    ys = [w.y for w in tail]
    # last few metres hold the full lateral offset (parallel to the lane line)
    assert max(ys) - min(ys) < 0.08, f"no straight-in section: {ys}"


def test_completion_waits_for_straight_heading():
    b = BehaviorSystem(); b.set_mission()
    slow = Pose(healthy=True); slow.speed = 0.3
    out = b.update(PerceptionOutput(), slow, 1.2, True, park_heading_ok=False)
    assert out.behavior != DrivingBehavior.MISSION_COMPLETE, "must straighten before declaring parked"
    out = b.update(PerceptionOutput(), slow, 0.4, True, park_heading_ok=False)
    assert out.behavior == DrivingBehavior.MISSION_COMPLETE   # dead on the spot: accept
    b2 = BehaviorSystem(); b2.set_mission()
    out = b2.update(PerceptionOutput(), slow, 1.2, True, park_heading_ok=True)
    assert out.behavior == DrivingBehavior.MISSION_COMPLETE


# ---------------- FIND PARKING: slots, occupancy geometry, retarget ----------------

def _mock_bays(p, y=2.8, width=2.5, span=(100.0, 150.0)):
    def fake_bay(x, _y, _z):
        if span[0] <= x <= span[1]:
            return (x, y, 0.0, width)
        return None
    p._right_bay = fake_bay


def test_slots_are_sliced_along_the_bay():
    p = planner()
    r = straight_route(80)          # 158 m route along x
    _mock_bays(p)                   # 50 m of bay inside the 70 m search window -> ~7 slots
    slots = p.find_parking_slots(r)
    assert 5 <= len(slots) <= 7
    assert all(abs(s["y"] - 2.8) < 0.01 and s["width"] == 2.5 for s in slots)
    xs = [s["x"] for s in slots]
    assert xs == sorted(xs), "slots must be ordered along the route (far -> near destination)"
    c = slots[0]["corners"]
    assert len(c) == 4 and abs(max(p_[0] for p_ in c) - min(p_[0] for p_ in c) - 7.0) < 0.1


def test_point_in_slot_and_van_in_slot():
    slot = {"x": 10.0, "y": 2.8, "yaw": 0.0, "length": 7.0, "width": 2.5, "corners": []}
    P = RoutePlanner
    assert P.point_in_slot(11.0, 3.0, slot)
    assert not P.point_in_slot(15.0, 3.0, slot)
    inside, m_along, m_side = P.van_in_slot(10.0, 2.8, 0.0, 2.9, 1.0, slot)
    assert inside and abs(m_along - 0.6) < 0.01 and abs(m_side - 0.25) < 0.01
    inside, m_along, _ = P.van_in_slot(13.0, 2.8, 0.0, 2.9, 1.0, slot)
    assert not inside and m_along < 0


def test_retarget_to_slot_ends_route_in_the_slot():
    p = planner()
    r = straight_route(80)
    slot = {"x": 100.0, "y": 2.8, "yaw": 0.0, "length": 7.0, "width": 2.5}
    assert p.retarget_to_slot(r, slot)
    last = r.waypoints[-1]
    assert abs(last.x - 100.0) < 0.2 and abs(last.y - 2.8) < 0.05
    assert abs(last.yaw) < 0.01
    ys = [w.y for w in r.waypoints[-4:]]
    assert max(ys) - min(ys) < 0.15, "must arrive straight into the slot"


def test_no_slots_on_curved_bay_sections():
    p = planner()
    # bay following a bend: quarter-circle R=20 -> every slot spans >8 deg -> none allowed
    def curved_bay(x, y, z):
        return None
    p._right_bay = curved_bay
    run = [(20 * math.sin(t / 20), 20 - 20 * math.cos(t / 20), 0.0, 2.5)
           for t in range(0, 30, 2)]
    assert p._slice_run_into_slots(run) == []
    # and a straight run of the same length still yields slots
    straight = [(float(t), 2.8, 0.0, 2.5) for t in range(0, 30, 2)]
    assert len(p._slice_run_into_slots(straight)) == 4


def test_no_slots_near_junctions():
    p = planner()
    _mock_bays(p, span=(100.0, 150.0))
    r = straight_route(80)
    for i, w in enumerate(r.waypoints):
        if 118.0 <= w.x <= 126.0:          # junction in the middle of the bay
            w.is_junction = True
    slots = p.find_parking_slots(r)
    assert slots, "straight sections should still produce slots"
    for sl in slots:
        assert not (112.0 <= sl["x"] <= 132.0), f"slot at {sl['x']} too close to the junction"
