"""A pass is drawn on the road, not on the parking tail the route was cut to.

The end of the route is bent to a parking spot and cut there. The ROAD goes on: _route_base
is the route as planned before any pull-in was drawn on it, carried PARK_FAR_PAST_PIN_M past
the pin.

Live in E3 on 2026-09-15 the van stood 280 s behind a car 7.7 m in front of it. All five ways
round were refused, sixteen weighings out of seventeen, with the same line: "only 19.0 m of
route remains, 26.7 m required (blocker at 7.7 m + 16 m to rejoin + 3 m)". Nineteen metres
because its own parking slot was nineteen metres away -- not because the street ended. The
van refused to go round a car because of where it had decided to park.

A pass that finishes past the spot is one the van can make. It parks further on afterwards,
which is what a driver does: get past the obstruction first, then find somewhere to stop.
"""
import test_go_around_diagnostics as G
from warp_av.planning.planner import RoutePlanner


def van(route_points, base_points, **kw):
    """The E3 shape: a short route cut to a parking spot, on a road that carries on."""
    s = G.van(route=G.road(points=route_points), **kw)
    s._route_base = list(G.road(points=base_points).waypoints) if base_points else None
    s._pass_past_the_spot = False          # WarpAV.__init__ sets this; the namespace must too
    return s


def codes(s):
    return {r["option"]: r["reason_code"] for r in (s._go_around or {}).get("options", [])}


# ---- the bug, reproduced then fixed ---------------------------------------------------

def test_a_route_cut_short_for_parking_no_longer_refuses_the_pass():
    s = van(route_points=14, base_points=60)          # 26 m of route, 118 m of road
    assert s.planner.route_left_m(s._route, 0.0, 0.0) < G.LEAD_D + 16.0 + 3.0, \
        "the route really is too short on its own"
    s.go()
    assert s._go_around["taken"] == "NUDGE_LEFT", codes(s)
    assert s._overtake_point is not None
    assert s._pass_past_the_spot is True, "and the van knows it will end past the spot"


def test_without_the_fix_that_same_route_is_the_old_refusal():
    """No road beyond the tail: exactly the behaviour E3 showed, unchanged."""
    s = van(route_points=14, base_points=None)
    s.go()
    assert set(codes(s).values()) == {"ROUTE_TOO_SHORT"}
    assert s._overtake_point is None


def test_the_check_is_not_deleted_only_asked_of_the_road():
    """A road that is genuinely too short is still too short. The van is not simply allowed
    to swing out wherever it likes."""
    s = van(route_points=14, base_points=15)          # 26 m of route, 28 m of road
    s.go()
    assert set(codes(s).values()) == {"ROUTE_TOO_SHORT"}
    assert s._overtake_point is None and s._pass_past_the_spot is False


def test_the_boundary_is_the_rejoin_distance_and_nothing_else():
    """Blocker 10 m out: a pass needs 10 + 16 + 3 = 29 m. The road decides."""
    needs = G.LEAD_D + RoutePlanner.OVERTAKE_REJOIN_M + 3.0
    assert abs(needs - 29.0) < 1e-6
    short = van(route_points=14, base_points=15)      # 28 m of road: one metre short
    short.go()
    assert short._overtake_point is None
    just_enough = van(route_points=14, base_points=17)  # 32 m of road
    just_enough.go()
    assert just_enough._overtake_point is not None


# ---- and the ordinary case is untouched ----------------------------------------------------

def test_a_route_long_enough_on_its_own_never_looks_at_the_road_behind_it():
    s = van(route_points=60, base_points=60)
    s.go()
    assert s._go_around["taken"] == "NUDGE_LEFT"
    assert s._pass_past_the_spot is False, "the parking tail was never left"


def test_the_route_the_van_follows_is_the_one_the_pass_was_drawn_on():
    s = van(route_points=14, base_points=60)
    s.go()
    assert len(s._route.waypoints) == 60, "the pass runs on the road, so the route does too"
    ends_at = s._route.waypoints[-1].x
    assert ends_at > 26.0, "past where the parking tail used to stop it"


def test_nothing_is_taken_when_every_option_is_refused_for_another_reason():
    """The road being long enough is not permission: the lane checks still decide."""
    s = van(route_points=14, base_points=60, lane_ok=False, same_way_ok=False, shoulder_ok=False)
    s.go()
    assert s._overtake_point is None
    assert "ROUTE_TOO_SHORT" not in set(codes(s).values()), "the road was long enough"


def test_the_rejoin_distance_itself_did_not_move():
    """This fix asks the question of a longer road. It does not shorten the rejoin, which is
    what would have made the swing sharper."""
    assert RoutePlanner.OVERTAKE_REJOIN_M == 16.0
    assert RoutePlanner.OVERTAKE_PASS_M == 8.0


def test_the_flag_only_says_what_happened_and_never_decides_anything():
    """Two runs, identical but for the road behind the tail: the verdicts differ only where
    the old rule would have refused."""
    with_road = van(route_points=14, base_points=60)
    without = van(route_points=14, base_points=None)
    with_road.go()
    without.go()
    assert with_road._go_around["taken"] == "NUDGE_LEFT"
    assert without._go_around["taken"] is None
    long_route = van(route_points=60, base_points=60)
    long_route.go()
    assert long_route._go_around["taken"] == "NUDGE_LEFT", "unchanged where it always worked"
