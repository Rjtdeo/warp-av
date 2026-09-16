"""When may the van go round a whole blocked road?

The map knows other streets. Whether one of them is usable is plan_route_avoiding's answer;
whether the van could take it at all is a cheaper question asked first, and until 2026-09-15
that question was impossible to answer yes to.

It asked for a junction nearer than the blocker. But a road only counts as blocked once the
van has stopped at it, and by then the blocker is a metre or two away -- so the test wanted a
junction under the van's own wheels. Live in F_reroute on 2026-09-15 the reroute logic was
reached 19 times over a 300 s drive with the blocker 1.1 to 2.6 m ahead, and refused all 19
times at that one gate. Every other gate passed.

Now the junction is looked for from where the van could still turn off FROM. It backs out up
to REVERSE_MAX_M (_start_backing_out), so a turn-off it has just crept past is still one it
can reach; a junction BEYOND the blockage is not, because it cannot drive through it.

These tests drive the gate rather than reading it. test_reroute.py keeps the source-text
pins; the behaviour is here.
"""
import time
from types import SimpleNamespace

from warp_av.main import WarpAV
from warp_av.planning.planner import Route, RoutePlanner, Waypoint

BLOCKED_WHY = WarpAV.BLOCKED_REASONS[0]          # whatever "something is in the way" is called


def road(n=80, step=2.0, junction_at=None, junction_m=6.0):
    """A straight road from (0, 0) along +x, optionally with a junction on it."""
    wps = []
    for i in range(n):
        x = i * step
        wps.append(Waypoint(x=x, y=0.0, yaw=0.0,
                            is_junction=bool(junction_at is not None
                                             and junction_at <= x < junction_at + junction_m)))
    return Route(waypoints=wps, total_distance=(n - 1) * step)


def van(route=None, van_x=20.0, blocker_m=1.5, blocked_for_s=30.0, asked_ago_s=60.0,
        blocked=True, alternate=None, overtaking=False):
    """A WarpAV with only the parts _maybe_reroute touches, and the real method bound to it."""
    s = SimpleNamespace()
    s.said, s.logged, s.asked = [], [], []
    for k in ("BLOCKED_REASONS", "REROUTE_AFTER_S", "REROUTE_EVERY_S", "REROUTE_WITHIN_M",
              "REVERSE_MAX_M", "REROUTE_LOG_EVERY_S", "REROUTE_LOOK_FAR_M"):
        setattr(s, k, getattr(WarpAV, k))
    s.REROUTE_LOG_EVERY_S = 0.0                  # in a test every change is written
    now = time.time()
    s._route = route if route is not None else road()
    s._overtake_point = object() if overtaking else None
    s._blocked_road_since = None if blocked_for_s is None else now - blocked_for_s
    s._reroute_asked_at = now - asked_ago_s
    s._reroute = None
    s._signal_lookahead = None
    # _maybe_reroute reads exactly these two fields off the path record
    s._path = SimpleNamespace(blocked=blocked, closest_distance_m=blocker_m)
    s.mission_manager = SimpleNamespace(
        current_mission=SimpleNamespace(destination_x=400.0, destination_y=0.0))
    s.planner = RoutePlanner.__new__(RoutePlanner)        # real junction_span, no CARLA

    def avoiding(sx, sy, ex, ey, ax, ay, clear_m=4.0, why=None):
        s.asked.append((round(sx, 2), round(sy, 2), round(ax, 2), round(ay, 2)))
        if alternate is None and why is not None:
            why["code"], why["text"] = "NO_ALTERNATE_ROUTE", "no other way"
        return alternate

    s.planner.plan_route_avoiding = avoiding
    s.logger = SimpleNamespace(log_event=lambda *a, **k: s.logged.append((a, k)))
    s._note_move = lambda what, text: s.said.append((what, text))
    s._dress_route_for_parking = lambda: s.said.append(("dressed", ""))
    s._record_reroute = lambda *a, **k: WarpAV._record_reroute(s, *a, **k)
    s.pose = SimpleNamespace(x=van_x, y=0.0, yaw=0.0, speed=0.0)
    s.beh = SimpleNamespace(why=BLOCKED_WHY if blocked else "clear")
    s.go = lambda: WarpAV._maybe_reroute(s, s.pose, SimpleNamespace(objects=[]), s.beh)
    return s


def code(s):
    return (s._reroute or {}).get("code")


# ---- TEST 1: the bug, reproduced then fixed -------------------------------------------

def test_a_turn_off_the_van_has_just_crept_past_is_still_one_it_can_take():
    """The live case. Blocker 1.5 m ahead, a junction 2 m BEHIND the van. The old rule wanted
    a junction within 1.5 m and refused; nothing about the road had changed."""
    s = van(route=road(junction_at=18.0), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    s.go()
    assert code(s) == "REROUTE_ACCEPTED", s._reroute
    assert s.asked, "the map was actually asked"
    # ...and the old rule really would have refused this
    old_span = s.planner.junction_span(s._route, s.pose.x, s.pose.y)
    assert old_span is None or old_span[0] > 1.5, \
        "the old gate wanted a junction nearer than the blocker, which this is not"


def test_the_gate_no_longer_shrinks_to_nothing_as_the_van_creeps_up():
    """The same road at four stopping distances: the verdict must not depend on how close the
    van happens to have crept, which is what made the old gate unsatisfiable."""
    for blocker_m in (1.1, 1.5, 2.6, 5.0):
        s = van(route=road(junction_at=18.0), van_x=20.0, blocker_m=blocker_m,
                alternate=road(n=30))
        s.go()
        assert code(s) == "REROUTE_ACCEPTED", f"{blocker_m} m: {s._reroute}"


# ---- TEST 2: the F shape -- fully blocked, a legitimate junction --------------------------

def test_a_road_blocked_end_to_end_may_still_be_gone_round_by_the_map():
    other = road(n=40)
    s = van(route=road(junction_at=17.0), van_x=20.0, blocker_m=2.0, alternate=other)
    s.go()
    assert s._route is other, "the route was swapped for the other way"
    assert ("dressed", "") in s.said, "a new route is dressed for parking, as before"
    assert any(w == "rerouted" for w, _ in s.said), "and the move is announced as before"
    assert s._blocked_road_since is None, "the block watch restarts on the new route"
    r = s._reroute
    assert r["code"] == "REROUTE_ACCEPTED"
    assert r["old_route_m"] and r["new_route_m"], r
    assert r["blocker_distance_m"] == 2.0 and r["diversion_at_m"] is not None


# ---- TEST 3: no diversion -- the safety gate is still there --------------------------------

def test_a_blocked_road_with_nowhere_to_turn_off_is_not_gone_round():
    """The gate was relaxed, not deleted."""
    s = van(route=road(junction_at=None), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    s.go()
    assert code(s) == "NO_DIVERSION_POINT"
    assert s.asked == [], "the map is not even asked when there is nowhere to turn off"
    assert any(w == "no_way_round" for w, _ in s.said)
    assert s._route.waypoints[0].x == 0.0, "the route is untouched"


def test_a_junction_beyond_the_blockage_is_no_use_and_is_still_refused():
    """It is on the far side of the thing the van cannot drive through."""
    s = van(route=road(junction_at=40.0), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    s.go()
    assert code(s) == "NO_DIVERSION_POINT", s._reroute
    assert "past the blockage" in s._reroute["reason"]
    assert s.asked == []


def test_the_reach_is_the_blocker_plus_what_the_van_can_back_out_and_no_more():
    """The boundary of the one number this rule rests on. REVERSE_MAX_M is what
    _start_backing_out already uses; nothing new was invented."""
    blocker_m = 1.5
    reach = blocker_m + WarpAV.REVERSE_MAX_M
    inside = van(route=road(junction_at=20.0 - WarpAV.REVERSE_MAX_M + 1.0), van_x=20.0,
                 blocker_m=blocker_m, alternate=road(n=30))
    inside.go()
    assert code(inside) == "REROUTE_ACCEPTED", "a junction inside the reach is taken"
    beyond = van(route=road(junction_at=20.0 + reach + 4.0), van_x=20.0,
                 blocker_m=blocker_m, alternate=road(n=30))
    beyond.go()
    assert code(beyond) == "NO_DIVERSION_POINT", "one outside it is not"


# ---- TEST 4: the map's own answer is still the authority -------------------------------------

def test_an_alternate_that_still_runs_through_the_blockage_is_rejected():
    """plan_route_avoiding's own rule, driven for real: the cost is bumped, the search is run,
    the cost is put back, and a route that still passes within clear_m is not a way round."""
    p = RoutePlanner.__new__(RoutePlanner)
    edge = {"length": 100.0}
    p.carla_map = SimpleNamespace(get_waypoint=lambda loc: SimpleNamespace(
        road_id=1, section_id=0, lane_id=-1))
    p._grp = SimpleNamespace(_road_id_to_edge={1: {0: {-1: (7, 8)}}},
                             _graph=SimpleNamespace(edges={(7, 8): edge}))
    through = Route(waypoints=[Waypoint(x=float(i), y=0.0) for i in range(40)],
                    total_distance=39.0)
    p.plan_route = lambda sx, sy, ex, ey: through
    why = {}
    got = p.plan_route_avoiding(0.0, 0.0, 39.0, 0.0, 21.5, 0.0, why=why)
    assert got is None, "it goes straight through the blockage"
    assert why["code"] == "ALTERNATE_ROUTE_STILL_HITS_BLOCKER"
    assert edge["length"] == 100.0, "the cost is always put back"

    clear = Route(waypoints=[Waypoint(x=float(i), y=40.0) for i in range(40)], total_distance=39.0)
    p.plan_route = lambda sx, sy, ex, ey: clear
    why = {}
    got = p.plan_route_avoiding(0.0, 0.0, 39.0, 0.0, 21.5, 0.0, why=why)
    assert got is clear and why["code"] == "REROUTE_ACCEPTED"
    assert edge["length"] == 100.0


def test_the_van_reports_the_maps_refusal_rather_than_inventing_its_own():
    s = van(route=road(junction_at=18.0), van_x=20.0, blocker_m=1.5, alternate=None)
    s.go()
    assert code(s) == "NO_ALTERNATE_ROUTE"
    assert s.asked, "the map WAS asked -- the gate let it through"
    assert any(w == "no_way_round" for w, _ in s.said)
    assert s._route.waypoints[0].x == 0.0, "and the route is untouched"


# ---- TEST 5, 6, 7, 8: the gates that were already right, still right ---------------------------

def test_a_road_only_just_blocked_is_watched_before_the_map_is_asked():
    s = van(route=road(junction_at=18.0), blocked_for_s=2.0, alternate=road(n=30))
    s.go()
    assert code(s) == "REROUTE_NOT_READY" and s.asked == []
    assert s._reroute["blocked_for_s"] < WarpAV.REROUTE_AFTER_S


def test_the_first_blocked_tick_only_starts_the_clock():
    s = van(route=road(junction_at=18.0), blocked_for_s=None, alternate=road(n=30))
    s.go()
    assert code(s) == "REROUTE_NOT_READY" and s.asked == []
    assert s._blocked_road_since is not None, "the clock started"


def test_the_map_is_not_asked_more_often_than_the_cooldown_allows():
    s = van(route=road(junction_at=18.0), asked_ago_s=1.0, alternate=road(n=30))
    s.go()
    assert code(s) == "REROUTE_COOLDOWN" and s.asked == []
    assert s._reroute["asked_ago_s"] < WarpAV.REROUTE_EVERY_S


def test_a_blockage_further_off_than_the_reroute_range_is_not_one_yet():
    s = van(route=road(junction_at=18.0), blocker_m=WarpAV.REROUTE_WITHIN_M + 5.0,
            alternate=road(n=30))
    s.go()
    assert code(s) == "NO_BLOCKER" and s.asked == []


def test_a_road_called_blocked_with_nothing_measured_is_not_gone_round():
    s = van(route=road(junction_at=18.0), blocker_m=None, alternate=road(n=30))
    s.go()
    assert code(s) == "NO_BLOCKER" and s.asked == []


def test_a_clear_road_leaves_the_route_alone_and_forgets_the_block():
    route = road(junction_at=18.0)
    s = van(route=route, blocked=False, alternate=road(n=30))
    s.go()
    assert code(s) == "NOT_BLOCKED"
    assert s._blocked_road_since is None and s.asked == []
    assert s._route is route and s.said == [], "nothing said, nothing changed"


def test_nothing_is_gone_round_while_a_pass_is_being_driven():
    route = road(junction_at=18.0)
    s = van(route=route, overtaking=True, alternate=road(n=30))
    s.go()
    assert s._route is route and s.asked == [] and s._reroute is None


# ---- TEST 9: the local way round is untouched -------------------------------------------------

def test_the_local_go_around_still_decides_exactly_what_it_did():
    """This task changed the whole-road reroute. Weighing the ways round one blocker must be
    bit for bit what it was: the same option accepted for the same five answers."""
    import test_go_around_diagnostics as G
    table = [(dict(), "NUDGE_LEFT"),
             (dict(lane_ok=False), "LANE_RIGHT"),
             (dict(lane_ok=False, same_way_ok=False), "SHOULDER"),
             (dict(lane_ok=False, same_way_ok=False, shoulder_ok=False), None),
             (dict(same_way_ok=False), "NUDGE_LEFT"),
             (dict(shoulder_ok=False), "NUDGE_LEFT")]
    for kwargs, expected in table:
        g = G.van(**kwargs)
        g.go()
        assert g._go_around["taken"] == expected, kwargs
        assert (g._overtake_point is not None) == (expected is not None), kwargs


# ---- where the decision is kept ----------------------------------------------------------------

def test_every_reroute_decision_lands_in_the_mission_log_as_structured_data():
    s = van(route=road(junction_at=18.0), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    s.go()
    events = [(a, k) for a, k in s.logged if a[0] == "reroute_decision"]
    assert len(events) == 1
    (_, said), kw = events[0]
    assert kw["data"]["code"] == "REROUTE_ACCEPTED" and "REROUTE_ACCEPTED" in said
    assert "new_route_clears_blocker_m" in kw["data"]


def test_the_same_verdict_tick_after_tick_is_written_once():
    s = van(route=road(junction_at=None), van_x=20.0, blocker_m=1.5)
    s.REROUTE_LOG_EVERY_S = 30.0
    for _ in range(12):
        s._reroute_asked_at = time.time() - 60.0      # keep it asking, as a blocked van does
        s.go()
    assert len([a for a, _ in s.logged if a[0] == "reroute_decision"]) == 1


def test_a_broken_recorder_can_never_stop_a_reroute():
    other = road(n=30)
    s = van(route=road(junction_at=18.0), van_x=20.0, blocker_m=1.5, alternate=other)
    s.logger.log_event = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    s.go()
    assert s._route is other, "the reroute still happened"


# ---- a dead end should not read like a near miss (F_reroute, 2026-09-15) -------------------

def test_a_junction_out_of_reach_and_a_road_with_no_exit_read_differently():
    """F sits on a road whose first junction is 132 m on, the far side of the truck, and whose
    nearest one behind is 38 m back -- further than the van will ever reverse. The refusal
    said only "no junction within 16 m", which reads like a near miss rather than a street the
    van cannot leave. Those are different situations and an operator needs to be told which."""
    beyond = van(route=road(junction_at=60.0), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    beyond.go()
    r = beyond._reroute
    assert r["code"] == "NO_DIVERSION_POINT"
    assert r["first_junction_on_road_m"] is not None, "there IS one, further up the road"
    assert "the far side of the blockage" in r["reason"]

    none_at_all = van(route=road(junction_at=None), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    none_at_all.go()
    r2 = none_at_all._reroute
    assert r2["code"] == "NO_DIVERSION_POINT"
    assert r2["first_junction_on_road_m"] is None
    assert "no exit the van can reach" in r2["reason"]
    assert r["reason"] != r2["reason"], "the two noes say different things"


def test_how_far_it_looks_to_report_is_not_how_far_it_will_reroute():
    """The long look is diagnostics only: a junction 60 m off is still refused."""
    s = van(route=road(junction_at=60.0), van_x=20.0, blocker_m=1.5, alternate=road(n=30))
    s.go()
    assert s._reroute["code"] == "NO_DIVERSION_POINT"
    assert s.asked == [], "the map was never asked"
    assert s._route.waypoints[0].x == 0.0, "and the route is untouched"
    assert WarpAV.REROUTE_LOOK_FAR_M > WarpAV.REROUTE_WITHIN_M
