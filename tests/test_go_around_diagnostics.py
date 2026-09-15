"""Why EACH way round was refused -- not only the last one that happened to be tried.

Before 2026-09-15 `_maybe_overtake` kept one `refused` string that four branches overwrote in
turn. The shoulder is tried last, so in a ten-drive campaign 38 of 72 recorded refusals read
"1.80 m over onto the shoulder is refused by the geometry (bend/junction/no lane of ours to
use/route end)" -- one sentence covering four quite different causes, about the LAST option,
saying nothing at all about the four before it. There was no way to tell a route that ran out
from a bend, a junction, or a lane the van may not borrow.

Nothing here changes what the van does. Every test that fixes a decision lives beside the
behaviour it guards (test_overtake, test_overtake_clearance); these only check that the
reasons survive, and the last two check that the decisions did not move.
"""
import math
import time
from types import SimpleNamespace

from warp_av.main import WarpAV, SQUEEZE_ABORT_M, SQUEEZE_SPEED_MPS
from warp_av.behavior.behavior import DrivingBehavior
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PlannerDecision
from warp_av.planning.planner import Route, RoutePlanner, Waypoint, pass_options

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)   # the live body
LEAD_D = 10.0                    # the dead car's back, metres ahead of the van
EVERY = ["NUDGE_LEFT", "NUDGE_RIGHT", "LANE_RIGHT", "LANE_LEFT", "SHOULDER"]      # pass_options


def road(points=60, step=2.0, junction_from=None, bend_from=None, bend_deg=30.0):
    """A straight road from (0, 0) along +x, optionally with a junction or a bend on it."""
    wps = []
    for i in range(points):
        wps.append(Waypoint(
            x=i * step, y=0.0,
            yaw=(math.radians(bend_deg) if (bend_from is not None and i >= bend_from) else 0.0),
            is_junction=bool(junction_from is not None and i >= junction_from)))
    return Route(waypoints=wps, total_distance=(points - 1) * step)


def standing(x, y, kind=ObjectType.VEHICLE, length=4.5, w=1.8):
    """Something parked: x ahead, y to the RIGHT of the van, as perception reports it."""
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=0.0,
                          stationary=True, vx_world=0.0, vy_world=0.0, height_m=1.5,
                          width_m=w, length_m=length, box_length_m=length, box_width_m=w,
                          box_yaw_deg=0.0)


def moving(x, y, vx, kind=ObjectType.VEHICLE, oid=None):
    o = standing(x, y, kind=kind)
    o.speed, o.stationary, o.vx_world, o.id = abs(vx), False, vx, oid
    return o


class Grid:
    """The laser's own answer about a strip of ground: (free, blocked, unseen)."""
    def __init__(self, counts, updated=True):
        self.counts, self.updated = counts, updated

    def strip_ahead(self, a, b, half, offset_m=0.0):
        return self.counts


def van(route=None, lane_ok=True, same_way_ok=True, shoulder_ok=True, grid=None,
        ground_block=False, standing_things=(), lead_d=LEAD_D, kind="vehicle",
        blocker_speed=0.0, traffic_light="green", degraded=False, real_pull_in=False):
    """A WarpAV with only the parts _maybe_overtake touches, and the real methods bound to it.

    The three lane checks are answers, not maps: what each one decides is fixed elsewhere
    (test_overtake_clearance). Here they are set so a chosen option can be made to fail.
    """
    s = SimpleNamespace()
    s.said, s.logged = [], []
    s._overtake_point = None
    s._blocked_since = time.time() - 60.0        # long past the 10 s wait
    s._overtake_retry_at = 0.0
    s._overtake_why_at = 0.0
    s._go_around = None
    s._ground_block = ground_block
    s.OVERTAKE_STATES = WarpAV.OVERTAKE_STATES
    s.OVERTAKE_AFTER_S = WarpAV.OVERTAKE_AFTER_S
    s.WAY_ROUND_SEEN_SHARE = WarpAV.WAY_ROUND_SEEN_SHARE
    s.GO_AROUND_LOG_EVERY_S = 0.0                # in a test every change is written
    s.planner = RoutePlanner.__new__(RoutePlanner)          # no CARLA: geometry only
    s.behavior = SimpleNamespace(cruise_speed=4.0, front_offset_m=2.95)
    s.footprint_blocking = SimpleNamespace(footprint=FOOT)
    s.perception = SimpleNamespace(grid=grid)
    s._route = route if route is not None else road()
    s.logger = SimpleNamespace(log_event=lambda *a, **k: s.logged.append((a, k)))
    s._note_move = lambda what, said: s.said.append((what, said))
    s._lane_width = lambda pose: 3.5

    def answer(ok, bad_code, bad_text):
        good = (True, "CLEAR", "fine")
        bad = (False, bad_code, bad_text)
        return (lambda *a: (ok and good or bad)[0], lambda *a: (ok and good or bad)[1:])

    s._lane_ok, s._lane_ok_why_2 = answer(
        lane_ok, "LANE_INVALID", "(9.0, 1.2) is not on a driving lane")
    s._same_way_lane_ok, s._same_way_why_2 = answer(
        same_way_ok, "LANE_NOT_OURS", "the lane at (9.0, 3.6) runs 178 deg off our heading")
    s._shoulder_ok, s._shoulder_why_2 = answer(
        shoulder_ok, "SHOULDER_INVALID",
        "what lies right of the lane is sidewalk, not ground the van may use")
    s._lane_ok_why = lambda x, y: (s._lane_ok(x, y),) + s._lane_ok_why_2(x, y)
    s._same_way_lane_ok_why = lambda x, y, yaw: (s._same_way_lane_ok(x, y, yaw),) + s._same_way_why_2(x, y)
    s._shoulder_ok_why = lambda x, y: (s._shoulder_ok(x, y),) + s._shoulder_why_2(x, y)

    s._way_round_is_seen_free = lambda o, t: WarpAV._way_round_is_seen_free(s, o, t)
    s._way_round_seen_why = lambda o, t: WarpAV._way_round_seen_why(s, o, t)
    s._record_go_around = lambda *a, **k: WarpAV._record_go_around(s, *a, **k)
    if not real_pull_in:
        s.planner.pull_in_blocker = lambda *a, **k: None

    s._path = PlannerDecision(closest_distance_m=lead_d,
                              closest_kind=(ObjectType(kind) if kind else None),
                              closest_speed_mps=blocker_speed)
    s.pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    s.per = SimpleNamespace(objects=list(standing_things), traffic_light=traffic_light,
                            degraded=degraded)
    s.beh = SimpleNamespace(behavior=DrivingBehavior.STOPPED_VEHICLE)
    s.go = lambda junction=None: WarpAV._maybe_overtake(s, s.pose, s.per, s.beh, junction)
    return s


def by_option(s):
    return {r["option"]: r for r in (s._go_around or {}).get("options", [])}


# ---- TEST 1: every option is kept, not just the last -----------------------------------

def test_every_way_round_keeps_its_own_verdict():
    """The whole point. Five options are offered on a 3.5 m lane; five verdicts come back."""
    s = van(same_way_ok=False, shoulder_ok=False, lane_ok=False)
    s.go()
    rows = s._go_around["options"]
    assert [r["option"] for r in rows] == EVERY, "all five, in the order pass_options offers them"
    assert len({r["option"] for r in rows}) == 5, "no option overwrites another"
    for r in rows:
        assert r["status"] in ("ACCEPTED", "REJECTED", "NOT_EVALUATED")
        assert r["reason_code"] and r["reason"], f"{r['option']} says nothing"
        assert isinstance(r["shift_m"], float) and r["side"] in ("left", "right")
        assert r["kind"] in ("nudge", "lane", "shoulder")
    # ...and the shapes match what pass_options actually offered
    offered = list(pass_options(3.5, FOOT.half_width, s.planner.OVERTAKE_SHIFT_M))
    assert [r["shift_m"] for r in rows] == [round(o[0], 2) for o in offered]
    assert [r["side"] for r in rows] == ["left" if o[0] > 0 else "right" for o in offered]


def test_the_old_single_string_could_only_ever_describe_the_last_one():
    """What this replaced: the one line the van prints is still the LAST option's, and that is
    exactly why it was never enough on its own."""
    s = van(same_way_ok=False, shoulder_ok=False, lane_ok=False)
    s.go()
    said = [t for what, t in s.said][-1]
    assert "1.80 m over" in said and "shoulder" in said, "the printed line is the shoulder's"
    assert by_option(s)["NUDGE_LEFT"]["reason"] != said, "the first option's reason is a different one"


# ---- TEST 2: the route running out is its own reason ------------------------------------

def test_a_route_that_runs_out_says_so_with_the_numbers():
    s = van(route=road(points=14))                 # 26 m of road; the pass needs 10 + 16 + 3
    s.go()
    for name, r in by_option(s).items():
        assert r["reason_code"] == "ROUTE_TOO_SHORT", name
        assert "26.0 m of route remains" in r["reason"] and "29.0 m required" in r["reason"]
        assert "16 m to rejoin" in r["reason"]


# ---- TEST 3 and 4: a junction and a bend are not the same thing --------------------------

def test_a_junction_on_the_detour_is_named_and_placed():
    s = van(route=road(junction_from=9))            # 18 m along, inside the 28 m needed
    s.go()
    r = by_option(s)["NUDGE_LEFT"]
    assert r["reason_code"] == "JUNCTION_ON_DETOUR"
    assert "18.0 m along" in r["reason"] and "28.0 m the pass needs" in r["reason"]


def test_a_bend_is_named_with_how_far_it_turns():
    s = van(route=road(bend_from=6, bend_deg=30.0))
    s.go()
    r = by_option(s)["NUDGE_LEFT"]
    assert r["reason_code"] == "BEND_TOO_SHARP"
    assert "turns 30 deg" in r["reason"] and "14 deg a pass allows" in r["reason"]


def test_route_end_junction_and_bend_are_three_different_codes():
    """The four causes the one old sentence lumped together, now told apart. This is the
    question the whole task exists to answer: "bend/junction/no lane of ours to use/route
    end" could not be acted on, because it never said which."""
    vans = {"end": van(route=road(points=14)),
            "junction": van(route=road(junction_from=9)),
            "bend": van(route=road(bend_from=6)),
            "lane": van(lane_ok=False)}
    got = {}
    for name, s in vans.items():
        s.go()
        got[name] = by_option(s)["NUDGE_LEFT"]["reason_code"]
    assert got == {"end": "ROUTE_TOO_SHORT", "junction": "JUNCTION_ON_DETOUR",
                   "bend": "BEND_TOO_SHARP", "lane": "LANE_INVALID"}
    assert len(set(got.values())) == 4, "four causes, four codes"


# ---- TEST 5: no lane to borrow, and WHICH no it was --------------------------------------

def test_a_lane_the_van_may_not_borrow_is_told_from_no_lane_at_all():
    s = van(lane_ok=False, same_way_ok=False, shoulder_ok=False)
    s.go()
    got = by_option(s)
    assert got["NUDGE_LEFT"]["reason_code"] == "LANE_INVALID", "a nudge asks _lane_ok"
    assert got["NUDGE_RIGHT"]["reason_code"] == "LANE_INVALID"
    assert got["LANE_LEFT"]["reason_code"] == "LANE_NOT_OURS", "a whole lane asks _same_way_lane_ok"
    assert got["LANE_RIGHT"]["reason_code"] == "LANE_NOT_OURS"
    assert got["SHOULDER"]["reason_code"] == "SHOULDER_INVALID"
    assert "178 deg off our heading" in got["LANE_LEFT"]["reason"], "the oncoming lane, in words"


def test_the_lane_checks_and_their_explanations_never_disagree():
    """The boolean the planner is handed is the explaining twin's own first answer."""
    for ok in (True, False):
        s = van(lane_ok=ok, same_way_ok=ok, shoulder_ok=ok)
        assert s._lane_ok(1.0, 2.0) is s._lane_ok_why(1.0, 2.0)[0]
        assert s._same_way_lane_ok(1.0, 2.0, 0.0) is s._same_way_lane_ok_why(1.0, 2.0, 0.0)[0]
        assert s._shoulder_ok(1.0, 2.0) is s._shoulder_ok_why(1.0, 2.0)[0]


# ---- TEST 6: what the van's body would touch, kept whole ---------------------------------

def test_a_body_in_the_way_keeps_its_id_its_distance_and_where_it_would_be_touched():
    hit = SimpleNamespace(along_m=12.4, lateral_m=-0.6)
    obj = standing(12.0, -0.6)
    obj.id = 4471
    s = van()
    s.planner.pull_in_blocker = lambda *a, **k: (obj, 12.03, hit, (12.0, -0.6), None)
    s.go()
    r = by_option(s)["NUDGE_LEFT"]
    assert r["status"] == "REJECTED" and r["reason_code"] == "PULL_IN_BLOCKED"
    assert r["blocker_id"] == 4471 and r["blocker_kind"] == "vehicle"
    assert r["blocker_distance_m"] == 12.03
    assert r["touch_along_m"] == 12.4 and r["touch_lateral_m"] == -0.6
    assert "4471" in r["reason"], "and the sentence still reads as it did"


# ---- TEST 7: moving traffic, in parts ----------------------------------------------------

def test_oncoming_traffic_keeps_the_option_the_vehicle_and_both_times():
    coming = moving(30.0, -3.5, vx=-6.0, oid=908)        # 30 m off, closing at 6 m/s
    s = van(standing_things=[coming])
    s.go()
    r = by_option(s)["NUDGE_LEFT"]                        # a nudge left looks into that lane
    assert r["option"] == "NUDGE_LEFT" and r["reason_code"] == "ONCOMING_CONFLICT"
    assert r["blocker_id"] == 908 and r["blocker_kind"] == "vehicle"
    assert r["blocker_distance_m"] == round(math.hypot(30.0, 3.5), 2)
    assert r["meets_in_s"] == 5.0, "when it would meet us"
    assert r["pass_needs_s"] == 8.5, "and how long the pass needs"
    assert r["gap_margin_s"] == 3.0
    assert "meets us in 5 s" in r["reason"] and "pass needs 8 s" in r["reason"]


def test_traffic_coming_up_behind_is_its_own_code_and_keeps_the_vehicle():
    behind = moving(-20.0, -3.5, vx=5.0, oid=77)
    s = van(standing_things=[behind])
    s.go()
    r = by_option(s)["NUDGE_LEFT"]
    assert r["reason_code"] == "REAR_GAP" and r["blocker_id"] == 77
    assert r["look_back_m"] == 40.0
    assert r["blocker_distance_m"] == round(math.hypot(20.0, 3.5), 2)


def test_the_side_a_pass_would_take_still_decides_which_traffic_counts():
    """Behaviour, not diagnostics: a car in the lane on the LEFT refuses the ways round that
    go left and leaves the ones that go right alone (2026-09-14)."""
    s = van(standing_things=[moving(30.0, -3.5, vx=-6.0, oid=908)])
    s.go()
    got = by_option(s)
    assert got["NUDGE_LEFT"]["reason_code"] == "ONCOMING_CONFLICT"
    assert got["NUDGE_RIGHT"]["reason_code"] != "ONCOMING_CONFLICT"


# ---- the ground the laser has or has not seen --------------------------------------------

def test_ground_seen_with_something_on_it_is_told_from_ground_never_looked_at():
    seen_blocked = van(ground_block=True, grid=Grid((300, 6, 0)))
    seen_blocked.go()
    r = by_option(seen_blocked)["NUDGE_LEFT"]
    assert r["reason_code"] == "OCCUPANCY_BLOCKED" and "6 of the 306" in r["reason"]

    mostly_unseen = van(ground_block=True, grid=Grid((100, 0, 300)))
    mostly_unseen.go()
    r = by_option(mostly_unseen)["NUDGE_LEFT"]
    assert r["reason_code"] == "OCCUPANCY_UNKNOWN" and "100 of the 400" in r["reason"]

    no_map = van(ground_block=True, grid=None)
    no_map.go()
    assert by_option(no_map)["NUDGE_LEFT"]["reason_code"] == "OCCUPANCY_UNKNOWN"


# ---- TEST 8: the success path is untouched ------------------------------------------------

def test_a_way_round_that_works_is_taken_and_the_rest_are_marked_not_evaluated():
    s = van()
    before = list(s._route.waypoints)
    s.go()
    got = by_option(s)
    assert got["NUDGE_LEFT"]["status"] == "ACCEPTED" and got["NUDGE_LEFT"]["reason_code"] == "CLEAR"
    assert s._go_around["taken"] == "NUDGE_LEFT"
    for name in EVERY[1:]:
        assert got[name]["status"] == "NOT_EVALUATED" and got[name]["reason_code"] == "NOT_REACHED"
    # and the van actually went: the same three things the pass has always set
    assert s._overtake_point is not None
    assert s._route.waypoints != before, "the route was swapped for the way round"
    assert s._overtake_tight_m == SQUEEZE_ABORT_M and s._overtake_cap_mps == SQUEEZE_SPEED_MPS
    assert s._blocked_since is None
    assert [w for w, _ in s.said] == ["go_around_start"], "one move announced, as before"


def test_a_taken_pass_is_not_weighed_again_while_it_is_being_driven():
    s = van()
    s.go()
    s.said.clear()
    s.go()                                   # _overtake_point is set now
    assert s._go_around["gate"]["reason_code"] == "ACTIVE_PASS"
    assert s._go_around["options_weighed_now"] is False
    assert s._go_around["taken"] == "NUDGE_LEFT", "the accepted option is still readable"
    assert s.said == [], "and nothing new was announced"


# ---- the gates: why no option was weighed at all -------------------------------------------

def test_the_reasons_no_option_is_even_weighed_are_each_their_own_code():
    red = van(traffic_light="red")
    red.go()
    assert red._go_around["gate"]["reason_code"] == "SIGNAL_RESTRICTION"

    junction = van()
    junction.go(junction=12.0)
    assert junction._go_around["gate"]["reason_code"] == "JUNCTION_NEAR"

    far = van(lead_d=31.0)
    far.go()
    assert far._go_around["gate"]["reason_code"] == "BLOCKER_TOO_FAR"
    assert "31.0 m away" in far._go_around["gate"]["reason"]

    person = van(kind="pedestrian")
    person.go()
    assert person._go_around["gate"]["reason_code"] == "VRU"

    rolling = van(blocker_speed=2.0)
    rolling.go()
    assert rolling._go_around["gate"]["reason_code"] == "MOVING_BLOCKER"

    blind = van(kind="obstacle", degraded=True)
    blind.go()
    assert blind._go_around["gate"]["reason_code"] == "CAMERA_DEGRADED"

    waiting = van(same_way_ok=False, shoulder_ok=False, lane_ok=False)
    waiting.go()                                     # refuses, and sets the 10 s retry timer
    waiting.go()
    assert waiting._go_around["gate"]["reason_code"] == "RETRY_TIMER"
    assert waiting._go_around["options_weighed_now"] is False
    assert len(waiting._go_around["options"]) == 5, "the last weighing is still readable"


# ---- where it is kept ----------------------------------------------------------------------

def test_the_record_goes_into_the_mission_log_as_structured_data():
    s = van(same_way_ok=False, shoulder_ok=False, lane_ok=False)
    s.go()
    events = [(a, k) for a, k in s.logged if a[0] == "go_around_attempts"]
    assert len(events) == 1, "one event per change, not one per tick"
    (kind, said), kw = events[0]
    data = kw["data"]
    assert len(data["options"]) == 5 and data["taken"] is None
    assert data["blocker_kind"] == "vehicle" and data["blocker_distance_m"] == 10.0
    assert all(o in said for o in EVERY), "the one-line summary names every option"
    assert "NUDGE_LEFT=REJECTED:LANE_INVALID" in said


def test_the_same_verdicts_tick_after_tick_are_written_once():
    s = van(same_way_ok=False, shoulder_ok=False, lane_ok=False)
    s.GO_AROUND_LOG_EVERY_S = 5.0
    for _ in range(20):
        s._overtake_retry_at = 0.0               # keep it weighing, as a blocked van does
        s.go()
    assert len([a for a, _ in s.logged if a[0] == "go_around_attempts"]) == 1


# ---- TEST 9: none of this moved a decision -------------------------------------------------

def test_asking_the_planner_to_explain_itself_never_changes_its_answer():
    """plan_overtake with and without the two diagnostic arguments, over every option and
    every shape of road this file can build."""
    p = RoutePlanner.__new__(RoutePlanner)
    roads = [road(), road(points=14), road(junction_from=9), road(bend_from=6),
             road(points=200), road(points=11)]
    for r in roads:
        for over_m, in_lane, on_shoulder in pass_options(3.5, FOOT.half_width, p.OVERTAKE_SHIFT_M):
            for ok in (True, False):
                a = Route(waypoints=list(r.waypoints))
                b = Route(waypoints=list(r.waypoints))
                plain = p.plan_overtake(a, 0.0, 0.0, LEAD_D, shift_m=over_m,
                                        lane_ok=(lambda x, y: ok))
                said = {}
                loud = p.plan_overtake(b, 0.0, 0.0, LEAD_D, shift_m=over_m,
                                       lane_ok=(lambda x, y: ok), why=said,
                                       lane_ok_why=(lambda x, y: ("LANE_INVALID", "no")))
                assert (plain is None) == (loud is None)
                if plain is not None:
                    assert (plain.x, plain.y) == (loud.x, loud.y)
                assert [(w.x, w.y) for w in a.waypoints] == [(w.x, w.y) for w in b.waypoints]


def test_asking_the_traffic_check_to_explain_itself_never_changes_its_answer():
    from warp_av.planning.planner import overtake_blocker
    cases = [[], [moving(30.0, -3.5, vx=-6.0)], [moving(-20.0, -3.5, vx=5.0)],
             [moving(6.0, 0.0, vx=6.0)], [moving(30.0, -3.5, vx=6.0)],
             [standing(12.0, 0.0)], [moving(150.0, -3.5, vx=-6.0)]]
    for objs in cases:
        for side in (-1, 1):
            plain = overtake_blocker(objs, LEAD_D, 34.0, 0.0, pass_takes_s=9.0, side=side)
            said = {}
            loud = overtake_blocker(objs, LEAD_D, 34.0, 0.0, pass_takes_s=9.0, side=side, why=said)
            assert plain == loud
            assert (said.get("text") if said else None) == plain


def test_which_way_round_is_taken_is_exactly_what_it_was():
    """The option chosen, for each way the five checks can answer -- the table the old code
    produced, recomputed through the new one."""
    table = [
        (dict(), "NUDGE_LEFT"),
        (dict(lane_ok=False), "LANE_RIGHT"),
        (dict(lane_ok=False, same_way_ok=False), "SHOULDER"),
        (dict(lane_ok=False, same_way_ok=False, shoulder_ok=False), None),
        (dict(same_way_ok=False), "NUDGE_LEFT"),
        (dict(shoulder_ok=False), "NUDGE_LEFT"),
    ]
    for kwargs, expected in table:
        s = van(**kwargs)
        s.go()
        assert s._go_around["taken"] == expected, kwargs
        assert (s._overtake_point is not None) == (expected is not None), kwargs


def test_a_refusal_still_sets_the_ten_second_retry_timer_and_says_one_line():
    s = van(lane_ok=False, same_way_ok=False, shoulder_ok=False)
    now = time.time()
    s.go()
    assert 9.0 < s._overtake_retry_at - now < 11.0, "the 10 s wait is unchanged"
    assert len(s.said) == 1 and s.said[0][0] == "go_around_wait"


def test_a_note_about_a_pass_can_never_stop_one():
    """_maybe_overtake runs inside a try at the tick, so a throw in here would not crash the
    van -- it would quietly skip the whole go-around check, and the van would stop passing
    things. Nothing about recording a reason may reach the decision."""
    # the mission log is a file on a laptop; it can fail
    s = van()
    s.logger.log_event = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    s.go()
    assert s._overtake_point is not None, "the pass still happened"
    assert s._go_around["taken"] == "NUDGE_LEFT", "and the record was still made in memory"

    refusing = van(lane_ok=False, same_way_ok=False, shoulder_ok=False)
    refusing.logger.log_event = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    refusing.go()
    assert len(refusing.said) == 1, "and a refusal still says its one line"

    # an explaining twin can throw where its plain boolean did not
    angry = van(ground_block=True, grid=Grid((300, 6, 0)))
    angry._way_round_seen_why = lambda *a: (_ for _ in ()).throw(RuntimeError("no grid"))
    angry.go()
    assert angry._go_around is not None and len(angry._go_around["options"]) == 5

    hostile = van()
    hostile._lane_ok_why = lambda *a: (_ for _ in ()).throw(RuntimeError("no map"))
    hostile.go()
    assert hostile._overtake_point is not None, "plan_overtake swallows a thrown explanation"


def test_the_recorder_is_a_real_method_and_swallows_everything_it_can():
    """The one failure the call sites cannot survive is the method not being there at all --
    every path INSIDE it is already contained, so this is the whole of the risk."""
    assert callable(getattr(WarpAV, "_record_go_around", None))
    s = van()
    s.logger = None                     # about as broken as it gets
    WarpAV._record_go_around(s, "vehicle", 10.0, [])
    assert s._go_around["options"] == [] and s._go_around["blocker_kind"] == "vehicle"
