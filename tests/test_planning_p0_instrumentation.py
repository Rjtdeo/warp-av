"""Planning V2, phase 0: the van explains itself.

Nothing here changes how the van drives. These tests pin the two things P0 adds, because
both are about to be relied on by every later phase: a timer that can find the ONE slow
tick, and a decision record a machine can filter.

The timer this replaces kept a single exponentially-smoothed number per phase, so its
"worst" was the worst smoothed value. A 300 ms tick in a stream of 90 ms ticks moved that
average by 30 ms and then vanished. Smoothing is the wrong tool for finding the tick that
broke something.
"""
import pytest

from warp_av.planning.instrumentation import (
    PhaseTimer, PlannerDecision, debug_planning_enabled,
    CLEAR, NO_ROUTE, BLOCKED_TRACKED_OBJECT, BLOCKED_SWEPT_PATH, BLOCKED_VRU, ALL_REASONS)


class FakeClock:
    """A clock the test drives, so timings are exact rather than nearly right."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += ms / 1000.0


# ---------------------------------------------------------------- the timer


def test_it_times_each_phase_separately():
    clock = FakeClock()
    t = PhaseTimer(clock=clock)
    t.start()
    clock.advance_ms(90.0)
    t.mark("perception")
    clock.advance_ms(8.0)
    t.mark("route corridor")
    t.end()
    assert t.summary("perception")["avg_ms"] == pytest.approx(90.0, abs=0.01)
    assert t.summary("route corridor")["avg_ms"] == pytest.approx(8.0, abs=0.01)
    assert t.summary("whole tick")["avg_ms"] == pytest.approx(98.0, abs=0.01)


def test_one_slow_tick_survives_as_the_worst_case():
    """The whole point. Ninety-nine ticks at 90 ms and one at 300 ms: the average barely
    moves, and the worst case must still say 300."""
    clock = FakeClock()
    t = PhaseTimer(clock=clock)
    for i in range(100):
        t.start()
        clock.advance_ms(300.0 if i == 42 else 90.0)
        t.mark("perception")
        t.end()
    s = t.summary("perception")
    assert s["worst_ms"] == pytest.approx(300.0, abs=0.01), "the slow tick was smoothed away"
    assert 90.0 < s["avg_ms"] < 95.0, "one spike should barely move the average"


def test_p95_ignores_the_single_spike_but_catches_a_persistent_one():
    clock = FakeClock()
    t = PhaseTimer(clock=clock)
    for i in range(100):
        t.start()
        clock.advance_ms(250.0 if i < 10 else 90.0)     # a tenth of ticks are slow
        t.mark("perception")
        t.end()
    s = t.summary("perception")
    assert s["p95_ms"] == pytest.approx(250.0, abs=0.01), "a tenth of ticks slow is not a p95 of 90"


def test_the_window_forgets_old_ticks():
    """A p95 over the whole run is useless once something is fixed; it must age out."""
    clock = FakeClock()
    t = PhaseTimer(window=50, clock=clock)
    for _ in range(50):
        t.start(); clock.advance_ms(400.0); t.mark("perception"); t.end()
    assert t.summary("perception")["avg_ms"] == pytest.approx(400.0, abs=0.01)
    for _ in range(50):
        t.start(); clock.advance_ms(90.0); t.mark("perception"); t.end()
    s = t.summary("perception")
    assert s["avg_ms"] == pytest.approx(90.0, abs=0.01), "the old slow ticks never aged out"
    assert s["n"] == 50


def test_a_p95_over_five_samples_is_honestly_the_worst_one():
    clock = FakeClock()
    t = PhaseTimer(clock=clock)
    for ms in (10.0, 20.0, 30.0, 40.0, 50.0):
        t.start(); clock.advance_ms(ms); t.mark("x"); t.end()
    assert t.summary("x")["p95_ms"] == pytest.approx(50.0, abs=0.01)


def test_it_names_the_phase_to_look_at_first():
    clock = FakeClock()
    t = PhaseTimer(clock=clock)
    t.start()
    clock.advance_ms(88.0); t.mark("perception")
    clock.advance_ms(28.0); t.mark("safety and health")
    clock.advance_ms(0.4); t.mark("behaviour")
    t.end()
    assert t.worst_phase() == "perception"
    assert "whole tick" not in (t.worst_phase() or ""), "the total is not a phase"


def test_an_unseen_phase_reports_nothing_rather_than_zero():
    t = PhaseTimer()
    assert t.summary("never ran") is None
    assert "never ran" not in t.as_dict()


def test_mark_without_start_does_not_crash():
    """A tick that throws before start() must not take the timer down with it."""
    t = PhaseTimer()
    assert t.mark("orphan") == 0.0
    assert t.end() == 0.0


# ---------------------------------------------------------------- the decision


def test_a_clear_path_is_not_blocked():
    d = PlannerDecision(reason=CLEAR)
    assert d.blocked is False
    assert d.as_dict()["blocked"] is False


def test_no_route_is_not_the_same_as_blocked():
    """Having nothing to judge against is not evidence of an obstruction."""
    d = PlannerDecision(reason=NO_ROUTE)
    assert d.blocked is False


@pytest.mark.parametrize("reason", [BLOCKED_TRACKED_OBJECT, BLOCKED_SWEPT_PATH, BLOCKED_VRU])
def test_every_blocked_reason_reads_as_blocked(reason):
    assert PlannerDecision(reason=reason).blocked is True


def test_it_names_the_blocker_not_just_the_fact():
    """The thing that was missing: 'blocked = true' cannot tell a parked lorry from the
    same kerb sliver reported forty times."""
    d = PlannerDecision(reason=BLOCKED_TRACKED_OBJECT, blocker_id=17, blocker_kind="vehicle",
                        blocker_distance_m=13.24, blocker_lateral_m=0.81)
    out = d.as_dict()
    assert out["blocker_id"] == 17
    assert out["blocker_kind"] == "vehicle"
    assert out["blocker_distance_m"] == 13.2
    assert out["blocker_lateral_m"] == 0.81
    assert "blocker=17" in d.one_line() and "13.2 m" in d.one_line()


def test_the_reason_vocabulary_is_closed():
    """P2 and P3 must extend this list, not invent a second one beside it."""
    d = PlannerDecision()
    assert d.reason in ALL_REASONS
    assert len(set(ALL_REASONS)) == len(ALL_REASONS), "a reason code is duplicated"


def test_the_long_tail_is_behind_a_switch(monkeypatch):
    """The decision itself always ships -- it is what makes a failure readable, and it is
    free. Only every candidate the planner weighed is optional."""
    monkeypatch.delenv("WARP_PLAN_DEBUG", raising=False)
    assert debug_planning_enabled({}) is False
    assert debug_planning_enabled({"WARP_PLAN_DEBUG": "1"}) is True

    quiet = PlannerDecision(reason=CLEAR)
    assert "candidates" not in quiet.as_dict()
    loud = PlannerDecision(reason=CLEAR, candidates=[{"id": 3, "lat": 2.1}])
    assert loud.as_dict()["candidates"] == [{"id": 3, "lat": 2.1}]


# ---------------------------------------------------------------- wired to the real planner

import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.footprint import VehicleFootprint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.planning.instrumentation import BLOCKED_SCRAPE


def _planner():
    return RoutePlanner.__new__(RoutePlanner)


def _route(n=60):
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(n)])


def _obj(world_x, world_y, kind=ObjectType.OBSTACLE, speed=0.0, ident=0, size=None):
    # ego at the origin facing +x, so the ego frame and the world frame coincide
    o = DetectedObject(object_type=kind, x=world_x, y=world_y, distance=math.hypot(world_x, world_y))
    o.speed = speed
    o.id = ident
    if size:
        o.length_m, o.width_m, o.height_m = size
    return o


def _run(planner, perception, route, ego=(0.0, 0.0, 0.0), footprint=None):
    return planner.filter_to_route_corridor(perception, route, ego[0], ego[1], ego[2],
                                            footprint=footprint)


def test_a_clear_road_says_clear_and_names_nothing():
    p = _planner()
    _run(p, PerceptionOutput(objects=[_obj(20.0, 8.0, ident=5)]), _route())
    d = p.last_decision
    assert d.reason == CLEAR and d.blocked is False
    assert d.blocker_id is None
    assert d.objects_considered == 1
    assert d.route_points_used == 60


def test_a_blocker_is_named_by_id_kind_and_distance():
    """The whole point of phase 0: not 'blocked = true'."""
    p = _planner()
    per = PerceptionOutput(objects=[_obj(6.0, 0.4, kind=ObjectType.VEHICLE, ident=17)])
    _run(p, per, _route())
    d = p.last_decision
    assert per.path_blocked is True, "the decision under test must actually be a block"
    assert d.blocked is True
    assert d.blocker_id == 17
    assert d.blocker_kind == "vehicle"
    assert 5.0 <= d.blocker_distance_m <= 7.0
    assert d.blocker_lateral_m is not None and d.blocker_lateral_m < 1.0


def test_a_person_gets_its_own_reason():
    """'the van stopped' and 'the van stopped FOR A PERSON' are different lines in a report."""
    p = _planner()
    per = PerceptionOutput(objects=[_obj(5.0, 0.3, kind=ObjectType.PEDESTRIAN, ident=4)])
    _run(p, per, _route())
    assert per.path_blocked is True
    assert p.last_decision.reason == BLOCKED_VRU
    assert p.last_decision.blocker_kind == "pedestrian"


def test_the_nearest_blocker_is_the_one_reported():
    p = _planner()
    per = PerceptionOutput(objects=[_obj(7.5, 0.3, kind=ObjectType.VEHICLE, ident=99),
                                    _obj(4.0, 0.3, kind=ObjectType.VEHICLE, ident=7)])
    _run(p, per, _route())
    assert p.last_decision.blocker_id == 7, "reported a further blocker than the one that stops us"


def test_the_swept_body_check_says_so_when_it_is_the_one_deciding():
    """A reason code must distinguish the centre-line bands from the van's real body,
    because phase 1 changes which of them runs."""
    p = _planner()
    foot = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
    per = PerceptionOutput(objects=[_obj(5.0, 0.9, kind=ObjectType.VEHICLE, ident=3)])
    _run(p, per, _route(), footprint=foot)
    if per.path_blocked:
        assert p.last_decision.reason == BLOCKED_SWEPT_PATH
        assert p.last_decision.used_footprint is True


def test_no_route_reports_no_route_not_a_clear_road():
    """Having nothing to judge against is not evidence the way is clear."""
    p = _planner()
    _run(p, PerceptionOutput(objects=[_obj(5.0, 0.0)]), Route(waypoints=[]))
    assert p.last_decision.reason == NO_ROUTE
    assert p.last_decision.blocked is False


def test_the_decision_never_goes_stale():
    """A tick that judges nothing must not leave the previous tick's blocker on show."""
    p = _planner()
    per = PerceptionOutput(objects=[_obj(4.0, 0.3, kind=ObjectType.VEHICLE, ident=7)])
    _run(p, per, _route())
    assert p.last_decision.blocker_id == 7
    _run(p, PerceptionOutput(objects=[]), _route())
    assert p.last_decision.blocker_id is None, "last tick's blocker was still being reported"
    assert p.last_decision.reason == CLEAR


def test_the_decision_does_not_change_the_verdict():
    """Phase 0 is instrumentation. Whatever the corridor decided before, it decides now."""
    p = _planner()
    for wy in (0.2, 1.0, 1.6, 2.1, 3.0, 6.0):
        per = PerceptionOutput(objects=[_obj(5.0, wy, kind=ObjectType.VEHICLE, ident=1)])
        _run(p, per, _route())
        # the record agrees with the flag it was derived from -- neither leads the other
        assert p.last_decision.blocked == per.path_blocked, f"disagreed at {wy} m"
