"""
Planning V2, phase 1C-VIS: footprint / swept-path debug drawing. Off by
default, independent of the blocking flag, never able to touch a decision.
"""
import math
import os
import types

from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.footprint_config import FootprintBlockingConfig
from warp_av.planning.footprint_debug import (FootprintDebugConfig, FootprintDebugDrawer, build_frame,
                                              rectangle_corners)
from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
MAIN = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "main.py")


class RecordingDebug:
    def __init__(self):
        self.calls = []

    def draw_line(self, *a, **kw): self.calls.append(("line", a, kw))
    def draw_point(self, *a, **kw): self.calls.append(("point", a, kw))
    def draw_string(self, *a, **kw): self.calls.append(("string", a, kw))


class FailingDebug:
    def draw_line(self, *a, **kw): raise RuntimeError("RPC dropped")
    def draw_point(self, *a, **kw): raise RuntimeError("RPC dropped")
    def draw_string(self, *a, **kw): raise RuntimeError("RPC dropped")


def world_with(debug):
    return types.SimpleNamespace(debug=debug)


def straight_route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def obj(x, y, kind=ObjectType.OBSTACLE, speed=0.0):
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed)


def test_debug_visualisation_defaults_off():
    cfg = FootprintDebugConfig()
    assert cfg.enabled is False and cfg.state() == {"footprint_debug_enabled": False}
    assert cfg.set(enabled=True)["footprint_debug_enabled"] is True
    assert cfg.set(enabled="off")["footprint_debug_enabled"] is False
    assert cfg.set(enabled="maybe")["success"] is False


def test_debug_switch_does_not_touch_the_blocking_switch():
    """Drawing what the sweep sees and USING what the sweep sees are separate switches.
    Set explicitly rather than leaning on either default, so a change to one default
    cannot quietly make this test stop testing anything."""
    blocking = FootprintBlockingConfig(enabled=False)
    debug = FootprintDebugConfig()
    debug.set(enabled=True)
    assert blocking.enabled is False and blocking.active_footprint() is None
    blocking.set(enabled=True)
    debug.set(enabled=False)
    assert blocking.enabled is True and blocking.active_footprint() is not None
    # all four combinations are representable
    for b, d in ((False, False), (False, True), (True, True), (True, False)):
        blocking.set(enabled=b); debug.set(enabled=d)
        assert (blocking.enabled, debug.enabled) == (b, d)


def test_frame_shows_body_margin_sweep_obstacles_and_labels():
    route = straight_route()
    frame = build_frame((10.0, 0.0, 0.0), FOOT, route.waypoints,
                        [obj(6.0, -1.0), obj(6.0, -1.9), obj(8.0, 0.0, speed=6.0)],
                        blocking_enabled=False, planner_blocked=True, planner_distance=6.0)
    # body and protected rectangles: 4 corners each, protected wider by the margin
    assert len(frame.body) == 4 and len(frame.protected) == 4
    assert max(abs(p[1]) for p in frame.body) - 0.99 < 1e-9
    assert max(abs(p[1]) for p in frame.protected) - 1.29 < 1e-9
    # swept path: one protected rectangle per metre over the 12 m horizon (13 stations)
    assert len(frame.stations) == 13
    # the moving car is not drawn as a checked obstacle; the two stationary ones are
    tags = sorted(t for (_, _, _, t) in frame.obstacles)
    assert tags == ["checked", "hit"]                      # 1.9 m off (planter, r 0.5) clear; 1.0 m off hit
    assert frame.conflict is not None and frame.conflict[2] > 0
    kinds = dict(frame.texts)
    assert kinds["mode"] == "FOOTPRINT BLOCKING: OFF"
    assert kinds["sweep"].startswith("SWEPT PATH CONFLICT @")
    assert kinds["planner"] == "PLANNER: BLOCKED (6.0 m)"


def test_frame_reports_clear_and_on():
    frame = build_frame((10.0, 0.0, 0.0), FOOT, straight_route().waypoints, [obj(6.0, -2.5)],
                        blocking_enabled=True, planner_blocked=False)
    kinds = dict(frame.texts)
    assert kinds["mode"] == "FOOTPRINT BLOCKING: ON" and kinds["sweep"] == "SWEPT PATH CLEAR"
    assert kinds["planner"] == "PLANNER: CLEAR" and frame.conflict is None
    # no route: still a frame, still safe
    empty = build_frame((0.0, 0.0, 0.0), FOOT, [], [], blocking_enabled=False)
    assert dict(empty.texts)["sweep"] == "SWEPT PATH: no route" and empty.stations == []


def test_drawing_failure_is_counted_and_never_raised():
    drawer = FootprintDebugDrawer(world_with(FailingDebug()))
    frame = build_frame((0.0, 0.0, 0.0), FOOT, straight_route().waypoints, [obj(6.0, 0.0)], blocking_enabled=True)
    assert drawer.draw(frame) is False
    assert drawer.failures == 1 and drawer.frames_drawn == 0
    assert drawer.draw(frame) is False and drawer.failures == 2
    # a world with no debug attribute at all
    assert FootprintDebugDrawer(object()).draw(frame) is False


def test_drawing_uses_short_lifetimes_and_stays_light():
    rec = RecordingDebug()
    drawer = FootprintDebugDrawer(world_with(rec), lifetime_s=0.3)
    frame = build_frame((10.0, 0.0, 0.0), FOOT, straight_route().waypoints,
                        [obj(6.0, -1.0), obj(6.0, -1.9)], blocking_enabled=False, planner_blocked=False)
    assert drawer.draw(frame) is True and drawer.frames_drawn == 1
    assert all(kw.get("life_time") == 0.3 for _, _, kw in rec.calls)
    assert len(rec.calls) < 120                              # a few dozen primitives, not hundreds
    kinds = [c[0] for c in rec.calls]
    assert kinds.count("string") == 3 and kinds.count("point") >= 3


def test_planner_verdict_is_identical_with_visualisation_on_and_off():
    planner = RoutePlanner.__new__(RoutePlanner)
    route = straight_route()
    ego = (10.0, 0.0, 0.0)
    objects = [obj(6.0, -1.6, kind=ObjectType.VEHICLE), obj(6.0, -1.9), obj(9.0, 0.0, speed=5.0)]

    def verdict(footprint):
        p = PerceptionOutput(objects=list(objects))
        out = planner.filter_to_route_corridor(p, route, *ego, footprint=footprint)
        return (out.path_blocked, out.closest_obstacle_distance, out.closest_obstacle_type,
                out.closest_obstacle_speed, out.closest_obstacle_lateral_m)

    for footprint in (None, FOOT):
        before = verdict(footprint)
        # draw in between, with a recording world and a failing world
        for dbg in (RecordingDebug(), FailingDebug()):
            frame = build_frame(ego, FOOT, route.waypoints, objects,
                                blocking_enabled=footprint is not None, planner_blocked=before[0])
            FootprintDebugDrawer(world_with(dbg)).draw(frame)
        after = verdict(footprint)
        assert before == after
        # and the objects handed to the frame were not mutated
        assert (objects[0].x, objects[0].y, objects[0].speed) == (6.0, -1.6, 0.0)


def test_main_guards_the_drawing_and_exposes_the_switch():
    src = open(MAIN).read()
    assert "self.footprint_debug = FootprintDebugConfig()" in src
    assert "if self.footprint_debug.enabled" in src
    # the draw block sits after the corridor filter and is wrapped
    i_filter = src.index("footprint=self.footprint_blocking.active_footprint()")
    i_draw = src.index("self._footprint_drawer.draw(")
    assert i_draw > i_filter
    block = src[src.rfind("try:", 0, i_draw):i_draw + 400]
    assert "except Exception" in block and "note_failure" in block
    assert '"footprint_debug_enabled"' in src or "self.footprint_debug.state()" in src
    assert 'data.get("debug_enabled")' in src


def test_rectangle_corners_geometry():
    c = rectangle_corners(0.0, 0.0, math.pi / 2, 2.0, 1.0)
    xs = sorted(round(p[0], 6) for p in c); ys = sorted(round(p[1], 6) for p in c)
    assert xs == [-1.0, -1.0, 1.0, 1.0] and ys == [-2.0, -2.0, 2.0, 2.0]
