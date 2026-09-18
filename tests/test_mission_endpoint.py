"""Mission endpoint / parking target (2026-09-18): the chain from the requested pin to the place the van stops,
offline, on the road the rig drives -- the recorded route of WAV-0888 (x = 109.5, through junctions 675 and 566)
with the map facts the replay on the rig reported: a 2.0 m shoulder to the right everywhere, the junction spans,
the road ids. Live (every batch) the pin (109.43, 64.22) sat INSIDE junction 675, the pull-over found the nearest
stop its rules allow 78 m on, the van stopped there and the mission said "Arrived at destination"."""
import json
import math
import os
from types import SimpleNamespace

import pytest

from test_v2a_end_to_end import van, place, see  # noqa: F401
from warp_av.planning.planner import Route, Waypoint

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "endpoint", "recorded_routes.json"), encoding="utf-8"))
PIN_0888 = tuple(FIX["WAV-0888__2"]["pin"])            # (109.43, 64.22), inside junction 675
PIN_0001 = tuple(FIX["WAV-0001"]["pin"])               # (109.25, 90.22), plain road 1
PIN_BEND5 = tuple(FIX["WAV-V15-BEND5"]["pin"])         # (109.59, 40.23), inside junction 566


def as_route(pts):
    wps = [Waypoint(x=p["x"], y=p["y"], z=0.0, yaw=p["yaw"], speed=8.0, is_junction=bool(p["is_junction"]),
                    road_id=p.get("road_id"), lane_id=p.get("lane_id")) for p in pts]
    total = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(wps, wps[1:]))
    return Route(waypoints=wps, total_distance=total)


class FakeWP:
    """A CARLA waypoint, as far as the planner asks: on the lane line, with its neighbours."""
    def __init__(self, table, i, x, y, yaw_rad, road_id, lane_id, is_junction, lane_type="driving", width=3.5):
        self._table, self._i = table, i
        self.transform = SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=0.0),
                                         rotation=SimpleNamespace(yaw=math.degrees(yaw_rad)),
                                         get_right_vector=lambda: SimpleNamespace(x=-math.sin(yaw_rad), y=math.cos(yaw_rad), z=0.0))
        self.road_id, self.lane_id, self.is_junction, self.lane_type, self.lane_width = road_id, lane_id, is_junction, lane_type, width
        self.section_id = 0
        self._yaw = yaw_rad

    def _beside(self, offset_right, lane_id, lane_type, width):
        return FakeWP(self._table, self._i, self.transform.location.x - math.sin(self._yaw) * offset_right,
                      self.transform.location.y + math.cos(self._yaw) * offset_right, self._yaw,
                      self.road_id, lane_id, self.is_junction, lane_type, width)

    def get_right_lane(self):
        if self.lane_type != "driving":
            return None
        return self._beside(2.75, -3, "shoulder", 2.0)        # 3.5 / 2 + 2.0 / 2: the rig's shoulder

    def get_left_lane(self):
        return self._beside(-3.5, -1, "driving", 3.5) if self.lane_type == "driving" else None

    def next(self, d):
        j = self._i + max(1, int(round(d / 2.0)))
        return [self._table.wp(j)] if j < len(self._table.pts) else []


class EndpointMap:
    """The road as a table of lane-line points (a recorded route), carried on past its end the way the
    rig's map does: road 0 straight to y = -21.5, then road 10 bending away west (the replay on the rig:
    x 110.0 -> 104.5 between y -23.9 and -44.8, where extend_past_pin gives up), so that -- as on the rig --
    no 30 m straight run exists there for a shoulder pull-in."""
    def __init__(self, pts, carry_on_m=60.0):
        pts = list(pts)
        x, y, yaw = pts[-1]["x"], pts[-1]["y"], pts[-1]["yaw"]
        for k in range(1, int(carry_on_m / 2.0) + 1):
            if y <= -21.5:
                yaw -= 0.03                                  # 1.7 deg per 2 m: the bend of road 10
            x, y = x + 2.0 * math.cos(yaw), y + 2.0 * math.sin(yaw)
            pts.append({"x": x, "y": y, "yaw": yaw, "is_junction": False,
                        "road_id": 0 if y > -21.5 else 10, "lane_id": -2})
        self.pts = pts

    def wp(self, i):
        p = self.pts[i]
        return FakeWP(self, i, p["x"], p["y"], p["yaw"], p["road_id"], p["lane_id"], bool(p["is_junction"]))

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        i = min(range(len(self.pts)), key=lambda k: (self.pts[k]["x"] - loc.x) ** 2 + (self.pts[k]["y"] - loc.y) ** 2)
        return self.wp(i)


def straight_table(length_m=300.0):
    return [{"x": -20.0 + 2.0 * k, "y": 0.0, "yaw": 0.0, "is_junction": False, "road_id": 1, "lane_id": -2}
            for k in range(int(length_m / 2.0) + 1)]


def arm(v, monkeypatch, route_pts, table=None):
    """The real WarpAV.start_mission, with the map and the planned route handed in."""
    import carla
    monkeypatch.setattr(carla, "LaneType", SimpleNamespace(Driving="driving", Shoulder="shoulder",
                                                           Parking="parking", Sidewalk="sidewalk"), raising=False)
    m = EndpointMap(table if table is not None else FIX["WAV-0888__2"]["route"])
    v.planner.carla_map = m
    v.vehicle_adapter.get_map = lambda: m
    monkeypatch.setattr(v.planner, "plan_route", lambda sx, sy, ex, ey: as_route(route_pts))
    monkeypatch.setattr(v, "api_find_parking", lambda **k: {"success": False, "reason": "harness"})
    place(v, x=-18.0, y=140.6, yaw=0.0)
    return m


def drive_to_the_spot(v):
    sp = v._parking_spot
    place(v, x=sp["x"], y=sp["y"], yaw=sp["yaw"], speed=0.0)
    for _ in range(3):
        see(v, [])
        v.tick()
        v.clock.advance(0.25)
        if v._current_state["behavior"] == "mission_complete" or not v.mission_manager.current_mission:
            break
    return v.mission_manager.get_history()[-1]


# ---------------------------------------------------------------- Case B: the recorded 78 m
def test_case_b_the_recorded_pin_inside_junction_675_ends_78_m_on_and_the_mission_says_so(van, monkeypatch):
    pts = FIX["WAV-0888__2"]["route"][:FIX["WAV-0888__2"]["plan_route_points"]]
    arm(van, monkeypatch, pts)
    assert van.start_mission(*PIN_0888) is True
    judged = van._destination_judged
    assert judged["in_junction"] and judged["road_id"] == 675 and judged["off_road_m"] < 3.0, judged
    sp = van._parking_spot
    assert sp["kind"] == "lane" and abs(sp["past_pin_m"] - 78.0) < 0.6, sp             # the recorded stop, reproduced
    assert abs(sp["x"] - 109.94) < 0.3 and abs(sp["y"] + 11.05) < 0.6, sp
    assert math.hypot(sp["x"] - PIN_0888[0], sp["y"] - PIN_0888[1]) > 70.0
    note = van._stop_note
    assert "inside a junction (road 675)" in note and "78 m past it" in note, note
    assert van.mission_manager.get_status()["stop_note"] == note                          # on /api/state from tick one
    rec = drive_to_the_spot(van)
    assert rec["state"] == "completed", rec
    assert rec["reason_ended"].startswith("Stopped 75 m from the destination:") and "junction" in rec["reason_ended"], rec
    assert abs(rec["from_destination_m"] - 75.3) < 1.0 and rec["stopped_at"]["y"] < -10.0, rec


# ---------------------------------------------------------------- Case A / G: a good pin is untouched
def test_case_a_a_pin_on_plain_road_still_ends_at_the_pin_as_arrived(van, monkeypatch):
    pts = FIX["WAV-0001"]["route"]
    arm(van, monkeypatch, pts)
    assert van.start_mission(*PIN_0001) is True
    assert not van._destination_judged["in_junction"] and van._stop_note is None
    sp = van._parking_spot
    assert sp["kind"] == "lane" and math.hypot(sp["x"] - PIN_0001[0], sp["y"] - PIN_0001[1]) < 3.0, sp
    rec = drive_to_the_spot(van)
    assert rec["state"] == "completed" and rec["reason_ended"] == "Arrived at destination", rec
    assert rec["from_destination_m"] < 3.5 and rec["stop_note"] == "", rec


# ---------------------------------------------------------------- Case C: valid but not a stopping place
def test_case_c_a_pin_inside_junction_566_gets_the_kerb_60_m_on_explicitly(van, monkeypatch):
    pts = FIX["WAV-V15-BEND5"]["route"][:FIX["WAV-V15-BEND5"]["plan_route_points"]]
    arm(van, monkeypatch, pts)
    assert van.start_mission(*PIN_BEND5) is True
    assert van._destination_judged["in_junction"] and van._destination_judged["road_id"] == 566
    sp = van._parking_spot
    assert sp["kind"] == "kerb" and abs(sp["past_pin_m"] - 60.0) < 1.5, sp          # live: 60.0 m (the fake's shoulder line is 0.7 m out)
    assert "inside a junction (road 566)" in van._stop_note and "the kerb" in van._stop_note, van._stop_note
    assert "%.0f m past it" % sp["past_pin_m"] in van._stop_note, van._stop_note


def test_case_c2_a_pin_inside_a_junction_with_no_stop_in_the_window_is_refused_not_stopped_in_the_junction(van, monkeypatch):
    """WAV-0371's pin (109.36, 74.22), 10 m north of WAV-0888's: the same lane stop is 88 m on, outside the 80 m
    window, and the old answer was "the pin itself" -- a mission completed INSIDE junction 675."""
    pts = FIX["WAV-0888__2"]["route"][:84]                      # plan_route's 84 points to (109.34, 76.96)
    arm(van, monkeypatch, pts)
    assert van.start_mission(109.36, 74.22) is False
    rec = van.mission_manager.get_history()[-1]
    assert rec["state"] == "failed" and "inside a junction (road 675)" in rec["reason_ended"] and "80 m" in rec["reason_ended"], rec
    assert van._route is None and van.mission_manager.current_mission is None


# ---------------------------------------------------------------- Case D: the projection is named, and it is the route's lane
def test_case_d_the_judgement_names_the_lane_and_it_is_the_one_the_route_ends_in(van, monkeypatch):
    pts = FIX["WAV-0888__2"]["route"][:FIX["WAV-0888__2"]["plan_route_points"]]
    arm(van, monkeypatch, pts)
    j = van.planner.judge_destination(*PIN_0888)
    end = as_route(pts).waypoints[-1]
    assert (j["road_id"], j["lane_id"]) == (end.road_id, end.lane_id) == (675, -2), (j, end)
    # a pin on the shoulder beside the lane is judged against the DRIVING lane, 2.75 m off
    right = (PIN_0001[0] - math.sin(-math.pi / 2) * 2.75, PIN_0001[1] + math.cos(-math.pi / 2) * 2.75)
    j2 = van.planner.judge_destination(*right)
    assert j2["lane_id"] == -2 and 2.0 < j2["off_road_m"] < 4.0 and not j2["in_junction"], j2


# ---------------------------------------------------------------- Case E: nearer safe stop wins
def test_case_e_a_bay_at_the_pin_beats_any_stop_farther_on(van, monkeypatch):
    table = straight_table()
    arm(van, monkeypatch, table[:111], table=table)             # route to x = 200
    assert van.start_mission(200.0, 0.0) is True
    sp = van._parking_spot
    assert sp["kind"] == "bay" and sp["from_pin_m"] < 3.0 and sp["past_pin_m"] < 0.5, sp
    assert van._stop_note is None


# ---------------------------------------------------------------- Case F: off the map
def test_case_f_an_off_map_pin_is_refused_before_any_route_is_planned(van, monkeypatch):
    arm(van, monkeypatch, FIX["WAV-0888__2"]["route"][:89])
    monkeypatch.setattr(van.planner, "plan_route", lambda *a: (_ for _ in ()).throw(AssertionError("planned a route to nowhere")))
    assert van.start_mission(9000.0, 9000.0) is False
    rec = van.mission_manager.get_history()[-1]
    assert rec["state"] == "failed" and "from the nearest road" in rec["reason_ended"] and "40 m" in rec["reason_ended"], rec
    assert " 125" in rec["reason_ended"] or " 126" in rec["reason_ended"], rec["reason_ended"]     # ~12,5xx m
    assert van._route is None


# ---------------------------------------------------------------- without a map: nothing changes
def test_without_a_map_the_mission_runs_as_before(van, monkeypatch):
    assert van.planner.judge_destination(*PIN_0001) is None
    monkeypatch.setattr(van.planner, "plan_route", lambda sx, sy, ex, ey: as_route(FIX["WAV-0001"]["route"]))
    monkeypatch.setattr(van, "api_find_parking", lambda **k: {"success": False, "reason": "harness"})
    place(van, x=-18.0, y=140.6, yaw=0.0)
    assert van.start_mission(*PIN_0001) is True
    assert van._destination_judged is None and van._stop_note is None
    rec = drive_to_the_spot(van) if van._parking_spot else None
    if rec is not None:
        assert rec["reason_ended"] == "Arrived at destination"
