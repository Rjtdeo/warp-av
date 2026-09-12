"""A street that stays blocked is a street to go round.

Until 2026-09-11 a road the van could not get past ended the mission one way: stand there and
wait for a person, for ever. The map knows other streets; the only question is whether one of
them still reaches the destination without going through the blockage -- and whether the van
can take it, which needs a junction between it and the block, because there is no reverse
gear.
"""
from pathlib import Path

from warp_av.behavior import transitions as T


def test_going_round_and_finding_no_way_round_are_both_in_the_story():
    assert T.REROUTED in T.ALL_MOVES and T.NO_WAY_ROUND in T.ALL_MOVES


def test_the_map_is_asked_by_making_the_blocked_road_expensive():
    """The road graph CARLA's route search walks is the map's own; a blocked street is one
    edge of it. Made expensive, any other way round wins; still finite, so a street with no
    other way out still plans and the caller can see the answer goes through it anyway."""
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "planning" / "planner.py").read_text()
    i = src.index("def plan_route_avoiding")
    body = src[i:i + 2200]
    assert "_road_id_to_edge[wp.road_id][wp.section_id][wp.lane_id]" in body
    assert 'edge["length"] = was + self.AVOID_COST_M' in body
    assert 'finally:' in body and 'edge["length"] = was' in body, "the cost is always put back"
    assert "if near <= clear_m:" in body and "return None" in body, \
        "a route that still goes through the blockage is not a way round"


def test_the_van_only_asks_when_there_is_somewhere_to_turn_off():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("def _maybe_reroute")
    body = src[i:i + 3200]
    assert "span = self.planner.junction_span" in body
    assert "if span is None or span[0] > at_m:" in body, \
        "no junction between us and the block means no way round without reversing"
    assert "NO_WAY_ROUND" in body and "REROUTED" in body
    assert "self._dress_route_for_parking()" in body, "a new route needs a new parking spot"
    assert "REROUTE_AFTER_S" in body and "REROUTE_EVERY_S" in body, \
        "watch it a while, and do not ask the map every tick"


def test_a_new_route_is_dressed_for_parking_the_same_way_a_new_mission_is():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert src.count("self._dress_route_for_parking()") >= 2
    i = src.index("def _dress_route_for_parking")
    body = src[i:i + 2500]
    assert "extend_past_pin" in body and "_route_base" in body and "_choose_spot" in body
