"""Troy fix #1: obey traffic lights (red/yellow stop, green go, hazards still outrank)."""
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput, ObjectType
from warp_av.localization.localization import Pose


def out(b, **kw):
    return b.update(PerceptionOutput(**kw), Pose(healthy=True), 500, True)


def test_red_and_yellow_stop_green_goes():
    b = BehaviorSystem(); b.set_mission()
    r = out(b, traffic_light="red")     # unknown stop-line distance -> stop now
    assert r.behavior == DrivingBehavior.STOPPED_RED_LIGHT and r.should_stop
    assert "RED" in r.reason and "green" in r.reason.lower()
    y = out(b, traffic_light="yellow")
    assert y.behavior == DrivingBehavior.STOPPED_RED_LIGHT and y.should_stop
    g = out(b, traffic_light="green")
    assert g.behavior == DrivingBehavior.FOLLOWING_ROUTE and not g.should_stop
    n = out(b, traffic_light="none")
    assert n.behavior == DrivingBehavior.FOLLOWING_ROUTE


def test_red_light_far_away_rolls_up_to_the_line():
    b = BehaviorSystem(); b.set_mission()
    far = out(b, traffic_light="red", traffic_light_distance_m=18.0)
    assert far.behavior == DrivingBehavior.FOLLOWING_ROUTE and not far.should_stop
    assert far.desired_speed_mps <= 4.0 and "rolling up" in far.reason
    near = out(b, traffic_light="red", traffic_light_distance_m=2.6)
    assert near.behavior == DrivingBehavior.STOPPED_RED_LIGHT and near.should_stop


def test_green_light_releases_red_stop():
    b = BehaviorSystem(); b.set_mission()
    assert out(b, traffic_light="red").should_stop
    after = out(b, traffic_light="green")
    assert not after.should_stop and after.desired_speed_mps == b.cruise_speed


def test_physical_hazards_outrank_the_light():
    b = BehaviorSystem(); b.set_mission()
    # pedestrian in path at a red light: the reason must be the pedestrian
    o = out(b, traffic_light="red", path_blocked=True,
            closest_obstacle_type=ObjectType.PEDESTRIAN, closest_obstacle_distance=6.0)
    assert o.behavior == DrivingBehavior.STOPPED_PEDESTRIAN
    # red light outranks car-following (no creeping into the junction behind traffic)
    o = out(b, traffic_light="red", closest_obstacle_type=ObjectType.VEHICLE,
            closest_obstacle_distance=20.0, closest_obstacle_speed=5.0)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT


def test_camera_mode_default_is_unchanged_behavior():
    # camera mode doesn't classify lights yet -> traffic_light stays "none"
    assert PerceptionOutput().traffic_light == "none"
    b = BehaviorSystem(); b.set_mission()
    assert out(b).behavior == DrivingBehavior.FOLLOWING_ROUTE


def test_red_light_without_stop_line_uses_junction_edge():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    # no stop-line data, but the map says the junction starts 14 m ahead: roll up
    o = b.update(PerceptionOutput(traffic_light="red"), pose, 500, True, junction_ahead_m=14.0)
    assert o.behavior == DrivingBehavior.FOLLOWING_ROUTE and not o.should_stop
    assert "junction edge" in o.reason and o.desired_speed_mps <= 4.0
    # close to the junction: hold
    o = b.update(PerceptionOutput(traffic_light="red"), pose, 500, True, junction_ahead_m=2.5)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT and o.should_stop
    # no data at all: stop immediately (early is the safe direction)
    o = b.update(PerceptionOutput(traffic_light="red"), pose, 500, True, junction_ahead_m=None)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT and o.should_stop


def test_red_light_holds_relative_to_junction_entry_not_carla_stop_wp():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    # CARLA says the stop line is 2 m away (early data) but the junction is
    # actually 10 m ahead: keep rolling — junction entry wins
    o = b.update(PerceptionOutput(traffic_light="red", traffic_light_distance_m=2.0),
                 pose, 500, True, junction_ahead_m=10.0)
    assert not o.should_stop and "junction edge" in o.reason
    # at 3.0 m from the junction entry: hold (operator-tuned hold 2.6)
    o = b.update(PerceptionOutput(traffic_light="red", traffic_light_distance_m=2.0),
                 pose, 500, True, junction_ahead_m=3.0)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT


def test_committed_crossing_never_stops_inside_the_junction():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    o = b.update(PerceptionOutput(traffic_light="red"), pose, 500, True,
                 junction_ahead_m=0.0)
    assert o.behavior != DrivingBehavior.STOPPED_RED_LIGHT, "must clear the junction, not freeze in it"


def test_white_line_outranks_junction_edge_reference():
    """Operator/Troy: hold at the PAINTED line, not the (earlier) junction
    edge polygon. When the crosswalk paint is known, it wins."""
    b = BehaviorSystem(); b.set_mission()
    far = b.update(PerceptionOutput(traffic_light="red"), Pose(healthy=True), 500, True,
                   junction_ahead_m=4.0, white_line_m=20.0)
    assert far.behavior == DrivingBehavior.FOLLOWING_ROUTE, \
        "junction edge says 4 m but the paint is 20 m ahead — keep rolling"
    assert "white line" in far.reason
    near = b.update(PerceptionOutput(traffic_light="red"), Pose(healthy=True), 500, True,
                    junction_ahead_m=9.0, white_line_m=3.1)
    assert near.behavior == DrivingBehavior.STOPPED_RED_LIGHT and near.should_stop
    assert "white line" in near.reason


def test_no_paint_falls_back_to_junction_edge():
    b = BehaviorSystem(); b.set_mission()
    r = b.update(PerceptionOutput(traffic_light="red"), Pose(healthy=True), 500, True,
                 junction_ahead_m=3.0, white_line_m=None)
    assert r.behavior == DrivingBehavior.STOPPED_RED_LIGHT
    assert "junction edge" in r.reason
