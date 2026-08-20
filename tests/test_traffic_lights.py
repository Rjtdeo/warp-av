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
