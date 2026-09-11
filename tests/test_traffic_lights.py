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
    b._block_memory = None          # priority test, not a latch-timing test
    o = out(b, traffic_light="red", closest_obstacle_type=ObjectType.VEHICLE,
            closest_obstacle_distance=20.0, closest_obstacle_speed=5.0)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT


def test_camera_mode_default_is_unchanged_behavior():
    # camera mode doesn't classify lights yet -> traffic_light stays "none"
    assert PerceptionOutput().traffic_light == "none"
    b = BehaviorSystem(); b.set_mission()
    assert out(b).behavior == DrivingBehavior.FOLLOWING_ROUTE


def at(b, state, stop_line_m=None, speed=0.0, light_id=7, **kw):
    """One decision with the bumper `stop_line_m` from the stop line, doing `speed`."""
    return b.update(PerceptionOutput(traffic_light=state), Pose(healthy=True, speed=speed),
                    500, True, stop_line_m=stop_line_m, light_id=light_id, **kw)


def test_a_red_light_with_no_known_line_stops_at_once():
    b = BehaviorSystem(); b.set_mission()
    o = at(b, "red", stop_line_m=None)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT and o.should_stop


def test_the_stop_is_measured_from_the_front_bumper():
    b = BehaviorSystem(); b.set_mission()
    far = at(b, "red", stop_line_m=5.0)
    assert far.behavior == DrivingBehavior.FOLLOWING_ROUTE and not far.should_stop
    assert "5.0 m to the stop line" in far.reason
    near = at(b, "red", stop_line_m=0.6)
    assert near.behavior == DrivingBehavior.STOPPED_RED_LIGHT and near.should_stop


def test_without_a_bumper_distance_the_middle_of_the_van_is_turned_into_one():
    """Old callers give only the distance from the van's middle: the bumper is 2.95 m ahead."""
    b = BehaviorSystem(); b.set_mission()
    o = b.update(PerceptionOutput(traffic_light="red", traffic_light_distance_m=3.5),
                 Pose(healthy=True), 500, True)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT, "bumper is 0.55 m from the line"


def test_rolling_up_ends_with_the_bumper_short_of_the_line_not_over_it():
    """Drive the approach: the van does what behaviour asks, then rolls 0.1 m once told to stop.
    The old rule ended with the bumper 0.35 m past its line -- and its line was the zebra."""
    b = BehaviorSystem(); b.set_mission()
    d, v, dt = 30.0, 8.0, 0.1
    for _ in range(600):
        o = at(b, "red", stop_line_m=d, speed=v)
        if o.should_stop:
            d -= 0.1                           # the roll after the brake
            break
        v = o.desired_speed_mps
        d -= v * dt
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT
    assert 0.2 <= d <= 1.0, f"stopped with the bumper {d:.2f} m from the line"


def test_already_over_the_line_when_it_changes_means_clear_the_junction():
    b = BehaviorSystem(); b.set_mission()
    at(b, "green", stop_line_m=-1.0, speed=6.0)
    o = at(b, "red", stop_line_m=-1.2, speed=6.0)
    assert o.behavior != DrivingBehavior.STOPPED_RED_LIGHT, "must clear the junction, not freeze in it"
    assert b.light_status["choice"] == "go"


def test_a_creep_over_the_line_at_a_red_never_turns_into_going():
    """Decided once: a van that chose to stop stays stopped, even a few centimetres over.
    The old rule -- carry on through any colour once within 1 m of the junction edge -- let it
    roll on into the junction."""
    b = BehaviorSystem(); b.set_mission()
    assert not at(b, "red", stop_line_m=10.0, speed=3.0).should_stop       # rolling up
    o = at(b, "red", stop_line_m=-0.3, speed=0.2)                            # crept over
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT and o.should_stop


def test_yellow_too_close_to_stop_goes_through_and_keeps_going_on_red():
    b = BehaviorSystem(); b.set_mission()
    at(b, "green", stop_line_m=20.0, speed=8.0)
    o = at(b, "yellow", stop_line_m=8.0, speed=8.0)     # needs 12.8 m to stop at 2.5 m/s2
    assert not o.should_stop and o.behavior != DrivingBehavior.STOPPED_RED_LIGHT
    o = at(b, "red", stop_line_m=2.0, speed=8.0)
    assert not o.should_stop, "the choice to go on was made at the yellow"


def test_yellow_with_room_to_stop_stops():
    b = BehaviorSystem(); b.set_mission()
    at(b, "green", stop_line_m=40.0, speed=8.0)
    o = at(b, "yellow", stop_line_m=30.0, speed=8.0)
    assert o.desired_speed_mps <= 3.0 and "YELLOW" in o.reason
    assert b.light_status["choice"] == "stop"


def test_an_unreadable_light_is_never_a_reason_to_go_on():
    """Unknown is not yellow: it gets no 'too close to stop' pass."""
    b = BehaviorSystem(); b.set_mission()
    at(b, "green", stop_line_m=10.0, speed=8.0)
    o = at(b, "unknown", stop_line_m=3.0, speed=8.0)
    assert o.should_stop or o.desired_speed_mps <= 3.0
    assert b.light_status["choice"] == "stop"


def test_the_choice_is_made_fresh_after_every_green():
    b = BehaviorSystem(); b.set_mission()
    at(b, "red", stop_line_m=0.5)
    assert not at(b, "green", stop_line_m=0.5).should_stop
    at(b, "yellow", stop_line_m=20.0, speed=0.5)
    assert b.light_status["choice"] == "stop"


def test_a_new_light_is_a_new_choice():
    b = BehaviorSystem(); b.set_mission()
    at(b, "green", stop_line_m=-1.0, speed=6.0, light_id=1)
    at(b, "red", stop_line_m=-1.2, speed=6.0, light_id=1)        # committed through light 1
    o = at(b, "red", stop_line_m=30.0, speed=6.0, light_id=2)    # the next light
    assert b.light_status["choice"] == "stop" and o.desired_speed_mps <= 3.0


def test_a_predicted_crosser_never_lets_the_van_skip_a_red_light():
    """The 'slowing for a predicted crosser' branch returned 2.5 m/s without ever reaching the
    light. At a junction with cross traffic, that is how a van rolls through a red."""
    crosser = {"t": 3.0, "along_m": 20.0, "type": "vehicle"}
    b = BehaviorSystem(); b.set_mission()
    o = at(b, "red", stop_line_m=0.5, speed=1.0, predicted_conflict=crosser)
    assert o.behavior == DrivingBehavior.STOPPED_RED_LIGHT and o.should_stop
    b = BehaviorSystem(); b.set_mission()
    o = at(b, "red", stop_line_m=20.0, speed=3.0, predicted_conflict=crosser)
    assert o.desired_speed_mps <= 2.5, "the stricter of the two must win"
