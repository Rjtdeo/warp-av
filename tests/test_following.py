"""
Troy fix #6: follow a moving vehicle at a time gap — no more stop-and-go.

Includes a closed-loop simulation: a lead car drives a straight road at
constant speed, the van starts 30 m behind at cruise. With the old logic the
van sawtoothed (slow to 3, lead escapes, speed to 8, repeat). Now it must
settle at the lead's speed with a stable gap and never come to a halt.
"""
import math

from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput, ObjectType
from warp_av.localization.localization import Pose
from warp_av.control.controller import VehicleController
from test_controller_stability import SimVan, DT


def perc(gap, lead_speed, kind=ObjectType.VEHICLE):
    return PerceptionOutput(
        closest_obstacle_distance=gap,
        closest_obstacle_type=kind,
        closest_obstacle_speed=lead_speed,
        path_blocked=gap < 8.0,
    )


def test_gap_law_directions():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    lead = 5.0
    want = b.follow_standstill_m + b.follow_time_gap_s * lead   # 15.5 m

    far = b.update(perc(25.0, lead), pose, 500, True)           # too far: close in
    assert far.behavior == DrivingBehavior.FOLLOWING_VEHICLE
    assert far.desired_speed_mps > lead

    at = b.update(perc(want, lead), pose, 500, True)            # at gap: match speed
    assert abs(at.desired_speed_mps - lead) < 0.01

    close = b.update(perc(10.0, lead), pose, 500, True)         # too close: drop back
    assert 0.0 < close.desired_speed_mps < lead


def test_following_never_applies_to_pedestrians_or_stopped_vehicles():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    # moving pedestrian ahead: slow zone, never "following"
    out = b.update(perc(15.0, 1.4, kind=ObjectType.PEDESTRIAN), pose, 500, True)
    assert out.behavior != DrivingBehavior.FOLLOWING_VEHICLE
    assert out.desired_speed_mps <= b.slow_speed
    # stopped vehicle (or camera mode with unknown speed=0): slow zone as before
    out = b.update(perc(15.0, 0.0), pose, 500, True)
    assert out.behavior == DrivingBehavior.FOLLOWING_ROUTE
    assert out.desired_speed_mps == b.slow_speed
    # blocked range still stops hard
    out = b.update(perc(7.0, 0.0), pose, 500, True)
    assert out.behavior == DrivingBehavior.STOPPED_VEHICLE and out.should_stop


def test_closed_loop_no_stop_and_go():
    LEAD_SPEED = 4.0
    b = BehaviorSystem(); b.set_mission()
    ctrl = VehicleController()
    van = SimVan(speed=8.0)
    lead_x = 30.0
    pose = Pose(healthy=True)

    gaps, speeds = [], []
    for i in range(int(60 / DT)):
        lead_x += LEAD_SPEED * DT
        gap = lead_x - van.x
        out = b.update(perc(gap, LEAD_SPEED), pose, 500, True)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed,
                                   van.x + 12.0, 0.0, out.desired_speed_mps, out.should_stop)
        van.step(cmd)
        gaps.append(gap); speeds.append(van.speed)

    assert min(gaps) > 5.0, f"got dangerously close: {min(gaps):.1f} m"
    settle = int(25 / DT)
    want = 8.0 + 1.5 * LEAD_SPEED   # 14 m
    late_gaps = gaps[settle:]
    late_speeds = speeds[settle:]
    assert min(late_speeds) > 2.0, f"stop-and-go still happening: dropped to {min(late_speeds):.1f} m/s"
    assert abs(sum(late_speeds) / len(late_speeds) - LEAD_SPEED) < 0.5, "not matching lead speed"
    assert want - 4 < sum(late_gaps) / len(late_gaps) < want + 4, f"gap not settling near {want} m: {sum(late_gaps)/len(late_gaps):.1f}"
    # gap must be stable, not oscillating metres back and forth
    assert max(late_gaps) - min(late_gaps) < 4.0, f"gap oscillates {max(late_gaps)-min(late_gaps):.1f} m"
