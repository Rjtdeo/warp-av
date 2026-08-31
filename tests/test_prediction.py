"""Prediction: yield to crossers/cut-ins BEFORE they are in the path."""
from warp_av.planning.prediction import predict_route_conflict
from warp_av.planning.planner import Route, Waypoint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.localization.localization import Pose


def wps():
    return [Waypoint(x=i * 2.0, y=0.0) for i in range(40)]


def obj(x, y, vx=0.0, vy=0.0, kind=ObjectType.VEHICLE):
    import math
    return DetectedObject(object_type=kind, x=x, y=y,
                          distance=math.hypot(x, y),
                          speed=math.hypot(vx, vy),
                          vx_world=vx, vy_world=vy)


def test_side_crosser_is_predicted():
    # car 8 m to our right, 15 m ahead, driving toward our lane at 4 m/s
    c = predict_route_conflict([obj(15.0, 8.0, vy=-4.0)], wps(), 0, 0, 0.0, 5.0)
    assert c is not None
    assert c["t"] <= 2.5 and 10.0 < c["along_m"] < 20.0


def test_oncoming_in_its_own_lane_is_ignored():
    # opposite-lane car staying 3.5 m left the whole time
    c = predict_route_conflict([obj(20.0, -3.5, vx=-8.0)], wps(), 0, 0, 0.0, 5.0)
    assert c is None


def test_lead_car_in_our_lane_is_following_logics_job():
    c = predict_route_conflict([obj(10.0, 0.3, vx=5.0)], wps(), 0, 0, 0.0, 5.0)
    assert c is None


def test_parked_cars_never_trigger_prediction():
    c = predict_route_conflict([obj(12.0, 1.8)], wps(), 0, 0, 0.0, 5.0)
    assert c is None


def test_behavior_yields_close_and_slows_far():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    close = b.update(PerceptionOutput(), pose, 500, True,
                     predicted_conflict={"t": 1.6, "along_m": 9.0, "type": "vehicle"})
    assert close.behavior == DrivingBehavior.YIELDING_PREDICTED and close.should_stop
    assert "Yielding" in close.reason
    far = b.update(PerceptionOutput(), pose, 500, True,
                   predicted_conflict={"t": 2.8, "along_m": 22.0, "type": "pedestrian"})
    assert far.behavior == DrivingBehavior.YIELDING_PREDICTED and not far.should_stop
    assert far.desired_speed_mps == 2.5
    none = b.update(PerceptionOutput(), pose, 500, True, predicted_conflict=None)
    assert none.behavior == DrivingBehavior.FOLLOWING_ROUTE
