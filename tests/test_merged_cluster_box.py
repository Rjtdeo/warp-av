"""A rectangle no thing of that kind could be is a merge, not a body.

Perception already throws away a SIGHTING that is impossible for its class -- a vehicle is at
most 4.0 m across, however long a bus is (tracking.CLASS_SIZE_LIMITS). The fitted rectangle it
hands the planner was never given the same test.

Live in F_reroute on 2026-09-15 a stopped truck, 5.20 x 2.63 m in CARLA's own record, merged
with the kerb beside it. The van tracked it as one "vehicle" 3.8 to 5.4 m wide on every one of
1350 ticks, mean 4.69 m, centred 5 m from where the truck actually was and overlapping the
van's own body -- so all four swept-path refusals reported contact at nought metres along, and
every way round was refused by construction rather than by geometry.
"""
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.planner import fit_is_believable, obstacle_box_for


def thing(kind=ObjectType.VEHICLE, points=(6.20, 3.40), box=(5.06, 4.40), height=2.0):
    """points = the spread of the laser points; box = the rectangle perception fitted to them."""
    return DetectedObject(object_type=kind, x=2.0, y=1.7, distance=2.7, speed=0.0,
                          stationary=True, height_m=height,
                          length_m=points[0], width_m=points[1],
                          box_length_m=box[0], box_width_m=box[1], box_yaw_deg=172.5)


def test_a_vehicle_wider_than_any_vehicle_is_not_believed():
    """The live merge. 4.40 m across is wider than the 4.0 m a vehicle may be."""
    assert fit_is_believable(thing()) is False


def test_the_planner_then_uses_the_points_the_laser_actually_saw():
    """Both paths widen what they are given to cover a heading error, so compare the answers,
    not the inputs: the impossible rectangle blocked 5.30 m of road, the points 4.46 m."""
    merged = thing()                                          # judged: not believable
    same_but_believable = thing(kind=ObjectType.OBSTACLE)      # same numbers, no size table
    from_points = obstacle_box_for(merged, 0.0)
    from_fit = obstacle_box_for(same_but_believable, 0.0)
    assert from_points is not None and from_fit is not None
    assert 2 * from_points.half_width < 2 * from_fit.half_width, \
        "the merge blocks less road than the rectangle it was refused for"
    assert 4.0 < 2 * from_points.half_width < 5.0, 2 * from_points.half_width


def test_it_is_still_far_too_wide_to_pass_and_F_stays_refused():
    """Narrower is not clear. The cluster sat 1.71 m to the van's right and the van is 1.29 m
    of half width plus margin, so anything over 0.42 m of half width still overlaps it. This
    fix makes the shape honest; it does not open a way past a truck."""
    box = obstacle_box_for(thing(), 0.0)
    assert box.half_width > 0.42 + 1.0, "still overlapping the van where it stood"


def test_a_real_car_is_still_believed():
    """The guard must not touch anything that is actually a vehicle."""
    car = thing(points=(4.79, 2.16), box=(4.79, 2.16), height=1.49)
    assert fit_is_believable(car) is True
    box = obstacle_box_for(car, 0.0)
    assert box is not None and 2 * box.half_width < 3.0, "a car is still car-shaped"


def test_a_lorry_is_long_and_is_still_believed():
    """Length is not the test: a bus is 12 m long and still a vehicle."""
    lorry = thing(points=(12.0, 2.55), box=(12.0, 2.55), height=3.2)
    assert fit_is_believable(lorry) is True


def test_the_truck_in_F_at_its_real_size_is_believed():
    truck = thing(points=(5.20, 2.63), box=(5.20, 2.63), height=2.5)
    assert fit_is_believable(truck) is True


def test_the_class_is_what_decides_and_only_the_class():
    """The same numbers, twice. Nothing claims to know how wide an obstacle may be, so the
    new rule must leave that one alone -- it is the class table doing the work, not the size."""
    assert fit_is_believable(thing(kind=ObjectType.VEHICLE)) is False
    assert fit_is_believable(thing(kind=ObjectType.OBSTACLE)) is True


def test_the_post_in_E3_was_already_refused_by_the_older_area_rule():
    """Not this change: a 1.05 x 0.10 m rectangle over 0.87 x 0.06 m of points claims twice
    the ground, which the area test has refused since 2026-09-11."""
    post = thing(kind=ObjectType.OBSTACLE, points=(0.87, 0.06), box=(1.05, 0.10), height=2.2)
    assert fit_is_believable(post) is False


def test_the_area_test_that_was_there_before_still_works():
    """A fit claiming far more ground than the points is still a fit over empty ground: the
    51 m facade that came out a 22.6 x 12.4 m block (2026-09-11)."""
    facade = thing(kind=ObjectType.OBSTACLE, points=(51.1, 2.8), box=(22.6, 12.4), height=8.0)
    assert fit_is_believable(facade) is False


def test_an_impossible_width_is_caught_even_when_the_area_test_passes():
    """The live merge slipped through precisely because its POINTS were huge too: fitted area
    22.3 m2 against 21.1 m2 of points is well inside the area slack."""
    o = thing()
    fit_area = 5.06 * 4.40
    point_area = 6.20 * 3.40
    assert fit_area <= 1.6 * point_area, "the old test had no reason to complain"
    assert fit_is_believable(o) is False, "the new one does"


def test_nothing_measured_is_still_the_cautious_answer():
    bare = DetectedObject(object_type=ObjectType.VEHICLE, x=5.0, y=0.0, distance=5.0,
                          speed=0.0, stationary=True)
    assert fit_is_believable(bare) is True
    assert obstacle_box_for(bare, 0.0) is None, "no rectangle: the caller falls back to the circle"


def test_the_guard_never_throws_on_an_odd_object():
    class Odd:
        object_type = None
        box_length_m = float("nan")
        box_width_m = 2.0
        length_m = 1.0
        width_m = 1.0
        height_m = 1.0
    assert isinstance(fit_is_believable(Odd()), bool)
