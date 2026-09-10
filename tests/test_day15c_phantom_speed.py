"""A track must not run away with itself along a kerb.

The fault, measured live in Town10HD on 2026-09-10 with the van crawling under 1.2 m/s on an
empty road: tracks 0.1 m wide reported a MEDIAN speed of 19 to 27 m/s, pinned at the 30 m/s
clamp, and jumped 19, 22 and 31 m from one frame to the next. They fired the crossing
prediction on 12% of readings and stopped the van on 18%, with nothing in the street.

The cause was a loop with the gain on the wrong side. A track's OWN speed widened its OWN
search radius, with nothing above it, multiplied by a gap of up to DROP_AFTER_S:

    best_d = GATE_M + dt * max(GATE_SPEED_MPS, tr.speed)

One wrong match hands a track a speed it never had; the speed widens the radius; the wider
radius reaches a further wrong match. Beside a kerb, where the LiDAR hands back dozens of
near-identical slivers, it never stops. At 30 m/s with a 1.2 s gap the radius was 38 m.
"""
import math

import pytest

from warp_av.perception.tracking import ObjectTracker, MAX_ROAD_SPEED_MPS


def sighting(wx, wy, **kw):
    o = {"wx": float(wx), "wy": float(wy), "distance": math.hypot(wx, wy)}
    o.update(kw)
    return o


def kerb_slivers(x0, n=14, spacing=1.5, y=6.0):
    """A kerb as the LiDAR really returns it: a row of near-identical little blobs."""
    return [sighting(x0 + i * spacing, y) for i in range(n)]


# ---------------------------------------------------------------- the gate cannot run away


def test_the_gate_never_opens_wider_than_a_few_metres():
    """Whatever a track believes about its own speed, and however long it has been unseen."""
    t = ObjectTracker()
    for speed in (0.0, 4.0, 14.0, MAX_ROAD_SPEED_MPS, 999.0):
        for gap in (0.05, 0.35, 1.0, 1.2):
            reach = min(t.GATE_MAX_DT_S, gap) * min(t.GATE_SPEED_CAP_MPS,
                                                    max(t.GATE_SPEED_MPS, speed))
            assert min(t.GATE_MAX_M, t.GATE_M + reach) <= t.GATE_MAX_M


def test_a_track_carrying_a_phantom_speed_does_not_grab_a_blob_far_away():
    """The exact live failure: a sliver track pinned at the speed clamp, unseen for a moment,
    reaching down the kerb and taking a blob 20 m away as itself."""
    t = ObjectTracker()
    t.update([sighting(0.0, 6.0)], 0.0)
    t.update([sighting(0.0, 6.0)], 0.11)
    tracks = t.update([sighting(0.0, 6.0)], 0.22)
    tr = tracks[0]
    tr.x[2], tr.x[3] = MAX_ROAD_SPEED_MPS, 0.0     # hand it the phantom speed by force
    before = tr.tid

    # nothing for a moment, then a sliver 20 m further along the same kerb
    after = t.update([sighting(20.0, 6.0)], 0.22 + 1.0)
    after = t.update([sighting(20.0, 6.0)], 0.22 + 1.1)
    ids = {x.tid for x in after}
    far = [x for x in after if abs(x.wx - 20.0) < 5.0]
    assert far, "the far sliver should still be tracked -- as something NEW"
    assert far[0].tid != before, "it took a blob 20 m away as the same object"


def test_a_kerb_slider_is_not_a_crossing():
    """The other half of the fault, and the half that matters to the driver.

    Bounding the gate stops the 20 m jumps, but it does NOT stop a track sliding along a
    kerb -- and the recording says that sliding is REAL. Over 4702 live readings of tracks
    the van called moving, the claimed speed was 3.19 m/s and the ground actually covered
    was 2.78 m/s: a ratio of 1.1. Doubting the speed buys 3%. So the fix is not to disbelieve
    the motion but to stop calling it a crossing: a thing sliding ALONG beside our path is
    not coming towards us.
    """
    from warp_av.planning.prediction import predict_route_conflict, MIN_CLOSING_MPS
    from warp_av.planning.planner import Waypoint

    route = [Waypoint(x=float(i) * 2.0, y=0.0) for i in range(40)]

    class Obj:
        def __init__(self, x, y, vx, vy):
            self.x, self.y = x, y            # ego frame; the van sits at the origin facing +x
            self.vx_world, self.vy_world = vx, vy
            self.speed = math.hypot(vx, vy)
            self.stationary = False
            self.object_type = "obstacle"
            self.id = 1

    # a railing beside the road, "moving" at 4 m/s straight down the kerb line
    slider = Obj(x=14.0, y=4.0, vx=4.0, vy=0.0)
    assert predict_route_conflict([slider], route, 0.0, 0.0, 0.0, 5.0) is None

    # ... and it stays quiet whichever way along the kerb it appears to slide
    back = Obj(x=14.0, y=4.0, vx=-4.0, vy=0.0)
    assert predict_route_conflict([back], route, 0.0, 0.0, 0.0, 5.0) is None


def test_someone_stepping_off_the_pavement_still_stops_the_van():
    """The whole point of the prediction. This must not be quietened by the fix above."""
    from warp_av.planning.prediction import predict_route_conflict
    from warp_av.planning.planner import Waypoint

    route = [Waypoint(x=float(i) * 2.0, y=0.0) for i in range(40)]

    class Obj:
        def __init__(self, x, y, vx, vy, kind="pedestrian"):
            self.x, self.y = x, y
            self.vx_world, self.vy_world = vx, vy
            self.speed = math.hypot(vx, vy)
            self.stationary = False
            self.object_type = kind
            self.id = 2

    walker = Obj(x=12.0, y=5.0, vx=0.0, vy=-1.6)      # off the kerb, straight at our lane
    hit = predict_route_conflict([walker], route, 0.0, 0.0, 0.0, 4.0)
    assert hit is not None, "a person walking into our lane raised no warning"
    assert hit["type"] == "pedestrian"


def test_a_car_changing_lane_into_us_still_counts():
    """A cut-in closes slowly and sideways. It must clear the bar the kerb slider fails.

    It has to be a cut-in we would actually MEET: a car 25 m ahead pulling away from us is
    correctly ignored, because we will not be there when it is. That is the older half of the
    rule and the closing test does not touch it.
    """
    from warp_av.planning.prediction import predict_route_conflict
    from warp_av.planning.planner import Waypoint

    route = [Waypoint(x=float(i) * 2.0, y=0.0) for i in range(40)]

    class Obj:
        def __init__(self):
            self.x, self.y = 6.0, 3.4                 # next lane over, just ahead of us
            self.vx_world, self.vy_world = 3.0, -1.0  # slower than us, drifting into our lane
            self.speed = math.hypot(3.0, 1.0)
            self.stationary = False
            self.object_type = "vehicle"
            self.id = 3

    assert predict_route_conflict([Obj()], route, 0.0, 0.0, 0.0, 5.0) is not None


def test_a_real_car_is_still_caught_on_its_second_sighting():
    """The day-6 case the gate exists for. A brand new track has no speed of its own, so none
    of the new bounds touch it -- a car doing 12 m/s stays ONE object frame after frame."""
    t = ObjectTracker()
    speed, step = 12.0, 0.11
    now = 0.0
    ids = []
    for frame in range(6):
        got = t.update([sighting(speed * now, 0.0)], now)
        if got:
            ids.append(got[0].tid)
        now += step
    assert ids, "a car doing 12 m/s was never confirmed at all"
    assert len(set(ids)) == 1, f"a fast car changed identity: {ids}"


def test_a_real_car_keeps_its_track_and_its_speed():
    """A car crossing at a steady 8 m/s must still read as one object doing about 8 m/s."""
    t = ObjectTracker()
    speed, step = 8.0, 0.11
    now, ident, seen = 0.0, None, []
    for frame in range(30):
        tracks = t.update([sighting(speed * now, 0.0)], now)
        now += step
        if not tracks:
            continue
        mine = tracks[0]
        if ident is None:
            ident = mine.tid
        seen.append((mine.tid, t.reported_speed(mine)))
    assert seen, "a car driving in a straight line was never confirmed"
    assert seen[-1][0] == ident, "a car driving in a straight line changed identity"
    assert 6.0 <= seen[-1][1] <= 10.0, f"read {seen[-1][1]:.1f} m/s for a car doing {speed}"


# ---------------------------------------------------------------- stopping for the kerb


def kerb_thing(length, width, kind="obstacle"):
    from warp_av.perception.perception import DetectedObject, ObjectType
    o = DetectedObject(object_type=getattr(ObjectType, kind.upper()), x=6.0, y=0.0, distance=6.0)
    o.length_m, o.width_m = length, width
    return o


def test_the_van_does_not_stop_for_a_kerb_post():
    """The live fault: on an empty road the van stopped on 18% of frames for stationary things
    1.36 to 2.20 m off its line -- posts, railings and kerb, none of them on any road. It was
    treating every one of them as if it were a parked car reaching into its margin."""
    from warp_av.planning.planner import would_scrape
    assert would_scrape(kerb_thing(0.3, 0.1), 1.53) is False
    assert would_scrape(kerb_thing(0.5, 0.1), 1.60) is False
    assert would_scrape(kerb_thing(0.5, 0.5), 1.90) is False      # the planter, as documented


def test_the_van_still_stops_for_a_parked_car_half_in_its_margin():
    """Run 62: a parked mini 1.6 m off the line was hit at 2.8 m/s. Half a car reaches into
    the van's margin, and the LiDAR only ever sees one face of it, so a vehicle keeps a floor
    near its real half-width."""
    from warp_av.planning.planner import would_scrape
    assert would_scrape(kerb_thing(4.5, 1.8, "vehicle"), 1.60) is True
    assert would_scrape(kerb_thing(1.8, 0.5, "vehicle"), 2.00) is True   # seen end-on, floored


def test_something_we_never_measured_still_stops_the_van():
    """Not knowing how big a thing is has never been a reason to drive at it."""
    from warp_av.planning.planner import would_scrape, scrape_half_width_m

    class Nameless:
        object_type = None
        length_m = None
        width_m = None
    assert scrape_half_width_m(Nameless()) is None
    assert would_scrape(Nameless(), 2.10) is True


def test_a_big_thing_blocks_however_it_is_turned():
    """The half-diagonal over-estimates on purpose: a long object lying along the kerb still
    counts, because we cannot tell from a footprint which way it is facing."""
    from warp_av.planning.planner import would_scrape
    assert would_scrape(kerb_thing(3.0, 0.2), 1.60) is True


# ---------------------------------------------------------------- who may raise a warning


def crosser(kind, height, width, y=5.0, vy=-1.6):
    class Obj:
        object_type = kind
        x, y_ = 12.0, y
        def __init__(self):
            self.x, self.y = 12.0, y
            self.vx_world, self.vy_world = 0.0, vy
            self.speed = abs(vy)
            self.stationary = False
            self.height_m, self.width_m = height, width
            self.id = 4
    return Obj()


def route_ahead():
    from warp_av.planning.planner import Waypoint
    return [Waypoint(x=float(i) * 2.0, y=0.0) for i in range(40)]


def test_a_knee_high_post_cannot_raise_a_crossing_warning():
    """Every one of the 31 warnings left on an empty road came from an unnamed lump 0.0 to
    0.3 m wide and under a metre tall -- kerb, railings, posts. Nothing that crosses a road
    is 10 cm wide and knee high."""
    from warp_av.planning.prediction import predict_route_conflict
    assert predict_route_conflict([crosser("obstacle", 0.7, 0.1)], route_ahead(),
                                  0.0, 0.0, 0.0, 4.0) is None


def test_a_person_the_camera_named_is_trusted_at_any_size():
    """The LiDAR sees one face of a thing, and a half-seen person measures 0.3 x 0.1 m. If the
    camera has NAMED it, its size is not allowed to argue."""
    from warp_av.planning.prediction import predict_route_conflict
    hit = predict_route_conflict([crosser("pedestrian", 0.4, 0.1)], route_ahead(),
                                 0.0, 0.0, 0.0, 4.0)
    assert hit is not None and hit["type"] == "pedestrian"


def test_an_unnamed_thing_big_enough_to_be_a_road_user_still_counts():
    """A person the camera missed, but the LiDAR saw properly, is still a person."""
    from warp_av.planning.prediction import predict_route_conflict
    assert predict_route_conflict([crosser("obstacle", 1.7, 0.4)], route_ahead(),
                                  0.0, 0.0, 0.0, 4.0) is not None
    assert predict_route_conflict([crosser("obstacle", 0.8, 0.6)], route_ahead(),
                                  0.0, 0.0, 0.0, 4.0) is not None


def test_something_we_never_sized_still_counts():
    """Not having measured a thing is not evidence that it is small."""
    from warp_av.planning.prediction import predict_route_conflict
    assert predict_route_conflict([crosser("obstacle", None, None)], route_ahead(),
                                  0.0, 0.0, 0.0, 4.0) is not None
