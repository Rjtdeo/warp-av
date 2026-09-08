"""
Perception V2 day 6: is it moving, or is it parked?

Before today the van worked out speed by subtracting two sightings and
smoothing. The middle of a LiDAR blob jumps about as different parts of a
thing come back, so a parked bin appeared to be doing 6.7 m/s. Now each track
carries a motion filter: a sighting nudges the estimate as much as its own
noise deserves, and a thing is only called moving once it has actually gone
somewhere, in one direction, over several sightings.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.tracking import (MOVING_SPEED_MPS, ObjectTracker, Track,  # noqa: E402
                                         measurement_noise_m)

DT = 0.1


def feed(tracker, path, jitter=0.0, distance=20.0, dt=DT):
    """Run a list of true positions through the tracker, with blob jitter on top."""
    t, seen = 0.0, []
    for k, (x, y) in enumerate(path):
        t += dt
        w = jitter * math.sin(k * 2.1)
        tracks = tracker.update([{"wx": x + w, "wy": y + 0.3 * w, "distance": distance}], t)
        if tracks:
            seen.append(tracks[0])
    return seen


def straight(speed, n=30, start=40.0):
    return [(start - speed * (k + 1) * DT, 0.0) for k in range(n)]


def test_a_parked_thing_reads_zero_however_much_its_blob_jitters():
    tr = ObjectTracker()
    seen = feed(tr, [(12.0, 0.0)] * 30, jitter=0.30)
    assert seen, "it should be tracked"
    assert all(t.stationary for t in seen[4:]), "a parked thing must stay parked"
    assert tr.reported_speed(seen[-1]) == 0.0
    assert seen[-1].speed < 0.6, f"filtered speed {seen[-1].speed:.2f} m/s on a parked thing"


@pytest.mark.parametrize("speed", [1.5, 3.0, 8.0, 15.0])
def test_a_moving_thing_gets_its_speed_right(speed):
    tr = ObjectTracker()
    seen = feed(tr, straight(speed), jitter=0.15, distance=25.0)
    assert len(seen) >= 25
    assert seen[-1].speed == pytest.approx(speed, abs=0.6)
    assert not seen[-1].stationary
    assert tr.reported_speed(seen[-1]) == pytest.approx(speed, abs=0.6)
    assert len({t.tid for t in seen}) == 1, "it should keep one track number throughout"


def test_it_notices_movement_quickly():
    tr = ObjectTracker()
    seen = feed(tr, straight(5.0), jitter=0.15, distance=25.0)
    first = next(i for i, t in enumerate(seen) if not t.stationary)
    assert first * DT <= 0.6, f"took {first * DT:.1f} s to call a 5 m/s car moving"


def test_a_thing_that_stops_is_called_parked_again():
    """A car doing 6 m/s brakes over a second, then waits. The van should notice both."""
    tr = ObjectTracker()
    t, x, v, moved_at, parked_at, track = 0.0, 40.0, 6.0, None, None, None
    for k in range(60):
        t += DT
        v = max(0.0, 6.0 - 6.0 * max(0.0, t - 1.0))          # braking between t=1 and t=2
        x -= v * DT
        got = tr.update([{"wx": x + 0.1 * math.sin(k * 2.1), "wy": 0.0, "distance": 20.0}], t)
        if not got:
            continue
        track = got[0]
        if not track.stationary and moved_at is None:
            moved_at = t
        if v == 0.0 and track.stationary and parked_at is None:
            parked_at = t - 2.0
    assert moved_at is not None and moved_at <= 0.7, "it should spot a 6 m/s car quickly"
    assert parked_at is not None, "after stopping it must read parked again"
    assert parked_at <= 1.5, f"took {parked_at:.1f} s to believe the car had stopped"
    assert tr.reported_speed(track) == 0.0


def test_shuffling_on_the_spot_is_not_movement():
    """A blob whose middle hops between two parts of the same object travels, but
    gets nowhere. That is what a wandering cluster looks like, and it is not motion."""
    tr = ObjectTracker()
    path = [(12.0 + (0.55 if k % 2 else -0.55), 0.0) for k in range(30)]
    seen = feed(tr, path, distance=15.0)
    assert all(t.stationary for t in seen[4:]), "hopping back and forth is not going anywhere"


def test_two_parked_things_side_by_side_keep_their_own_tracks():
    """A bin and a planter 2.4 m apart must not swap identities."""
    tr = ObjectTracker()
    ids_a, ids_b = set(), set()
    t = 0.0
    for k in range(25):
        t += DT
        w = 0.25 * math.sin(k * 2.1)
        tracks = tr.update([{"wx": 12.0 + w, "wy": -1.6, "distance": 12.1},
                            {"wx": 13.0, "wy": 0.8 + w, "distance": 13.0}], t)
        for tk_ in tracks:
            (ids_a if tk_.wy < 0 else ids_b).add(tk_.tid)
    assert len(ids_a) == 1 and len(ids_b) == 1, f"identities wandered: {ids_a}, {ids_b}"


def test_a_far_blob_is_trusted_less_than_a_near_one():
    assert measurement_noise_m({"distance": 5.0}) < measurement_noise_m({"distance": 30.0})
    assert measurement_noise_m({"distance": 10.0, "weak": True}) > measurement_noise_m({"distance": 10.0})


def test_a_brand_new_track_is_never_called_moving():
    tr = ObjectTracker()
    t = 0.0
    for k in range(3):
        t += DT
        for track in tr.update([{"wx": 10.0 + k * 0.9, "wy": 0.0, "distance": 10.0}], t):
            assert track.stationary, "two or three jumpy sightings are not proof of movement"


def test_the_filter_carries_on_through_a_missed_frame():
    tr = ObjectTracker()
    t = 0.0
    last = None
    for k in range(24):
        t += DT
        if k in (10, 11):                     # the thing is hidden for two frames
            tr.update([], t)
            continue
        got = tr.update([{"wx": 40.0 - 6.0 * t, "wy": 0.0, "distance": 25.0}], t)
        if got:
            last = got[0]
    assert last is not None and last.speed == pytest.approx(6.0, abs=0.8)
    assert not last.stationary


def test_predict_moves_the_estimate_and_widens_it():
    tr = Track(1, 10.0, 0.0, 0.0)
    tr.x[2] = 4.0
    before = tr.P[0][0]
    tr.predict(0.5)
    assert tr.wx == pytest.approx(12.0)
    assert tr.P[0][0] > before, "waiting makes the van less sure where the thing is"


# ---------------------------------------------------------------------------
# Found on day 7, when the world model started publishing "moving" for every
# object in the scene rather than only the six placed ones: the far background
# was full of walls and hedges the van believed were driving about.
# ---------------------------------------------------------------------------

def test_an_impossible_jump_is_not_a_speed():
    """Two different things confused for one another is not a 250 m/s car."""
    tr = ObjectTracker()
    tr.update([{"wx": 10.0, "wy": 0.0, "distance": 40.0}], 0.1)
    got = tr.update([{"wx": 35.0, "wy": 0.0, "distance": 40.0}], 0.2)
    for track in tr._tracks:
        assert track.speed <= 30.0, f"{track.speed:.0f} m/s is not something a road can hold"


def test_a_wall_is_never_moving():
    """A 20 m long blob is scenery. Scenery does not drive."""
    tr = ObjectTracker()
    t = 0.0
    for k in range(30):
        t += DT
        tr.update([{"wx": 40.0 + 0.35 * k, "wy": 0.0, "distance": 40.0,
                    "length_m": 20.0, "width_m": 3.0, "height_m": 4.0}], t)
    assert all(track.stationary for track in tr._tracks), "a wall was called moving"


def test_a_far_thing_must_travel_further_before_it_counts():
    near, far = Track(1, 0.0, 0.0, 0.0), Track(2, 0.0, 0.0, 0.0)
    near.range_m, far.range_m = 8.0, 40.0
    # the rule is the travel a track must show; far asks for more, but never more than
    # a walking person covers in the window
    from warp_av.perception.tracking import STILL_TRAVEL_M, STILL_TRAVEL_MAX_M, STILL_TRAVEL_PER_M
    need = lambda t: min(STILL_TRAVEL_MAX_M, max(STILL_TRAVEL_M, STILL_TRAVEL_PER_M * t.range_m))
    assert need(near) < need(far) <= STILL_TRAVEL_MAX_M


def test_a_person_walking_at_twenty_five_metres_is_still_caught():
    tr = ObjectTracker()
    t, first = 0.0, None
    for k in range(90):
        t += DT
        d = 25.0 - 1.4 * t
        got = tr.update([{"wx": d, "wy": 0.0, "distance": d}], t)
        if got and not got[0].stationary and first is None:
            first = t
    assert first is not None and first <= 2.5, "a walking person must not be mistaken for scenery"


def test_a_car_rounding_a_bend_is_moving():
    """The straight-line test must not punish a car for turning."""
    tr = ObjectTracker()
    t, first = 0.0, None
    for k in range(60):
        t += DT
        a = 0.06 * k
        got = tr.update([{"wx": 25.0 - 20.0 * math.sin(a), "wy": 20.0 * (1 - math.cos(a)),
                          "distance": 25.0}], t)
        if got and not got[0].stationary and first is None:
            first = t
    assert first is not None and first <= 1.0
