"""
A thing that moves covers ground on every sighting (track-hop phase, 2026-09-21).

The broad sweep's van braked for nothing 36 times in 49 runs: "obstacle will cross our path".
Nothing was crossing. Each time a track had taken a blob that was not its own -- the next
parked car as the van passed the first, the next pole, another sliver of kerb -- and the
still-or-moving decision called that ONE step movement: the travel was the jump, the path was
the jump (perfectly straight), and the filter turned the jump into a speed that took a second
to die away. Prediction then ran a parked car across the road, and the van put the brake to 1.0.

The first wrong stage is the still -> moving decision, and that is where the fix is: a still
track turns moving only if it ALSO covered ground on each of its last two sightings. One jump
to somewhere else, however far, is one step. Association, the gate, the filter, prediction and
behaviour are not touched, so every test here runs the chain the van runs:

    sightings -> ObjectTracker.update -> the objects perception reports
              -> predict_route_conflict -> BehaviorSystem.update

Every false case is run twice: with the new test switched off it must reproduce the failure
(the fake mover, the warning, the hard yield), so that the test is known to have teeth; with
it on, the failure must be gone. Every true case must come out the same frame as before.

The scenes are fed at the rate the van really sights things (about 0.3 s a frame on the rig)
and, where it holds, at 10 Hz too. At 10 Hz an even share of the travel per step is small, and
the new test filters less; that is said where it matters (CASE J) rather than hidden.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior  # noqa: E402
from warp_av.localization.localization import Pose  # noqa: E402
from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput  # noqa: E402
from warp_av.perception.tracking import PROGRESS_SHARE, ObjectTracker, Track  # noqa: E402
from warp_av.planning.planner import Waypoint  # noqa: E402
from warp_av.planning.prediction import predict_route_conflict  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "trackhop"
LIVE_DT = 0.3                       # the rig's sighting rate (3-4 Hz); the day-6 tests use 0.1
STRAIGHT_ROAD = [Waypoint(x=float(x), y=0.0) for x in range(-10, 140, 2)]
CAR = (4.2, 1.8, 1.5)
POLE = (0.15, 0.15, 3.5)
PERSON = (0.5, 0.5, 1.7)


@pytest.fixture
def old_rule(monkeypatch):
    """The decision as it was before this phase: the new test always says yes."""
    monkeypatch.setattr(Track, "covered_ground_twice", lambda self: True)


# ---- the chain, as the van runs it ----------------------------------------------------------

def blob(x, y, ego, cls=None, size=CAR, k=0, jitter=0.0):
    """One sighting, with the blob's middle wandering as a LiDAR blob's does."""
    w = jitter * math.sin(k * 2.1)
    return {"wx": x + w, "wy": y + 0.3 * w, "cls": cls, "cls_source": "camera" if cls else None,
            "confidence": 0.8 if cls else 0.0, "weak": False,
            "distance": math.hypot(x - ego[0], y - ego[1]),
            "length_m": size[0], "width_m": size[1], "height_m": size[2]}


def as_reported(tracker, tracks, ex, ey, yaw):
    """Tracks -> the objects perception hands on (camera_lidar_perception, 'tracks ->
    DetectedObjects'): the van's frame, the speed and velocity the tracker REPORTS."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for tr in tracks:
        dx, dy = tr.wx - ex, tr.wy - ey
        otype = (ObjectType.PEDESTRIAN if tr.cls == "pedestrian" else ObjectType.VEHICLE
                 if tr.cls == "vehicle" else ObjectType.OBSTACLE)
        vx, vy = tracker.reported_velocity(tr)
        out.append(DetectedObject(
            object_type=otype, x=dx * c + dy * s, y=-dx * s + dy * c, distance=math.hypot(dx, dy),
            speed=tracker.reported_speed(tr), vx_world=vx, vy_world=vy,
            length_m=tr.length_m, width_m=tr.width_m, height_m=tr.height_m,
            stationary=bool(tr.stationary), motion_class=tr.motion.state, id=tr.tid))
    return out


class Frame:
    def __init__(self, k, t, tracks, warning, decision):
        self.k, self.t, self.tracks, self.warning, self.decision = k, t, tracks, warning, decision

    @property
    def movers(self):
        return [tr_id for tr_id, still in self.tracks if not still]


def drive(frames, route):
    """frames: (t, (ex, ey, yaw, speed), sightings). The whole chain, one Frame per frame."""
    tracker, behaviour = ObjectTracker(), BehaviorSystem()
    behaviour.set_mission()
    out = []
    for k, (t, (ex, ey, yaw, speed), sightings) in enumerate(frames):
        tracks = tracker.update(sightings, t)
        warning = None
        if speed > 0.5:                                          # as main.py asks it
            warning = predict_route_conflict(as_reported(tracker, tracks, ex, ey, yaw), route,
                                             ex, ey, yaw, speed)
        decision = behaviour.update(PerceptionOutput(), Pose(x=ex, y=ey, yaw=yaw, speed=speed, healthy=True),
                                    500, True, predicted_conflict=warning)
        out.append(Frame(k, t, [(tr.tid, bool(tr.stationary)) for tr in tracks], warning, decision))
    return out


def scene(sightings_at, seconds, ego_speed, dt=LIVE_DT):
    """A van driving along the straight road at ego_speed; sightings_at(k, t, ego) -> sightings."""
    frames = []
    for k in range(int(round(seconds / dt))):
        t = (k + 1) * dt
        ego = (ego_speed * t, 0.0)
        frames.append((t, (ego[0], ego[1], 0.0, ego_speed), sightings_at(k, t, ego)))
    return frames


def hard_yields(run):
    return [f.k for f in run if f.decision.behavior == DrivingBehavior.YIELDING_PREDICTED and f.decision.should_stop]


def first_moving(run):
    return next((f.k for f in run if f.movers), None)


def first_warning(run):
    return next((f.k for f in run if f.warning), None)


def ids(run):
    return sorted({tr_id for f in run for tr_id, _ in f.tracks})


# ---- CASE A: two parked cars, and the track of the first takes the second ---------------------

def parked_car_hop(dt, both_seen):
    """Two cars parked nose-in beside the road, 2.97 m apart, the second nearer the road. The
    first is seen for 3.6 s; then the van has passed it, its blob is gone, and the second
    car's blob is inside the first track's gate."""
    n_first = int(round(3.6 / dt))

    def sightings(k, t, ego):
        out = [blob(30.0, 5.5, ego, "vehicle", k=k, jitter=0.1)] if k < n_first else []
        if both_seen or k >= n_first:
            out.append(blob(32.1, 3.4, ego, "vehicle", k=k + 3, jitter=0.1))
        return out
    return scene(sightings, 6.6, 5.0, dt)


@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
@pytest.mark.parametrize("both_seen", [False, True])
def test_case_a_before_the_change_a_hop_between_parked_cars_was_a_crossing_car(old_rule, dt, both_seen):
    run = drive(parked_car_hop(dt, both_seen), STRAIGHT_ROAD)
    assert first_moving(run) is not None, "the failure must reproduce: one jump was called movement"
    assert first_warning(run) is not None and hard_yields(run), "...and the van braked hard for it"
    # the same track number carries through the hop: nothing downstream could tell
    assert run[first_moving(run)].movers == [1]


@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
@pytest.mark.parametrize("both_seen", [False, True])
def test_case_a_a_hop_between_two_parked_cars_is_not_movement(dt, both_seen):
    run = drive(parked_car_hop(dt, both_seen), STRAIGHT_ROAD)
    assert first_moving(run) is None, f"a parked car was called moving at frame {first_moving(run)}"
    assert first_warning(run) is None
    assert all(f.decision.behavior == DrivingBehavior.FOLLOWING_ROUTE for f in run), "the van just drives"


# ---- CASE B: a row of poles; one is missed for a frame and every track shifts one along -------

def pole_row(dt):
    poles = [(28.0 + 1.84 * i, 7.0 - 1.84 * i) for i in range(4)]        # 2.6 m apart, as a kerb line runs on a bend
    n_before = int(round(3.0 / dt))
    missed = range(n_before, n_before + max(1, int(round(0.3 / dt))))

    def sightings(k, t, ego):
        return [blob(x, y, ego, None, POLE, k=k + 5 * i, jitter=0.05)
                for i, (x, y) in enumerate(poles) if not (i == 0 and k in missed)]
    return scene(sightings, 6.0, 5.0, dt)


@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
def test_case_b_before_the_change_a_row_of_poles_set_off_walking(old_rule, dt):
    assert first_moving(drive(pole_row(dt), STRAIGHT_ROAD)) is not None


@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
def test_case_b_poles_that_swap_tracks_stay_still(dt):
    run = drive(pole_row(dt), STRAIGHT_ROAD)
    assert first_moving(run) is None
    assert first_warning(run) is None


# ---- CASE C: ordinary blob wobble keeps the track, and keeps it still -------------------------

@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
def test_case_c_a_wobbling_parked_car_keeps_one_still_track(dt):
    run = drive(scene(lambda k, t, ego: [blob(30.0, 4.0, ego, "vehicle", k=k, jitter=0.30)], 9.0, 3.0, dt),
                STRAIGHT_ROAD)
    assert ids(run) == [1], "the association is not touched: no new fragments"
    assert first_moving(run) is None and first_warning(run) is None


# ---- CASE D: a few missed frames break neither a still track nor a moving one -----------------

@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
def test_case_d_a_parked_car_missed_for_a_moment_is_the_same_still_track(dt):
    gap = range(int(round(3.0 / dt)), int(round(3.0 / dt)) + int(round(0.6 / dt)))
    run = drive(scene(lambda k, t, ego: [] if k in gap else [blob(30.0, 4.0, ego, "vehicle", k=k, jitter=0.15)],
                      7.0, 3.0, dt), STRAIGHT_ROAD)
    assert ids(run) == [1]
    assert first_moving(run) is None


@pytest.mark.parametrize("dt", [LIVE_DT, 0.1])
def test_case_d_a_moving_car_missed_for_a_moment_is_the_same_moving_track(dt):
    gap = range(int(round(2.4 / dt)), int(round(2.4 / dt)) + int(round(0.6 / dt)))
    run = drive(scene(lambda k, t, ego: [] if k in gap else [blob(60.0 - 6.0 * t, 3.5, ego, "vehicle", k=k, jitter=0.15)],
                      6.0, 3.0, dt), STRAIGHT_ROAD)
    assert ids(run) == [1], "one car, one track, through the gap"
    after = [f for f in run if f.k > gap[-1] + 1]
    assert after and all(f.movers == [1] for f in after), "it is still a moving car when it is seen again"


def test_case_d_a_car_that_sets_off_is_caught_though_a_frame_is_missed(monkeypatch):
    """The last two steps are between SIGHTINGS: a missed frame makes one step longer, never
    a step of nothing."""
    def sightings(k, t, ego):
        if k == 12:
            return []
        return [blob(22.0, 6.0 - 0.5 * 2.0 * max(0.0, t - 3.0) ** 2, ego, "vehicle", k=k, jitter=0.1)]
    with monkeypatch.context() as as_it_was:
        as_it_was.setattr(Track, "covered_ground_twice", lambda self: True)
        before = first_moving(drive(scene(sightings, 6.0, 4.0), STRAIGHT_ROAD))
    now = first_moving(drive(scene(sightings, 6.0, 4.0), STRAIGHT_ROAD))
    assert now is not None and now == before, f"called moving at frame {now}, was {before}"


# ---- CASE E, F, G: things that really move are called moving on the same frame as before ------

def crossing_car(speed, x=20.0, y=22.0):
    """Out of a side road on the left, across the van's path x metres up the road (placed so
    that the van and the car would really meet)."""
    return lambda k, t, ego: [blob(x, y - speed * t, ego, "vehicle", k=k, jitter=0.15)]


def pulling_out(k, t, ego):
    """Parked at the kerb for 3 s, then out across the lane at 2 m/s^2."""
    return [blob(22.0, 6.0 - 0.5 * 2.0 * max(0.0, t - 3.0) ** 2, ego, "vehicle", k=k, jitter=0.12)]


def stepping_off(k, t, ego):
    """Someone standing at the kerb for 2 s, then walking into the road at 1.4 m/s."""
    return [blob(18.0, 5.0 - 1.4 * max(0.0, t - 2.0), ego, "pedestrian", PERSON, k=k, jitter=0.06)]


MOVERS = [("a car crossing at 3 m/s", crossing_car(3.0, 28.0, 12.0), 4.5, 5.0, LIVE_DT),
          ("a car crossing at 8 m/s", crossing_car(8.0), 2.7, 5.0, LIVE_DT),
          ("a car crossing at 8 m/s, 10 Hz", crossing_car(8.0), 2.7, 5.0, 0.1),
          ("a car crossing at 12 m/s, 10 Hz", crossing_car(12.0, 14.0, 22.0), 1.8, 5.0, 0.1),
          ("a parked car pulling out", pulling_out, 6.0, 4.0, LIVE_DT),
          ("a parked car pulling out, 10 Hz", pulling_out, 6.0, 4.0, 0.1),
          ("a walker stepping off", stepping_off, 6.0, 2.0, LIVE_DT),
          ("a walker stepping off, 10 Hz", stepping_off, 6.0, 2.0, 0.1)]
def outcome(run):
    return first_moving(run), first_warning(run), hard_yields(run)[:1], ids(run)


@pytest.mark.parametrize("name,sightings,seconds,ego_speed,dt", MOVERS, ids=[m[0] for m in MOVERS])
def test_case_efg_real_movers_lose_no_frame(monkeypatch, name, sightings, seconds, ego_speed, dt):
    with monkeypatch.context() as as_it_was:
        as_it_was.setattr(Track, "covered_ground_twice", lambda self: True)
        before = outcome(drive(scene(sightings, seconds, ego_speed, dt), STRAIGHT_ROAD))
    assert before[0] is not None and before[1] is not None and before[2], \
        "the scene must hold a real mover that the van was warned about and stopped for"
    now = outcome(drive(scene(sightings, seconds, ego_speed, dt), STRAIGHT_ROAD))
    assert now == before, (f"{name}: called moving at frame {now[0]} (was {before[0]}), first warned at frame "
                           f"{now[1]} (was {before[1]}), tracks {now[3]} (were {before[3]})")
    assert now[3] == [1], "CASE F: one real car is one track -- it is not broken into pieces"


def test_case_g_a_car_pulling_out_is_moving_within_a_second_and_a_half():
    """In absolute terms, so that a slow drift in BOTH rules could not hide: from the moment
    it starts to roll, at the live rate."""
    run = drive(scene(pulling_out, 6.0, 4.0), STRAIGHT_ROAD)
    assert (first_moving(run) + 1) * LIVE_DT - 3.0 <= 1.5


# ---- CASE H: recorded false yields, end to end ------------------------------------------------
# tests/fixtures/trackhop/README.md says where these came from and how they were checked.

def recorded(name):
    fx = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    frames = [(f["t"], tuple(f["pose"]),
               [dict(o, static_shapes=tuple(o.get("static_shapes") or ()),
                     road_gap_fn=(lambda v=o.get("gap"): v) if "gap" in o else None) for o in f["obs"]])
              for f in fx["frames"]]
    return fx, frames, [Waypoint(x=x, y=y) for x, y in fx["route"]]


FALSE_YIELDS = ["false_young_track_reach.json", "false_at_speed_0166.json"]
TRUE_YIELDS = ["true_crossing_car_0145.json", "true_young_fast_car_0148.json"]


@pytest.mark.parametrize("name", FALSE_YIELDS)
def test_case_h_the_recorded_sightings_reproduce_the_recorded_false_yield(old_rule, name):
    fx, frames, route = recorded(name)
    assert fx["truth_for_scoring_only"]["label"].startswith("FALSE"), "CARLA: nothing was moving there"
    assert fx["live_record"]["peak_brake"] == 1.0 and "yielding_predicted" in fx["live_record"]["behaviors"]
    run = drive(frames, route)
    assert [f.k for f in run if f.warning] == fx["before_the_change"]["warning_frames"]
    first = run[first_warning(run)].warning
    live = fx["live_record"]["predicted_conflict"]
    assert (first["along_m"], first["t"], first["type"]) == (live["along_m"], live["t"], live["type"]), \
        "the very warning the van recorded live"
    assert hard_yields(run), "and the van stops for it: YIELDING_PREDICTED, should_stop"


@pytest.mark.parametrize("name", FALSE_YIELDS)
def test_case_h_the_recorded_false_yield_is_gone(name):
    fx, frames, route = recorded(name)
    run = drive(frames, route)
    assert [f.k for f in run if f.warning] == [], "no warning about a thing that never moved"
    assert all(f.decision.behavior == DrivingBehavior.FOLLOWING_ROUTE for f in run)


# ---- CASE I: recorded true yields, end to end --------------------------------------------------

@pytest.mark.parametrize("name", TRUE_YIELDS)
def test_case_i_the_recorded_true_yield_is_still_raised_on_the_same_frame(name):
    fx, frames, route = recorded(name)
    truth = fx["truth_for_scoring_only"]
    assert truth["label"] == "TRUE" and truth["its_true_speed_mps"] > 7.0, "CARLA: a traffic car doing 28 km/h"
    assert fx["live_record"]["peak_brake"] == 1.0, "the van braked for it live, and was right to"
    run = drive(frames, route)
    assert [f.k for f in run if f.warning] == fx["before_the_change"]["warning_frames"], "not a frame later"
    first = run[first_warning(run)].warning
    want = fx["before_the_change"]["first_warning"]
    assert (first["along_m"], first["t"], first["type"]) == (want["along_m"], want["t"], want["type"])
    assert hard_yields(run) and hard_yields(run)[0] == first_warning(run)


# ---- CASE J: neighbours of the same class -----------------------------------------------------

def flickering_neighbours(dt):
    """Two parked cars 2.6 m apart, both 'vehicle'. The first one's blob comes and goes (every
    0.9 s), so its track keeps reaching for the second car's blob."""
    def sightings(k, t, ego):
        out = [blob(30.0, 5.0, ego, "vehicle", k=k, jitter=0.15)]
        if t < 2.0 or int(t / 0.9) % 2 == 0:
            out.insert(0, blob(27.4, 5.0, ego, "vehicle", k=k + 7, jitter=0.15))
        return out
    return scene(sightings, 8.0, 3.0, dt)


def test_case_j_before_the_change_flickering_neighbours_made_a_mover(old_rule):
    assert first_moving(drive(flickering_neighbours(LIVE_DT), STRAIGHT_ROAD)) is not None


def test_case_j_neighbours_of_the_same_class_stay_parked_cars():
    run = drive(flickering_neighbours(LIVE_DT), STRAIGHT_ROAD)
    assert first_moving(run) is None and first_warning(run) is None


def test_case_j_known_limit_at_ten_hertz_the_new_test_filters_less():
    """Said out loud, not hidden: with ten sightings a second an even share of the travel is a
    few centimetres a step, and ordinary blob wobble can pass for the second step of a
    journey. The van's perception runs at 3-4 Hz on the rig; if that ever changes, the share
    should be stated over time and not over steps. This pins today's behaviour."""
    run = drive(flickering_neighbours(0.1), STRAIGHT_ROAD)
    assert first_warning(run) is None, "even then, nothing reaches the brake in this scene"


# ---- the rule itself ---------------------------------------------------------------------------

def _track(sightings, dt=LIVE_DT):
    tr = Track(1, *sightings[0], dt)
    for k, (x, y) in enumerate(sightings[1:], start=2):
        tr.predict(dt)
        tr.correct(x, y, 0.3, k * dt)
    return tr


def test_one_jump_is_one_step_however_far():
    tr = _track([(0.0, 0.0)] * 6 + [(3.0, 0.0)] * 2)
    assert not tr.covered_ground_twice()
    assert tr.stationary


def test_two_even_steps_are_a_journey():
    tr = _track([(0.0, 0.0)] * 4 + [(0.9, 0.0), (1.8, 0.0), (2.7, 0.0)])
    assert tr.covered_ground_twice()


def test_uneven_steps_still_count_if_each_carries_a_quarter_of_its_share():
    assert PROGRESS_SHARE == 0.25
    walker = _track([(0.0, 0.0), (0.45, 0.0), (0.6, 0.0), (1.1, 0.0), (1.3, 0.0), (1.8, 0.0)])
    assert walker.covered_ground_twice(), "a walker's blob does not advance evenly, and need not"


def test_going_back_is_not_going_on():
    tr = _track([(0.0, 0.0)] * 5 + [(3.0, 0.0), (2.6, 0.0)])
    assert not tr.covered_ground_twice(), "it jumped, then came back a little: the second step is backwards"


def test_the_sightings_kept_are_no_older_than_the_history():
    tr = _track([(0.1 * k, 0.0) for k in range(40)])
    assert len(tr._seen) <= len(tr._history) + 1
    assert tr._seen[-1][0] - tr._seen[0][0] <= tr.window_s + LIVE_DT
