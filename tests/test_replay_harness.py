"""
Perception V2 day 3: the scored replay harness.

Every fixture under tests/fixtures/perception/ is replayed through the REAL
pipeline (sweep accumulator -> ground filter -> clustering -> tracker) with a
stand-in camera model, and scored against its answer key. The thresholds live
next to each fixture in expected.json (written by tools/replay_score.py
--set-baseline: the measured numbers minus a margin). A change that makes the
pipeline miss an object, invent one in the lane, or eat the road fails here,
on any machine, without CARLA.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="the perception module needs OpenCV")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from warp_av.perception import replay as rp  # noqa: E402

FIXTURES = rp.all_fixtures()
needs_fixtures = pytest.mark.skipif(not FIXTURES, reason="no recorded fixtures")


@pytest.fixture(scope="module")
def results():
    return {p.name: rp.replay(rp.load_fixture(p)) for p in FIXTURES}


def _expected(p: Path) -> dict:
    f = p / "expected.json"
    if not f.exists():
        pytest.skip(f"{p.name}: no expected.json (run tools/replay_score.py --set-baseline)")
    return json.load(open(f))


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_replay_runs_the_whole_pipeline(path, results):
    r = results[path.name]
    assert r.unhealthy == 0, "update() raised or flagged itself unhealthy"
    assert r.updates - rp.WARMUP_UPDATES >= 8, "two seconds of deliveries should give at least eight scored 10 Hz updates"
    assert r.reports_total > 0
    assert 0 < np.mean(r.update_ms) < 200


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_every_placed_object_is_found(path, results):
    r, exp = results[path.name], _expected(path)
    for o in r.objects:
        want = exp["objects"].get(o.name)
        assert want is not None, f"{o.name} missing from expected.json"
        if not o.visible:
            continue                     # even the dense labelled LiDAR could not see it
        assert o.recall >= want["min_recall"], f"{path.name}: {o.name} at {o.distance:.1f} m recall {o.recall:.2f} < {want['min_recall']}"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_positions_are_close(path, results):
    r = results[path.name]
    for o in r.objects:
        if o.median_error_m is None:
            continue
        # the pipeline reports the visible part of a thing, not its centre: allow the object's own
        # half-size plus the clustering cell
        limit = max(0.6, o.size_m + 0.5)
        assert o.median_error_m <= limit, f"{path.name}: {o.name} median error {o.median_error_m:.2f} m > {limit:.2f}"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_blobs_do_not_glue_two_objects_together(path, results):
    r, exp = results[path.name], _expected(path)
    per_update = r.merged / max(1, r.updates - rp.WARMUP_UPDATES - r.unhealthy)
    assert per_update <= exp.get("max_merged_per_update", 0.5), \
        f"{per_update:.2f} blobs per update cover two placed objects at once"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_measured_height_matches_the_answer_key(path, results):
    """Day 4: the van reports how tall a thing is. Width and length are lower bounds (it
    sees one face), but height is measured against the road and should be close."""
    r = results[path.name]
    for o in r.objects:
        # only for things the van actually holds: a single sighting of a rarely seen object
        # can be a blob shared with a taller neighbour, and the track keeps the biggest view
        if not o.visible or o.labelled_points < 8 or o.median_height_m is None or o.recall < 0.5:
            continue
        assert abs(o.median_height_m - o.true_height_m) <= 0.4, \
            f"{path.name}: {o.name} measured {o.median_height_m:.2f} m tall, answer key {o.true_height_m:.2f} m"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_phantoms_do_not_grow(path, results):
    r, exp = results[path.name], _expected(path)
    assert r.phantoms_in_lane <= exp["max_phantoms_in_lane"], "a ground leftover or ghost inside the lane"
    per_update = r.phantoms / max(1, r.updates - rp.WARMUP_UPDATES)
    assert per_update <= exp["max_phantoms_per_update"], f"{per_update:.1f} phantoms per update"


def test_hidden_objects_are_marked():
    o = rp.ObjectScore("cone", 14.0, 4.5, 14.7, 1.5, labelled_points=0)
    assert not o.visible
    o = rp.ObjectScore("cone", 14.0, 4.5, 14.7, 1.5, labelled_points=7)
    assert o.visible
    o = rp.ObjectScore("cone", 14.0, 4.5, 14.7, 1.5)          # no answer key: score everything
    assert o.visible


def test_classify_report():
    solid = np.array([[10.0, 0.0]] * 3)
    walk = np.array([[5.0, 3.0]] * 3)
    assert rp.classify_report(10.2, 0.1, solid, walk) == "solid"
    assert rp.classify_report(5.0, 3.2, solid, walk) == "kerb"
    assert rp.classify_report(7.0, -3.0, solid, walk) == "phantom"
    assert rp.classify_report(10.2, 0.1, solid[:2], walk) == "phantom", "two points are not enough evidence"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_ground_removal(path, results):
    r, exp = results[path.name], _expected(path)
    assert r.ground["road_deleted"] >= exp["min_road_deleted"]
    assert r.ground["object_kept"] >= exp["min_object_kept"]


def test_ego_frame_matches_the_stack():
    van = {"x": 10.0, "y": 5.0, "z": 0.0, "yaw_deg": 90.0}
    x, y = rp.ego_frame(van, 10.0, 15.0)      # 10 m along +y world = straight ahead for yaw 90
    assert math.isclose(x, 10.0, abs_tol=1e-9) and math.isclose(y, 0.0, abs_tol=1e-9)
    x, y = rp.ego_frame(van, 7.0, 5.0)        # 3 m along -x world = 3 m to the right for yaw 90
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 3.0, abs_tol=1e-9)


@needs_fixtures
def test_stub_detector_is_used_inline():
    det = rp.StubDetector()
    rp.replay(rp.load_fixture(FIXTURES[0]), detector=det)
    assert det.calls >= 6, "the stand-in camera model must be asked every update"


@needs_fixtures
def test_replay_is_repeatable():
    fx = rp.load_fixture(FIXTURES[0])
    a, b = rp.replay(fx), rp.replay(fx)
    assert a.reports_total == b.reports_total and a.phantoms == b.phantoms
    assert [o.hits for o in a.objects] == [o.hits for o in b.objects]


@needs_fixtures
def test_labels_are_the_answer_key():
    fx = rp.load_fixture(FIXTURES[0])
    g = rp.ground_metrics(fx)
    assert set(g) >= {"road_deleted", "object_kept"}
    assert all(0.0 <= v <= 1.0 for v in g.values())
    assert g["road_deleted"] > 0.9, "the ground filter should delete nearly all road points of the labelled sweep"


@needs_fixtures
@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_the_camera_names_things_right(path):
    """Day 5: with a camera that boxes every person and car exactly where the geometry
    says, the van must give every object its right name. This tests the fusion, not the
    detector: whether the box lands on the right blob and the name reaches the output."""
    r = rp.replay(rp.load_fixture(path), scripted_camera=True)
    for o in r.objects:
        if not o.visible or o.named_right is None:
            continue
        assert o.named_right >= 0.9, \
            f"{path.name}: {o.name} should be a {o.expected_type}, was called {o.types}"


def test_box_distance():
    # a 4 m long, 2 m wide box pointing straight ahead, centred 10 m in front
    assert rp.box_distance(10.0, 0.0, 10.0, 0.0, 0.0, 4.0, 2.0) == 0.0
    assert rp.box_distance(12.5, 0.0, 10.0, 0.0, 0.0, 4.0, 2.0) == pytest.approx(0.5)
    assert rp.box_distance(10.0, 2.0, 10.0, 0.0, 0.0, 4.0, 2.0) == pytest.approx(1.0)
    # turned 90 degrees, the long side now runs across
    assert rp.box_distance(12.5, 0.0, 10.0, 0.0, 90.0, 4.0, 2.0) == pytest.approx(1.5)


def test_one_report_is_credited_once():
    """A long planter's wide reach must not be credited with the bin's report next to it."""
    binn = rp.ObjectScore("bin", 12.0, -1.6, 12.1, 1.5, size_m=0.5,
                          true_length_m=0.65, true_width_m=0.53, true_yaw_deg=0.0)
    planter = rp.ObjectScore("planter", 13.0, 0.6, 13.0, 3.0, size_m=2.5,
                             true_length_m=4.95, true_width_m=0.86, true_yaw_deg=90.0)
    got = rp.assign([binn, planter], [(11.8, -1.6)])
    assert 0 in got and 1 not in got, "the bin's report was credited to the planter as well"
    got = rp.assign([binn, planter], [(11.8, -1.6), (13.1, 0.5)])
    assert got[0][0] == 0 and got[1][0] == 1, "each object should take its own report"
    assert rp.assign([binn, planter], [(30.0, 9.0)]) == {}, "a far report belongs to nobody"
