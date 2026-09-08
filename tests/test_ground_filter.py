"""
Perception fix 2: road removal by local patches, the road-edge rule, and the
wiring in camera+LiDAR perception (switchable back to the old flat line).
"""
import os
import time
import warnings

import numpy as np

from warp_av.perception.ground_filter import (GroundFilter, flat_cut, is_road_edge,
                                              ground_filter_mode_from_env, remove_road_edge_points)
from warp_av.perception.tracking import cluster_points

PERC = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "perception", "camera_lidar_perception.py")
LIDAR = 2.5     # metres above the road under the van; road = z -2.5 in the sensor frame


def road_points(x0=3.0, x1=40.0, step=0.5, width=6.0, hill_from=None, slope=0.10):
    """A carpet of road returns: rows across the lane every `step` metres.
    From `hill_from` onwards the road climbs with `slope`."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(-width / 2, width / 2 + 1e-6, 1.0)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, -LIDAR)
    if hill_from is not None:
        Z = Z + np.where(X > hill_from, slope * (X - hill_from), 0.0)
    return np.c_[X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, 0.5)].astype(np.float32)


def column(x, y, heights_above_road, road_z=-LIDAR):
    """Points stacked on an object standing on the road at (x, y)."""
    return np.array([[x, y, road_z + h, 0.7] for h in heights_above_road], dtype=np.float32)


def run(points):
    gf = GroundFilter(lidar_height_m=LIDAR)
    return gf.apply(points), gf


# ---------------------------------------------------------------- flat road

def test_flat_road_is_deleted_and_objects_are_kept():
    road = road_points()
    barrel = column(10.0, 0.0, [0.10, 0.20, 0.45, 0.68, 0.90])
    planter = column(15.0, 0.5, [0.05, 0.14, 0.31])
    person = column(20.0, -0.3, [0.05, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7])
    pts = np.vstack([road, barrel, planter, person])
    res, gf = run(pts)
    n_road = len(road)
    assert not res.keep[:n_road].any()                       # every road point gone
    kept = res.keep[n_road:]
    heights = np.r_[[0.10, 0.20, 0.45, 0.68, 0.90], [0.05, 0.14, 0.31], [0.05, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7]]
    assert (kept == (heights > 0.12)).all()                   # only the 5 and 10 cm points are lost
    assert np.allclose(res.above[n_road:], heights, atol=1e-3)
    assert res.borrowed_tiles == 0 and res.ground_tiles > 100


def test_old_flat_cut_loses_the_low_objects():
    """The rule being replaced, for the record."""
    planter = column(15.0, 0.5, [0.05, 0.14, 0.31])
    barrel = column(10.0, 0.0, [0.10, 0.20, 0.45, 0.68, 0.90])
    res = flat_cut(np.vstack([planter, barrel]), lidar_height_m=LIDAR, min_above_m=0.35)
    assert not res.keep[:3].any()                              # planter invisible
    assert list(res.keep[3:]) == [False, False, True, True, True]


# ---------------------------------------------------------------- hills and levels

def test_hill_road_is_still_road_and_an_object_on_the_hill_is_kept():
    road = road_points(hill_from=25.0, slope=0.10)             # 10 % climb from 25 m
    hill_z = -LIDAR + 0.10 * (32.0 - 25.0)                     # the road at 32 m is 0.7 m up
    barrel = column(32.0, 0.0, [0.2, 0.5, 0.8], road_z=hill_z)
    pts = np.vstack([road, barrel])
    res, _ = run(pts)
    assert not res.keep[:len(road)].any()                      # no fake wall on the hill
    assert res.keep[len(road):].all()
    assert np.allclose(res.above[len(road):], [0.2, 0.5, 0.8], atol=0.16)   # within one tile's climb


def test_long_steep_hill_stays_road_all_the_way_out():
    """8 % climb from 10 m to 50 m: the road at 45 m is 2.8 m up and must
    still be road, not a wall; a barrel on it is still an obstacle."""
    road = road_points(x0=3.0, x1=49.5, hill_from=10.0, slope=0.08)
    hill_z = -LIDAR + 0.08 * (45.0 - 10.0)
    barrel = column(45.0, 0.0, [0.3, 0.6, 0.9], road_z=hill_z)
    res, _ = run(np.vstack([road, barrel]))
    assert not res.keep[:len(road)].any()
    assert res.keep[len(road):].all()
    # and a tile whose only points are 6 m up (a bridge / a tree canopy) is not believed to be road
    canopy = np.array([[20.0, 0.0, -LIDAR + 6.0, 0.1], [20.5, 0.2, -LIDAR + 6.2, 0.1]], dtype=np.float32)
    res, _ = run(np.vstack([road_points(), canopy]))
    assert not res.keep[-2:].any()                              # above the 4 m ceiling anyway
    assert np.allclose(res.above[-2:], [6.0, 6.2], atol=0.05)   # measured from the borrowed road, not from itself


def test_steep_hills_leave_no_sliver_inside_a_tile():
    """On a 10 % or 12 % grade the road rises 0.15-0.18 m inside one 1.5 m
    tile, more than the 12 cm keep line: the uphill sliver must still be
    ground (each point is measured against the highest neighbouring tile)."""
    for slope in (0.10, 0.12):
        road = road_points(x0=3.0, x1=45.0, step=0.25, hill_from=8.0, slope=slope)
        res, _ = run(road)
        assert not res.keep.any(), f"{int(res.keep.sum())} road points kept on a {slope:.0%} grade"


def test_shallow_ditch_inside_a_road_tile_is_ground():
    """A 0.4 m ditch beside the road, its edge in the middle of a tile: the
    road points sharing that tile sit 0.4 m above the tile's lowest point and
    must still be ground (the reviewer's phantom 'parked vehicle')."""
    road = road_points(width=7.0)                               # y -3.5 .. 3.5
    ditch = road_points(width=10.0)
    ditch = ditch[(ditch[:, 1] > 3.6) & (ditch[:, 1] < 4.6)]
    ditch[:, 2] -= 0.4
    res, _ = run(np.vstack([road, ditch]))
    assert not res.keep.any()
    # the same with the field edge NOT on a tile boundary
    road = road_points(width=6.0)
    field = road_points(width=16.0)
    field = field[field[:, 1] > 3.2]
    field[:, 2] -= 2.0
    res, _ = run(np.vstack([road, field]))
    assert not res.keep.any()


def test_far_object_on_a_sparse_ring_is_not_its_own_ground():
    """Beyond 30 m the 32 beams land 8-15 m apart. A car at 40 m may have no
    road ring in its tile or the neighbouring ones; its lowest ring must not
    become 'the road here'. The fitted road plane is the reference."""
    rings = np.vstack([road_points(x0=r, x1=r + 0.1, step=1.0, width=8.0) for r in (4, 5, 6.5, 8, 10, 13, 17, 21, 26, 34, 49)])
    car = np.array([[40.0, 0.0, -LIDAR + 0.5, 0.3], [40.2, 0.4, -LIDAR + 0.75, 0.3], [40.1, -0.3, -LIDAR + 1.2, 0.3],
                    [41.0, 0.0, -LIDAR + 1.4, 0.3]], dtype=np.float32)
    res, gf = run(np.vstack([rings, car]))
    assert gf.last_plane is not None
    assert res.keep[-4:].all()
    assert np.allclose(res.above[-4:], [0.5, 0.75, 1.2, 1.4], atol=0.1)
    # and a far road ring on a hill (mates of one height across the road) is still road
    hill_ring = road_points(x0=42.0, x1=42.1, step=1.0, width=8.0)
    hill_ring[:, 2] += 1.2                                       # 1.2 m up: a hill, not a wall
    res, _ = run(np.vstack([rings, hill_ring]))
    assert not res.keep[-len(hill_ring):].any()


def test_far_wall_face_does_not_eat_a_far_car():
    """At 40 m a building face next to a parked car has its lowest return
    1.8 m up (no road ring in its tile). That tile must not become a 1.8 m
    'road' that the car's points are measured against."""
    rings = np.vstack([road_points(x0=r, x1=r + 0.1, step=1.0, width=8.0) for r in (4, 5, 6.5, 8, 10, 13, 17, 21, 26, 34, 49)])
    wall = np.array([[40.0, 4.0 + 0.1 * k, -LIDAR + 1.8 + 0.3 * k, 0.2] for k in range(6)], dtype=np.float32)
    car = np.array([[40.0, 2.4, -LIDAR + 0.5, 0.3], [40.3, 2.6, -LIDAR + 0.8, 0.3], [40.6, 2.5, -LIDAR + 1.3, 0.3],
                    [41.0, 2.5, -LIDAR + 1.4, 0.3]], dtype=np.float32)
    res, _ = run(np.vstack([rings, wall, car]))
    assert res.keep[-4:].all()                                  # the wall's height never became the car's road
    assert np.allclose(res.above[-4:], [0.5, 0.8, 1.3, 1.4], atol=0.15)


def test_blind_circle_near_the_van():
    """The lowest beam first touches the road 4.3 m out, so no tile within
    that radius ever has road of its own. A parking neighbour beside the van
    must still be seen there."""
    road = road_points(x0=-30.0, x1=30.0, step=0.5, width=8.0)
    road = road[np.hypot(road[:, 0], road[:, 1]) > 4.3]
    neighbour = np.array([[x, 2.7, -LIDAR + h, 0.4] for x in np.arange(-5.0, 8.0, 0.5) for h in (0.3, 0.8, 1.3)],
                         dtype=np.float32)
    res, _ = run(np.vstack([road, neighbour]))
    kept = res.keep[len(road):]
    heights = np.array([0.3, 0.8, 1.3] * 26)
    # inside the blind circle a car's own lowest band can pass for ground (no
    # road anywhere near to say otherwise); everything above it must survive
    assert kept[heights >= 0.8].all()
    assert kept.mean() > 0.8
    assert not res.keep[:len(road)].any()


def test_flat_mode_equals_the_old_mask_and_bad_input_is_quiet():
    rng = np.random.default_rng(3)
    pts = np.c_[rng.uniform(-50, 50, 5000), rng.uniform(-50, 50, 5000), rng.uniform(-3, 2, 5000),
                rng.uniform(0, 1, 5000)].astype(np.float32)
    old_mask = (pts[:, 2] > -2.15) & (pts[:, 2] < 1.5)
    res = flat_cut(pts, lidar_height_m=LIDAR, min_above_m=-2.15 + LIDAR, ceiling_above_road_m=1.5 + LIDAR)
    assert (res.keep == old_mask).sum() >= 4995                 # equal up to float rounding at the edges
    bad = np.array([[float("nan"), 0.0, -1.0, 0.1], [10.0, float("inf"), -1.0, 0.1], [10.0, 0.0, -1.0, 0.1]],
                   dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = run(np.vstack([road_points(), bad]))[0]
    assert list(r.keep[-3:]) == [False, False, True]


def test_tile_covered_by_a_car_borrows_the_road_height():
    road = road_points()
    # remove the road returns under the car (x 19.5 .. 24, the car's shadow)
    shadow = (road[:, 0] > 19.4) & (road[:, 0] < 24.1) & (np.abs(road[:, 1]) < 1.6)
    road = road[~shadow]
    car = np.array([[20.0, 0.0, -LIDAR + 0.30, 0.6], [20.0, 0.5, -LIDAR + 0.55, 0.6],
                    [21.0, 0.0, -LIDAR + 0.90, 0.6], [22.0, 0.3, -LIDAR + 1.40, 0.6],
                    [23.0, -0.4, -LIDAR + 1.45, 0.6]], dtype=np.float32)
    pts = np.vstack([road, car])
    res, _ = run(pts)
    assert res.keep[len(road):].all()                          # the whole car survives ...
    assert np.allclose(res.above[len(road):], [0.30, 0.55, 0.90, 1.40, 1.45], atol=0.05)   # ... measured from the real road
    assert res.borrowed_tiles >= 1


def test_bridge_deck_does_not_borrow_the_ground_far_below():
    """Road at our level; the tiles beside it see the ground 5 m lower. The
    deck must stay road, not become a 5 m wall."""
    deck = road_points(width=4.0)
    below = road_points(width=14.0)
    below = below[np.abs(below[:, 1]) > 2.5]
    below[:, 2] -= 5.0
    pts = np.vstack([deck, below])
    res, _ = run(pts)
    assert not res.keep[:len(deck)].any()


def test_embankment_road_does_not_borrow_the_field_beside_it():
    """Road 2 m above a field on one side: the road stays road."""
    road = road_points(width=6.0)
    field = road_points(width=16.0)
    field = field[field[:, 1] > 3.5]                            # one side only
    field[:, 2] -= 2.0
    pts = np.vstack([road, field])
    res, _ = run(pts)
    assert not res.keep[:len(road)].any()
    assert not res.keep[len(road):].any()                       # the field is ground too


def test_out_of_range_and_ceiling():
    pts = np.array([[60.0, 0.0, -1.0, 0.1],                     # beyond 50 m
                    [10.0, 0.0, -LIDAR + 4.5, 0.1],             # 4.5 m up: a bridge or a tree
                    [10.0, 0.0, -LIDAR + 1.0, 0.1],             # a normal object point
                    [10.0, 0.0, -LIDAR, 0.1]], dtype=np.float32)  # the road
    res, _ = run(pts)
    assert list(res.keep) == [False, False, True, False]
    assert np.isnan(res.above[0])


def test_empty_and_speed():
    res, gf = run(np.zeros((0, 4), dtype=np.float32))
    assert res.keep.shape == (0,)
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-50, 50, 7500), rng.uniform(-50, 50, 7500), rng.uniform(-2.6, 1.0, 7500),
                rng.uniform(0, 1, 7500)].astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(5):
        gf.apply(pts)
    ms = (time.perf_counter() - t0) / 5 * 1000.0
    assert ms < 60.0, f"{ms:.1f} ms per 7,500-point scan is too slow"


# ---------------------------------------------------------------- clusters and the road-edge rule

def test_cluster_points_reports_height_and_length_when_given_heights():
    pts = [(10.0, 0.0), (10.2, 0.1), (10.1, -0.1), (30.0, 5.0), (30.3, 5.2), (30.1, 5.1)]
    hs = [0.2, 0.9, 0.5, 0.1, 0.15, 0.12]
    cl = cluster_points(pts, heights=hs)
    assert len(cl) == 2
    assert abs(cl[0]["height"] - 0.9) < 1e-9 and abs(cl[1]["height"] - 0.15) < 1e-9
    assert cl[0]["length"] == 2.0 * cl[0]["extent"]
    # without heights nothing changes for the old callers
    old = cluster_points(pts)
    assert old[0]["height"] is None and old[0]["n"] == 3


def test_kerb_is_a_road_edge_and_a_planter_is_not():
    kerb = [(x, 1.9) for x in np.arange(4.0, 14.0, 0.5)]        # 10 m of kerb line, 15 cm tall
    kerb_h = [0.15] * len(kerb)
    planter = [(15.0, 0.4), (15.3, 0.4), (15.1, 0.7), (15.4, 0.7)]
    planter_h = [0.3, 0.32, 0.28, 0.33]
    cl = cluster_points(kerb + planter, heights=kerb_h + planter_h)
    kinds = {round(c["x"]): is_road_edge(c) for c in cl}
    assert kinds[9] is True                                     # the kerb blob (centre ~9 m)
    assert kinds[15] is False                                   # the planter stays an obstacle
    assert is_road_edge({"extent": 3.0, "height": None}) is False   # no height known: never dropped
    # a long, low thing IN the lane (a traffic island, a slab) is not a road edge
    assert is_road_edge({"extent": 2.3, "height": 0.28, "y": 0.2}) is False
    assert is_road_edge({"extent": 2.3, "height": 0.28, "y": 1.9}) is True


def test_kerb_strip_cannot_drag_a_lamp_post():
    """The reviewer's case: a 0.75 m strip of pavement (13-20 cm) survives
    beside the kerb; clustered together with a lamp post on the pavement
    it made one big blob whose centre sat in the lane's wide-body band.
    Pass 1 removes the strip's points first."""
    strip = [(x, y) for x in np.arange(2.0, 14.0, 0.5) for y in (1.9, 2.3, 2.6)]
    strip_h = [0.15] * len(strip)
    post = [(12.0, 2.6), (12.0, 2.65), (12.05, 2.6), (12.0, 2.62), (12.0, 2.58), (12.02, 2.6)]
    post_h = [0.5, 1.5, 2.5, 3.0, 3.5, 3.9]
    slab = [(6.0, 0.0), (6.5, 0.0), (7.0, 0.3), (7.5, -0.3), (8.0, 0.0), (8.5, 0.2)]   # low, long, IN the lane
    slab_h = [0.25] * len(slab)
    sel = np.array(strip + post + slab, dtype=np.float32)
    heights = np.array(strip_h + post_h + slab_h, dtype=np.float32)
    # without pass 1: one merged blob whose centre is far from the post
    merged = cluster_points(sel.tolist(), heights=heights.tolist())
    big = max(merged, key=lambda c: c["n"])
    assert big["height"] > 3.0 and abs(big["x"] - 12.0) > 3.0        # the post's centroid dragged into the strip
    # with pass 1
    sel2, h2, dropped = remove_road_edge_points(sel, heights)
    assert dropped == 1 and sel2.shape[0] == len(post) + len(slab)
    cl = cluster_points(sel2.tolist(), heights=h2.tolist())
    posts = [c for c in cl if c["height"] > 3.0]
    assert len(posts) == 1 and abs(posts[0]["x"] - 12.0) < 0.1 and abs(posts[0]["y"] - 2.6) < 0.1
    slabs = [c for c in cl if c["height"] < 0.3]
    assert len(slabs) == 1 and abs(slabs[0]["y"]) < 0.5           # the in-lane slab is still an obstacle
    # nothing low at all: untouched
    s3, h3, d3 = remove_road_edge_points(np.array(post, dtype=np.float32), np.array(post_h, dtype=np.float32))
    assert d3 == 0 and s3.shape[0] == len(post)


# ---------------------------------------------------------------- wiring

def test_mode_from_env_and_perception_wiring():
    assert ground_filter_mode_from_env({}) == "patches"
    assert ground_filter_mode_from_env({"WARP_GROUND_FILTER": "flat"}) == "flat"
    assert ground_filter_mode_from_env({"WARP_GROUND_FILTER": "nonsense"}) == "patches"
    src = open(PERC).read()
    assert 'if self.ground_filter_mode == "patches":' in src
    assert "ground = self.ground_filter.apply(pts)" in src
    assert "flat_cut(pts" in src
    assert "cluster_points(xy.tolist(), heights=heights.tolist())" in src
    assert "remove_road_edge_points(sel, heights, cluster_points)" in src
    # the old numbers are still the flat fallback, unchanged
    assert "self.minimum_lidar_z = -2.15" in src and "self.maximum_lidar_z = 1.5" in src
