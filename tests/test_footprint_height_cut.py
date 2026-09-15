"""What the van could drive into is what is below its own roof.

A street light has a column about 0.4 m across and a lamp arm three metres up and three
metres out. The laser sees both. Folded into one footprint they came out as a single body
two and a half metres long, lying across a parking approach -- and a box that long, turned
across the way in, reaches about a metre sideways.

Live in E3 on 2026-09-15 that refused the way into a parking bay. CARLA's own map says what
is there: BP_StreetLight_simple72, whose arm is 3.04 m across at 3.05 m up. The van is 2.56 m
tall. It cannot touch the arm, so the arm is not ground it has to steer around.

A wall is not affected: the points below the roof span its length just the same.
"""
from warp_av.perception.tracking import (FOOTPRINT_MAX_HEIGHT_M, FOOTPRINT_MIN_POINTS,
                                         cluster_points)


def column_and_arm():
    """A street light: a narrow column at ground level, an arm 3 m up reaching 3 m out."""
    pts, hs = [], []
    for i in range(8):                                    # the column, 0.4 m across
        pts.append((10.0 + 0.05 * (i % 2), 4.0 + 0.05 * (i // 2)))
        hs.append(0.3 + 0.35 * i)                         # up to 2.75 m
    for i in range(10):                                   # the arm, 3 m up, 3 m out
        pts.append((10.0 + 0.3 * i, 4.0))
        hs.append(3.05)
    return pts, hs


def wall(length_m=4.8, height_m=3.9):
    """A wall: points all along it, at every height."""
    pts, hs = [], []
    n = 12
    for i in range(n):
        for h in (0.4, 1.5, 2.4, height_m):
            pts.append((10.0 + length_m * i / (n - 1), 4.0))
            hs.append(h)
    return pts, hs


def one(pts, hs=None, cell=2.0):
    cl = cluster_points(pts, heights=hs, cell=cell, min_points=3, max_range=80.0)
    assert len(cl) == 1, f"expected one blob, got {len(cl)}"
    return cl[0]


# ---- the street light ------------------------------------------------------------------

def test_a_lamp_arm_three_metres_up_is_not_part_of_the_footprint():
    pts, hs = column_and_arm()
    c = one(pts, hs)
    assert c["length_m"] < 0.6, f"the column, not the arm: got {c['length_m']:.2f} m"
    assert c["box_len"] < 0.8, f"and the fitted box too: got {c['box_len']:.2f} m"


def test_and_without_the_cut_it_really_was_the_arm():
    """The same points with no heights at all: the old answer, kept as the cautious fallback."""
    pts, _ = column_and_arm()
    c = one(pts, None)
    assert c["length_m"] > 2.0, f"all the points, arm included: {c['length_m']:.2f} m"


def test_how_tall_the_thing_is_is_still_reported_in_full():
    """Only the FOOTPRINT is cut. Everything that asks how tall a thing is still gets the
    truth -- the kerb and road-edge rules read it."""
    pts, hs = column_and_arm()
    c = one(pts, hs)
    assert c["height"] > 3.0, f"the arm is still up there: {c['height']}"


# ---- a wall is untouched -----------------------------------------------------------------

def test_a_wall_keeps_its_length():
    pts, hs = wall()
    c = one(pts, hs)
    assert c["length_m"] > 4.0, f"a wall is really that long: got {c['length_m']:.2f} m"


def test_a_wall_taller_than_the_van_still_keeps_its_length():
    pts, hs = wall(length_m=6.0, height_m=7.0)
    c = one(pts, hs)
    assert c["length_m"] > 5.0, "height is not the test; where the points are is"


def test_a_car_is_unchanged():
    """Everything at driving height is below the cut, so nothing about it moves."""
    pts, hs = [], []
    for i in range(10):
        for j in range(4):
            pts.append((10.0 + 0.5 * i, 0.5 * j))
            hs.append(0.3 + 0.3 * j)
    with_h = one(pts, hs)
    without = one(pts, None)
    assert abs(with_h["length_m"] - without["length_m"]) < 1e-9
    assert abs(with_h["box_len"] - without["box_len"]) < 1e-9


# ---- the fallbacks -------------------------------------------------------------------------

def test_too_few_points_below_the_roof_falls_back_to_all_of_them():
    """A thing seen only over the top of something else: measuring it from one point is worse
    than measuring it from all of them."""
    pts = [(10.0 + 0.4 * i, 4.0) for i in range(8)]
    hs = [3.5] * 7 + [1.0]                       # only one point below the roof
    c = one(pts, hs)
    assert c["length_m"] > 2.0, "all the points were used, as before"
    assert FOOTPRINT_MIN_POINTS == 2


def test_a_point_with_no_height_is_kept_rather_than_thrown_away():
    pts = [(10.0 + 0.4 * i, 4.0) for i in range(8)]
    hs = [None] * 8
    c = one(pts, hs)
    assert c["length_m"] > 2.0


def test_the_cut_is_the_vans_own_roof_and_is_written_down():
    """2.56 m is the Sprinter; the rest is for pitch and the laser's own error. The lamp arm
    that started this is at 3.05 m, clear of it."""
    assert 2.6 <= FOOTPRINT_MAX_HEIGHT_M < 3.05


def test_where_the_thing_is_still_comes_from_all_its_points():
    """The cut decides the SHAPE. Where the blob is, how far off, and how tall, are still
    measured from everything the laser saw."""
    pts, hs = column_and_arm()
    with_h = one(pts, hs)
    without = one(pts, None)
    assert abs(with_h["x"] - without["x"]) < 1e-9 and abs(with_h["y"] - without["y"]) < 1e-9
    assert abs(with_h["distance"] - without["distance"]) < 1e-9
    assert with_h["n"] == without["n"], "no point was dropped from the blob itself"
