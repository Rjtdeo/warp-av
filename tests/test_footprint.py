"""
Swept-path geometry (Planning V2, phase 1A): the van as a rectangle slid
along the route, instead of a point compared with a line. Pure math.
"""
import math

from warp_av.planning.footprint import VehicleFootprint, sweep_conflict

VAN = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
# body edge 0.99 m from the centre line; with the margin, 1.29 m


def straight(length=60.0, step=2.0):
    return [(x, 0.0) for x in _frange(0.0, length, step)]


def left_bend(radius=8.0, straight_in=10.0, degrees=90.0, step=1.0):
    """A straight run, then a left-hand arc of `radius`. Heading starts along +x."""
    pts = [(x, 0.0) for x in _frange(0.0, straight_in, step)]
    cx, cy = straight_in, radius          # centre of the arc, to the LEFT (+y)
    n = int(radius * math.radians(degrees) / step)
    for k in range(1, n + 1):
        a = math.radians(degrees) * k / n
        pts.append((cx + radius * math.sin(a), cy - radius * math.cos(a)))
    return pts


def _frange(a, b, step):
    out = []
    x = a
    while x <= b + 1e-9:
        out.append(x)
        x += step
    return out


def test_obstacle_clearly_outside_a_straight_path_is_no_conflict():
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, 2.5)) is None
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, -2.5)) is None


def test_obstacle_inside_the_van_width_is_a_conflict():
    hit = sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, 0.4))
    assert hit is not None
    # contact when the nose (2.96 m + 0.30 m margin ahead of the centre) reaches it
    assert abs(hit.along_m - (12.0 - VAN.swept_half_length)) < 0.5
    assert hit.lateral_m == 0.4


def test_the_safety_margin_decides_the_near_misses():
    # 1.15 m off the line: outside the metal (0.99), inside the margin (1.29) -> conflict
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, 1.15)) is not None
    # 1.40 m off the line: outside the margin -> no conflict
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, 1.40)) is None
    # the same 1.15 m point is clear when the margin is switched off
    bare = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.0)
    assert sweep_conflict(straight(), (0.0, 0.0), bare, (12.0, 1.15)) is None
    # a fat obstacle (radius 0.5) at 1.40 m reaches back into the margin -> conflict
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (12.0, 1.40), obstacle_radius=0.5) is not None


def test_van_corner_sweeps_into_an_obstacle_on_the_outside_of_a_bend():
    route = left_bend(radius=8.0)
    # a point 1.6 m to the RIGHT of the route line, half-way round the bend:
    # on a straight that would be clear (1.6 > 1.29) ...
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (14.0, -1.6)) is None
    # ... but in the bend the front-right corner swings about 1.46 m wide
    # of the line (sqrt((R + 0.99)^2 + 2.96^2) - R), and with the margin it
    # reaches 1.76 m: the corner clips the obstacle.
    cx, cy = 10.0, 8.0
    a = math.radians(45.0)
    mid_x, mid_y = cx + 8.0 * math.sin(a), cy - 8.0 * math.cos(a)
    # outward normal at that point (away from the arc centre)
    nx, ny = (mid_x - cx) / 8.0, (mid_y - cy) / 8.0
    obstacle = (mid_x + 1.6 * nx, mid_y + 1.6 * ny)
    hit = sweep_conflict(route, (0.0, 0.0), VAN, obstacle, horizon_m=40.0)
    assert hit is not None, "the outside corner must be caught in the bend"
    assert hit.lateral_m < -1.3                 # it really is off to the right of the line
    # the same obstacle is missed by a pure centre-line rule with the old 1.40 m block band
    assert abs(hit.lateral_m) > 1.40


def test_obstacle_behind_the_van_is_ignored():
    assert sweep_conflict(straight(), (20.0, 0.0), VAN, (14.0, 0.0)) is None
    assert sweep_conflict(straight(), (20.0, 0.0), VAN, (19.0, 0.3)) is None   # already alongside/behind the centre


def test_obstacle_beyond_the_horizon_is_ignored_until_it_is_in_range():
    far = (28.0, 0.0)
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, far, horizon_m=20.0) is None
    hit = sweep_conflict(straight(), (0.0, 0.0), VAN, far, horizon_m=30.0)
    assert hit is not None and abs(hit.along_m - (28.0 - VAN.swept_half_length)) < 0.5


def test_degenerate_inputs_return_none():
    assert sweep_conflict([(0.0, 0.0)], (0.0, 0.0), VAN, (5.0, 0.0)) is None      # one point, no direction
    assert sweep_conflict(straight(), (0.0, 0.0), VAN, (5.0, 0.0), horizon_m=0.0) is None


def test_route_points_may_be_objects_with_x_and_y():
    class P:
        def __init__(self, x, y):
            self.x, self.y = x, y
    route = [P(x, 0.0) for x in range(0, 40, 2)]
    hit = sweep_conflict(route, P(0.0, 0.0), VAN, P(10.0, 0.2))
    assert hit is not None and hit.along_m > 0
