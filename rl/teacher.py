"""
The rule-based teacher for round 9: a stateless parking driver that works in
the slot frame from exactly what the brain sees (position along and across
the bay, heading error, speed, four feelers), so its actions can be copied.

    Waymo's recipe (ChauffeurNet 2019, "Imitation is not enough" 2022, RL
    fine-tuning of sim agents 2024): copy a competent driver first, then
    practise. Rounds 1-8 practised from zero and spent a week un-learning
    the habit of swinging into the parking lane early.

Behaviour, all in the slot frame (ax along the bay, negative = before it;
ay across, the driving lane is at ay = LANE_Y < 0; steer > 0 turns toward +ay):
  1. hold the driving lane while more than TURN_START_M before the bay, while a
     car is beside the van (right feeler), or while the bay's approach is
     blocked by a car ahead-right (a car right behind the bay);
  2. otherwise follow an S-curve from the lane into the bay line and stop
     inside the slot;
  3. blocked bay: keep going, pull past to PULL_PAST_M, then reverse along an
     S-curve into the slot (reverse gear = third control).

Also holds a kinematic bicycle model so the teacher can be exercised in tests
without CARLA (`simulate`).
"""
import math

from rl.parking_math import (van_corners_in_slot, SLOT_LEN, SLOT_WID, VAN_HALF_LEN,
                             VAN_HALF_WID, FEELER_SECTORS, feelers)

LANE_Y = -3.1            # driving-lane centre across the bay (measured: 3.1 m, kerb side positive)
TURN_START_M = 12.0      # start the S-curve this far before the bay (= LANE_HOLD_UNTIL_M)
TURN_END_M = -3.2        # the S-curve reaches the bay line here (van centre), straight; then creep in
BESIDE_CLOSE = 0.30      # right feeler below this (3 m): a car is beside us, keep holding;
                         # released about 2 m after the van's centre passes the car's nose
BLOCKED_CLOSE = 0.7      # ahead-right feeler below this while entering the turn: bay approach blocked
PULL_PAST_M = 9.5        # blocked: pull past to here (van centre), then reverse in; a long,
                         # gentle reverse S-curve arrives straight (the car behind sits at -4.65)
REVERSE_END_M = -1.1     # the reverse S-curve reaches the bay line here (van centre): the van's
                         # tail is then level with the slot's rear edge, 1.15 m clear of a car right behind
REVERSE_TOO_LATE_M = -6.5  # still in the lane this close to the bay: a forward turn-in no longer fits
APPROACH_SPEED = 2.8
TURN_SPEED = 1.8
FINAL_SPEED = 0.9
REVERSE_SPEED = 1.2
STOP_WINDOW_M = 0.35     # |ax| below this while lined up: brake to a stop
SHUFFLE_FWD_M = 0.9      # correcting a sideways error: forward strokes may run to here (nose 3.25 m, slot 3.5)
SHUFFLE_TOL_M = 0.3      # sideways error below this: just straighten and stop
K_HEADING = 4.0          # steer per radian of heading error
K_LATERAL = 0.5          # lateral error gain (Stanley-style aim = atan2(K*e, lookahead))
LOOKAHEAD_M = 4.5        # (grid-searched in the paper model: 10/10 forward scenarios park)
WHEELBASE_M = 3.5
MAX_WHEEL_RAD = math.radians(60.0) * 0.8   # CARLA vans lock to ~60-70 deg; the arena multiplies steer by 0.8
CONSERVATIVE_WHEEL_RAD = math.radians(35.0) * 0.8   # a stiff van, for robustness checks


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def reference_lateral(ax, start_m, end_m, from_y=LANE_Y, to_y=0.0):
    """Lateral position of the S-curve at `ax`: from_y at start_m, to_y at end_m."""
    t = (ax - start_m) / (end_m - start_m) if end_m != start_m else 1.0
    return from_y + (to_y - from_y) * smoothstep(t)


def reference_slope(ax, start_m, end_m, from_y=LANE_Y, to_y=0.0, h=0.05):
    """d(lateral)/d(along) of that S-curve, by central difference."""
    return (reference_lateral(ax + h, start_m, end_m, from_y, to_y)
            - reference_lateral(ax - h, start_m, end_m, from_y, to_y)) / (2 * h)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _steer_to_path(ax, ay, herr, y_ref, slope, reverse=False, lookahead=None, k_lateral=None):
    """Stanley-style: aim along the path's slope, corrected toward it. In
    reverse the van's motion is opposite to its heading, so the error sign flips.
    A shorter lookahead (used when creeping) corrects sideways error harder."""
    path_heading = math.atan(slope)
    lateral_aim = math.atan2((k_lateral if k_lateral is not None else K_LATERAL) * (y_ref - ay),
                             lookahead or LOOKAHEAD_M)
    if not reverse:
        err = _wrap(path_heading + lateral_aim - herr)
        steer = K_HEADING * err
    else:
        # Backing up along the path the nose still points along the path's
        # tangent, but to move the TAIL toward the path the nose must swing the
        # other way, so the lateral correction is mirrored; and with the van
        # travelling backwards a right wheel turns the heading LEFT, so the
        # wheel sign flips too.
        err = _wrap(path_heading - lateral_aim - herr)
        steer = -K_HEADING * err
    return max(-1.0, min(1.0, steer))


def _pedal(speed, target, reverse=False):
    """Accelerator/brake in [-1, 1] toward a target speed (magnitude)."""
    v = abs(speed)
    if target <= 0.0:
        return -1.0
    d = target - v
    if d > 0.3:
        return min(1.0, 0.35 + 0.5 * d)
    if d < -0.3:
        return max(-1.0, 0.6 * d)
    return 0.15


def teacher_action(ax, ay, herr, speed, feeler_readings=None, reverse_ok=True):
    """(steer, pedal, gear) for one step. gear is 1.0 for reverse, else 0.0.
    Stateless: everything is derived from what the brain itself observes."""
    f = list(feeler_readings) if feeler_readings is not None else [1.0, 1.0, 1.0, 1.0]
    ahead_right, right = f[1], f[3]
    lane_side = -1.0 if LANE_Y < 0 else 1.0
    in_lane = (ay * lane_side) > 1.6          # still (mostly) in the driving lane

    inside, m_len, m_wid = van_corners_in_slot(ax, ay, herr)
    lined_up = abs(ay) < 0.35 and abs(herr) < math.radians(8)
    standing = abs(speed) <= 0.02
    moving_fwd = speed > 0.02
    moving_back = speed < -0.02

    # ---- at the bay ----
    if not in_lane:
        off_side = abs(ay) >= SHUFFLE_TOL_M
        if moving_fwd:
            limit = SHUFFLE_FWD_M if off_side else -STOP_WINDOW_M
            if ax > limit:
                return 0.0, -1.0, 0.0                 # end of the forward stroke
        if standing and inside and ax > -STOP_WINDOW_M - 0.3:
            return 0.0, -1.0, 0.0                     # parked
        if reverse_ok and ax > REVERSE_END_M and (moving_back or (standing and not inside and ax > -0.6)):
            # back up: from far right along the reverse S-curve to the slot's
            # rear, correcting sideways error with the tail (mirrored aim)
            y_ref = reference_lateral(ax, PULL_PAST_M, REVERSE_END_M)
            slope = reference_slope(ax, PULL_PAST_M, REVERSE_END_M)
            k_lat = 0.8 if ax < 2.0 else None
            steer = _steer_to_path(ax, ay, herr, y_ref, slope, reverse=True,
                                   lookahead=1.2 if ax < 2.0 else None, k_lateral=k_lat)
            target = REVERSE_SPEED if ax > 1.5 else FINAL_SPEED
            return steer, _pedal(speed, target), 1.0

    beside = right < BESIDE_CLOSE
    blocked = (ax >= -TURN_START_M) and (ahead_right < BLOCKED_CLOSE) and in_lane and ax <= REVERSE_TOO_LATE_M

    # ---- hold the lane (only while in it): far out, or a car beside us, or the bay's approach blocked ----
    if in_lane and (ax < -TURN_START_M or beside or blocked):
        steer = _steer_to_path(ax, ay, herr, LANE_Y, 0.0)
        target = APPROACH_SPEED if ax < -TURN_START_M - 4.0 else TURN_SPEED
        return steer, _pedal(speed, target), 0.0

    # ---- reverse manoeuvre: we are (nearly) past the bay's centre, still in the lane ----
    if reverse_ok and in_lane and ax > REVERSE_TOO_LATE_M:
        room_ahead = ahead_right > 0.5            # do not pull past into a car parked ahead
        if not room_ahead and ax < 2.0 and not moving_back:
            return 0.0, -1.0, 0.0                 # boxed in (cars behind AND ahead): stop, do not force it
        pull_more = ((moving_fwd and ax < PULL_PAST_M - 0.4) or (standing and ax < PULL_PAST_M - 0.6)) and room_ahead
        if pull_more and not moving_back:
            steer = _steer_to_path(ax, ay, herr, LANE_Y, 0.0)
            target = REVERSE_SPEED if ax > PULL_PAST_M - 2.5 else TURN_SPEED
            return steer, _pedal(speed, target), 0.0
        y_ref = reference_lateral(ax, PULL_PAST_M, REVERSE_END_M)
        slope = reference_slope(ax, PULL_PAST_M, REVERSE_END_M)
        steer = _steer_to_path(ax, ay, herr, y_ref, slope, reverse=True)
        return steer, _pedal(speed, REVERSE_SPEED), 1.0

    # ---- forward S-curve into the bay, then a straight creep to the centre ----
    y_ref = reference_lateral(ax, -TURN_START_M, TURN_END_M)
    slope = reference_slope(ax, -TURN_START_M, TURN_END_M)
    creeping = ax > TURN_END_M
    lookahead = 1.5 if creeping else None
    k_lat = None
    if in_lane and ax > -TURN_START_M + 1.0:
        # the hold ended late (a car beside us): no room for the gentle S. Drive
        # a straight diagonal to the S-curve's end point; the creep straightens.
        room = max(1.0, TURN_END_M - ax)
        slope = (0.0 - ay) / room
        y_ref = ay
        lookahead, k_lat = 2.0, 1.2
    if creeping:
        if abs(ay) >= SHUFFLE_TOL_M:
            lookahead, k_lat = 1.2, 0.8       # still off to the side: keep correcting through the stroke
        else:
            lookahead, k_lat = 2.0, 0.15      # within tolerance: straighten, the width margin absorbs the rest
    steer = _steer_to_path(ax, ay, herr, y_ref, slope, lookahead=lookahead, k_lateral=k_lat)
    target = TURN_SPEED if ax < TURN_END_M - 1.5 else FINAL_SPEED
    return steer, _pedal(speed, target), 0.0


# ----------------------------------------------------------------------
# A kinematic bicycle model, so the teacher can be exercised without CARLA.
# ----------------------------------------------------------------------

def box_points(cx, cy, yaw, half_len, half_wid):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(cx, cy)]
    for fx, fy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)):
        lx, ly = fx * half_len, fy * half_wid
        pts.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return pts


def _van_hits_box(ax, ay, herr, box):
    """Separating-axis test between the van and a car box (cx, cy, yaw, hl, hw)."""
    def corners(cx, cy, yaw, hl, hw):
        c, s = math.cos(yaw), math.sin(yaw)
        return [(cx + fx * hl * c - fy * hw * s, cy + fx * hl * s + fy * hw * c)
                for fx, fy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    a = corners(ax, ay, herr, VAN_HALF_LEN, VAN_HALF_WID)
    b = corners(*box)
    for poly in (a, b):
        for i in range(4):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % 4]
            nx, ny = -(y2 - y1), (x2 - x1)
            pa = [px * nx + py * ny for px, py in a]
            pb = [px * nx + py * ny for px, py in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def simulate(start_ax, start_ay, start_herr, cars=(), dt=0.1, max_s=60.0, reverse_ok=True,
             wheel_rad=MAX_WHEEL_RAD, wheelbase=WHEELBASE_M):
    """Drive the teacher in a kinematic bicycle model. `cars` are boxes
    (cx, cy, yaw, half_len, half_wid) in the slot frame. Returns a dict with
    result ('parked' / 'collision' / 'timeout' / 'lost'), the path, margins."""
    ax, ay, herr, v = start_ax, start_ay, start_herr, 0.0
    path = [(ax, ay, herr)]
    t = 0.0
    reverse_steps = 0
    pts = [p for car in cars for p in box_points(*car)]
    while t < max_s:
        f = feelers(ax, ay, herr, pts)
        steer, pedal, gear = teacher_action(ax, ay, herr, v, f, reverse_ok=reverse_ok)
        rev = gear > 0.5
        cap = 1.5 if rev else 3.0
        throttle = max(0.0, pedal) * 0.7 if abs(v) < cap else 0.0
        brake = max(0.0, -pedal)
        want = -1.0 if rev else 1.0
        if abs(v) > 0.02 and (v > 0) != (want > 0):
            # gear opposite to the motion: brake to a stop first
            v -= math.copysign(6.0 * dt, v)
            if abs(v) < 0.06:
                v = 0.0
        else:
            mag = abs(v) + (3.2 * throttle - 6.0 * brake) * dt
            mag = max(0.0, min(cap, mag))
            v = want * mag
        delta = steer * wheel_rad
        ax += v * math.cos(herr) * dt
        ay += v * math.sin(herr) * dt
        herr = _wrap(herr + (v / wheelbase) * math.tan(delta) * dt)
        if rev:
            reverse_steps += 1
        t += dt
        path.append((ax, ay, herr))
        for car in cars:
            if _van_hits_box(ax, ay, herr, car):
                return dict(result="collision", t=t, path=path, reverse_steps=reverse_steps)
        inside, m_len, m_wid = van_corners_in_slot(ax, ay, herr)
        if inside and abs(v) < 0.3:
            return dict(result="parked", t=t, path=path, m_len=m_len, m_wid=m_wid,
                        herr_deg=math.degrees(herr), reverse_steps=reverse_steps)
        if abs(ay) > 7.0 or ax > 14.0:
            return dict(result="lost", t=t, path=path, reverse_steps=reverse_steps)
    return dict(result="timeout", t=t, path=path, reverse_steps=reverse_steps)


def car_box(bays_back):
    """A parked car `bays_back` bays behind the target bay (negative = behind), in the slot frame."""
    return (-7.0 * bays_back, 0.0, 0.0, 2.35, 1.0)
