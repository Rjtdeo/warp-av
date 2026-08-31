"""
Pure math for the RL parking student: observations and rewards.

Everything is expressed in the SLOT's own frame (x along the slot, y across
it, origin at the slot centre) so every parking bay in every town looks
identical to the student — learn once, park anywhere.

No CARLA imports: the same code runs in offline tests and in training.
"""
from __future__ import annotations

import math

SLOT_LEN = 7.0
SLOT_WID = 2.5
VAN_HALF_LEN = 2.35
VAN_HALF_WID = 0.98

SUCCESS_SPEED = 0.3          # must be (nearly) stopped to count as parked
OUT_OF_BOUNDS_M = 18.0       # truly lost (the van STARTS ~11 m from the slot)
OVERSHOOT_M = 5.0            # drove past the slot -> attempt over
LATERAL_LOST_M = 7.0         # wandered sideways off the road -> attempt over


def to_slot_frame(van_x, van_y, van_yaw, slot_x, slot_y, slot_yaw):
    """Van pose -> (dx along slot, dy across slot, heading error rad)."""
    dx = van_x - slot_x
    dy = van_y - slot_y
    c, s = math.cos(-slot_yaw), math.sin(-slot_yaw)
    ax = dx * c - dy * s
    ay = dx * s + dy * c
    herr = (van_yaw - slot_yaw + math.pi) % (2 * math.pi) - math.pi
    return ax, ay, herr


def observation(van_x, van_y, van_yaw, speed, prev_steer,
                slot_x, slot_y, slot_yaw):
    """The 5 numbers the student sees. Scaled to roughly [-1, 1]."""
    ax, ay, herr = to_slot_frame(van_x, van_y, van_yaw, slot_x, slot_y, slot_yaw)
    return [max(-1.5, min(1.5, ax / 15.0)),
            max(-1.5, min(1.5, ay / 6.0)),
            herr / math.pi,
            speed / 5.0,
            prev_steer]


def van_corners_in_slot(ax, ay, herr):
    """(inside_fully, worst length margin, worst width margin) for a van at
    slot-frame position (ax, ay) with heading error herr."""
    c, s = math.cos(herr), math.sin(herr)
    worst_l = worst_w = 0.0
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        px = ax + sx * c * VAN_HALF_LEN - sy * s * VAN_HALF_WID
        py = ay + sx * s * VAN_HALF_LEN + sy * c * VAN_HALF_WID
        worst_l = max(worst_l, abs(px))
        worst_w = max(worst_w, abs(py))
    m_len = SLOT_LEN / 2.0 - worst_l
    m_wid = SLOT_WID / 2.0 - worst_w
    return (m_len >= 0.0 and m_wid >= 0.0), m_len, m_wid


def step_outcome(ax, ay, herr, speed, dist_prev, steer, steer_prev, t_s,
                 collided, timeout_s=30.0):
    """One training step's (reward, done, info).

    dist_prev is last step's distance-to-centre (progress shaping).
    """
    dist = math.hypot(ax, ay)
    inside, m_len, m_wid = van_corners_in_slot(ax, ay, herr)

    if collided:
        return -200.0, True, {"result": "collision"}
    if dist > OUT_OF_BOUNDS_M or abs(ay) > LATERAL_LOST_M:
        return -50.0, True, {"result": "out_of_bounds"}
    if ax > OVERSHOOT_M:
        return -40.0, True, {"result": "overshoot"}
    if t_s > timeout_s:
        return -30.0, True, {"result": "timeout"}

    if inside and abs(speed) < SUCCESS_SPEED:
        # Parked! Base prize + bonuses for being centred and straight.
        r = 100.0
        r += 40.0 * max(0.0, min(m_len, 1.0))       # length margin bonus
        r += 60.0 * max(0.0, min(m_wid / 0.25, 1.0))  # width is the hard part
        r += 20.0 * max(0.0, 1.0 - abs(herr) / math.radians(6))
        return r, True, {"result": "parked", "m_len": round(m_len, 2),
                         "m_wid": round(m_wid, 2),
                         "herr_deg": round(math.degrees(herr), 1)}

    # Ongoing: reward progress, charge for time and jerky steering.
    r = 2.0 * (dist_prev - dist)
    r -= 0.05
    r -= 0.10 * abs(steer - steer_prev)
    if inside:
        r += 1.5          # being IN the box pays every moment (taste of success)
    return r, False, {"result": "driving", "dist": round(dist, 2)}
