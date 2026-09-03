"""
Pure math for the RL parking student: what it sees, what it scores, and how
hard the next attempt should be.

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

APPROACH_PAY = 8.0           # points per metre closed on the slot. Round 3 paid
                             # 2.0, so driving the whole 16 m was worth +32 while
                             # standing still cost -45 and a crash cost -200. The
                             # student took the safe option and never moved: 1,626
                             # episodes, not one failed attempt covered any ground.

ALIGN_PAY = 20.0             # points per metre of lateral corner error removed.
                             # Round 5 paid only for closing DISTANCE, so the van
                             # drove to the bay and hovered there crooked or off-
                             # centre: 5 parks in 2,430 attempts, none in the last
                             # 1,140. Nothing had ever paid it for lining up.

COLLISION_PENALTY = -80.0    # still the worst outcome, but no longer so far below
                             # the parking prize (100-220) that the last metre is
                             # never worth risking. Round 5's -200 froze the van a
                             # step short of the box.

INSIDE_PAY = 0.4             # per step spent inside the box while still moving.
                             # Round 2's 1.5 was a "taste of success"; at 1.5 a van
                             # crawling the slot's length at the parked-speed limit
                             # would earn ~350 — more than any park — so the moment
                             # it learned to enter the box it would learn never to
                             # stop. 0.4 keeps the taste below the smallest prize.

SUCCESS_SPEED = 0.3          # must be (nearly) stopped to count as parked
OUT_OF_BOUNDS_M = 18.0       # FLOOR only — see bounds_for(). The real limit is
                             # measured from where THIS attempt started, because
                             # a full-distance start is already ~16.3 m out.
OVERSHOOT_M = 5.0            # drove past the slot -> attempt over
LATERAL_LOST_M = 7.0         # wandered sideways off the road -> attempt over

LANE_START_M = 16.0          # how far back down the lane a p=0 attempt begins


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


def lateral_error(ay, herr):
    """How far the van's corners reach sideways of the slot's centre line, in
    metres: its own offset plus what its heading swings the far corners out by.
    Zero means centred and straight — the pose the 0.27 m side clearance needs."""
    return abs(ay) + VAN_HALF_LEN * abs(math.sin(herr))


def step_outcome(ax, ay, herr, speed, dist_prev, steer, steer_prev, t_s,
                 collided, timeout_s=30.0, bounds_m=None, align_prev=None):
    """One training step's (reward, done, info).

    dist_prev is last step's distance-to-centre (progress shaping).
    align_prev is last step's lateral_error; pass it to also pay for getting
    straighter and more centred, not only closer. None disables that term.
    bounds_m is how far out this attempt is allowed to stray; pass
    bounds_for(start_dist) so a far start is not disqualified for barely
    moving. None keeps the old fixed limit.
    """
    limit = OUT_OF_BOUNDS_M if bounds_m is None else bounds_m
    dist = math.hypot(ax, ay)
    inside, m_len, m_wid = van_corners_in_slot(ax, ay, herr)

    if collided:
        return COLLISION_PENALTY, True, {"result": "collision"}
    if dist > limit or abs(ay) > LATERAL_LOST_M:
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

    # Ongoing: pay for getting closer AND for getting lined up; charge for
    # time and jerky steering. Both shaping terms are differences, so hovering
    # in place earns nothing — only improving does.
    r = APPROACH_PAY * (dist_prev - dist)
    if align_prev is not None:
        r += ALIGN_PAY * (align_prev - lateral_error(ay, herr))
    r -= 0.05
    r -= 0.10 * abs(steer - steer_prev)
    if inside:
        r += INSIDE_PAY   # being IN the box pays a little every moment
    return r, False, {"result": "driving", "dist": round(dist, 2)}


# ---------------------------------------------------------------------------
# Where an attempt starts, how far it may stray, and how long it gets
# ---------------------------------------------------------------------------

def spawn_pose(p, lane_x, lane_y, lane_yaw, slot_x, slot_y, slot_yaw):
    """Start pose for difficulty p, sliding along the ideal pull-in.

    p = 1.0 -> already in the box, only needs to stop (the free lesson)
    p = 0.0 -> the full exam: back at the lane start, must drive and turn in
    """
    p = max(0.0, min(1.0, p))
    x = slot_x * p + lane_x * (1.0 - p)
    y = slot_y * p + lane_y * (1.0 - p)
    yaw = math.atan2(math.sin(slot_yaw) * p + math.sin(lane_yaw) * (1.0 - p),
                     math.cos(slot_yaw) * p + math.cos(lane_yaw) * (1.0 - p))
    return x, y, yaw


def bounds_for(start_dist_m, slack_m=8.0, floor_m=OUT_OF_BOUNDS_M):
    """How far from the slot this attempt may stray before it is written off.

    Round 1-3 used a flat 18 m for every attempt. A full-distance start is
    already ~16.3 m out, so the student had under 2 m of room: one wrong
    second of steering ended the attempt before it had driven anywhere.
    """
    return max(floor_m, start_dist_m + slack_m)


def timeout_for(start_dist_m, base_s=12.0, per_m_s=2.0, cap_s=60.0):
    """Seconds allowed. A far start needs longer than a nudge-in-the-box."""
    return min(cap_s, base_s + per_m_s * max(0.0, start_dist_m))


class Curriculum:
    """Picks the next attempt's difficulty, and gets harder as the student wins.

    Rounds 1-3 drew all six difficulties with equal chance for ever, so a third
    of every training run was spent re-proving lessons already mastered. Here a
    'focus' level walks down the ladder once the recent success rate is good
    enough, with occasional easy runs mixed in so old skills are not forgotten.
    """

    # Difficulty as a fraction of the way along the ideal pull-in: 0.0 is the
    # full ~16 m exam. Rounds 1-4 started this ladder at 1.0 — the van spawned
    # ALREADY INSIDE the box, where doing nothing scored ~220 instantly and
    # driving would have left the box for an overshoot. Lesson one taught the
    # exact habit the rest of the ladder needs broken, and the student never
    # unlearned it (measured 2026-09-03: 10/10 parked at p=1.0, 0/30 at p=0.0).
    # The easiest rung is now a genuine ~4 m pull-in.
    LEVELS = (0.75, 0.6, 0.45, 0.3, 0.15, 0.0)

    def __init__(self, window=40, promote_at=0.55, demote_at=0.15,
                 review_p=0.25, start_level=0):
        self.window = window
        self.promote_at = promote_at
        self.demote_at = demote_at
        self.review_p = review_p
        self.focus = max(0, min(len(self.LEVELS) - 1, start_level))
        self._recent = []
        self.last_move = None

    @property
    def focus_p(self):
        return self.LEVELS[self.focus]

    def next_p(self, rng):
        """Difficulty for the next attempt."""
        if self.focus > 0 and rng.random() < self.review_p:
            return self.LEVELS[rng.randrange(self.focus)]   # revisit an easier one
        return self.focus_p

    def record(self, p, parked):
        """Log an attempt. Returns the new focus level if the ladder moved."""
        if p != self.focus_p:
            return None                    # review runs never move the ladder
        self._recent.append(bool(parked))
        if len(self._recent) < self.window:
            return None
        seen = len(self._recent)
        rate = sum(self._recent) / float(seen)
        was = self.focus
        if rate >= self.promote_at and self.focus < len(self.LEVELS) - 1:
            self.focus += 1
        elif rate <= self.demote_at and self.focus > 0:
            self.focus -= 1
        else:
            # no move: slide the window so we re-check soon instead of waiting
            # for another full window of attempts
            self._recent = self._recent[seen // 2:]
            return None
        # Remember WHY we moved. The move is the only moment anything gets
        # logged, and it is also the moment the window is emptied — without
        # this the log line would for ever read "0 recent, 0% parked".
        self.last_move = {"from": was, "to": self.focus, "rate": rate, "n": seen}
        self._recent = []
        return self.focus

    def describe(self):
        # Name the rung the same way --start-level does (0-indexed) as well as
        # human 1-of-6, so a number read out of the log cannot be fed back in
        # one rung too hard.
        head = (f"level {self.focus + 1}/{len(self.LEVELS)} "
                f"(--start-level {self.focus}, p={self.focus_p:.2f})")
        if self._recent:
            seen = len(self._recent)
            rate = sum(self._recent) / float(seen)
            return f"{head}, {seen} recent at this level, {rate * 100:.0f}% parked"
        if self.last_move:
            m = self.last_move
            verb = "promoted" if m["to"] > m["from"] else "eased back"
            return (f"{head}, just {verb} after {m['rate'] * 100:.0f}% parked "
                    f"over {m['n']} attempts")
        return f"{head}, no attempts yet"
