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

IDLE_COST = 0.2              # extra charge per step for standing still OUTSIDE the
IDLE_SPEED = 0.2             # box (below this speed). Round 7a relearned round 3's
                             # standstill within 1,200 attempts once a parked car
                             # ahead turned overshoots into -80 crashes: sitting
                             # still cost -40, so it sat. With this, standing still
                             # for a whole easy-rung attempt costs about -80 too -
                             # never the safe option.

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


ALIGN_ZONE_M = 9.5           # lining-up only pays inside the last slot-and-a-bit; paid
                             # everywhere it rewarded the early cut-in that clips a car
REVERSE_COST = 0.02          # per step in reverse: reversing is allowed, dithering is not


def step_outcome(ax, ay, herr, speed, dist_prev, steer, steer_prev, t_s,
                 collided, timeout_s=30.0, bounds_m=None, align_prev=None,
                 feeler_readings=None, ax_prev=None, overshoot_m=None, reversing=False):
    """One training step's (reward, done, info).

    dist_prev is last step's distance-to-centre (progress shaping); when
    ax_prev is given, progress is paid ALONG THE BAY (|ax| shrinking) instead -
    round 7 showed centre-distance progress rewards the early cut-in that
    clips a parked car, and reversing into a slot shrinks |ax| just as well.
    align_prev is last step's lateral_error; pass it to also pay for getting
    straighter and more centred, not only closer. None disables that term.
    overshoot_m overrides OVERSHOOT_M (a reversing parker must be allowed to
    pull past the slot first).
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
    if ax > (OVERSHOOT_M if overshoot_m is None else overshoot_m):
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
    if ax_prev is not None:
        r = APPROACH_PAY * (abs(ax_prev) - abs(ax))
    else:
        r = APPROACH_PAY * (dist_prev - dist)
    if align_prev is not None and abs(ax) <= ALIGN_ZONE_M:
        r += ALIGN_PAY * (align_prev - lateral_error(ay, herr))
    if feeler_readings is not None:
        r += proximity_penalty(feeler_readings)
    if reversing:
        r -= REVERSE_COST
    r -= 0.05
    r -= 0.10 * abs(steer - steer_prev)
    if inside:
        r += INSIDE_PAY   # being IN the box pays a little every moment
    elif speed < IDLE_SPEED:
        r -= IDLE_COST    # standing still outside the box is never the safe play
    return r, False, {"result": "driving", "dist": round(dist, 2)}


# ---------------------------------------------------------------------------
# Where an attempt starts, how far it may stray, and how long it gets
# ---------------------------------------------------------------------------

KEEP_CLEAR_BACK_M = 18.0     # slot frame: how far behind the slot centre must be empty
KEEP_CLEAR_AHEAD_M = 4.5     # ... and ahead (the slot's own front edge is at +3.5)
KEEP_CLEAR_SIDE_M = 4.0      # ... and to either side


def bay_is_clear(slot_x, slot_y, slot_yaw, static_outlines,
                 back_m=KEEP_CLEAR_BACK_M, ahead_m=KEEP_CLEAR_AHEAD_M,
                 side_m=KEEP_CLEAR_SIDE_M):
    """False if any point of any static outline lies in the bay's keep-clear
    zone: the slot itself, the bays behind it, and the approach strip.

    Town10 bakes decorative parked cars and motorcycles into the map. They are
    not actors, the student's five observations carry nothing about them, and
    it cannot steer round what it cannot see. Round-6 exam, seed 2026: all nine
    misses were static.car / static.motorcycle — one of them with the van
    parked dead centre, 0.3 deg off, at 0.02 m/s, on top of a motorcycle.
    static_outlines is a list of point lists (world x, y), as returned by the
    main stack's static-vehicle scan.
    """
    for outline in static_outlines:
        for px, py in outline:
            ax, ay, _ = to_slot_frame(px, py, 0.0, slot_x, slot_y, slot_yaw)
            if -back_m <= ax <= ahead_m and abs(ay) <= side_m:
                return False
    return True


# ---------------------------------------------------------------------------
# Round 7: eyes for obstacles, and neighbours to practise against
# ---------------------------------------------------------------------------

FEELER_REACH_M = 10.0
# Four sectors in the van's own frame, degrees, 0 = straight ahead, positive =
# the van's right. The two ahead sectors overlap by 20 deg so something dead
# ahead shows up in both. The side sectors reach back to the rear quarters:
# round 7c's crashes were the TAIL clipping a car the van had already passed -
# the right feeler read 2-3 m alongside it, then "clear" while the rear was
# still beside it, so no warning ever fired. Straight behind stays unseen.
FEELER_SECTORS = ((-70.0, 10.0),     # ahead-left
                  (-10.0, 70.0),     # ahead-right
                  (-160.0, -70.0),   # left, back to the rear quarter
                  (70.0, 160.0))     # right, back to the rear quarter


def feelers(van_x, van_y, van_yaw, points, reach_m=FEELER_REACH_M):
    """Distance to the nearest obstacle point in each sector, as a fraction of
    reach_m: 1.0 means nothing within reach, 0.3 means something 3 m away.

    Rounds 1-6 gave the student five numbers, none of them about anything
    around it. It parked dead centre on top of a motorcycle once, and would hit
    a real neighbour every time. These four numbers are the fix - measured from
    the van's centre, so the student learns its own half-length.
    """
    best = [reach_m] * len(FEELER_SECTORS)
    c, s = math.cos(-van_yaw), math.sin(-van_yaw)
    for px, py in points:
        dx, dy = px - van_x, py - van_y
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        d = math.hypot(lx, ly)
        if d == 0.0 or d >= reach_m:
            continue
        ang = math.degrees(math.atan2(ly, lx))
        for i, (lo, hi) in enumerate(FEELER_SECTORS):
            if lo <= ang <= hi and d < best[i]:
                best[i] = d
    return [b / reach_m for b in best]


NEIGHBOUR_BAY_PITCH_M = 7.0          # the next bay along is one slot length away
NEIGHBOUR_BEHIND_BAYS = 2            # the practice car sits TWO bays back (-14 m).
                                     # Round 7b put it one bay back: parallel bays,
                                     # a 4.7 m van with a 7 m turning circle and no
                                     # reverse gear cannot nose into a 7 m gap with a
                                     # car right behind - 0 parks in ~1,800 tries. Two
                                     # bays back is where the round-6 pull-in really
                                     # clips cars (crashes at 11-15 m before the bay),
                                     # and turning in later avoids it.
NEIGHBOUR_BEHIND_CLEARANCE_M = 6.0   # a car `n` bays back may only be placed when the
                                     # van starts at least n*7 + 6 m back: half a car
                                     # (2.35) + half a van (2.6) + a metre of road, so
                                     # the van is born BEHIND the car and has to drive
                                     # past it. Round 8a/8b used a flat 13 m, written
                                     # for one bay back, with the car TWO bays back
                                     # (-14 m): from a 16 m start the van was born
                                     # alongside the car's rear, 1.4 m away, and its
                                     # first move (a steer to the right) hit it -
                                     # 313/313 crashes at 1.4 s, nothing to learn from.
NEIGHBOUR_BEHIND_MIN_START_M = 1 * NEIGHBOUR_BAY_PITCH_M + NEIGHBOUR_BEHIND_CLEARANCE_M   # 13 m, one bay back

PROXIMITY_CLOSE = 0.15               # feeler reading (x10 m) below which it is "too
PROXIMITY_PAY = 15.0                 # close": charge PAY * (CLOSE - reading) per step,
                                     # Was 0.25 (2.5 m): a lawful pass down a 3.5 m lane
                                     # beside a parked car is a 1.4 m gap, and that was
                                     # being charged -1.65 a step for driving correctly.
                                     # 1.5 m leaves the lawful pass nearly free and
                                     # still charges a swerve towards the car.
                                     # so the warning arrives BEFORE the crash. Cannot
                                     # be farmed: it is never positive. Was 3.0 in
                                     # round 7c's first half hour: the +8/m approach
                                     # pay rewards cutting the corner early, which is
                                     # the crash path, and a -0.6/step warning lost that
                                     # tug-of-war (26% with the car, flat from the first
                                     # block). At 15 a close pass costs about a crash,
                                     # but before contact, where it can still steer.
NEIGHBOUR_AHEAD_MIN_START_M = 9.0    # hazards after the skill, not before: round 7a
                                     # put a car 7 m ahead on a third of its 4 m
                                     # starts and the student learned to freeze


def proximity_penalty(feeler_readings, close=PROXIMITY_CLOSE, pay=PROXIMITY_PAY):
    """Per-step charge for anything inside the 'too close' band of any feeler.
    Zero when nothing is near; grows as a reading falls towards contact."""
    return -pay * sum(max(0.0, close - f) for f in feeler_readings)


def neighbour_pose(slot_x, slot_y, slot_yaw, bays_away):
    """Pose of the bay `bays_away` slots along (negative = behind the target)."""
    d = bays_away * NEIGHBOUR_BAY_PITCH_M
    return (slot_x + d * math.cos(slot_yaw), slot_y + d * math.sin(slot_yaw), slot_yaw)


def neighbour_ahead_fits(start_dist_m):
    """A car in the bay ahead only once the van starts far enough back to have
    a real approach to practise, so an overshoot is a lesson and not a trap."""
    return start_dist_m >= NEIGHBOUR_AHEAD_MIN_START_M


def neighbour_behind_fits(start_dist_m, bays_back=1):
    """A car `bays_back` bays behind the target may only be placed when the van
    starts far enough back to be born behind it, not beside it: at least
    bays_back * 7 m + 6 m (half a car, half a van, a metre of road)."""
    return start_dist_m >= bays_back * NEIGHBOUR_BAY_PITCH_M + NEIGHBOUR_BEHIND_CLEARANCE_M


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


class Stages:
    """Hazards in stages, unlocked by success: 0 = an empty bay, 1 = a car two
    bays back (avoidable forward) and sometimes a car ahead, 2 = also a car
    right behind, with the bay ahead free - only a pull-past-and-reverse gets
    in. Round 7 put a car right behind on day one and the student froze."""

    NAMES = ("empty", "car two bays back", "car right behind")

    def __init__(self, window=40, promote_at=0.6, demote_at=0.2, start=0):
        self.window, self.promote_at, self.demote_at = window, promote_at, demote_at
        self.level = max(0, min(len(self.NAMES) - 1, start))
        self._recent = []

    def record(self, parked):
        self._recent.append(bool(parked))
        if len(self._recent) < self.window:
            return None
        rate = sum(self._recent) / float(len(self._recent))
        if rate >= self.promote_at and self.level < len(self.NAMES) - 1:
            self.level += 1
        elif rate <= self.demote_at and self.level > 0:
            self.level -= 1
        else:
            self._recent = self._recent[len(self._recent) // 2:]
            return None
        self._recent = []
        return self.level

    def describe(self):
        return f"obstacle stage {self.level}: {self.NAMES[self.level]}"
