"""Where the tick's time goes, and why the planner decided what it did.

Planning V2, phase 0. This module changes NOTHING about how the van drives. It exists so
that when a planning scenario fails, the van can say which part of itself was responsible
instead of somebody bisecting it by hand.

Two things live here.

**PhaseTimer** -- a stopwatch per phase of the tick, keeping a rolling window so it can
report an average, a 95th percentile and a genuine worst case. The old timer kept one
exponentially-smoothed number per phase, so "worst" was the worst SMOOTHED value and a
single 300 ms tick disappeared into the average. Smoothing is the wrong tool for finding
the tick that broke something.

**PlannerDecision** -- what the planner decided and why, in a form a machine can filter.
Until now a blocked path reported this:

    blocked = true

which is true, and useless. There is no way to tell from a log whether the van stopped for
a parked lorry or for the same kerb sliver forty times. What it reports now:

    state    = follow_route
    reason   = blocked_tracked_object
    blocker  = 17 (vehicle) at 13.2 m, 0.8 m off the line

The reason codes are deliberately a small closed set. A prose sentence is written for the
operator and read by nobody else; a code can be counted, grouped and regression-tested.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


def debug_planning_enabled(env=None) -> bool:
    """WARP_PLAN_DEBUG=1 adds the per-object detail to the telemetry.

    The decision itself -- state, reason, blocker -- is ALWAYS published: it costs nothing
    and it is what makes a failure readable. Only the long tail (every candidate the
    planner considered) is behind the switch.
    """
    env = os.environ if env is None else env
    return str(env.get("WARP_PLAN_DEBUG", "0")).strip().lower() in ("1", "true", "on", "yes")


# ---------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------

DEFAULT_WINDOW = 200          # about 20 s at 10 Hz: long enough for a p95 that means something


class PhaseTimer:
    """Per-phase durations over a rolling window.

    Usage mirrors the timer it replaces, so call sites read the same way:

        timer.start()
        ...work...
        timer.mark("perception")
        ...work...
        timer.mark("behaviour")
        timer.end()                      # closes the tick and records the total
    """

    def __init__(self, window: int = DEFAULT_WINDOW, clock=None):
        self.window = int(window)
        self._clock = clock or time.perf_counter
        self._samples: Dict[str, Deque[float]] = {}
        self._order: list = []            # phases in the order first seen, for readable output
        self._t0 = None
        self._tick_t0 = None
        self.ticks = 0

    def start(self) -> None:
        now = self._clock()
        self._t0 = now
        self._tick_t0 = now

    def mark(self, name: str) -> float:
        """Close the phase that has been running and start the next. Returns its ms."""
        now = self._clock()
        if self._t0 is None:              # mark() without start(): begin here rather than crash
            self._t0 = now
            return 0.0
        ms = (now - self._t0) * 1000.0
        self._t0 = now
        self._record(name, ms)
        return ms

    def end(self, name: str = "whole tick") -> float:
        """Record the whole tick. Does not disturb the phase clock."""
        if self._tick_t0 is None:
            return 0.0
        ms = (self._clock() - self._tick_t0) * 1000.0
        self._record(name, ms)
        self.ticks += 1
        return ms

    def _record(self, name: str, ms: float) -> None:
        q = self._samples.get(name)
        if q is None:
            q = self._samples[name] = deque(maxlen=self.window)
            self._order.append(name)
        q.append(float(ms))

    # ---- reading it back ---------------------------------------------------------------
    def summary(self, name: str) -> Optional[dict]:
        q = self._samples.get(name)
        if not q:
            return None
        values = sorted(q)
        n = len(values)
        # nearest-rank p95, and with fewer than 20 samples that is simply the worst one --
        # which is honest: a p95 over 5 samples is not a p95.
        idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
        return {"avg_ms": round(sum(values) / n, 2),
                "p95_ms": round(values[idx], 2),
                "worst_ms": round(values[-1], 2),
                "n": n}

    def as_dict(self) -> dict:
        out = {}
        for name in self._order:
            s = self.summary(name)
            if s is not None:
                out[name] = s
        return out

    def worst_phase(self) -> Optional[str]:
        """Which phase has the largest average. What to look at first when a tick is slow."""
        best, best_avg = None, -1.0
        for name in self._order:
            if name == "whole tick":
                continue
            s = self.summary(name)
            if s and s["avg_ms"] > best_avg:
                best, best_avg = name, s["avg_ms"]
        return best


# ---------------------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------------------

# Why the path is or is not usable. A small closed set on purpose: these get counted and
# regression-tested, and a free-text reason cannot be.
CLEAR = "clear"
BLOCKED_TRACKED_OBJECT = "blocked_tracked_object"   # something the tracker holds is in the way
BLOCKED_SWEPT_PATH = "blocked_swept_path"           # the van's own body would touch it
BLOCKED_SCRAPE = "blocked_scrape"                   # its body reaches into ours from beside the line
BLOCKED_VRU = "blocked_vru"                         # a person or someone riding
NO_ROUTE = "no_route"                               # nothing to judge against

#: Reasons that exist in the code but are not produced yet. Named here so phases P2 and P3
#: extend one vocabulary instead of inventing a second.
BLOCKED_OCCUPANCY = "blocked_occupancy"             # P2: the free-space map disagrees
UNKNOWN_SPACE = "unknown_space"                     # P2: too much unseen where we must go
ROAD_BOUNDARY = "road_boundary"                     # P2: kerb or road edge
BLOCKED_SIGNAL = "blocked_signal"                   # P3: a traffic light, owned by behaviour today

ALL_REASONS = (CLEAR, BLOCKED_TRACKED_OBJECT, BLOCKED_SWEPT_PATH, BLOCKED_SCRAPE,
               BLOCKED_VRU, NO_ROUTE, BLOCKED_OCCUPANCY, UNKNOWN_SPACE, ROAD_BOUNDARY,
               BLOCKED_SIGNAL)


@dataclass
class PlannerDecision:
    """What the planner concluded about the path this tick, and on what evidence."""

    reason: str = NO_ROUTE
    blocker_id: Optional[int] = None
    blocker_kind: Optional[str] = None
    blocker_distance_m: Optional[float] = None
    blocker_lateral_m: Optional[float] = None
    #: how much was looked at, so a cheap tick and a busy one can be told apart
    objects_considered: int = 0
    objects_in_corridor: int = 0
    route_points_used: int = 0
    #: set when the swept-body check decided it, rather than the centre-line bands
    used_footprint: bool = False
    #: everything the planner weighed, only when WARP_PLAN_DEBUG is on
    candidates: list = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.reason not in (CLEAR, NO_ROUTE)

    def as_dict(self) -> dict:
        out = {"reason": self.reason,
               "blocked": self.blocked,
               "blocker_id": self.blocker_id,
               "blocker_kind": self.blocker_kind,
               "blocker_distance_m": (None if self.blocker_distance_m is None
                                      else round(self.blocker_distance_m, 1)),
               "blocker_lateral_m": (None if self.blocker_lateral_m is None
                                     else round(self.blocker_lateral_m, 2)),
               "objects_considered": self.objects_considered,
               "objects_in_corridor": self.objects_in_corridor,
               "route_points_used": self.route_points_used,
               "used_footprint": self.used_footprint}
        if self.candidates:
            out["candidates"] = self.candidates
        return out

    def one_line(self) -> str:
        """For a log or a console, when a table is too much."""
        if not self.blocked:
            return f"reason={self.reason}"
        where = "" if self.blocker_distance_m is None else f" at {self.blocker_distance_m:.1f} m"
        who = "" if self.blocker_id is None else f" blocker={self.blocker_id}"
        kind = "" if self.blocker_kind is None else f" ({self.blocker_kind})"
        return f"reason={self.reason}{who}{kind}{where}"
