"""Why the van's behaviour changed, from a fixed short list (Planning V2, P3).

The behaviour layer has always said what it is doing and written a sentence beside it. A
sentence cannot be counted and cannot be tested for, which is why the planner was given a
closed set of reason codes first (planning/instrumentation.py, P0). This is the same idea for
the state machine: every change of state carries one code from ALL_WHY, the sentence stays
beside it, and the changes are kept in order, so a whole drive reads as a list -- what it was
doing, what it changed to, why, and when.

Nothing here decides anything. It records.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# --- why the state changed. A small closed set on purpose: these get counted and tested. ---

# stopped, and not because of the road
SAFETY_HOLD = "safety_hold"                 # the safety supervisor said stop
LOCALIZATION_LOST = "localization_lost"     # the van does not know where it is
PERCEPTION_LOST = "perception_lost"         # it cannot see
NO_MISSION = "no_mission"                   # nowhere to go

# the road ahead
VRU_IN_PATH = "vru_in_path"                 # a person or someone riding
VEHICLE_IN_PATH = "vehicle_in_path"
OBSTACLE_IN_PATH = "obstacle_in_path"
ROUTE_BLOCKED_TOO_LONG = "route_blocked_too_long"
CONFIRMING_CLEAR = "confirming_clear"       # it blinked out of sight: wait before moving
FOLLOWING_LEAD = "following_lead"           # a moving vehicle ahead, kept at a time gap
OBJECT_AHEAD_SLOW = "object_ahead_slow"     # something in sight, not in the way: slow down
ROUTE_CLEAR = "route_clear"

# what is about to be in the way
PREDICTED_CROSSER_STOP = "predicted_crosser_stop"
PREDICTED_CROSSER_SLOW = "predicted_crosser_slow"

# traffic lights
LIGHT_ROLL_UP = "light_roll_up"             # not green: rolling up to the stop line
LIGHT_HOLD = "light_hold"                   # holding at the line, waiting for green

# junctions without lights
JUNCTION_ROLL_UP = "junction_roll_up"
JUNCTION_PAUSE = "junction_pause"           # stopped at the line to look
JUNCTION_GIVE_WAY = "junction_give_way"     # something is coming
JUNCTION_TIMEOUT = "junction_timeout"       # waited long enough: creeping out

# the end of the mission
DESTINATION_NEAR = "destination_near"
PARKING_PULL_IN = "parking_pull_in"
PARKED = "parked"
PARKED_OVERSHOT = "parked_overshot"         # went past the spot and stopped rather than hunt

ALL_WHY = (
    SAFETY_HOLD, LOCALIZATION_LOST, PERCEPTION_LOST, NO_MISSION,
    VRU_IN_PATH, VEHICLE_IN_PATH, OBSTACLE_IN_PATH, ROUTE_BLOCKED_TOO_LONG, CONFIRMING_CLEAR,
    FOLLOWING_LEAD, OBJECT_AHEAD_SLOW, ROUTE_CLEAR,
    PREDICTED_CROSSER_STOP, PREDICTED_CROSSER_SLOW,
    LIGHT_ROLL_UP, LIGHT_HOLD,
    JUNCTION_ROLL_UP, JUNCTION_PAUSE, JUNCTION_GIVE_WAY, JUNCTION_TIMEOUT,
    DESTINATION_NEAR, PARKING_PULL_IN, PARKED, PARKED_OVERSHOT,
)


@dataclass(frozen=True)
class Transition:
    """One change of what the van is doing."""
    t: float
    was: str                # the state it left
    now: str                # the state it moved to
    why: str                # one of ALL_WHY
    said: str               # the sentence the behaviour wrote at the time
    speed_mps: float = 0.0  # what it asked for
    stopping: bool = False
    rule: str = ""          # which rule answered (BehaviorSystem.RULES)
    rank: int = 0           # and where it sits in that order: 1 is asked first

    def as_dict(self) -> dict:
        return {"t": round(self.t, 2), "was": self.was, "now": self.now, "why": self.why,
                "said": self.said, "speed_mps": round(self.speed_mps, 2),
                "stopping": self.stopping, "rule": self.rule, "rank": self.rank}

    def __str__(self) -> str:
        place = f" [rule {self.rank} {self.rule}]" if self.rule else ""
        return f"{self.was} -> {self.now} ({self.why}){place}: {self.said}"


@dataclass
class TransitionLog:
    """The changes, in order, and how many times each reason has come up.

    A change is a change of STATE or of REASON: holding at a red light and then holding for a
    car that pulled across are the same state and not the same thing. Ticks that repeat both
    are not changes and are not recorded -- 10 a second of them would say nothing.
    """
    keep: int = 400
    changes: deque = field(default_factory=lambda: deque(maxlen=400))
    counts: dict = field(default_factory=dict)
    total: int = 0                          # every change ever, not just the ones still kept

    def note(self, was: str, now: str, why: str, said: str, speed_mps: float = 0.0,
             stopping: bool = False, t: Optional[float] = None, rule: str = "",
             rank: int = 0) -> Optional[Transition]:
        if why not in ALL_WHY:
            raise ValueError(f"unknown reason for a change of behaviour: {why!r}")
        if self.changes and self.changes[-1].now == now and self.changes[-1].why == why:
            return None
        change = Transition(t=t if t is not None else time.time(), was=was, now=now, why=why,
                            said=said, speed_mps=float(speed_mps), stopping=bool(stopping),
                            rule=rule, rank=int(rank))
        self.changes.append(change)
        self.counts[why] = self.counts.get(why, 0) + 1
        self.total += 1
        return change

    @property
    def last(self) -> Optional[Transition]:
        return self.changes[-1] if self.changes else None

    def recent(self, n: int = 10):
        return list(self.changes)[-n:]

    def as_dict(self) -> dict:
        return {"total": self.total, "counts": dict(self.counts),
                "recent": [c.as_dict() for c in self.recent(10)]}
