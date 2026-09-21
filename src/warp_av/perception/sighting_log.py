"""
Evidence: the sightings the tracker is FED, one JSON line per tracker update.

The run records hold what the tracker reported -- a filter position per track -- not the blobs
it was handed, so an association decision could only be inferred afterwards, never replayed
(static-object track-hop phase, 2026-09-21: a parked car's track taking the next car's blob
had to be reconstructed by inverting the motion filter). With this on, feeding the logged
sightings to a fresh ObjectTracker reproduces the live tracks exactly, offline.

Off unless the file ``logs/track_obs.on`` exists when perception starts; then each stack
process writes ``logs/track_obs/obs_<start time>.jsonl``. Nothing in the van reads it back,
and nothing about what the tracker receives changes: the one thing wrapped is the lazy
road-gap lookup, whose answer is passed through untouched and noted beside its sighting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

FLAG_NAME = "track_obs.on"
KEPT = ("wx", "wy", "cls", "cls_source", "confidence", "weak", "distance", "length_m", "width_m", "height_m",
        "yaw_deg", "yaw_world_deg", "box_wx", "box_wy", "box_len", "box_wid", "box_yaw_world_deg")


def _r(v):
    return round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class SightingLog:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"obs_{int(time.time())}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")

    @classmethod
    def open_if_enabled(cls, logs_dir) -> Optional["SightingLog"]:
        """The log, when ``<logs_dir>/track_obs.on`` exists; otherwise None (the normal case)."""
        try:
            logs_dir = Path(logs_dir)
            if not (logs_dir / FLAG_NAME).exists():
                return None
            return cls(logs_dir / "track_obs")
        except Exception as e:                      # evidence must never stop perception
            print(f"[SightingLog] not started: {e}")
            return None

    @staticmethod
    def watch(observation: dict) -> None:
        """Note the road gap beside its sighting IF the tracker asks for it (it asks once per
        track, lazily -- asking here instead would cost a map lookup per blob per frame)."""
        fn = observation.get("road_gap_fn")
        if fn is None:
            return

        def asked(fn=fn, o=observation):
            o["_gap"] = fn()
            return o["_gap"]
        observation["road_gap_fn"] = asked

    def write(self, t: float, pose, observations, tracks) -> None:
        try:
            # the sightings and the time at full precision: a replay must land on the same floats
            line = {"t": float(t), "pose": [_r(v) for v in pose],
                    "obs": [dict({k: o[k] for k in KEPT if k in o},
                                 static_shapes=list(o.get("static_shapes") or ()),
                                 **({"gap": o["_gap"]} if "_gap" in o else {}))
                            for o in observations],
                    "tracks": [[tr.tid, _r(tr.wx), _r(tr.wy), _r(tr.vx), _r(tr.vy), bool(tr.stationary),
                                _r(tr.hits), tr.cls, tr.motion.state] for tr in tracks]}
            self._f.write(json.dumps(line) + "\n")
            self._f.flush()
        except Exception as e:
            print(f"[SightingLog] line dropped: {e}")
