"""
Evidence log of the tracker's sightings (static-object track-hop phase, 2026-09-21).

The run records hold what the tracker REPORTED, not what it was fed, so an association could never be replayed.
With logs/track_obs.on present the perception writes every sighting; fed back to a fresh ObjectTracker they must
reproduce the live tracks exactly. Off by default, and it may not change what the tracker receives.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.sighting_log import FLAG_NAME, SightingLog  # noqa: E402
from warp_av.perception.tracking import ObjectTracker  # noqa: E402


def frames(n=30):
    """A parked thing with a wobbling middle, a car driving past at 6 m/s, and a far pole that is asked for its road gap."""
    out = []
    for k in range(n):
        t = 0.3 * (k + 1)
        w = 0.2 * math.sin(k * 2.1)
        obs = [{"wx": 12.0 + w, "wy": -3.0, "cls": "vehicle", "cls_source": "shape", "confidence": 0.45, "weak": False,
                "distance": 12.4, "length_m": 4.4, "width_m": 1.8, "height_m": 1.4, "yaw_deg": 2.0,
                "static_shapes": (), "road_gap_fn": None},
               {"wx": 40.0 - 6.0 * t, "wy": 3.5, "cls": "vehicle", "cls_source": "camera", "confidence": 0.8, "weak": False,
                "distance": abs(40.0 - 6.0 * t), "length_m": 4.2, "width_m": 1.8, "height_m": 1.5, "yaw_deg": 0.0,
                "static_shapes": (), "road_gap_fn": None},
               {"wx": 25.0, "wy": 9.0 + 0.5 * w, "cls": None, "cls_source": None, "confidence": 0.5, "weak": False,
                "distance": 26.6, "length_m": 0.3, "width_m": 0.3, "height_m": 3.4, "yaw_deg": 0.0,
                "static_shapes": ("pole",), "road_gap_fn": (lambda: 6.5)}]
        out.append((t, obs))
    return out


def test_it_is_off_unless_the_flag_file_is_there(tmp_path):
    assert SightingLog.open_if_enabled(tmp_path) is None
    (tmp_path / FLAG_NAME).write_text("on", encoding="utf-8")
    log = SightingLog.open_if_enabled(tmp_path)
    assert log is not None and log.path.parent == tmp_path / "track_obs"


def test_the_gap_is_passed_through_untouched_and_noted_only_when_asked():
    asked = {"n": 0}

    def gap():
        asked["n"] += 1
        return 4.25
    o = {"road_gap_fn": gap}
    SightingLog.watch(o)
    assert asked["n"] == 0 and "_gap" not in o            # nothing looked up on its own
    assert o["road_gap_fn"]() == 4.25 and asked["n"] == 1 and o["_gap"] == 4.25
    none = {"road_gap_fn": None}
    SightingLog.watch(none)
    assert none["road_gap_fn"] is None


def test_the_logged_sightings_replay_to_exactly_the_live_tracks(tmp_path):
    (tmp_path / FLAG_NAME).write_text("on", encoding="utf-8")
    log = SightingLog.open_if_enabled(tmp_path)
    live = ObjectTracker()
    plain = ObjectTracker()                                   # the same frames with no log at all
    for t, obs in frames():
        bare = [dict(o) for o in obs]
        for o in obs:
            SightingLog.watch(o)
        live.update(obs, t)
        log.write(t, (0.0, 0.0, 0.0, 5.0), obs, live._tracks)
        plain.update(bare, t)
    # logging changed nothing the tracker did
    assert [(tr.tid, tr.wx, tr.wy, tr.stationary, tr.motion.state) for tr in live._tracks] == \
           [(tr.tid, tr.wx, tr.wy, tr.stationary, tr.motion.state) for tr in plain._tracks]
    lines = [json.loads(l) for l in open(log.path, encoding="utf-8")]
    assert len(lines) == 30 and any("gap" in o for ln in lines for o in ln["obs"])
    again = ObjectTracker()
    for ln in lines:
        obs = [dict(o, static_shapes=tuple(o["static_shapes"]),
                    road_gap_fn=(lambda v=o.get("gap"): v) if "gap" in o else None) for o in ln["obs"]]
        again.update(obs, ln["t"])
        got = [[tr.tid, round(tr.wx, 4), round(tr.wy, 4), round(tr.vx, 4), round(tr.vy, 4), bool(tr.stationary),
                round(tr.hits, 4), tr.cls, tr.motion.state] for tr in again._tracks]
        assert got == ln["tracks"], (ln["t"], got, ln["tracks"])       # the same ids, places, speeds and flags, frame by frame


def test_the_hook_sits_around_the_tracker_update_and_does_nothing_when_off():
    src = (Path(__file__).resolve().parents[1] / "src/warp_av/perception/camera_lidar_perception.py").read_text(encoding="utf-8")
    i = src.index("tracks = self.tracker.update(observations, now)")
    around = src[i - 260:i + 420]
    assert around.count("if self._sighting_log is not None:") == 2
    assert "self._sighting_log.watch(o)" in around and "self._sighting_log.write(now" in around
