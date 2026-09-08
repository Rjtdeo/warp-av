"""
Offline replay and scoring of the perception pipeline (Perception V2, day 3).

A fixture (tests/fixtures/perception/<name>/, recorded with
tools/record_fixture.py) holds a second of the van's real LiDAR deliveries,
a few camera frames, the labelled LiDAR as the answer key and the true pose
of every object that was placed. This module:

  1. replays the deliveries through the REAL sweep accumulator and the REAL
     CameraLidarPerception.update() (with a stand-in detector, since the
     camera model cannot run here), at the loop's 10 Hz;
  2. scores what came out against the answer key:
       * recall per placed object: share of updates in which an object was
         reported within reach of it (after the tracker's warm-up);
       * position error of those reports (to the object's origin);
       * false alarms: reported objects with nothing solid in the labelled
         scan near them (a ground leftover), and how many sit in the lane;
       * ground removal on the labelled sweeps: road deleted, object kept;
       * time per update.

No CARLA, no camera model, no network (OpenCV itself is still needed: the
perception module imports it). Used by tests/test_replay_harness.py and
tools/replay_score.py.

Two things are deliberately different from the van: the stand-in detector is
asked every update (the van asks YOLOX every 0.25 s), and the tracker is fed
the sweep's simulation time instead of the wall clock, so a replay is
repeatable and its drop-outs happen at the van's pace.
"""
from __future__ import annotations

import json
import math
import os
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..adapters.carla_sensor_adapter import CameraFrame, LidarScan
from ..adapters.lidar_sweep import LidarSweepAccumulator, azimuth_coverage_bins
from .camera_lidar_perception import CameraLidarPerception
from .ground_filter import GroundFilter, flat_cut

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "perception"
MIN_ADVANCE_S = 0.08              # the van's own rule: re-run the tracker when the sweep has moved on
WARMUP_UPDATES = 3                # the tracker needs two sightings; ignore the first updates
GROUND_TAGS = {0, 1, 2, 10, 24, 25}   # unlabeled, road, sidewalk, terrain, road line, ground
OBJECT_TAGS = (12, 13, 14, 15, 16, 18, 19, 20, 21)   # people, vehicles, props
MIN_VISIBLE_POINTS = 3
BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 35.0))


# ---------------------------------------------------------------- fixture

@dataclass
class Fixture:
    name: str
    path: Path
    points: np.ndarray            # plain LiDAR, Nx5 (x, y, z, intensity, ring)
    index: np.ndarray             # delivery index per point
    times: np.ndarray             # sim time per delivery
    matrices: np.ndarray          # 4x4 sensor pose per delivery
    labels: Optional[np.ndarray]  # labelled LiDAR, Mx6 (x, y, z, cos, ring, tag)
    label_index: Optional[np.ndarray]
    label_times: Optional[np.ndarray]
    label_matrices: Optional[np.ndarray]
    objects: List[dict]
    meta: dict
    frames: List[Tuple[float, Path]] = field(default_factory=list)


def load_fixture(path) -> Fixture:
    path = Path(path)
    if not path.is_dir():
        path = FIXTURE_ROOT / str(path)
    d = np.load(path / "deliveries.npz")
    meta = json.loads((path / "meta.json").read_text())
    objects = json.loads((path / "objects.json").read_text())
    labels = label_index = label_times = label_matrices = None
    if (path / "labels.npz").exists():
        lab = np.load(path / "labels.npz")
        labels, label_index = lab["points"], lab["delivery_index"]
        label_times, label_matrices = lab["delivery_time"], lab["delivery_matrix"]
    frames = []
    if (path / "frames.json").exists():
        for fr in json.loads((path / "frames.json").read_text()):
            frames.append((float(fr["sim_time"]), path / fr["file"]))
    return Fixture(path.name, path, d["points"], d["delivery_index"], d["delivery_time"], d["delivery_matrix"],
                   labels, label_index, label_times, label_matrices, objects, meta, frames)


# ---------------------------------------------------------------- stand-ins

class StubDetector:
    """The camera model's stand-in: returns scripted detections (none by default)."""

    def __init__(self, detections=None):
        self.detections = list(detections or [])
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.detections)


class ReplayAdapter:
    """What CameraLidarPerception reads: latest_camera, latest_lidar, vehicle."""

    def __init__(self, van: dict, width: int = 800, height: int = 600):
        self.latest_camera = None
        self.latest_lidar = None
        self._van = van
        self.width, self.height = width, height
        loc = types.SimpleNamespace(x=van["x"], y=van["y"], z=van["z"])
        rot = types.SimpleNamespace(yaw=van["yaw_deg"], pitch=0.0, roll=0.0)
        tf = types.SimpleNamespace(location=loc, rotation=rot)
        self.vehicle = types.SimpleNamespace(get_transform=lambda: tf)

    def set_camera(self, image_bgra: Optional[np.ndarray]):
        if image_bgra is None:
            image_bgra = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.latest_camera = CameraFrame(image=image_bgra, width=image_bgra.shape[1], height=image_bgra.shape[0],
                                         fov=90.0, timestamp=time.time())


def _load_frame(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return np.dstack([bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)])
    except Exception:
        return None


def ego_frame(van: dict, wx: float, wy: float) -> Tuple[float, float]:
    """World -> van frame (x forward, y positive to the right, CARLA's convention)."""
    yaw = math.radians(van["yaw_deg"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx, dy = wx - van["x"], wy - van["y"]
    return dx * cy + dy * sy, -dx * sy + dy * cy


def object_label(blueprint: str) -> str:
    parts = blueprint.split(".")
    return "walker" if parts[0] == "walker" else parts[-1]


def object_reach(o: dict) -> float:
    ext = o.get("extent", [0.5, 0.5, 0.5])
    return max(1.5, math.hypot(ext[0], ext[1]) + 0.5)


# ---------------------------------------------------------------- replay

@dataclass
class ObjectScore:
    name: str
    x: float
    y: float
    distance: float
    reach: float
    size_m: float = 0.5           # half-diagonal of the object's footprint
    labelled_points: int = -1     # points the labelled LiDAR (no drop-off) returned from it: -1 = no answer key
    updates: int = 0
    hits: int = 0
    cluster_hits: int = 0
    errors_m: List[float] = field(default_factory=list)
    types: Dict[str, int] = field(default_factory=dict)

    @property
    def visible(self) -> bool:
        """False when even the dense labelled LiDAR barely saw it (hidden behind something): not scored."""
        return self.labelled_points < 0 or self.labelled_points >= MIN_VISIBLE_POINTS

    @property
    def recall(self) -> float:
        return self.hits / self.updates if self.updates else 0.0

    @property
    def cluster_recall(self) -> float:
        return self.cluster_hits / self.updates if self.updates else 0.0

    @property
    def median_error_m(self) -> Optional[float]:
        return float(np.median(self.errors_m)) if self.errors_m else None


@dataclass
class ReplayResult:
    fixture: str
    updates: int
    objects: List[ObjectScore]
    coverage: List[float] = field(default_factory=list)   # sectors of the 360 deg each fed sweep covered
    unhealthy: int = 0                  # updates the pipeline flagged unhealthy (should be 0)
    reports_total: int = 0              # everything the pipeline reported, all scored updates
    solid_reports: int = 0              # reports sitting on solid labelled points (walls, poles, placed things)
    kerb_reports: int = 0               # reports sitting on sidewalk points only (kerb tops)
    phantoms: int = 0                   # reports with nothing but road/terrain/nothing near them: ground leftovers, ghosts
    phantoms_in_lane: int = 0           # ... of which inside the lane corridor ahead, within 20 m
    phantoms_near: int = 0              # ... of which within 20 m anywhere
    ground: Dict[str, float] = field(default_factory=dict)
    update_ms: List[float] = field(default_factory=list)
    settings: Dict[str, object] = field(default_factory=dict)

    def as_rows(self) -> List[dict]:
        return [{"fixture": self.fixture, "object": o.name, "distance_m": round(o.distance, 1), "size_m": round(o.size_m, 2),
                 "labelled_points": o.labelled_points, "visible": o.visible,
                 "recall": round(o.recall, 2), "cluster_recall": round(o.cluster_recall, 2),
                 "median_error_m": None if o.median_error_m is None else round(o.median_error_m, 2),
                 "types": dict(o.types)} for o in self.objects]


def _labelled_sweep(fx: Fixture) -> Optional[np.ndarray]:
    """One whole labelled sweep (the answer key), in the labelled sensor's frame."""
    if fx.labels is None:
        return None
    acc = LidarSweepAccumulator()
    sweep = None
    for i in range(len(fx.label_times)):
        sweep = acc.add(fx.labels[fx.label_index == i], fx.label_matrices[i], float(fx.label_times[i]))
        if acc.span_s >= 0.09:
            break
    return sweep


def ground_metrics(fx: Fixture, ground_mode: str = "patches", sweep: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Road removal scored on the labelled sweep, like tools/ground_filter_score.py."""
    if sweep is None:
        sweep = _labelled_sweep(fx)
    if sweep is None:
        return {}
    pts = sweep[:, :4]
    tags = sweep[:, 5].astype(np.int64)
    keep = GroundFilter().apply(pts).keep if ground_mode == "patches" else flat_cut(pts).keep
    road = np.isin(tags, [1, 24, 25])
    obj = np.isin(tags, list(OBJECT_TAGS))
    dist = np.hypot(pts[:, 0], pts[:, 1])
    out = {"road_deleted": float(1.0 - keep[road].mean()) if road.any() else 1.0,
           "object_kept": float(keep[obj].mean()) if obj.any() else 1.0}
    for lo, hi in BANDS:
        m = obj & (dist >= lo) & (dist < hi)
        if m.any():
            out[f"object_kept_{int(lo)}_{int(hi)}"] = float(keep[m].mean())
    return out


def assign(targets: List["ObjectScore"], xy: List[Tuple[float, float]]) -> Dict[int, Tuple[int, float]]:
    """Give each placed object at most one of the pipeline's outputs, and each output to at
    most one object: smallest reach first, nearest free output. Without this a long planter's
    3 m reach is credited with the bin's report standing 2.4 m away."""
    out: Dict[int, Tuple[int, float]] = {}
    free = set(range(len(xy)))
    for i in sorted(range(len(targets)), key=lambda i: targets[i].reach):
        tgt = targets[i]
        near = [(math.hypot(xy[k][0] - tgt.x, xy[k][1] - tgt.y), k) for k in free]
        near = [n for n in near if n[0] <= tgt.reach]
        if near:
            d, k = min(near)
            out[i] = (k, d)
            free.discard(k)
    return out


def classify_report(x: float, y: float, solid_xy: np.ndarray, walk_xy: Optional[np.ndarray],
                    radius_m: float = 1.5, min_points: int = 3) -> str:
    """'solid' if solid labelled points sit within reach, 'kerb' if only sidewalk points do, else 'phantom'."""
    r2 = radius_m ** 2
    if ((solid_xy[:, 0] - x) ** 2 + (solid_xy[:, 1] - y) ** 2 <= r2).sum() >= min_points:
        return "solid"
    if walk_xy is not None and ((walk_xy[:, 0] - x) ** 2 + (walk_xy[:, 1] - y) ** 2 <= r2).sum() >= min_points:
        return "kerb"
    return "phantom"


def replay(fx: Fixture, ground_mode: str = "patches", thin: int = 1, detector=None,
           use_camera_frames: bool = True) -> ReplayResult:
    van = fx.meta["van"]
    adapter = ReplayAdapter(van)
    perc = CameraLidarPerception(adapter, detector=detector or StubDetector())
    perc.yolox_inline = True                 # the stand-in runs inline, every update
    perc.inference_interval = 0.0
    perc.ground_filter_mode = ground_mode
    perc.thin_step = thin

    # the tracker's clock: the van hands it the wall clock; here consecutive sweeps are
    # milliseconds apart in wall time but 0.1 s apart in simulation time, so feed it the
    # sweep's simulation time (repeatable, drop-outs at the van's pace)
    clock = {"t": 0.0}
    tracker_update = perc.tracker.update
    perc.tracker.update = lambda observations, now: tracker_update(observations, clock["t"])

    frames = sorted(fx.frames)
    frame_imgs = {}

    def camera_for(t: float):
        if not use_camera_frames or not frames:
            return None
        ts, path = min(frames, key=lambda f: abs(f[0] - t))
        if path not in frame_imgs:
            frame_imgs[path] = _load_frame(path)
        return frame_imgs[path]

    targets = []
    for o in fx.objects:
        ox, oy = ego_frame(van, o["x"], o["y"])
        ext = o.get("extent", [0.5, 0.5, 0.5])
        label = object_label(o["blueprint"])
        if any(t.name == label for t in targets):
            label = f"{label}_{sum(t.name.startswith(label) for t in targets) + 1}"
        targets.append(ObjectScore(label, ox, oy, math.hypot(ox, oy), object_reach(o),
                                   size_m=max(0.5, math.hypot(ext[0], ext[1]))))

    # the labelled scan says what is really there: solid things (not ground), and sidewalk (kerb tops)
    lab = _labelled_sweep(fx)
    solid_xy = walk_xy = None
    if lab is not None:
        tags = lab[:, 5].astype(np.int64)
        solid_xy = lab[~np.isin(tags, list(GROUND_TAGS))][:, :2]
        walk_xy = lab[tags == 2][:, :2]
        obj_xy = lab[np.isin(tags, list(OBJECT_TAGS))][:, :2]
        for tgt in targets:      # how much of each placed thing the sensor could see at all
            d2 = (obj_xy[:, 0] - tgt.x) ** 2 + (obj_xy[:, 1] - tgt.y) ** 2
            tgt.labelled_points = int((d2 <= tgt.reach ** 2).sum())

    acc = LidarSweepAccumulator()
    result = ReplayResult(fx.name, 0, targets, settings={"ground": ground_mode, "thin": thin})
    last_pub = None
    n_updates = 0
    for i in range(len(fx.times)):
        t = float(fx.times[i])
        sweep = acc.add(fx.points[fx.index == i], fx.matrices[i], t)
        # the van hands the pipeline whatever the accumulator holds after every delivery and lets
        # the pipeline's own 0.08 s rule decide when to look again: do exactly that
        if sweep is None or len(sweep) == 0:
            continue
        if last_pub is not None and t - last_pub < MIN_ADVANCE_S - 1e-6:
            continue
        last_pub = t
        clock["t"] = t
        result.coverage.append(float(azimuth_coverage_bins(sweep)))
        adapter.latest_lidar = LidarScan(points=sweep, timestamp=time.time(), frames=acc.frames_in_sweep,
                                         span_s=acc.span_s, sim_time=t, sensor_matrix=np.asarray(fx.matrices[i]))
        adapter.set_camera(camera_for(t))
        t0 = time.perf_counter()
        out = perc.update()
        result.update_ms.append((time.perf_counter() - t0) * 1000.0)
        n_updates += 1
        if not out.healthy:
            result.unhealthy += 1
            continue
        if n_updates <= WARMUP_UPDATES:
            continue
        clusters = getattr(perc, "last_clusters", []) or []
        result.reports_total += len(out.objects)
        by_report = assign(targets, [(ob.x, ob.y) for ob in out.objects])
        by_cluster = assign(targets, [(c["x"], c["y"]) for c in clusters])
        for i, tgt in enumerate(targets):
            tgt.updates += 1
            if i in by_report:
                k, d = by_report[i]
                ob = out.objects[k]
                tgt.hits += 1
                tgt.errors_m.append(d)
                tgt.types[ob.object_type.value] = tgt.types.get(ob.object_type.value, 0) + 1
            if i in by_cluster:
                tgt.cluster_hits += 1
        matched = {k for k, _ in by_report.values()}
        # every other report: is there something solid where it points? a kerb? or nothing but ground?
        for k, ob in enumerate(out.objects):
            if k in matched or solid_xy is None:
                continue
            kind = classify_report(ob.x, ob.y, solid_xy, walk_xy)
            if kind == "solid":
                result.solid_reports += 1
            elif kind == "kerb":
                result.kerb_reports += 1
            else:
                result.phantoms += 1
                if ob.distance < 20.0:
                    result.phantoms_near += 1
                    if ob.x > 0 and abs(ob.y) < 1.75:
                        result.phantoms_in_lane += 1
    result.updates = n_updates
    result.ground = ground_metrics(fx, ground_mode, sweep=lab)
    try:
        perc.close()
    except Exception:
        pass
    return result


def format_report(results: List[ReplayResult]) -> str:
    lines = []
    for r in results:
        g = r.ground
        scored = max(1, r.updates - WARMUP_UPDATES - r.unhealthy)
        bad = f", {r.unhealthy} UNHEALTHY" if r.unhealthy else ""
        cov = f", {np.mean(r.coverage):.0f}/36 sectors" if r.coverage else ""
        lines.append(f"== {r.fixture}: {r.updates} updates{bad}{cov}, {np.mean(r.update_ms):.1f} ms each; per update: "
                     f"{r.reports_total / scored:.1f} reports = placed objects + {r.solid_reports / scored:.1f} solid things "
                     f"+ {r.kerb_reports / scored:.1f} kerb + {r.phantoms / scored:.1f} phantoms "
                     f"({r.phantoms_near / scored:.1f} within 20 m, {r.phantoms_in_lane / scored:.1f} in lane); "
                     f"road deleted {100 * g.get('road_deleted', 0):.1f} %, object points kept {100 * g.get('object_kept', 0):.1f} %"
                     + "".join(f", {int(lo)}-{int(hi)} m {100 * g[f'object_kept_{int(lo)}_{int(hi)}']:.0f} %"
                               for lo, hi in BANDS if f'object_kept_{int(lo)}_{int(hi)}' in g))
        lines.append(f"   {'object':14s} {'dist':>5s} {'size':>5s} {'lidar pts':>9s} {'recall':>7s} {'cluster':>8s} {'err m':>6s}  named as")
        for o in sorted(r.objects, key=lambda o: o.distance):
            err = "-" if o.median_error_m is None else f"{o.median_error_m:.2f}"
            pts = "?" if o.labelled_points < 0 else str(o.labelled_points)
            note = "" if o.visible else "   (hidden from the sensor: not scored)"
            lines.append(f"   {o.name:14s} {o.distance:5.1f} {o.size_m:5.1f} {pts:>9s} {o.recall:7.2f} {o.cluster_recall:8.2f} {err:>6s}  {o.types}{note}")
    return "\n".join(lines)


def all_fixtures() -> List[Path]:
    if not FIXTURE_ROOT.is_dir():
        return []
    return sorted(p for p in FIXTURE_ROOT.iterdir() if (p / "deliveries.npz").exists())
