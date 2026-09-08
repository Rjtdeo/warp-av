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
from .camera_lidar_perception import (PERSON_CLASS, CameraDetection, CameraLidarPerception,
                                       VEHICLE_CLASSES)
from .camera_model import CameraModel
from .ground_filter import GroundFilter, flat_cut
from .tracking import VEHICLE_MIN_EXTENT_M, VEHICLE_MIN_HEIGHT_M, cluster_points

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "perception"
MIN_ADVANCE_S = 0.08              # the van's own rule: re-run the tracker when the sweep has moved on
WARMUP_UPDATES = 3                # the tracker needs two sightings; ignore the first updates
GROUND_TAGS = {0, 1, 2, 10, 24, 25}   # unlabeled, road, sidewalk, terrain, road line, ground
OBJECT_TAGS = (12, 13, 14, 15, 16, 18, 19, 20, 21)   # people, vehicles, props
MIN_VISIBLE_POINTS = 3
LIDAR_HEIGHT_M = 2.5
VEHICLE_NAMES = {"model3", "tesla", "audi", "mercedes", "sprinter", "cybertruck", "mkz"}
MATCH_SLACK_M = 1.0               # how far outside its real footprint a report still counts
GRAB_RADIUS_M = 2.0               # labelled points this far from a placed object belong to it
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


class ScriptedCamera:
    """A perfect camera, for testing the fusion instead of the model (day 5).

    It draws a box around every placed person and car exactly where the geometry says
    it must appear, and ignores props, which is what YOLOX does: the COCO classes have
    no barrel, cone or planter. What this measures is whether the van attaches the box
    to the right blob and gives it the right name.
    """

    def __init__(self, targets: List["ObjectScore"], camera: Optional[CameraModel] = None,
                 miss: float = 0.0):
        self.targets = targets
        self.camera = camera or CameraModel()
        self.miss = miss           # share of frames in which the camera sees nothing
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.miss and (self.calls % max(1, round(1.0 / self.miss)) == 0):
            return []
        out = []
        for t in self.targets:
            kind = PERSON_CLASS if t.name.startswith("walker") else (2 if t.name in VEHICLE_NAMES else None)
            if kind is None or t.true_length_m <= 0.0:
                continue
            box = self.box_for(t)
            if box is not None:
                out.append(CameraDetection(class_id=kind, label="person" if kind == PERSON_CLASS else "car",
                                           confidence=0.9, box=box))
        return out

    def box_for(self, t: "ObjectScore") -> Optional[Tuple[float, float, float, float]]:
        """The image box around an object, from the corners of its whole body.

        A real detector draws its box around the object it recognises, not around the
        part the LiDAR happens to hit, so this uses CARLA's own box, which is right for
        people and cars (for static props it is world-aligned and unusable, but the
        scripted camera never draws those).
        """
        cx, cy = t.x, t.y
        a = math.radians(t.body_yaw_deg)
        hl, hw = max(t.body_length_m, t.true_length_m) / 2.0, max(t.body_width_m, t.true_width_m) / 2.0
        us, vs = [], []
        for du in (-hl, hl):
            for dv in (-hw, hw):
                px = cx + du * math.cos(a) - dv * math.sin(a)
                py = cy + du * math.sin(a) + dv * math.cos(a)
                for pz in (-LIDAR_HEIGHT_M, -LIDAR_HEIGHT_M + max(0.3, max(t.body_height_m, t.true_height_m))):
                    uv = self.camera.project(px, py, pz)
                    if uv is None:
                        return None
                    us.append(uv[0]); vs.append(uv[1])
        if not us:
            return None
        x1, x2, y1, y2 = min(us), max(us), min(vs), max(vs)
        if x2 < 0 or y2 < 0 or x1 > self.camera.width or y1 > self.camera.height:
            return None            # outside the picture: the camera cannot see it
        return (x1, y1, x2 - x1, y2 - y1)


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


def oriented_box(xy: np.ndarray) -> Tuple[float, float, float]:
    """Long side, short side and the heading of the long side for a patch of points.
    The same method the pipeline uses on its own clusters, so the two are comparable."""
    if len(xy) < 2:
        return 0.0, 0.0, 0.0
    c = xy.mean(axis=0)
    d = xy - c
    sxx, syy, sxy = float((d[:, 0] ** 2).sum()), float((d[:, 1] ** 2).sum()), float((d[:, 0] * d[:, 1]).sum())
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    along = d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th)
    across = -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)
    length, width = float(along.max() - along.min()), float(across.max() - across.min())
    if width > length:
        length, width = width, length
        th += math.pi / 2
    return length, width, fold_angle(math.degrees(th))


def carla_footprint(o: dict, van: dict) -> Tuple[float, float, float, float]:
    """The box CARLA reported when the object was placed. Trustworthy for vehicles and
    walkers; for static props CARLA gives a world-axis-aligned box, so the same barrel
    measures differently at every heading. Used only as a fallback."""
    ex, ey, ez = (o.get("extent") or [0.5, 0.5, 0.5])[:3]
    length, width = (2.0 * ex, 2.0 * ey) if ex >= ey else (2.0 * ey, 2.0 * ex)
    yaw = float(o.get("yaw_deg", 0.0)) - float(van.get("yaw_deg", 0.0))
    if ey > ex:
        yaw += 90.0
    return length, width, 2.0 * ez, fold_angle(yaw)


def fold_angle(deg: float) -> float:
    """A footprint's long side has no front or back: fold any angle into -90..90."""
    a = (float(deg) + 90.0) % 180.0 - 90.0
    return a


def object_reach(o: dict) -> float:
    ext = o.get("extent", [0.5, 0.5, 0.5])
    return max(1.5, math.hypot(ext[0], ext[1]) + 0.5)


def box_distance(px: float, py: float, cx: float, cy: float, yaw_deg: float,
                 length_m: float, width_m: float) -> float:
    """How far a point lies outside an object's footprint (0 = on or inside it).

    Distance to the centre is not good enough: a 5 m planter's centre is 2.4 m from the bin
    beside it, so a circle around the centre credits the planter with the bin's report."""
    a = math.radians(yaw_deg)
    dx, dy = px - cx, py - cy
    along = abs(dx * math.cos(a) + dy * math.sin(a)) - length_m / 2.0
    across = abs(-dx * math.sin(a) + dy * math.cos(a)) - width_m / 2.0
    return math.hypot(max(0.0, along), max(0.0, across))


# ---------------------------------------------------------------- replay

@dataclass
class ObjectScore:
    name: str
    x: float
    y: float
    distance: float
    reach: float
    size_m: float = 0.5           # half-diagonal of the object's footprint
    true_length_m: float = 0.0    # from objects.json: the real box the recorder placed
    true_width_m: float = 0.0
    true_height_m: float = 0.0
    true_yaw_deg: float = 0.0     # heading of its long side in the van's frame
    body_length_m: float = 0.0    # the whole body from CARLA (right for people and cars)
    body_width_m: float = 0.0
    body_height_m: float = 0.0
    body_yaw_deg: float = 0.0
    box_x: Optional[float] = None  # centre of the outline the labelled scan shows (the visible face)
    box_y: Optional[float] = None
    labelled_points: int = -1     # points the labelled LiDAR (no drop-off) returned from it: -1 = no answer key
    updates: int = 0
    hits: int = 0
    cluster_hits: int = 0
    errors_m: List[float] = field(default_factory=list)
    types: Dict[str, int] = field(default_factory=dict)
    expected_type: str = "obstacle"        # what the van should call this thing
    seen_length_m: List[float] = field(default_factory=list)     # the footprint the van measured
    seen_width_m: List[float] = field(default_factory=list)
    seen_height_m: List[float] = field(default_factory=list)
    seen_yaw_err_deg: List[float] = field(default_factory=list)

    def box_distance(self, px: float, py: float) -> float:
        """How far a point lies outside this object's real footprint (0 = on it)."""
        if self.true_length_m <= 0.0:
            return max(0.0, math.hypot(px - self.x, py - self.y) - 0.5)
        cx = self.x if self.box_x is None else self.box_x
        cy = self.y if self.box_y is None else self.box_y
        return box_distance(px, py, cx, cy, self.true_yaw_deg, self.true_length_m, self.true_width_m)

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
    def named_right(self) -> Optional[float]:
        """Share of the sightings in which the van gave this thing the right name."""
        total = sum(self.types.values())
        return self.types.get(self.expected_type, 0) / total if total else None

    @property
    def median_error_m(self) -> Optional[float]:
        return float(np.median(self.errors_m)) if self.errors_m else None

    def _median(self, values) -> Optional[float]:
        return float(np.median(values)) if values else None

    @property
    def median_length_m(self) -> Optional[float]:
        return self._median(self.seen_length_m)

    @property
    def median_width_m(self) -> Optional[float]:
        return self._median(self.seen_width_m)

    @property
    def median_height_m(self) -> Optional[float]:
        return self._median(self.seen_height_m)

    @property
    def median_yaw_err_deg(self) -> Optional[float]:
        return self._median(self.seen_yaw_err_deg)


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
    merged: int = 0                     # clusters that cover two placed objects at once
    ground: Dict[str, float] = field(default_factory=dict)
    update_ms: List[float] = field(default_factory=list)
    settings: Dict[str, object] = field(default_factory=dict)

    def footprint_rows(self) -> List[dict]:
        rows = []
        for o in self.objects:
            if o.median_length_m is None:
                continue
            rows.append({"object": o.name, "distance_m": round(o.distance, 1),
                         "true_lwh": [round(o.true_length_m, 2), round(o.true_width_m, 2), round(o.true_height_m, 2)],
                         "seen_lwh": [round(o.median_length_m, 2), round(o.median_width_m, 2), round(o.median_height_m, 2)],
                         "yaw_err_deg": round(o.median_yaw_err_deg, 1)})
        return rows

    def as_rows(self) -> List[dict]:
        return [{"fixture": self.fixture, "object": o.name, "distance_m": round(o.distance, 1), "size_m": round(o.size_m, 2),
                 "labelled_points": o.labelled_points, "visible": o.visible,
                 "recall": round(o.recall, 2), "cluster_recall": round(o.cluster_recall, 2),
                 "median_error_m": None if o.median_error_m is None else round(o.median_error_m, 2),
                 "types": dict(o.types)} for o in self.objects]


def measure_targets(targets: List["ObjectScore"], lab: Optional[np.ndarray]) -> None:
    """Replace each object's size with what the dense labelled LiDAR actually shows of it.

    CARLA's own box is unusable for props (world-aligned, so it changes with heading), and in
    any case the fair question is how much of the *visible* object the pipeline recovered.
    Points are gathered around the placed position, kept if they carry an object tag, and
    measured with the same oriented box the pipeline computes."""
    if lab is None or not len(targets):
        return
    tags = lab[:, 5].astype(np.int64)
    obj = lab[np.isin(tags, list(OBJECT_TAGS))]
    road_z = float(np.median(lab[tags == 1, 2])) if (tags == 1).any() else float(np.median(lab[:, 2]))
    if not len(obj):
        return
    # the labelled object points are grouped into connected patches, and each placed object
    # takes the patch nearest to it: a radius grab would pull in the bin 2.4 m away, or a
    # piece of street furniture standing beside the barrel
    patches = cluster_points([(p[0], p[1]) for p in obj], cell=0.5, min_points=1,
                             max_range=80.0, max_clusters=4000, return_members=True)
    # each patch belongs to the object it sits closest to, and to no other: otherwise the
    # planter, which returns 5 points of its own, claims the bin's 19-point patch nearby
    claim: Dict[int, List[Tuple[float, int]]] = {}
    for pi, c in enumerate(patches):
        cand = []
        for i, t in enumerate(targets):
            # a big object's visible face sits well off its centre (a car's rear is 2.4 m from
            # the middle), so how far to look depends on how big the thing is
            grab = max(GRAB_RADIUS_M, 0.6 * math.hypot(t.true_length_m, t.true_width_m))
            d = math.hypot(c["x"] - t.x, c["y"] - t.y)
            if d <= grab:
                cand.append((d, i))
        if cand:
            _, i = min(cand)
            claim.setdefault(i, []).append((len(c["members"]), pi))
    for i, tgt in enumerate(targets):
        mine = claim.get(i, [])
        tgt.labelled_points = max((n for n, _ in mine), default=0)
        if tgt.labelled_points >= MIN_VISIBLE_POINTS:
            pi = max(mine)[1]                      # the fullest patch that chose this object
            pts = obj[np.asarray(patches[pi]["members"], dtype=int)]
            tgt.true_length_m, tgt.true_width_m, tgt.true_yaw_deg = oriented_box(pts[:, :2])
            tgt.true_height_m = float(pts[:, 2].max() - road_z)
            tgt.box_x, tgt.box_y = float(pts[:, 0].mean()), float(pts[:, 1].mean())


def lab_sweep(fx: Fixture) -> Optional[np.ndarray]:
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


def ground_metrics(fx: Fixture, ground_mode: str = "patches", sweep: Optional[np.ndarray] = None,
                   keep_above_m: Optional[float] = None) -> Dict[str, float]:
    """Road removal scored on the labelled sweep, like tools/ground_filter_score.py."""
    if sweep is None:
        sweep = lab_sweep(fx)
    if sweep is None:
        return {}
    pts = sweep[:, :4]
    tags = sweep[:, 5].astype(np.int64)
    gf = GroundFilter()
    if keep_above_m:
        gf.keep_above_m = float(keep_above_m)
    keep = gf.apply(pts).keep if ground_mode == "patches" else flat_cut(pts).keep
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


def assign(targets: List["ObjectScore"], xy: List[Tuple[float, float]],
           slack_m: float = MATCH_SLACK_M) -> Dict[int, Tuple[int, float]]:
    """Give each placed object at most one of the pipeline's outputs, and each output to at
    most one object.

    A report belongs to the object whose real footprint it sits closest to, and must land
    within `slack_m` of it. Both halves matter: distance to an object's centre credits a 5 m
    planter with the bin standing beside it, and without the "closest object wins" rule one
    report is counted for two objects at once. The value kept is the report's distance to the
    object's centre, so the error column stays comparable with day 3."""
    # step 1: every report belongs to the object whose footprint it sits closest to
    candidates: Dict[int, List[Tuple[float, int]]] = {}
    for k, (px, py) in enumerate(xy):
        dists = [(t.box_distance(px, py), i) for i, t in enumerate(targets)]
        if not dists:
            break
        bd, i = min(dists)
        if bd <= slack_m:
            candidates.setdefault(i, []).append((bd, k))
    # step 2: each object keeps the nearest report that chose it
    out: Dict[int, Tuple[int, float]] = {}
    for i, cands in candidates.items():
        _, k = min(cands)
        out[i] = (k, math.hypot(xy[k][0] - targets[i].x, xy[k][1] - targets[i].y))
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
           use_camera_frames: bool = True, cluster_cell_m: Optional[float] = None,
           keep_above_m: Optional[float] = None, far_range_m: Optional[float] = None,
           scripted_camera: bool = False, camera_miss: float = 0.0,
           shape_naming_outside_camera_only: Optional[bool] = None) -> ReplayResult:
    van = fx.meta["van"]
    adapter = ReplayAdapter(van)
    perc = CameraLidarPerception(adapter, detector=detector or StubDetector())
    if shape_naming_outside_camera_only is not None:
        # False = the old behaviour: shape names a vehicle anywhere, camera or not
        if not shape_naming_outside_camera_only:
            perc.strict_vehicle_extent_m = VEHICLE_MIN_EXTENT_M
            perc.strict_vehicle_points = 0
            perc.strict_vehicle_height_m = VEHICLE_MIN_HEIGHT_M
    perc.yolox_inline = True                 # the stand-in runs inline, every update
    perc.inference_interval = 0.0
    perc.ground_filter_mode = ground_mode
    perc.thin_step = thin
    if cluster_cell_m:
        perc.cluster_cell_m = float(cluster_cell_m)
    if keep_above_m:
        perc.ground_filter.keep_above_m = float(keep_above_m)
    if far_range_m:
        perc.far_range_m = float(far_range_m)

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

    # the labelled scan says what is really there: it sizes each placed object, and tells
    # solid things (not ground) from sidewalk (kerb tops)
    lab = lab_sweep(fx)
    targets = []
    for o in fx.objects:
        ox, oy = ego_frame(van, o["x"], o["y"])
        ext = o.get("extent", [0.5, 0.5, 0.5])
        label = object_label(o["blueprint"])
        if any(t.name == label for t in targets):
            label = f"{label}_{sum(t.name.startswith(label) for t in targets) + 1}"
        tl, tw, th, tyaw = carla_footprint(o, van)
        expected = ("pedestrian" if label.startswith("walker")
                    else "vehicle" if label in VEHICLE_NAMES else "obstacle")
        targets.append(ObjectScore(label, ox, oy, math.hypot(ox, oy), object_reach(o),
                                   size_m=max(0.5, math.hypot(ext[0], ext[1])),
                                   true_length_m=tl, true_width_m=tw, true_height_m=th, true_yaw_deg=tyaw,
                                   body_length_m=tl, body_width_m=tw, body_height_m=th, body_yaw_deg=tyaw,
                                   expected_type=expected))
    measure_targets(targets, lab)
    if scripted_camera and detector is None:
        perc.detector = ScriptedCamera(targets, perc.camera_model, miss=camera_miss)

    solid_xy = walk_xy = None
    if lab is not None:
        tags = lab[:, 5].astype(np.int64)
        solid_xy = lab[~np.isin(tags, list(GROUND_TAGS))][:, :2]
        walk_xy = lab[tags == 2][:, :2]

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
        # two placed objects whose nearest blob is the same blob: the cell glued them together
        owners: Dict[int, int] = {}
        for t in targets:
            if not t.visible or not clusters:
                continue
            k = min(range(len(clusters)), key=lambda k: t.box_distance(clusters[k]["x"], clusters[k]["y"]))
            if t.box_distance(clusters[k]["x"], clusters[k]["y"]) <= MATCH_SLACK_M:
                owners[k] = owners.get(k, 0) + 1
        result.merged += sum(1 for v in owners.values() if v >= 2)
        for i, tgt in enumerate(targets):
            tgt.updates += 1
            if i in by_report:
                k, d = by_report[i]
                ob = out.objects[k]
                tgt.hits += 1
                tgt.errors_m.append(d)
                tgt.types[ob.object_type.value] = tgt.types.get(ob.object_type.value, 0) + 1
                if getattr(ob, "length_m", 0.0):
                    tgt.seen_length_m.append(float(ob.length_m))
                    tgt.seen_width_m.append(float(ob.width_m))
                    tgt.seen_height_m.append(float(ob.height_m))
                    tgt.seen_yaw_err_deg.append(abs(fold_angle(float(ob.yaw_deg) - tgt.true_yaw_deg)))
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
    result.ground = ground_metrics(fx, ground_mode, sweep=lab, keep_above_m=keep_above_m)
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
                     f"merged blobs {r.merged / scored:.1f}; "
                     f"road deleted {100 * g.get('road_deleted', 0):.1f} %, object points kept {100 * g.get('object_kept', 0):.1f} %"
                     + "".join(f", {int(lo)}-{int(hi)} m {100 * g[f'object_kept_{int(lo)}_{int(hi)}']:.0f} %"
                               for lo, hi in BANDS if f'object_kept_{int(lo)}_{int(hi)}' in g))
        lines.append(f"   {'object':14s} {'dist':>5s} {'size':>5s} {'lidar pts':>9s} {'recall':>7s} {'cluster':>8s} {'err m':>6s}  named as")
        for o in sorted(r.objects, key=lambda o: o.distance):
            err = "-" if o.median_error_m is None else f"{o.median_error_m:.2f}"
            pts = "?" if o.labelled_points < 0 else str(o.labelled_points)
            nm = o.named_right
            names = f"{o.types} -> {'?' if nm is None else format(nm, '.2f')} right ({o.expected_type})"
            note = "" if o.visible else "   (hidden from the sensor: not scored)"
            lines.append(f"   {o.name:14s} {o.distance:5.1f} {o.size_m:5.1f} {pts:>9s} {o.recall:7.2f} {o.cluster_recall:8.2f} {err:>6s}  {names}{note}")
        rows = r.footprint_rows()
        if rows:
            lines.append(f"   footprint (long x short x tall, metres){'':6s} true{'':18s} measured      heading err")
            for row in rows:
                t, m = row["true_lwh"], row["seen_lwh"]
                lines.append(f"   {row['object']:14s} {row['distance_m']:5.1f}       "
                             f"{t[0]:5.2f} x {t[1]:4.2f} x {t[2]:4.2f}      "
                             f"{m[0]:5.2f} x {m[1]:4.2f} x {m[2]:4.2f}      {row['yaw_err_deg']:5.1f} deg")
    return "\n".join(lines)


def all_fixtures() -> List[Path]:
    if not FIXTURE_ROOT.is_dir():
        return []
    return sorted(p for p in FIXTURE_ROOT.iterdir() if (p / "deliveries.npz").exists())
