"""
Camera + LiDAR Perception
=========================

Phase 2 perception for Warp AV.

CAMERA:
    CARLA RGB camera
        -> OpenCV
        -> YOLOX
        -> person / vehicle classification

LIDAR:
    CARLA 32-channel LiDAR
        -> front driving corridor
        -> nearest stable group of points
        -> obstacle distance

FUSION:
    Camera tells us WHAT the object is.
    LiDAR tells us WHERE / HOW FAR it is.

Important:
    This is a simple forward-hazard fusion system for the demo.
    It is NOT full production 3D camera-LiDAR calibration/fusion.

    Ground-truth perception remains available separately as
    the stable fallback.
"""

import dataclasses
import math
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import tracking as _tracking
from .camera_model import (CameraModel, box_contains, box_edges, box_foot, box_overlap,
                           cluster_point, ground_point)
from .detection_worker import DetectionWorker, yolox_inline_from_env
from .tracking import (cluster_points, clearance_radius_m, ObjectTracker, MIN_POINTS_FAR,
                       FAR_RANGE_M, vehicle_shaped)
from .ground_filter import (GroundFilter, flat_cut, ground_filter_mode_from_env,
                            lidar_thin_step_from_env, remove_road_edge_points, DEFAULT_LIDAR_HEIGHT_M)

from .perception import (
    DetectedObject,
    ObjectType,
    PerceptionOutput,
)


# ============================================================
# COCO CLASSES
# ============================================================

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


# COCO IDs useful for an autonomous vehicle demo.
PERSON_CLASS = 0

VEHICLE_CLASSES = {
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
}

# A cyclist is not a class the camera model knows: it reports a person and a bicycle in the
# same place. When a person's box sits on top of one of these, that person is riding.
RIDDEN_CLASSES = {
    1,   # bicycle
    3,   # motorcycle
}
RIDER_OVERLAP = 0.35        # share of the person's box that must sit over the bike


# ============================================================
# CAMERA DETECTION RESULT
# ============================================================


# How close two points must be to belong to the same object. At 1.0 m a barrel and the
# planter beside it became one 4.4 m blob; 0.8 m separates them and cuts the measured-size
# error nearly in half, at about 1 ms per update (Perception V2 day 4).
DEFAULT_CLUSTER_CELL_M = 0.8
# How high above the road a point must be to count as an object. The old 12 cm threw away
# most of a 12.5 cm planter; 8 cm keeps 90 % of object points instead of 85 %, and adds no
# false object inside the lane on any recorded fixture (day 4).
DEFAULT_KEEP_ABOVE_M = 0.08

class CameraDetection:
    def __init__(
        self,
        class_id: int,
        label: str,
        confidence: float,
        box,
    ):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.box = box

    def __repr__(self):
        return (
            f"CameraDetection("
            f"label={self.label}, "
            f"confidence={self.confidence:.2f}, "
            f"box={self.box})"
        )


# ============================================================
# YOLOX USING ONLY OPENCV DNN
# ============================================================

class YoloXDetector:
    """
    Lightweight YOLOX detector using OpenCV DNN.

    No PyTorch.
    No Ultralytics.
    """

    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.50,
    ):
        if model_path is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

            model_path = (
                project_root
                / "models"
                / "yolox_s.onnx"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOX model not found: "
                f"{self.model_path}"
            )

        print(
            f"[CameraLidar] Loading YOLOX: "
            f"{self.model_path}"
        )

        self.net = cv2.dnn.readNetFromONNX(
            str(self.model_path)
        )
        # the model runs in a background thread now: leave two cores for the
        # driving loop and the sensor callbacks
        try:
            cv2.setNumThreads(max(1, (os.cpu_count() or 4) - 2))
        except Exception:
            pass

        # CPU is intentionally used for the first version.
        self.net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_OPENCV
        )

        self.net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CPU
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.nms_threshold = (
            nms_threshold
        )

        self.strides = [8, 16, 32]

        self.grids, self.expanded_strides = (
            self._generate_grids()
        )

        print(
            "[CameraLidar] YOLOX loaded successfully"
        )


    # --------------------------------------------------------
    # YOLOX GRID
    # --------------------------------------------------------

    def _generate_grids(self):

        grids = []
        expanded_strides = []

        for stride in self.strides:

            hsize = (
                self.INPUT_SIZE // stride
            )

            wsize = (
                self.INPUT_SIZE // stride
            )

            xv, yv = np.meshgrid(
                np.arange(wsize),
                np.arange(hsize),
            )

            grid = np.stack(
                (xv, yv),
                axis=2,
            ).reshape(
                1,
                -1,
                2,
            )

            grids.append(grid)

            expanded_strides.append(
                np.full(
                    (
                        1,
                        grid.shape[1],
                        1,
                    ),
                    stride,
                    dtype=np.float32,
                )
            )

        grids = np.concatenate(
            grids,
            axis=1,
        ).astype(np.float32)

        expanded_strides = np.concatenate(
            expanded_strides,
            axis=1,
        ).astype(np.float32)

        return grids, expanded_strides


    # --------------------------------------------------------
    # LETTERBOX
    # --------------------------------------------------------

    def _letterbox(self, image):

        height, width = image.shape[:2]

        scale = min(
            self.INPUT_SIZE / height,
            self.INPUT_SIZE / width,
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        resized = cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        padded = np.full(
            (
                self.INPUT_SIZE,
                self.INPUT_SIZE,
                3,
            ),
            114.0,
            dtype=np.float32,
        )

        # OpenCV Zoo YOLOX uses top-left placement.
        padded[
            :new_height,
            :new_width,
        ] = resized

        return padded, scale


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    def detect(
        self,
        bgr_image: np.ndarray,
    ) -> List[CameraDetection]:

        if bgr_image is None:
            return []

        # YOLOX (Megvii export / OpenCV Zoo yolox_s.onnx) is
        # trained and exported on BGR input with no channel
        # swap and no 0-1 normalization — only letterbox +
        # 114 padding. Converting to RGB here feeds the
        # network color-reversed images, which silently
        # tanks detection quality instead of erroring.
        input_image, scale = (
            self._letterbox(bgr_image)
        )

        # HWC -> CHW -> batch
        blob = np.transpose(
            input_image,
            (2, 0, 1),
        )

        blob = blob[
            np.newaxis,
            :,
            :,
            :
        ]

        self.net.setInput(blob)

        output = self.net.forward()

        detections = (
            self._postprocess(
                output,
                scale,
                bgr_image.shape[1],
                bgr_image.shape[0],
            )
        )

        return detections


    # --------------------------------------------------------
    # YOLOX POSTPROCESS
    # --------------------------------------------------------

    def _postprocess(
        self,
        output,
        scale,
        original_width,
        original_height,
    ):

        predictions = output.copy()

        predictions[..., :2] = (
            predictions[..., :2]
            + self.grids
        ) * self.expanded_strides

        predictions[..., 2:4] = (
            np.exp(
                predictions[..., 2:4]
            )
            * self.expanded_strides
        )

        predictions = predictions[0]

        objectness = predictions[:, 4:5]

        class_scores = predictions[:, 5:]

        scores = (
            objectness
            * class_scores
        )

        class_ids = np.argmax(
            scores,
            axis=1,
        )

        max_scores = np.max(
            scores,
            axis=1,
        )

        candidate_mask = (
            max_scores
            >= self.confidence_threshold
        )

        if not np.any(
            candidate_mask
        ):
            return []

        boxes = predictions[
            candidate_mask,
            :4,
        ].copy()

        scores_filtered = (
            max_scores[
                candidate_mask
            ]
        )

        class_ids_filtered = (
            class_ids[
                candidate_mask
            ]
        )

        # center x/y + width/height
        # -> x/y/width/height
        boxes_xywh = np.zeros_like(
            boxes
        )

        boxes_xywh[:, 0] = (
            boxes[:, 0]
            - boxes[:, 2] / 2
        )

        boxes_xywh[:, 1] = (
            boxes[:, 1]
            - boxes[:, 3] / 2
        )

        boxes_xywh[:, 2] = (
            boxes[:, 2]
        )

        boxes_xywh[:, 3] = (
            boxes[:, 3]
        )

        # Undo letterbox scale.
        boxes_xywh /= scale

        boxes_list = (
            boxes_xywh.tolist()
        )

        # Class-aware NMS.
        indices = (
            cv2.dnn.NMSBoxesBatched(
                boxes_list,
                scores_filtered.tolist(),
                class_ids_filtered.tolist(),
                self.confidence_threshold,
                self.nms_threshold,
            )
        )

        if len(indices) == 0:
            return []

        results = []

        for index in indices:

            index = int(index)

            class_id = int(
                class_ids_filtered[index]
            )

            score = float(
                scores_filtered[index]
            )

            x, y, w, h = (
                boxes_xywh[index]
            )

            # Clamp to real camera frame.
            x = max(
                0,
                min(
                    float(x),
                    original_width - 1,
                ),
            )

            y = max(
                0,
                min(
                    float(y),
                    original_height - 1,
                ),
            )

            w = max(
                0,
                min(
                    float(w),
                    original_width - x,
                ),
            )

            h = max(
                0,
                min(
                    float(h),
                    original_height - y,
                ),
            )

            label = (
                COCO_CLASSES[class_id]
                if (
                    0
                    <= class_id
                    < len(COCO_CLASSES)
                )
                else
                "unknown"
            )

            results.append(
                CameraDetection(
                    class_id=class_id,
                    label=label,
                    confidence=score,
                    box=(
                        x,
                        y,
                        w,
                        h,
                    ),
                )
            )

        return results


# ============================================================
# CAMERA + LIDAR PERCEPTION
# ============================================================

class CameraLidarPerception:
    """
    Simple forward Camera + LiDAR perception.

    Camera:
        classifies pedestrian / vehicle

    LiDAR:
        finds a stable obstacle in the forward corridor

    Fusion:
        if camera sees person/vehicle and LiDAR sees something
        ahead, use camera classification + LiDAR distance.

        if LiDAR sees something but camera does not classify it,
        safely call it a generic obstacle.
    """

    def __init__(
        self,
        sensor_adapter,
        model_path=None,
        detector=None,
    ):

        self.sensor_adapter = (
            sensor_adapter
        )

        self._enabled = True

        # `detector` lets the replay harness and tests run the whole pipeline
        # with a stand-in; the van always builds the real YOLOX model
        self.detector = detector if detector is not None else YoloXDetector(
            model_path=model_path,
        )

        # Same general limits as old perception.
        self.detection_range = 50.0
        self.danger_distance = 8.0   # Troy #4: was 5.0
        self.path_width = 3.5

        # Ignore extremely close points that could be
        # sensor mount / vehicle geometry.
        # Ignore the Sprinter's own body.
        #
        # Diagnostic testing showed stable self-returns at:
        #   ~1.5 m forward / y ±0.95 m
        #   ~2.4-2.5 m forward / center
        #
        # The LiDAR is roof-mounted, so parts of the van body
        # appear in the point cloud. Keep a small ego exclusion
        # zone in front of the sensor.
        self.minimum_lidar_x = 3.0

        # LiDAR is mounted around 2.5 m above ground.
        # Ground is therefore around z=-2.5 in sensor frame.
        # Keep points above most road surface returns.
        self.minimum_lidar_z = -2.15
        self.maximum_lidar_z = 1.5

        # Require multiple points so one stray LiDAR ray
        # does not become an obstacle.
        self.minimum_cluster_points = 3

        # Camera inference is expensive (~100ms on CPU).
        # We do not need to run YOLO 10 times every second
        # for this first demo.
        self.inference_interval = 0.25

        self._last_inference_time = 0.0
        self._cached_camera_detections = []

        self.last_inference_ms = 0.0

        # Perception V2 day 1 (fix 5): the detector runs in its own thread;
        # the loop reads the newest finished result. WARP_YOLO_INLINE=1 puts
        # it back inside the loop (A/B and rollback).
        self.yolox_inline = yolox_inline_from_env()
        self.detection_max_age_s = 1.0
        self.last_detection_age_s = 0.0
        self._worker = None
        self.detector_fail_after = 3          # consecutive detector errors -> perception unhealthy (as before)
        self.detector_stall_s = 5.0           # no first result this long after start -> unhealthy
        # at 10 Hz two ticks can see the same LiDAR rotation; the tracker only
        # runs when the sweep has advanced, otherwise the last output is reused
        self.min_sweep_advance_s = 0.08
        # how close two points must be to count as the same object. 1 m glues a barrel to the
        # planter beside it; the replay harness measures what a smaller cell costs and buys
        self.cluster_cell_m = DEFAULT_CLUSTER_CELL_M
        # beyond this range a two-point blob is allowed to be an object: nearer than it,
        # three points are required
        self.far_range_m = FAR_RANGE_M
        # where the camera sits and how it looks, for putting a LiDAR blob in the picture
        self.camera_model = CameraModel()
        self.last_camera_model = self.camera_model
        self.fusion_margin_px = 12.0
        # how far the box height may differ from what the blob's size and range imply
        self.fusion_size_ratio = 2.5
        # inside the camera's view the camera names things; the boxy-means-vehicle rule
        # is for the sides and the back, where the front camera cannot see
        # inside the camera's view a blob must be plainly car-sized before shape alone names
        # it a vehicle; outside the view the old, looser rule still applies
        self.strict_vehicle_extent_m = 1.6
        self.strict_vehicle_points = 25
        self.strict_vehicle_height_m = 0.7
        self.last_camera_labels = 0
        self.last_camera_detections = 0
        self._last_tracked_sim_time = None
        self._last_output = None

        # ----------------------------------------------------
        # SHORT LIDAR TEMPORAL MEMORY
        #
        # A pedestrian is narrow, so a 32-channel LiDAR may
        # occasionally miss it for one or two scans.
        #
        # If the camera still sees a relevant object, we may
        # reuse the most recent valid LiDAR hazard briefly.
        # ----------------------------------------------------

        self._last_valid_lidar_hazard = None
        self._last_valid_lidar_time = 0.0

        # Maximum age of a reused LiDAR measurement.
        self.lidar_hold_seconds = 0.8

        # v2: multi-object tracking in world frame (ids + speeds).
        self.tracker = ObjectTracker()
        self.last_track_count = 0

        # Perception fix 2: road removal by local patches instead of the flat
        # 35 cm line above. WARP_GROUND_FILTER=flat restores the old line.
        self.ground_filter_mode = ground_filter_mode_from_env()
        # Perception fix 3: keep every point the road filter leaves (the old
        # code kept one in three). WARP_LIDAR_THIN=3 restores the old thinning.
        self.thin_step = lidar_thin_step_from_env()
        self.ground_filter = GroundFilter(lidar_height_m=DEFAULT_LIDAR_HEIGHT_M,
                                          keep_above_m=DEFAULT_KEEP_ABOVE_M)
        self.last_ground_ms = 0.0
        self.last_ground_tiles = 0
        self.last_borrowed_tiles = 0
        self.last_road_edges_dropped = 0
        self.last_points_kept = 0
        self.last_clusters_before_cap = 0

        print(
            "[CameraLidar] Camera + LiDAR "
            "perception ready"
        )


    # --------------------------------------------------------
    # MAIN UPDATE
    # --------------------------------------------------------

    def update(self) -> PerceptionOutput:
        """v2: LiDAR clustering -> camera classification -> world-frame
        tracking. Produces a full object LIST with stable ids and speeds
        (what following, cut-in handling, and the moving/parked distinction
        need), instead of v1's single fused hazard. Traffic-light state is
        overlaid by the stack from the map/signal feed."""
        if not self._enabled:
            return PerceptionOutput(healthy=False, reason="PERCEPTION_DISABLED")
        try:
            now = time.time()
            camera = self.sensor_adapter.latest_camera
            lidar = self.sensor_adapter.latest_lidar
            if camera is None:
                return PerceptionOutput(healthy=False, reason="CAMERA_NO_DATA")
            if lidar is None:
                return PerceptionOutput(healthy=False, reason="LIDAR_NO_DATA")
            if now - camera.timestamp > 2.0:
                return PerceptionOutput(healthy=False,
                                        reason=f"CAMERA_STALE_{now - camera.timestamp:.1f}s")
            if now - lidar.timestamp > 2.0:
                return PerceptionOutput(healthy=False,
                                        reason=f"LIDAR_STALE_{now - lidar.timestamp:.1f}s")

            # ---- camera inference: worker thread (default) or inline (A/B) ----
            if self.yolox_inline:
                monotonic_now = time.monotonic()
                if monotonic_now - self._last_inference_time >= self.inference_interval:
                    t0 = time.perf_counter()
                    self._cached_camera_detections = self.detector.detect(camera.image[:, :, :3])
                    self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
                    self._last_inference_time = monotonic_now
                self.last_detection_age_s = time.monotonic() - self._last_inference_time
            else:
                if self._worker is None:
                    self._worker = DetectionWorker(self.detector.detect,
                                                   lambda: self.sensor_adapter.latest_camera,
                                                   interval_s=self.inference_interval, name="yolox")
                    self._worker.start()
                elif not self._worker.running:
                    self._worker.start()          # a died thread is restarted, never silently missing
                self._cached_camera_detections, self.last_detection_age_s = self._worker.latest(self.detection_max_age_s)
                self.last_inference_ms = self._worker.last_inference_ms
                # a failing or stalled detector stops the van, as an inline error did before
                if self._worker.consecutive_errors >= self.detector_fail_after:
                    return PerceptionOutput(healthy=False,
                                            reason=f"CAMERA_LIDAR_ERROR: detector failed {self._worker.consecutive_errors}x")
                if (self._worker.runs == 0 and self._worker.started_at is not None
                        and time.monotonic() - self._worker.started_at > self.detector_stall_s):
                    return PerceptionOutput(healthy=False, reason="CAMERA_LIDAR_ERROR: detector stalled")
            detections = self._cached_camera_detections
            # only a recent picture may overrule the shape rule: with a stale or missing
            # camera the van falls back to naming a car-sized blob a car
            camera_fresh = self.last_detection_age_s <= self.detection_max_age_s

            # ---- same LiDAR rotation as last tick? reuse the last output ----
            sim_time = getattr(lidar, "sim_time", None)
            if (sim_time is not None and self._last_tracked_sim_time is not None and self._last_output is not None
                    and 0.0 <= sim_time - self._last_tracked_sim_time < self.min_sweep_advance_s):
                return dataclasses.replace(self._last_output, timestamp=now)

            # ---- LiDAR -> 2D clusters (sensor frame: x fwd, y right) ----
            pts = lidar.points
            # road removal (fix 2): local patches, or the old flat line for A/B
            if self.ground_filter_mode == "patches":
                ground = self.ground_filter.apply(pts)
            else:
                ground = flat_cut(pts, lidar_height_m=DEFAULT_LIDAR_HEIGHT_M,
                                  min_above_m=self.minimum_lidar_z + DEFAULT_LIDAR_HEIGHT_M,
                                  ceiling_above_road_m=self.maximum_lidar_z + DEFAULT_LIDAR_HEIGHT_M)
            mask = ground.keep
            self.last_ground_ms = ground.ms
            self.last_ground_tiles = ground.ground_tiles
            self.last_borrowed_tiles = ground.borrowed_tiles
            self.last_points_kept = int(mask.sum())
            step = max(1, int(self.thin_step))
            sel = pts[mask][::step]          # fix 3: every point (step 1); the old code kept one in three
            heights = ground.above[mask][::step]
            # Pass 1 (fix 2): kerbs and road edges. Cluster the LOW points on
            # their own; a long, low blob clear of the lane is a road edge and
            # its POINTS are removed now, before the main clustering, so a
            # kerb strip can never glue itself to a lamp post, a bin or a
            # pedestrian standing on the pavement and drag their centroid.
            sel, heights, edges_dropped = remove_road_edge_points(sel, heights, cluster_points)
            # Pass 2: everything that is left
            xy = sel[:, :2]
            clusters = cluster_points(xy.tolist(), heights=heights.tolist(), cell=self.cluster_cell_m,
                                      min_points_far=MIN_POINTS_FAR, far_range_m=self.far_range_m)
            self.last_clusters_before_cap = _tracking.LAST_CLUSTER_TOTAL
            # a 2-point blob that is low and beside the lane is a kerb crumb, not an object
            clusters = [c for c in clusters
                        if not (c.get("weak") and (c.get("height") is not None and c["height"] < 0.30)
                                and abs(c["y"]) > 1.2)]
            self.last_road_edges_dropped = edges_dropped
            self.last_clusters = clusters     # sensor frame (x fwd, y right): the learned parker's feelers read these

            # ---- classify clusters by projecting them into the image ----
            # The two sensors sit in different places and the camera is tilted down,
            # so this is real geometry, not an angle-to-column guess (day 5).
            cam = self.camera_model.with_frame(camera.width, camera.height, camera.fov)
            self.last_camera_model = cam
            for c in clusters:
                c["cls"] = None
                c["conf"] = 0.0
                c["uv"] = cam.project(*cluster_point(c, DEFAULT_LIDAR_HEIGHT_M))
                c["uv_foot"] = cam.project(*ground_point(c, DEFAULT_LIDAR_HEIGHT_M))
            matched_boxes = 0
            ridden = [d for d in detections if d.class_id in RIDDEN_CLASSES]
            for det in detections:
                if det.class_id == PERSON_CLASS:
                    # a person standing on a bicycle's box is riding it, and a cyclist needs
                    # a vehicle's room while still being a person to give way to
                    cls = "cyclist" if any(box_overlap(det.box, b.box) >= RIDER_OVERLAP
                                           for b in ridden) else "pedestrian"
                elif det.class_id in VEHICLE_CLASSES:
                    cls = "vehicle"
                else:
                    cls = None
                if cls is None:
                    continue
                # Which blob is this box drawn around? Not simply the nearest one inside it:
                # a cone standing in front of a car sits inside the car's box and would steal
                # its name. The bottom of a box is where the thing touches the road, so the
                # blob whose own ground point lands nearest that edge is the right one, and
                # the box's height must suit the blob's range (day 5).
                fu, fv = box_foot(det.box)
                _, by1, _, by2 = box_edges(det.box)
                box_h = max(1.0, by2 - by1)
                best, best_score = None, None
                for c in clusters:
                    if c["uv"] is None or c["cls"] is not None or c["uv_foot"] is None:
                        continue
                    if not box_contains(det.box, c["uv"][0], c["uv"][1], self.fusion_margin_px):
                        continue
                    height_m = c.get("height") or 0.5
                    want_px = cam.focal_px * max(0.3, height_m) / max(1.0, c["distance"])
                    if not (1.0 / self.fusion_size_ratio) <= (want_px / box_h) <= self.fusion_size_ratio:
                        continue          # a thing of that size at that range cannot fill this box
                    score = math.hypot(c["uv_foot"][0] - fu, c["uv_foot"][1] - fv)
                    if best_score is None or score < best_score:
                        best, best_score = c, score
                if best is not None:
                    best["cls"] = cls
                    best["conf"] = float(det.confidence)
                    matched_boxes += 1
            self.last_camera_labels = matched_boxes
            self.last_camera_detections = len(detections)

            # Shape naming, for the sides and the back where the front camera cannot look:
            # a car-sized solid blob out there is a car. Inside the camera's view the
            # camera decides, so a bin is no longer called a vehicle for being boxy (day 5).
            base_points = 12 if self.thin_step <= 1 else 6
            for c in clusters:
                if c["cls"] is not None:
                    continue
                seen_by_camera = (camera_fresh and c.get("uv") is not None
                                  and cam.in_view(*cluster_point(c, DEFAULT_LIDAR_HEIGHT_M)))
                if seen_by_camera:
                    # the camera looked straight at it and did not call it a car, so only a
                    # blob that is unmistakably car-sized may still be named one. This is what
                    # stops a bin, or a bin glued to a planter, from becoming a "vehicle".
                    ok = vehicle_shaped(c, min_points=max(base_points, self.strict_vehicle_points),
                                        min_extent=self.strict_vehicle_extent_m,
                                        min_height=self.strict_vehicle_height_m)
                else:
                    # to the sides and behind, the camera never looked: shape is all we have
                    ok = vehicle_shaped(c, min_points=base_points)
                if ok:
                    c["cls"] = "vehicle"
                    c["conf"] = 0.45 if not seen_by_camera else 0.40

            # ---- ego -> world, then track ----
            tf = self.sensor_adapter.vehicle.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            ex0, ey0 = tf.location.x, tf.location.y
            observations = []
            for c in clusters:
                observations.append({
                    "wx": ex0 + c["x"] * cy - c["y"] * sy,
                    "wy": ey0 + c["x"] * sy + c["y"] * cy,
                    "cls": c["cls"],
                    "confidence": c["conf"],
                    "weak": bool(c.get("weak", False)),   # a 2-point far blob: three sightings before it counts
                    "distance": c["distance"],            # how much to trust this sighting (day 6)
                    "length_m": c.get("length_m", 0.0),
                    "width_m": c.get("width_m", 0.0),
                    "height_m": c.get("height") or 0.0,
                    "yaw_deg": c.get("yaw_deg", 0.0),
                })
            tracks = self.tracker.update(observations, now)

            # ---- tracks -> DetectedObjects (back to ego frame) ----
            objects = []
            for tr in tracks:
                dx, dy = tr.wx - ex0, tr.wy - ey0
                ex = dx * cy + dy * sy
                ey = -dx * sy + dy * cy
                dist = math.hypot(dx, dy)
                if dist > self.detection_range:
                    continue
                otype = (ObjectType.PEDESTRIAN if tr.cls == "pedestrian"
                         else ObjectType.CYCLIST if tr.cls == "cyclist"
                         else ObjectType.VEHICLE if tr.cls == "vehicle"
                         else ObjectType.OBSTACLE)
                objects.append(DetectedObject(
                    object_type=otype, x=ex, y=ey, distance=dist,
                    speed=self.tracker.reported_speed(tr),
                    vx_world=tr.vx, vy_world=tr.vy,
                    # a track built only from 2-point far sightings is reported, but with low confidence
                    confidence=(tr.confidence if tr.cls else 0.65) if not getattr(tr, "weak_only", False) else 0.35,
                    length_m=getattr(tr, "length_m", 0.0), width_m=getattr(tr, "width_m", 0.0),
                    height_m=getattr(tr, "height_m", 0.0), yaw_deg=getattr(tr, "yaw_deg", 0.0),
                    stationary=bool(getattr(tr, "stationary", True)),
                    size_uncertain=bool(getattr(tr, "size_uncertain", False)),
                    clearance_radius_m=clearance_radius_m(tr.cls, getattr(tr, "length_m", 0.0),
                                                          getattr(tr, "width_m", 0.0)),
                    id=tr.tid, timestamp=now))

            # ---- simple forward in-path summary (route corridor refines) ----
            closest_dist = 999.0
            closest_type = ObjectType.UNKNOWN
            closest_speed = 0.0
            path_blocked = False
            for obj in objects:
                if obj.x > 0 and abs(obj.y) < self.path_width / 2:
                    if obj.distance < closest_dist:
                        closest_dist = obj.distance
                        closest_type = obj.object_type
                        closest_speed = obj.speed
                    if obj.distance < self.danger_distance:
                        path_blocked = True

            self.last_track_count = len(objects)
            output = PerceptionOutput(
                objects=objects,
                closest_obstacle_distance=closest_dist,
                closest_obstacle_type=closest_type,
                closest_obstacle_speed=closest_speed,
                path_blocked=path_blocked,
                timestamp=now, healthy=True, reason="OK")
            self._last_tracked_sim_time = sim_time
            self._last_output = output
            return output

        except Exception as error:
            print("[CameraLidar] ERROR:", error)
            return PerceptionOutput(healthy=False,
                                    reason=f"CAMERA_LIDAR_ERROR: {error}")

    # --------------------------------------------------------
    # FIND FORWARD LIDAR HAZARD
    # --------------------------------------------------------

    def _find_front_lidar_hazard(
        self,
        points: np.ndarray,
    ):

        if (
            points is None
            or len(points) == 0
        ):
            return None


        xyz = points[:, :3]

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]


        mask = (
            (x > self.minimum_lidar_x)
            &
            (x < self.detection_range)
            &
            (
                np.abs(y)
                < self.path_width / 2.0
            )
            &
            (
                z
                > self.minimum_lidar_z
            )
            &
            (
                z
                < self.maximum_lidar_z
            )
        )


        front_points = (
            xyz[mask]
        )


        if len(front_points) < (
            self.minimum_cluster_points
        ):
            return None


        # ----------------------------------------------------
        # SIMPLE 0.5-METER RANGE BINS
        #
        # Instead of believing one isolated point,
        # find the nearest distance band containing
        # several LiDAR returns.
        # ----------------------------------------------------

        bin_size = 0.5

        x_values = (
            front_points[:, 0]
        )

        bin_ids = np.floor(
            x_values / bin_size
        ).astype(np.int32)


        unique_bins = np.unique(
            bin_ids
        )


        for bin_id in sorted(
            unique_bins
        ):

            cluster = (
                front_points[
                    bin_ids == bin_id
                ]
            )


            if (
                len(cluster)
                < self.minimum_cluster_points
            ):
                continue


            median_x = float(
                np.median(
                    cluster[:, 0]
                )
            )

            median_y = float(
                np.median(
                    cluster[:, 1]
                )
            )


            distance = float(
                np.sqrt(
                    median_x ** 2
                    + median_y ** 2
                )
            )


            return {
                "x": median_x,
                # Raw CARLA/UE4 sensor frame is Y-right (left-handed).
                # The rest of this codebase (see perception.py's
                # _actor_to_object) uses Y-left for vehicle-relative
                # objects, so flip the sign here to match.
                "y": -median_y,
                "distance": distance,
                "points": int(
                    len(cluster)
                ),
            }


        return None


    # --------------------------------------------------------
    # CHOOSE CAMERA OBJECT MOST LIKELY IN OUR PATH
    # --------------------------------------------------------

    def _choose_forward_camera_object(
        self,
        detections,
        image_width,
        image_height,
    ):

        candidates = []


        for detection in detections:

            if (
                detection.class_id
                != PERSON_CLASS
                and
                detection.class_id
                not in VEHICLE_CLASSES
            ):
                continue


            x, y, w, h = (
                detection.box
            )


            center_x = (
                x + w / 2.0
            )


            # Forward-driving region:
            # reject detections at extreme far edges
            # of the camera frame.
            if (
                center_x
                < image_width * 0.15
                or
                center_x
                > image_width * 0.85
            ):
                continue


            area = (
                w * h
            )


            candidates.append(
                (
                    area,
                    detection.confidence,
                    detection,
                )
            )


        if not candidates:
            return None


        # Larger image object usually means it is
        # more relevant/closer for this simple demo.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )


        return candidates[0][2]


    # --------------------------------------------------------
    # FAULT TEST SUPPORT
    # --------------------------------------------------------

    def close(self):
        """Stop the detector thread (shutdown)."""
        if self._worker is not None:
            self._worker.stop()

    def disable(self):

        self._enabled = False

        print(
            "[CameraLidar] DISABLED — "
            "will report unhealthy"
        )


    def enable(self):

        self._enabled = True

        print(
            "[CameraLidar] Re-enabled"
        )
