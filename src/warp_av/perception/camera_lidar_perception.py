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
from .camera_model import (CameraModel, camera_models, box_contains, box_edges, box_foot, box_overlap,
                           cluster_point, ground_point)
from .occupancy import OccupancyGrid
from .road_edges import RoadEdges, find_road_edges
from .motion_class import (NOT_A_VEHICLE_MIN_HEIGHT_M, NOT_A_VEHICLE_MIN_LENGTH_M,
                           vehicle_name_implausible,
                           carla_road_gap, high_share, sample_for_gap, shape_rules,
                           static_dynamic_wanted, STATIC)
from .ground_filter import ROAD_EDGE_MIN_CENTRE_LATERAL_M
from .detection_worker import DetectionWorker, yolox_inline_from_env
from .tracking import (cluster_points, clearance_radius_m, merge_split_clusters,
                       ObjectTracker, MIN_POINTS_FAR, FAR_RANGE_M, vehicle_shaped)
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
RIDDEN_CONFIDENCE_THRESHOLD = 0.15   # how sure the camera must be that it saw a bicycle,
                                     # when the answer is only used to tell a rider from a
                                     # walker. Both are people the van stops for.
CYCLIST_SPEED_MPS = 4.0     # nobody walks this fast. A person moving at cycling pace is on
                            # something with wheels, whether or not the camera saw them.
CYCLIST_TRAVEL_M = 3.0      # ... and they must have actually got somewhere, so that a thing
                            # which has just appeared cannot be renamed by one frame of it


# ============================================================
# CAMERA DETECTION RESULT
# ============================================================


# How close two points must be to belong to the same object. At 1.0 m a barrel and the
# planter beside it became one 4.4 m blob; 0.8 m separates them and cuts the measured-size
# error nearly in half, at about 1 ms per update (Perception V2 day 4).
# A chip of kerb is no taller than a kerb and sits on the kerb line the van already fitted.
MAX_CARRY_M = 5.0   # move further than this between sweeps and the old map is not worth
                    # sliding along -- that is a teleport or a long stall, so start afresh

KERB_CRUMB_MAX_HEIGHT_M = 0.30    # the same ceiling the ground filter's kerb rule uses
KERB_CRUMB_TOLERANCE_M = 0.50     # how close to the fitted line it must sit, at its own distance

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

#: How many cores the detector may use. 0 means "all but two", which is what it always did.
#:
#: I set this to 6 on a theory that the detector was starving the driving loop, and then
#: measured it: 20, 12, 8, 6, 4 and 2 cores. The detector got steadily slower exactly as you
#: would expect -- 150 ms a picture at 20 cores, 821 ms at 2 -- and the loop did not improve
#: at all, bouncing between 7.1 and 9.2 Hz with no pattern. So starving the detector costs a
#: lot and buys nothing. The default is back to what it was; the setting stays because being
#: able to run that experiment again is worth keeping.
DEFAULT_DETECTOR_THREADS = 0


def detector_threads(env=None) -> int:
    """Cores for the detector. WARP_DETECTOR_THREADS overrides; 0 or less means all of them."""
    env = os.environ if env is None else env
    try:
        want = int(str(env.get("WARP_DETECTOR_THREADS", DEFAULT_DETECTOR_THREADS)).strip())
    except (TypeError, ValueError):
        want = DEFAULT_DETECTOR_THREADS
    cores = os.cpu_count() or 4
    if want <= 0:
        return max(1, cores - 2)      # what it always did: all but two
    return max(1, min(want, cores))


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
        # How many cores the detector may take. This is the single biggest thing setting the
        # van's thinking rate, and it was not obvious: the detector runs in its own thread and
        # never blocks the loop, so it LOOKED free. It is not. Measured on the van, turning
        # the camera off so the detector idles took the loop from 7.8 Hz to 9.4 Hz and made
        # the GROUND FILTER -- which does not touch the camera at all -- run twice as fast,
        # 34.6 ms down to 17.2 ms. Same code, same data, half the time, purely because the
        # cores were free.
        #
        # It was taking all but two of the machine's 22 cores. Leaving two cores for CARLA,
        # the driving loop, the sensor callbacks and every numpy sum in perception is nowhere
        # near enough. WARP_DETECTOR_THREADS sets it; the default is measured, not guessed.
        try:
            cv2.setNumThreads(detector_threads())
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

        # bicycles and motorbikes are kept down to this, for the rider test only
        self.ridden_confidence_threshold = (
            RIDDEN_CONFIDENCE_THRESHOLD
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

        # A bicycle is kept at a lower bar than everything else. It is never reported as an
        # object on its own: it is used only to tell a person who is riding from a person
        # who is walking, and both of those are things the van stops for. So a hesitant
        # bicycle can only move a label between two vulnerable classes, never invent one.
        best_class = np.argmax(scores, axis=1)
        ridden = np.isin(best_class, list(RIDDEN_CLASSES))
        candidate_mask = np.where(
            ridden,
            max_scores >= self.ridden_confidence_threshold,
            max_scores >= self.confidence_threshold,
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
        # the free-space map (day 9), rebuilt from every sweep
        self.grid = OccupancyGrid()
        self.road_edges = RoadEdges()      # the kerb lines, when the laser can see them (day 10)
        self.merge_split_blobs = True      # put a thing seen as two blobs back together
        self.last_blobs_merged = 0
        self.last_grid_ms = 0.0
        self.last_grid_error = None
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
        # Planning V2: static or dynamic. Needs the map for "how far from the nearest lane",
        # read once on the first sweep. WARP_STATIC_DYNAMIC=0 labels everything dynamic.
        self.static_dynamic = static_dynamic_wanted()
        self._road_gap = None
        self._road_gap_tried = False
        self.last_static_count = 0
        self.last_static_candidates = 0
        self.last_static_by_rule = {}

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
        self.last_kerb_crumbs_dropped = 0
        self._last_pose = None            # day 12: where the van was at the previous sweep
        # Day 13: the van has five cameras and only the front one ever named anything. The
        # laser already finds things all round; what was missing was the name. Set
        # WARP_ALL_CAMERAS=0 to go back to the front camera alone.
        self.use_all_cameras = str(os.environ.get("WARP_ALL_CAMERAS", "1")).strip().lower() \
            not in ("0", "false", "off", "no")
        self._view_models = {k: v for k, v in camera_models().items() if k != "front"}
        self.last_view_detections = {}
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
            # Without the LiDAR the van is blind and must stop. Without the camera it can
            # still see everything solid; what it loses is the name on each thing, so it
            # carries on with no detections and says it is running degraded (day 8).
            if lidar is None:
                return PerceptionOutput(healthy=False, reason="LIDAR_NO_DATA")
            camera_fault = ""
            if camera is None:
                camera_fault = "CAMERA_NO_DATA"
            elif now - camera.timestamp > 2.0:
                camera_fault = f"CAMERA_STALE_{now - camera.timestamp:.1f}s"
            if now - lidar.timestamp > 2.0:
                return PerceptionOutput(healthy=False,
                                        reason=f"LIDAR_STALE_{now - lidar.timestamp:.1f}s")

            # ---- camera inference: worker thread (default) or inline (A/B) ----
            if camera_fault:
                # no usable picture: run on the laser alone, with nothing named
                self._cached_camera_detections = []
                self.last_detection_age_s = 999.0
            elif self.yolox_inline:
                monotonic_now = time.perf_counter()   # fine on every machine (detection_worker)
                if monotonic_now - self._last_inference_time >= self.inference_interval:
                    t0 = time.perf_counter()
                    self._cached_camera_detections = self.detector.detect(camera.image[:, :, :3])
                    self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
                    self._last_inference_time = monotonic_now
                self.last_detection_age_s = time.perf_counter() - self._last_inference_time
            else:
                if self._worker is None:
                    views = {"front": (lambda: self.sensor_adapter.latest_camera)}
                    if self.use_all_cameras:
                        for v in ("left", "right", "rear"):
                            views[v] = (lambda v=v: (getattr(self.sensor_adapter,
                                                             "latest_frames", {}) or {}).get(v))
                    self._worker = DetectionWorker(self.detector.detect,
                                                   lambda: self.sensor_adapter.latest_camera,
                                                   interval_s=self.inference_interval, name="yolox",
                                                   views=views)
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
                        and time.perf_counter() - self._worker.started_at > self.detector_stall_s):
                    return PerceptionOutput(healthy=False, reason="CAMERA_LIDAR_ERROR: detector stalled")
            if camera_fault:
                camera = self._blank_camera_frame(camera)
            detections = self._cached_camera_detections
            # only a recent picture may overrule the shape rule: with a stale or missing
            # camera the van falls back to naming a car-sized blob a car
            camera_fresh = self.last_detection_age_s <= self.detection_max_age_s

            # ---- same LiDAR rotation as last tick? reuse the last output ----
            sim_time = getattr(lidar, "sim_time", None)
            if (sim_time is not None and self._last_tracked_sim_time is not None and self._last_output is not None
                    and 0.0 <= sim_time - self._last_tracked_sim_time < self.min_sweep_advance_s):
                return dataclasses.replace(self._last_output, timestamp=now)

            # Where the van is, taken once and used twice: to carry the free-space map
            # forward with it (day 12), and to place the blobs on the map further down.
            tf = self.sensor_adapter.vehicle.get_transform()
            moved = self._moved_since(tf)
            self._last_pose = (tf.location.x, tf.location.y, tf.rotation.yaw)

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
            # Fit the kerb lines FIRST: the rule below deletes kerb POINTS by their distance
            # from the line, so it needs the line before it runs (day 11 follow-up). The fit
            # reads the full sweep and the same road decision, so moving it earlier changes
            # nothing about the lines themselves.
            try:
                self.road_edges = find_road_edges(pts[:, :2], ground.above)
            except Exception as edge_error:
                self.last_grid_error = repr(edge_error)
            sel, heights, edges_dropped = remove_road_edge_points(sel, heights, cluster_points,
                                                                  edges=self.road_edges)
            # Pass 2: everything that is left
            xy = sel[:, :2]
            clusters = cluster_points(xy.tolist(), heights=heights.tolist(), cell=self.cluster_cell_m,
                                      return_members=self.static_dynamic,    # static or dynamic reads them
                                      min_points_far=MIN_POINTS_FAR, far_range_m=self.far_range_m)
            self.last_clusters_before_cap = _tracking.LAST_CLUSTER_TOTAL
            # one thing seen as two: put it back together (day 10 follow-up)
            if self.merge_split_blobs:
                before = len(clusters)
                clusters = merge_split_clusters(clusters)
                self.last_blobs_merged = before - len(clusters)
            # a 2-point blob that is low and beside the lane is a kerb crumb, not an object
            clusters = [c for c in clusters
                        if not (c.get("weak") and (c.get("height") is not None and c["height"] < 0.30)
                                and abs(c["y"]) > 1.2)]
            self.last_road_edges_dropped = edges_dropped
            # Free space, not just a list of things (Perception V2 day 9). Built from the
            # same sweep and the same road decision, so it can never disagree with them.
            try:
                t_grid = time.perf_counter()
                self.grid.update(pts[:, :2], mask, moved=moved, now=now)
                # (the kerb lines were fitted above, before the rule that reads them)
                self.last_grid_ms = (time.perf_counter() - t_grid) * 1000.0
            except Exception as grid_error:
                self.last_grid_ms = 0.0
                self.last_grid_error = repr(grid_error)
            # ---- a crumb of the kerb is kerb, however short (Perception V2 day 11) ----
            # The kerb rule in the ground filter only recognises a piece longer than 3 m,
            # so that a short low thing IN the road -- a slab, a plank, a pallet -- is never
            # written off. That leaves the chips: seen live in Town10HD on 2026-09-09, a
            # 0.38 x 0.00 x 0.14 m sliver reported as an obstacle 12 m ahead, sitting at
            # 4.4 m to the right, while the van had ALREADY fitted a confident kerb line at
            # 4.46 m through 189 points over 27.7 m. It knew where the kerb was and did not
            # use it. Now it does: a low blob sitting on a confident kerb line, and safely
            # outside the corridor the van drives down, is that kerb.
            before_kerb = len(clusters)
            clusters = [c for c in clusters if not self._sits_on_the_kerb(c)]
            self.last_kerb_crumbs_dropped = before_kerb - len(clusters)

            self.last_clusters = clusters     # sensor frame (x fwd, y right): the learned parker's feelers read these

            # ---- classify clusters by projecting them into the image ----
            # The two sensors sit in different places and the camera is tilted down,
            # so this is real geometry, not an angle-to-column guess (day 5).
            cam = self.camera_model.with_frame(camera.width, camera.height, camera.fov)
            self.last_camera_model = cam
            for c in clusters:
                c["cls"] = None
                c["conf"] = 0.0
            # Every camera that has something to say, FRONT FIRST so the sharpest picture
            # gets first say and the others only speak for what it cannot see (day 13).
            lookers = [("front", cam, detections)]
            if self.use_all_cameras and not camera_fault and self._worker is not None:
                frames = getattr(self.sensor_adapter, "latest_frames", {}) or {}
                per_view = self._worker.latest_by_view(self.detection_max_age_s)
                for v in ("left", "right", "rear"):
                    fr, model = frames.get(v), self._view_models.get(v)
                    if fr is None or model is None:
                        continue
                    dets, _age = per_view.get(v, ([], float("inf")))
                    lookers.append((v, model.with_frame(fr.width, fr.height, fr.fov), dets))
            # which cameras can even SEE each blob: the shape rule below needs to know
            for c in clusters:
                here = cluster_point(c, DEFAULT_LIDAR_HEIGHT_M)
                # bearing first (cheap), full geometry only for the camera it points at
                c["views"] = [n for n, mdl, _d in lookers
                              if mdl.could_see(here[0], here[1]) and mdl.in_view(*here)]
            matched_boxes = sum(self._name_with_camera(clusters, mdl, dets)
                                for _n, mdl, dets in lookers)
            self.last_view_detections = {n: len(d) for n, _m, d in lookers}
            self.last_camera_labels = matched_boxes
            self.last_camera_detections = len(detections)

            # Shape naming, for the sides and the back where the front camera cannot look:
            # a car-sized solid blob out there is a car. Inside the camera's view the
            # camera decides, so a bin is no longer called a vehicle for being boxy (day 5).
            base_points = 12 if self.thin_step <= 1 else 6
            for c in clusters:
                if c["cls"] is not None:
                    continue
                # "did a camera actually look at this?" now means ANY of them (day 13)
                seen_by_camera = camera_fresh and bool(c.get("views"))
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
                    c["cls_from"] = "shape"
                    c["conf"] = 0.45 if not seen_by_camera else 0.40

            # ---- ego -> world, then track ---- (tf was taken at the top of this update)
            yaw = math.radians(tf.rotation.yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            ex0, ey0 = tf.location.x, tf.location.y
            road_gap = self._road_gap_reader() if self.static_dynamic else None
            candidates = 0
            names_dropped = 0
            observations = []
            for c in clusters:
                # Static or dynamic (Planning V2): the shape check is cheap and runs on every
                # blob; only a blob whose shape fits a rule gets a lookup on the map, and even
                # then only when its track asks (once, then reused -- see motion_class.py).
                shapes, gap_fn = (), None
                members = c.pop("members", None)
                # a camera's VEHICLE that stands tall, long and wholly off the road is street
                # furniture -- a bus shelter was named one (motion_class.vehicle_name_implausible)
                if (c.get("cls") == "vehicle" and road_gap is not None and members
                        and (c.get("height") or 0.0) >= NOT_A_VEHICLE_MIN_HEIGHT_M
                        and max(c.get("length_m", 0.0), c.get("width_m", 0.0)) >= NOT_A_VEHICLE_MIN_LENGTH_M):
                    idx = np.asarray(members, dtype=int)
                    wpts = [(ex0 + px * cy - py * sy, ey0 + px * sy + py * cy)
                            for px, py in sample_for_gap(sel[idx, :2])]
                    if vehicle_name_implausible(c.get("height"),
                                                max(c.get("length_m", 0.0), c.get("width_m", 0.0)),
                                                road_gap.gap_m(wpts, tf.location.z)):
                        c["cls"], c["cls_from"] = None, None
                        names_dropped += 1
                if road_gap is not None and members:
                    idx = np.asarray(members, dtype=int)
                    shapes = shape_rules(c.get("height") or 0.0,
                                         max(c.get("length_m", 0.0), c.get("width_m", 0.0)),
                                         high_share(heights[idx]), c["n"])
                    if shapes:
                        candidates += 1
                        pts = sample_for_gap(sel[idx, :2])
                        wpts = [(ex0 + px * cy - py * sy, ey0 + px * sy + py * cy) for px, py in pts]
                        gap_fn = (lambda w=wpts, z=tf.location.z: road_gap.gap_m(w, z))
                observations.append({
                    "static_shapes": shapes,
                    "road_gap_fn": gap_fn,
                    "wx": ex0 + c["x"] * cy - c["y"] * sy,
                    "wy": ey0 + c["x"] * sy + c["y"] * cy,
                    "cls": c["cls"],
                    "cls_source": c.get("cls_from"),
                    "confidence": c["conf"],
                    "weak": bool(c.get("weak", False)),   # a 2-point far blob: three sightings before it counts
                    "distance": c["distance"],            # how much to trust this sighting (day 6)
                    "length_m": c.get("length_m", 0.0),
                    "width_m": c.get("width_m", 0.0),
                    "height_m": c.get("height") or 0.0,
                    "yaw_deg": c.get("yaw_deg", 0.0),
                    # the same heading on the map, so it stays right after the van turns, and
                    # where the rectangle itself is centred (fix 2)
                    "yaw_world_deg": float(c.get("yaw_deg", 0.0) or 0.0) + tf.rotation.yaw,
                    "box_wx": ex0 + c.get("box_x", c["x"]) * cy - c.get("box_y", c["y"]) * sy,
                    "box_wy": ey0 + c.get("box_x", c["x"]) * sy + c.get("box_y", c["y"]) * cy,
                    "box_len": c.get("box_len", 0.0), "box_wid": c.get("box_wid", 0.0),
                    "box_yaw_world_deg": float(c.get("box_yaw_deg", c.get("yaw_deg", 0.0)) or 0.0) + tf.rotation.yaw,
                })
            tracks = self.tracker.update(observations, now)
            self.last_static_candidates = candidates
            self.last_vehicle_names_dropped = names_dropped

            # ---- tracks -> DetectedObjects (back to ego frame) ----
            objects = []
            for tr in tracks:
                dx, dy = tr.wx - ex0, tr.wy - ey0
                ex = dx * cy + dy * sy
                ey = -dx * sy + dy * cy
                dist = math.hypot(dx, dy)
                if dist > self.detection_range:
                    continue
                kind = tr.cls
                # Nobody walks at cycling pace. The test is on travel actually made, not on
                # a speed reading: a thing that has just appeared can show a large speed for
                # a moment, and that was enough to turn a standing person into a cyclist.
                if (kind == "pedestrian" and not getattr(tr, "stationary", True)
                        and self.tracker.reported_speed(tr) >= CYCLIST_SPEED_MPS
                        and getattr(tr, "travelled_m", 0.0) >= CYCLIST_TRAVEL_M):
                    kind = "cyclist"
                otype = (ObjectType.PEDESTRIAN if kind == "pedestrian"
                         else ObjectType.CYCLIST if kind == "cyclist"
                         else ObjectType.VEHICLE if kind == "vehicle"
                         else ObjectType.OBSTACLE)
                vx_w, vy_w = self.tracker.reported_velocity(tr)
                # heading in the van's frame NOW, from the map-frame one (fix 2): the sighting
                # it came from may be from before the van turned
                yaw_now = getattr(tr, "yaw_deg", 0.0)
                if getattr(tr, "yaw_world_deg", None) is not None:
                    yaw_now = (tr.yaw_world_deg - tf.rotation.yaw + 180.0) % 360.0 - 180.0
                ox, oy = getattr(tr, "box_off", (0.0, 0.0))
                box_len, box_wid = getattr(tr, "box_len", 0.0), getattr(tr, "box_wid", 0.0)
                box_yaw_world = getattr(tr, "box_yaw_world_deg", None)
                best = getattr(tr, "best_box", None) if getattr(tr, "stationary", False) else None
                if best is not None:             # standing still: its most complete look (fix 2)
                    ox, oy = best[1] - tr.wx, best[2] - tr.wy
                    box_len, box_wid, box_yaw_world = best[3], best[4], best[5]
                box_yaw_now = 0.0
                if box_yaw_world is not None:
                    box_yaw_now = (box_yaw_world - tf.rotation.yaw + 180.0) % 360.0 - 180.0
                objects.append(DetectedObject(
                    object_type=otype, x=ex, y=ey, distance=dist,
                    speed=self.tracker.reported_speed(tr),
                    vx_world=vx_w, vy_world=vy_w,
                    # a track built only from 2-point far sightings is reported, but with low confidence
                    confidence=(tr.confidence if tr.cls else 0.65) if not getattr(tr, "weak_only", False) else 0.35,
                    length_m=getattr(tr, "length_m", 0.0), width_m=getattr(tr, "width_m", 0.0),
                    height_m=getattr(tr, "height_m", 0.0), yaw_deg=yaw_now,
                    box_dx=ox * cy + oy * sy, box_dy=-ox * sy + oy * cy,
                    box_length_m=box_len, box_width_m=box_wid,
                    box_yaw_deg=box_yaw_now,
                    stationary=bool(getattr(tr, "stationary", True)),
                    size_uncertain=bool(getattr(tr, "size_uncertain", False)),
                    clearance_radius_m=clearance_radius_m(tr.cls, getattr(tr, "length_m", 0.0),
                                                          getattr(tr, "width_m", 0.0)),
                    motion_class=tr.motion.state,
                    static_rule=(tr.motion.rule or "") if tr.motion.is_static else "",
                    motion_why=tr.motion.why,
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
            by_rule = {}
            for obj in objects:
                if obj.motion_class == STATIC:
                    by_rule[obj.static_rule] = by_rule.get(obj.static_rule, 0) + 1
            self.last_static_count = sum(by_rule.values())
            self.last_static_by_rule = by_rule
            output = PerceptionOutput(
                objects=objects,
                closest_obstacle_distance=closest_dist,
                closest_obstacle_type=closest_type,
                closest_obstacle_speed=closest_speed,
                path_blocked=path_blocked,
                timestamp=now, healthy=True,
                reason="OK" if not camera_fault else f"LIDAR_ONLY: {camera_fault}",
                degraded=bool(camera_fault), degraded_reason=camera_fault)
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

    def _blank_camera_frame(self, camera):
        """A stand-in picture, so the geometry that needs the camera's shape still works
        while the camera itself is missing. Nothing is ever detected in it."""
        if camera is not None:
            return camera
        import numpy as _np
        from ..adapters.carla_sensor_adapter import CameraFrame
        w, h = self.camera_model.width, self.camera_model.height
        return CameraFrame(image=_np.zeros((h, w, 4), dtype=_np.uint8), width=w, height=h,
                           fov=self.camera_model.fov_deg, timestamp=time.time())

    def _name_with_camera(self, clusters, cam, detections) -> int:
        """Put names from ONE camera onto the blobs it can see. Returns how many stuck.

        Called once per camera, front first (day 13). A blob that already has a name is left
        alone, so the front camera -- the biggest, sharpest picture -- always gets first say,
        and the side and rear cameras only speak for what it cannot see.
        """
        if not detections:
            return 0
        for c in clusters:
            here = cluster_point(c, DEFAULT_LIDAR_HEIGHT_M)
            if not cam.could_see(here[0], here[1]):
                c["uv"] = c["uv_foot"] = None      # not this camera's business
                continue
            c["uv"] = cam.project(*here)
            c["uv_foot"] = cam.project(*ground_point(c, DEFAULT_LIDAR_HEIGHT_M))
        matched = 0
        ridden = [d for d in detections if d.class_id in RIDDEN_CLASSES]
        named = []
        for det in detections:
            if det.class_id == PERSON_CLASS:
                # A person standing over a bicycle's box is riding it, and a cyclist needs a
                # bicycle's room while still being a person to give way to.
                #
                # Asked BOTH WAYS ROUND, and that matters. It used to ask only how much of
                # the PERSON was covered by the bike, and a rider's box is taller than the
                # bicycle underneath them, so that number is small and marginal. Measured on
                # Town10HD 2026-09-09 with three bicycles and a motorbike: person-covered-by-
                # bike came out 0.33, 0.46, 0.56, 0.62, 0.65 -- and the 0.33 was a real rider
                # called a pedestrian, missing the bar by 0.02. The other way round gave
                # 0.58, 0.67, 0.72, 0.76, 0.91 on the same riders: always higher, never
                # marginal. Taking the larger of the two names every one of them correctly.
                #
                # What this test CANNOT do, measured rather than assumed: tell a rider from
                # somebody standing beside the bike. A person 2.5 m away scored 0.48 one way
                # and 0.73 the other -- the same as a real rider. That was already true
                # before this change, and it fails in the safe direction: a bystander called
                # a cyclist gets MORE room (0.8 m against 0.6 m) and is still someone the van
                # gives way to. Telling them apart needs the detector to place the rider ON
                # the saddle, which is class-list work, not a threshold.
                over = max((max(box_overlap(det.box, b.box), box_overlap(b.box, det.box))
                            for b in ridden), default=0.0)
                named.append((det, "cyclist" if over >= RIDER_OVERLAP else "pedestrian", over))
            elif det.class_id in VEHICLE_CLASSES:
                named.append((det, "vehicle", 0.0))
        # riders first: with a person beside the bike and a person on it, the one on it
        # must claim the blob, or the rider ends up named a pedestrian
        named.sort(key=lambda n: (n[1] != "cyclist", -n[2]))
        for det, cls, _over in named:
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
                best["cls_from"] = "camera"
                best["conf"] = float(det.confidence)
                matched += 1
        return matched

    def _moved_since(self, tf):
        """(forward, right, turned) since the last sweep, in the van's frame back then.

        None on the very first sweep, or after any gap long enough that carrying the old map
        forward would be guesswork -- then the map is simply rebuilt, as it always was.
        """
        last = getattr(self, "_last_pose", None)
        if last is None:
            return None
        lx, ly, lyaw = last
        dx, dy = tf.location.x - lx, tf.location.y - ly
        if not (abs(dx) < MAX_CARRY_M and abs(dy) < MAX_CARRY_M):
            return None                      # teleported, or a long stall: start again
        a = math.radians(lyaw)
        ca, sa = math.cos(a), math.sin(a)
        forward = dx * ca + dy * sa
        right = -dx * sa + dy * ca
        turned = (tf.rotation.yaw - lyaw + 180.0) % 360.0 - 180.0
        return (forward, right, turned)

    def _road_gap_reader(self):
        """How far things are from the nearest lane, from CARLA's map -- read once. Without a
        map (a replay, a fake sensor) there is no answer, and nothing is ever called static."""
        if not self._road_gap_tried:
            self._road_gap_tried = True
            try:
                self._road_gap = carla_road_gap(self.sensor_adapter.vehicle.get_world().get_map())
            except Exception as e:
                self._road_gap = None
                self.last_road_gap_error = repr(e)
        return self._road_gap

    def _sits_on_the_kerb(self, cluster) -> bool:
        """Is this low blob just a chip of the kerb the van has already found?

        Three things must all hold, and each one is there to stop this rule eating a real
        obstacle:
          * it is no taller than a kerb, so nothing that stands up is ever touched;
          * it is outside the corridor the van drives down, so nothing it could hit is
            touched however low (the same boundary the ground filter uses);
          * it lies on a CONFIDENT kerb line, at that blob's own distance ahead -- not
            merely somewhere off to the side.
        """
        height = cluster.get("height")
        if height is None or height >= KERB_CRUMB_MAX_HEIGHT_M:
            return False
        y = float(cluster.get("y", 0.0))
        if abs(y) < ROAD_EDGE_MIN_CENTRE_LATERAL_M:
            return False
        edges = getattr(self, "road_edges", None)
        if edges is None:
            return False
        edge = edges.right if y > 0 else edges.left
        if edge is None or not edge.confident:
            return False
        return abs(y - edge.lateral_at(float(cluster.get("x", 0.0)))) <= KERB_CRUMB_TOLERANCE_M

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
