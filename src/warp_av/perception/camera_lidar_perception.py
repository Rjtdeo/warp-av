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

import math
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .tracking import cluster_points, ObjectTracker

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


# ============================================================
# CAMERA DETECTION RESULT
# ============================================================

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
    ):

        self.sensor_adapter = (
            sensor_adapter
        )

        self._enabled = True

        self.detector = YoloXDetector(
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

            # ---- camera inference, cached at inference_interval ----
            monotonic_now = time.monotonic()
            if monotonic_now - self._last_inference_time >= self.inference_interval:
                t0 = time.perf_counter()
                self._cached_camera_detections = self.detector.detect(camera.image[:, :, :3])
                self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
                self._last_inference_time = monotonic_now
            detections = self._cached_camera_detections

            # ---- LiDAR -> 2D clusters (sensor frame: x fwd, y right) ----
            pts = lidar.points
            mask = (pts[:, 2] > self.minimum_lidar_z) & (pts[:, 2] < self.maximum_lidar_z)
            xy = pts[mask][::3, :2]          # downsample 3x: plenty for van-sized objects
            clusters = cluster_points(xy.tolist())

            # ---- classify clusters by projecting into the image ----
            # fov 90 deg -> fx = width/2. u grows to the RIGHT (y right in
            # sensor frame), matching the image axis directly.
            fx = camera.width / 2.0
            for c in clusters:
                c["cls"] = None
                c["conf"] = 0.0
                if c["x"] > 1.0:
                    c["u"] = camera.width / 2.0 + fx * (c["y"] / c["x"])
                else:
                    c["u"] = None
            for det in detections:
                cls = ("pedestrian" if det.class_id == PERSON_CLASS
                       else "vehicle" if det.class_id in VEHICLE_CLASSES
                       else None)
                if cls is None:
                    continue
                bx1, bx2 = det.box[0], det.box[2]
                best = None
                for c in clusters:
                    if c["u"] is None or c["cls"] is not None:
                        continue
                    if bx1 - 12 <= c["u"] <= bx2 + 12:
                        if best is None or c["distance"] < best["distance"]:
                            best = c
                if best is not None:
                    best["cls"] = cls
                    best["conf"] = float(det.confidence)

            # Shape naming: a car-sized solid blob on the road is a car,
            # camera confirmation or not (side/rear objects never enter the
            # front camera's view and were all labelled OBSTACLE).
            for c in clusters:
                if c["cls"] is None and c["extent"] >= 0.9 and c["n"] >= 6:
                    c["cls"] = "vehicle"
                    c["conf"] = 0.45

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
                         else ObjectType.VEHICLE if tr.cls == "vehicle"
                         else ObjectType.OBSTACLE)
                objects.append(DetectedObject(
                    object_type=otype, x=ex, y=ey, distance=dist,
                    speed=self.tracker.reported_speed(tr),
                    confidence=tr.confidence if tr.cls else 0.65,
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
            return PerceptionOutput(
                objects=objects,
                closest_obstacle_distance=closest_dist,
                closest_obstacle_type=closest_type,
                closest_obstacle_speed=closest_speed,
                path_blocked=path_blocked,
                timestamp=now, healthy=True, reason="OK")

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
