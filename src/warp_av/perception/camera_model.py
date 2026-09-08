"""
Where does a thing the LiDAR found appear in the camera picture?
(Perception V2, day 5.)

The two sensors sit in different places on the van and one of them is tilted,
so "12 m ahead, 2 m to the right" is not enough to find that thing in the
image. This module does the geometry properly:

    sensor (LiDAR) frame  ->  vehicle frame  ->  camera frame  ->  pixels

Frames, all CARLA's convention: x forward, y to the right, z up. A camera
looks along its own +x. The LiDAR sits on the roof centre, the front camera
sits ahead of it and lower, tilted down by ten degrees.

The old code skipped all of this: it turned an angle into a column with
`u = width/2 + (width/2) * y / x`, ignored the mounts, ignored the tilt, and
never worked out the row at all. It also read the detector's box as
(left, top, right, bottom) when the detector returns (left, top, width,
height), so the right edge was a width, the span was empty, and no camera
label ever reached a cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# where the sensors sit on the van, metres, from the CARLA sensor adapter
LIDAR_MOUNT = (0.0, 0.0, 2.5)
CAMERA_MOUNT = (2.0, 0.0, 1.8)
CAMERA_PITCH_DEG = -10.0        # negative = nose down


@dataclass(frozen=True)
class CameraModel:
    """The front camera: where it is, where it looks, and how wide it sees."""

    width: int = 800
    height: int = 600
    fov_deg: float = 90.0
    mount: Tuple[float, float, float] = CAMERA_MOUNT
    pitch_deg: float = CAMERA_PITCH_DEG
    lidar_mount: Tuple[float, float, float] = LIDAR_MOUNT

    @property
    def focal_px(self) -> float:
        """One number for both axes: square pixels, so the vertical field follows."""
        return (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    def sensor_to_camera(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """A point in the LiDAR's frame -> the camera's frame."""
        # LiDAR frame -> vehicle frame (the LiDAR is simply offset, never turned)
        vx = x + self.lidar_mount[0]
        vy = y + self.lidar_mount[1]
        vz = z + self.lidar_mount[2]
        # vehicle frame -> camera frame: shift to the camera, then undo its tilt
        dx = vx - self.mount[0]
        dy = vy - self.mount[1]
        dz = vz - self.mount[2]
        p = math.radians(self.pitch_deg)
        cp, sp = math.cos(p), math.sin(p)
        # the camera is turned by +pitch about the y axis, so undo it by -pitch
        return dx * cp + dz * sp, dy, -dx * sp + dz * cp

    def project(self, x: float, y: float, z: float) -> Optional[Tuple[float, float]]:
        """A point in the LiDAR's frame -> (column, row) in the picture, or None if
        it is behind the camera. Points outside the frame keep their coordinates:
        the caller decides whether a near-miss still counts."""
        cx, cy, cz = self.sensor_to_camera(x, y, z)
        if cx <= 0.05:
            return None
        f = self.focal_px
        return self.width / 2.0 + f * (cy / cx), self.height / 2.0 - f * (cz / cx)

    def in_view(self, x: float, y: float, z: float, margin_px: float = 0.0) -> bool:
        uv = self.project(x, y, z)
        if uv is None:
            return False
        u, v = uv
        return (-margin_px <= u <= self.width + margin_px
                and -margin_px <= v <= self.height + margin_px)

    def with_frame(self, width: int, height: int, fov_deg: float) -> "CameraModel":
        """The same mounting, for whatever picture the camera actually delivered."""
        if width == self.width and height == self.height and abs(fov_deg - self.fov_deg) < 1e-6:
            return self
        return CameraModel(width=width, height=height, fov_deg=fov_deg,
                           mount=self.mount, pitch_deg=self.pitch_deg, lidar_mount=self.lidar_mount)


def box_edges(box) -> Tuple[float, float, float, float]:
    """The detector reports (left, top, width, height). Return (left, top, right, bottom).

    Reading box[2] as the right edge is the day-5 bug: for a person 100 px wide at
    column 400 it gave the span 400..100, which nothing could ever fall inside.
    """
    x, y, w, h = (float(v) for v in box[:4])
    return x, y, x + w, y + h


def box_contains(box, u: float, v: float, margin_px: float = 12.0) -> bool:
    x1, y1, x2, y2 = box_edges(box)
    return (x1 - margin_px) <= u <= (x2 + margin_px) and (y1 - margin_px) <= v <= (y2 + margin_px)


def box_overlap(box, other) -> float:
    """What share of `box` sits over `other`. 0 = they do not touch, 1 = box is inside other.

    Used to tell a person standing beside a bicycle from a person riding one.
    """
    ax1, ay1, ax2, ay2 = box_edges(box)
    bx1, by1, bx2, by2 = box_edges(other)
    w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    return (w * h) / area


def box_centre(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box_edges(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_foot(box) -> Tuple[float, float]:
    """The middle of the box's bottom edge: where the thing meets the ground."""
    x1, _, x2, y2 = box_edges(box)
    return (x1 + x2) / 2.0, y2


def ground_point(cluster: dict, lidar_height_m: float = LIDAR_MOUNT[2]) -> Tuple[float, float, float]:
    """The road directly under a cluster: what the bottom of a camera box looks at."""
    return float(cluster["x"]), float(cluster["y"]), -lidar_height_m


def cluster_point(cluster: dict, lidar_height_m: float = LIDAR_MOUNT[2]) -> Tuple[float, float, float]:
    """A cluster is a footprint on the ground plus a height; aim at the middle of the
    thing, which is where a detector's box is centred, not at its feet."""
    h = cluster.get("height")
    h = 0.5 if h is None or h != h else float(h)
    return float(cluster["x"]), float(cluster["y"]), -lidar_height_m + h / 2.0
