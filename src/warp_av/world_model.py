"""
What the van knows right now, in one place (Perception V2, day 7).

Until today every part of the stack reached into perception's insides and
took what it liked: the behaviour layer read three loose fields, the web page
rebuilt world coordinates itself, the parking learner read raw blobs. Any
change inside perception risked breaking all of them at once, which is why
each of days 1 to 6 needed a careful hunt for who else might care.

This module is the sheet in between. Perception fills it in once a tick;
everyone else reads it and nothing else. It holds:

  * every thing the van is following, with its place in BOTH frames (metres
    ahead of and right of the van, and where it stands on the map), its
    measured size, what it is called, whether it is moving, and how sure and
    how fresh the answer is;
  * what is in the van's path and how far away;
  * whether the eyes are working;
  * when all of this was true.

It is a plain description, not a decision. Nothing here brakes, steers or
plans; it only says what is out there.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .perception.perception import DetectedObject, ObjectType, PerceptionOutput

DEFAULT_PATH_HALF_WIDTH_M = 1.75      # half the van's driving corridor
DEFAULT_PATH_AHEAD_M = 30.0


@dataclass
class WorldObject:
    """One thing the van is following."""

    id: int
    kind: str                          # 'vehicle' | 'pedestrian' | 'obstacle' | 'unknown'
    # where it is, from the van: x is metres ahead of the nose, y is metres to the right
    x: float
    y: float
    distance_m: float
    # where it is on the map
    world_x: float
    world_y: float
    # how it is moving, in map coordinates
    speed_mps: float = 0.0
    vx_world: float = 0.0
    vy_world: float = 0.0
    stationary: bool = True
    # how big it is, as measured by the LiDAR (0.0 = not measured)
    length_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    yaw_deg: float = 0.0
    confidence: float = 1.0
    size_uncertain: bool = False       # recent sightings disagreed about how big it is
    clearance_radius_m: float = 0.4    # room to leave around it, never less than its kind needs
    age_s: float = 0.0                 # how old the sighting behind this is

    @property
    def is_vehicle(self) -> bool:
        return self.kind == ObjectType.VEHICLE.value

    @property
    def is_pedestrian(self) -> bool:
        return self.kind == ObjectType.PEDESTRIAN.value

    @property
    def is_cyclist(self) -> bool:
        return self.kind == ObjectType.CYCLIST.value

    @property
    def is_vulnerable(self) -> bool:
        """A person, on foot or on a bike: always give way, never drive around."""
        return self.kind in (ObjectType.PEDESTRIAN.value, ObjectType.CYCLIST.value)

    @property
    def moving(self) -> bool:
        return not self.stationary

    def ahead(self) -> bool:
        return self.x > 0.0

    def in_corridor(self, half_width_m: float = DEFAULT_PATH_HALF_WIDTH_M,
                    ahead_m: float = DEFAULT_PATH_AHEAD_M) -> bool:
        """Is it in the strip of road straight in front of the van?

        This is the simple nose-forward test. The planner has a better one that
        follows the route round a bend; where that has run, `WorldModel.path`
        already carries its answer.
        """
        return 0.0 < self.x <= ahead_m and abs(self.y) <= half_width_m

    def as_dict(self) -> dict:
        def num(v):
            # a place on the map is unknown while the van does not know where it is;
            # say so with null rather than a NaN the browser cannot read
            return None if v is None or v != v or v in (float("inf"), float("-inf")) else round(v, 2)

        return {
            "id": self.id, "type": self.kind,
            "x": num(self.world_x), "y": num(self.world_y),
            "ego_x": round(self.x, 2), "ego_y": round(self.y, 2),
            "distance": round(self.distance_m, 1),
            "speed": round(self.speed_mps, 2), "stationary": self.stationary,
            "length_m": round(self.length_m, 2), "width_m": round(self.width_m, 2),
            "height_m": round(self.height_m, 2), "yaw_deg": round(self.yaw_deg, 1),
            "confidence": round(self.confidence, 2), "age_s": round(self.age_s, 2),
            "size_uncertain": self.size_uncertain,
            "clearance_radius_m": round(self.clearance_radius_m, 2),
        }


@dataclass
class PathState:
    """What lies in the van's way."""

    blocked: bool = False
    closest_distance_m: Optional[float] = None
    closest_kind: str = ObjectType.UNKNOWN.value
    closest_speed_mps: float = 0.0
    closest_lateral_m: Optional[float] = None
    closest_id: Optional[int] = None

    def as_dict(self) -> dict:
        return {"blocked": self.blocked,
                "closest_distance_m": None if self.closest_distance_m is None else round(self.closest_distance_m, 1),
                "closest_kind": self.closest_kind,
                "closest_speed_mps": round(self.closest_speed_mps, 2),
                "closest_lateral_m": None if self.closest_lateral_m is None else round(self.closest_lateral_m, 2),
                "closest_id": self.closest_id}


@dataclass
class WorldModel:
    """Everything the van knows about its surroundings at one moment."""

    timestamp: float = field(default_factory=time.time)
    van_x: float = 0.0
    van_y: float = 0.0
    van_yaw_rad: float = 0.0
    van_speed_mps: float = 0.0
    objects: List[WorldObject] = field(default_factory=list)
    path: PathState = field(default_factory=PathState)
    sensors_healthy: bool = True
    sensors_reason: str = "OK"
    traffic_light: str = "none"
    traffic_light_distance_m: Optional[float] = None
    source: str = "camera_lidar"       # which perception produced this
    # free space around the van (day 9): how far it can go before the map stops being free
    free_space: Optional[dict] = None

    # ---- reading it -----------------------------------------------------------------
    def age_s(self, now: Optional[float] = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.timestamp)

    def moving_objects(self) -> List[WorldObject]:
        return [o for o in self.objects if o.moving]

    def vulnerable(self) -> List[WorldObject]:
        """People, on foot or on bikes."""
        return [o for o in self.objects if o.is_vulnerable]

    def parked_objects(self) -> List[WorldObject]:
        return [o for o in self.objects if o.stationary]

    def of_kind(self, kind) -> List[WorldObject]:
        k = kind.value if isinstance(kind, ObjectType) else str(kind)
        return [o for o in self.objects if o.kind == k]

    def in_corridor(self, half_width_m: float = DEFAULT_PATH_HALF_WIDTH_M,
                    ahead_m: float = DEFAULT_PATH_AHEAD_M) -> List[WorldObject]:
        got = [o for o in self.objects if o.in_corridor(half_width_m, ahead_m)]
        got.sort(key=lambda o: o.distance_m)
        return got

    def nearest_ahead(self, half_width_m: float = DEFAULT_PATH_HALF_WIDTH_M,
                      ahead_m: float = DEFAULT_PATH_AHEAD_M) -> Optional[WorldObject]:
        got = self.in_corridor(half_width_m, ahead_m)
        return got[0] if got else None

    def crossing_vehicles(self, radius_m: float, lane_half_width_m: float = DEFAULT_PATH_HALF_WIDTH_M,
                          behind_m: float = -3.0, min_speed_mps: float = 1.0) -> List[WorldObject]:
        """Moving vehicles that could cross the van's path at a junction.

        Not the ones in our own lane straight ahead: those are the car-following
        problem. Not the parked ones, not the ones well behind us, not the far ones.
        """
        got = []
        for o in self.objects:
            if not o.is_vehicle or o.speed_mps < min_speed_mps:
                continue
            if o.distance_m > radius_m or o.x < behind_m:
                continue
            if o.x > 0 and abs(o.y) < lane_half_width_m:
                continue
            got.append(o)
        got.sort(key=lambda o: o.distance_m)
        return got

    def by_id(self, oid: int) -> Optional[WorldObject]:
        for o in self.objects:
            if o.id == oid:
                return o
        return None

    def as_dict(self) -> dict:
        return {"timestamp": self.timestamp, "age_s": round(self.age_s(), 3),
                "van": {"x": round(self.van_x, 2), "y": round(self.van_y, 2),
                        "yaw_rad": round(self.van_yaw_rad, 4), "speed_mps": round(self.van_speed_mps, 2)},
                "healthy": self.sensors_healthy, "reason": self.sensors_reason,
                "source": self.source,
                "traffic_light": self.traffic_light,
                "traffic_light_distance_m": self.traffic_light_distance_m,
                "path": self.path.as_dict(),
                "free_space": self.free_space,
                "counts": {"total": len(self.objects),
                           "moving": len(self.moving_objects()),
                           "vehicles": len(self.of_kind(ObjectType.VEHICLE)),
                           "pedestrians": len(self.of_kind(ObjectType.PEDESTRIAN)),
                           "cyclists": len(self.of_kind(ObjectType.CYCLIST))},
                "objects": [o.as_dict() for o in self.objects]}


def build_world_model(perception: PerceptionOutput, pose, source: str = "camera_lidar",
                      now: Optional[float] = None, free_space: Optional[dict] = None) -> WorldModel:
    """Turn one tick of perception, plus where the van is, into the sheet.

    `pose` is anything with x, y, yaw (radians) and optionally speed and healthy.
    Every number is carried across unchanged: this is a view of what perception
    already decided, never a second opinion.
    """
    now = time.time() if now is None else now
    van_x = float(getattr(pose, "x", 0.0))
    van_y = float(getattr(pose, "y", 0.0))
    yaw = float(getattr(pose, "yaw", 0.0))
    cy, sy = math.cos(yaw), math.sin(yaw)

    objects: List[WorldObject] = []
    pose_ok = bool(getattr(pose, "healthy", True))
    for o in (perception.objects or []):
        wx = van_x + cy * o.x - sy * o.y if pose_ok else float("nan")
        wy = van_y + sy * o.x + cy * o.y if pose_ok else float("nan")
        objects.append(WorldObject(
            id=int(getattr(o, "id", 0)),
            kind=o.object_type.value if isinstance(o.object_type, ObjectType) else str(o.object_type),
            x=float(o.x), y=float(o.y), distance_m=float(o.distance),
            world_x=wx, world_y=wy,
            speed_mps=float(getattr(o, "speed", 0.0)),
            vx_world=float(getattr(o, "vx_world", 0.0)),
            vy_world=float(getattr(o, "vy_world", 0.0)),
            stationary=bool(getattr(o, "stationary", True)),
            length_m=float(getattr(o, "length_m", 0.0)),
            width_m=float(getattr(o, "width_m", 0.0)),
            height_m=float(getattr(o, "height_m", 0.0)),
            yaw_deg=float(getattr(o, "yaw_deg", 0.0)),
            confidence=float(getattr(o, "confidence", 1.0)),
            size_uncertain=bool(getattr(o, "size_uncertain", False)),
            clearance_radius_m=float(getattr(o, "clearance_radius_m", 0.4)),
            age_s=max(0.0, now - float(getattr(o, "timestamp", now) or now)),
        ))

    closest = perception.closest_obstacle_distance
    if closest is not None and closest >= 900.0:
        closest = None                     # perception's "nothing there" sentinel
    kind = perception.closest_obstacle_type
    path = PathState(
        blocked=bool(perception.path_blocked),
        closest_distance_m=closest,
        closest_kind=kind.value if isinstance(kind, ObjectType) else str(kind or ObjectType.UNKNOWN.value),
        closest_speed_mps=float(perception.closest_obstacle_speed or 0.0),
        closest_lateral_m=perception.closest_obstacle_lateral_m,
    )
    if path.closest_distance_m is not None:
        near = [o for o in objects if abs(o.distance_m - path.closest_distance_m) < 1.5 and o.x > 0]
        if near:
            path.closest_id = min(near, key=lambda o: abs(o.distance_m - path.closest_distance_m)).id

    return WorldModel(
        timestamp=float(getattr(perception, "timestamp", now) or now),
        van_x=van_x, van_y=van_y, van_yaw_rad=yaw,
        van_speed_mps=float(getattr(pose, "speed", 0.0) or 0.0),
        objects=objects, path=path,
        sensors_healthy=bool(perception.healthy),
        sensors_reason=str(perception.reason),
        traffic_light=str(perception.traffic_light),
        traffic_light_distance_m=perception.traffic_light_distance_m,
        source=source,
        free_space=free_space,
    )
