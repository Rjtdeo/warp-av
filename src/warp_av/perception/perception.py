"""
Perception Module

YOUR ROVER equivalent:
    decision_node.py reads front/left/right distances and knows
    "something is 30cm in front of me".

THIS VERSION:
    Takes camera images and lidar point clouds.
    Detects vehicles, pedestrians, obstacles.
    Outputs a list of detected objects with positions.

    SHORTCUT: For Day 1-3, we use CARLA's built-in object ground truth.
    Later we can plug in YOLO or another detector on camera images.
    This is an honest shortcut — documented in OPEN_SOURCE.md.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ObjectType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """One thing the vehicle sees."""
    object_type: ObjectType
    x: float                    # position relative to vehicle (forward)
    y: float                    # position relative to vehicle (left)
    distance: float             # meters from vehicle
    speed: float = 0.0          # estimated speed m/s
    confidence: float = 1.0     # 0-1
    id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerceptionOutput:
    """Everything perception detected this frame."""
    objects: List[DetectedObject] = field(default_factory=list)
    closest_obstacle_distance: float = 999.0
    closest_obstacle_type: ObjectType = ObjectType.UNKNOWN
    closest_obstacle_speed: float = 0.0     # m/s; 0.0 also means "unknown" (camera mode has no tracking yet)
    path_blocked: bool = False
    timestamp: float = field(default_factory=time.time)
    healthy: bool = True
    reason: str = "OK"


class PerceptionSystem:
    """
    Detects objects around the vehicle.

    Right now: uses CARLA ground truth (world.get_actors()).
    Future: run YOLO on camera images, cluster lidar points.
    The output format stays the same either way — that's the point.
    """

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self._enabled = True
        # Fault-injection hooks (see testing/fault_injector.py). All default off.
        self._fault = {"freeze": False, "stale_age_s": 0.0, "latency_s": 0.0, "crash": False}
        self._last_output: Optional["PerceptionOutput"] = None

        # Detection parameters
        self.detection_range = 50.0          # meters
        self.danger_distance = 8.0           # meters — stop trigger (Troy #4: was 5.0; 8 m gives a real margin at 8 m/s cruise)
        self.path_width = 3.5                # meters — lane width to check

    def update(self) -> PerceptionOutput:
        """
        Run one perception cycle. Call this every tick.

        Like your decision_node reading front/left/right,
        but returns structured object list instead of raw distances.
        """
        if not self._enabled:
            return PerceptionOutput(healthy=False, reason="PERCEPTION_DISABLED")
        if self._fault["crash"]:
            self._fault["crash"] = False
            raise RuntimeError("INJECTED_PERCEPTION_CRASH")
        if self._fault["latency_s"] > 0:
            time.sleep(self._fault["latency_s"])
        if self._fault["freeze"] and self._last_output is not None:
            return self._last_output          # plausible but old data, timestamp not advancing

        try:
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

            objects = []

            # Get all actors in the world
            actors = self.world.get_actors()

            # Check vehicles
            for actor in actors.filter('vehicle.*'):
                if actor.id == self.vehicle.id:
                    continue
                obj = self._actor_to_object(actor, vehicle_location, vehicle_yaw, ObjectType.VEHICLE)
                if obj and obj.distance < self.detection_range:
                    objects.append(obj)

            # Check pedestrians (walkers)
            for actor in actors.filter('walker.*'):
                obj = self._actor_to_object(actor, vehicle_location, vehicle_yaw, ObjectType.PEDESTRIAN)
                if obj and obj.distance < self.detection_range:
                    objects.append(obj)

            # Check static obstacles (props)
            for actor in actors.filter('static.*'):
                obj = self._actor_to_object(actor, vehicle_location, vehicle_yaw, ObjectType.OBSTACLE)
                if obj and obj.distance < self.detection_range:
                    objects.append(obj)

            # Find closest object in our path
            closest_dist = 999.0
            closest_type = ObjectType.UNKNOWN
            closest_speed = 0.0
            path_blocked = False

            for obj in objects:
                # Is this object in our lane? (roughly ahead and within lane width)
                if obj.x > 0 and abs(obj.y) < self.path_width / 2:
                    if obj.distance < closest_dist:
                        closest_dist = obj.distance
                        closest_type = obj.object_type
                        closest_speed = obj.speed
                    if obj.distance < self.danger_distance:
                        path_blocked = True

            out = PerceptionOutput(
                objects=objects,
                closest_obstacle_distance=closest_dist,
                closest_obstacle_type=closest_type,
                closest_obstacle_speed=closest_speed,
                path_blocked=path_blocked,
                timestamp=time.time() - self._fault["stale_age_s"],
                healthy=True,
                reason="OK"
            )
            self._last_output = out
            return out

        except Exception as e:
            return PerceptionOutput(
                healthy=False,
                reason=f"PERCEPTION_ERROR: {e}"
            )

    def _actor_to_object(self, actor, vehicle_location, vehicle_yaw, obj_type) -> Optional[DetectedObject]:
        """Convert a CARLA actor to a DetectedObject in vehicle-relative coordinates."""
        try:
            actor_location = actor.get_location()

            # World-frame offset
            dx = actor_location.x - vehicle_location.x
            dy = actor_location.y - vehicle_location.y

            # Rotate into vehicle frame (x = forward, y = left)
            cos_yaw = math.cos(-vehicle_yaw)
            sin_yaw = math.sin(-vehicle_yaw)
            local_x = dx * cos_yaw - dy * sin_yaw
            local_y = dx * sin_yaw + dy * cos_yaw

            distance = math.sqrt(dx**2 + dy**2)

            # Get speed if it's a vehicle or walker
            speed = 0.0
            velocity = actor.get_velocity()
            if velocity:
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            return DetectedObject(
                object_type=obj_type,
                x=local_x,
                y=local_y,
                distance=distance,
                speed=speed,
                id=actor.id,
                timestamp=time.time()
            )
        except:
            return None

    def disable(self):
        """For testing Scenario 6: component failure."""
        self._enabled = False
        print("[Perception] DISABLED — will report unhealthy")

    def enable(self):
        self._enabled = True
        self._fault = {"freeze": False, "stale_age_s": 0.0, "latency_s": 0.0, "crash": False}
        print("[Perception] Re-enabled")

    def inject_fault(self, action: str, **params):
        """freeze | stale(age_s) | latency(latency_s) | crash. Cleared by enable()."""
        if action == "freeze":
            self._fault["freeze"] = True
        elif action == "stale":
            self._fault["stale_age_s"] = float(params.get("age_s", 2.0))
        elif action == "latency":
            self._fault["latency_s"] = float(params.get("latency_s", 0.3))
        elif action == "crash":
            self._fault["crash"] = True
        else:
            return False
        print(f"[Perception] FAULT INJECTED: {action} {params}")
        return True
