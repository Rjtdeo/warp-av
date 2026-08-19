"""
CARLA Sensor Adapter

YOUR ROVER equivalent:
    ESP32 reads ultrasonics, IMU, GPS → publishes JSON via MQTT
    sensor_node.py receives MQTT → publishes to ROS 2 topics

THIS VERSION:
    Attaches cameras, lidar, radar to the CARLA vehicle
    Publishes sensor data to ROS 2 topics
    Same pattern: hardware(sim) → adapter → ROS topics → perception

The rest of the stack never imports carla — only this adapter touches it.
"""

import carla
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable


@dataclass
class CameraFrame:
    image: np.ndarray        # HxWx4 BGRA
    width: int
    height: int
    fov: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LidarScan:
    points: np.ndarray       # Nx4 (x, y, z, intensity)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GnssReading:
    latitude: float
    longitude: float
    altitude: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ImuReading:
    accelerometer_x: float
    accelerometer_y: float
    accelerometer_z: float
    gyroscope_x: float
    gyroscope_y: float
    gyroscope_z: float
    compass: float           # heading in degrees
    timestamp: float = field(default_factory=time.time)


class CarlaSensorAdapter:
    """
    Manages all sensors attached to the CARLA vehicle.
    Think of this as your ESP32 — it reads raw sensor hardware
    and packages the data for the rest of the system.
    """

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []

        # Latest data (thread-safe via GIL for simple reads)
        self.latest_camera: Optional[CameraFrame] = None
        self.latest_lidar: Optional[LidarScan] = None
        self.latest_gnss: Optional[GnssReading] = None
        self.latest_imu: Optional[ImuReading] = None

        # Callbacks for when new data arrives
        self._camera_callbacks: List[Callable] = []
        self._lidar_callbacks: List[Callable] = []

        # Health tracking
        self._last_camera_time = 0.0
        self._last_lidar_time = 0.0
        self._last_gnss_time = 0.0
        self._last_imu_time = 0.0

        # Flag to simulate sensor failure (for Scenario 6)
        self.camera_enabled = True
        self.lidar_enabled = True
        self.gnss_enabled = True
        self.imu_enabled = True

    def setup_sensors(self):
        """Attach all sensors to the vehicle."""
        bp_lib = self.world.get_blueprint_library()

        # --- Front Camera ---
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.1')  # 10 Hz
        camera_transform = carla.Transform(
            carla.Location(x=2.0, z=1.8),  # front of vehicle, roof height
            carla.Rotation(pitch=-10)
        )
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(self._on_camera)
        self.sensors.append(camera)

        # --- LiDAR ---
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '150000')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('sensor_tick', '0.1')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar.listen(self._on_lidar)
        self.sensors.append(lidar)

        # --- GNSS (GPS) ---
        gnss_bp = bp_lib.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.1')
        gnss = self.world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        gnss.listen(self._on_gnss)
        self.sensors.append(gnss)

        # --- IMU ---
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')  # 20 Hz
        imu = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        imu.listen(self._on_imu)
        self.sensors.append(imu)

        print(f"[CarlaSensorAdapter] {len(self.sensors)} sensors attached")

    def _on_camera(self, image):
        if not self.camera_enabled:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_camera = CameraFrame(
            image=array, width=image.width, height=image.height,
            fov=float(image.fov), timestamp=time.time()
        )
        self._last_camera_time = time.time()
        for cb in self._camera_callbacks:
            cb(self.latest_camera)

    def _on_lidar(self, scan):
        if not self.lidar_enabled:
            return
        points = np.frombuffer(scan.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))  # x, y, z, intensity
        self.latest_lidar = LidarScan(points=points, timestamp=time.time())
        self._last_lidar_time = time.time()
        for cb in self._lidar_callbacks:
            cb(self.latest_lidar)

    def _on_gnss(self, gnss):
        if not self.gnss_enabled:
            return
        self.latest_gnss = GnssReading(
            latitude=gnss.latitude, longitude=gnss.longitude,
            altitude=gnss.altitude, timestamp=time.time()
        )
        self._last_gnss_time = time.time()

    def _on_imu(self, imu):
        if not self.imu_enabled:
            return
        self.latest_imu = ImuReading(
            accelerometer_x=imu.accelerometer.x,
            accelerometer_y=imu.accelerometer.y,
            accelerometer_z=imu.accelerometer.z,
            gyroscope_x=imu.gyroscope.x,
            gyroscope_y=imu.gyroscope.y,
            gyroscope_z=imu.gyroscope.z,
            compass=imu.compass,
            timestamp=time.time()
        )
        self._last_imu_time = time.time()

    def on_camera(self, callback):
        self._camera_callbacks.append(callback)

    def on_lidar(self, callback):
        self._lidar_callbacks.append(callback)

    # --- Health checks (used by Safety Supervisor) ---
    def is_camera_healthy(self, max_age_sec=2.0) -> bool:
        if not self.camera_enabled:
            return False
        return (time.time() - self._last_camera_time) < max_age_sec

    def is_lidar_healthy(self, max_age_sec=2.0) -> bool:
        if not self.lidar_enabled:
            return False
        return (time.time() - self._last_lidar_time) < max_age_sec

    def is_gnss_healthy(self, max_age_sec=2.0) -> bool:
        if not self.gnss_enabled:
            return False
        return (time.time() - self._last_gnss_time) < max_age_sec

    def is_imu_healthy(self, max_age_sec=2.0) -> bool:
        if not self.imu_enabled:
            return False
        return (time.time() - self._last_imu_time) < max_age_sec

    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()
        print("[CarlaSensorAdapter] All sensors destroyed")
