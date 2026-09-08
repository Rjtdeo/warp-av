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

try:
    import carla
except ImportError:  # offline replay/tests on a machine without the simulator
    carla = None
import numpy as np
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable

from .lidar_sweep import LidarSweepAccumulator

LIDAR_ROTATION_HZ = 10.0


def lidar_full_sweep_enabled(env=None) -> bool:
    """Perception fix 1 (2026-09-07): glue CARLA's per-frame LiDAR wedges into
    whole 360-degree sweeps. ON by default; WARP_LIDAR_SWEEP=0 restores the
    old one-wedge-per-scan behaviour for A/B comparison."""
    env = os.environ if env is None else env
    return str(env.get("WARP_LIDAR_SWEEP", "1")).strip().lower() not in ("0", "off", "false", "no")


@dataclass
class CameraFrame:
    image: np.ndarray        # HxWx4 BGRA
    width: int
    height: int
    fov: float
    timestamp: float = field(default_factory=time.time)


LIDAR_COLUMNS = ("x", "y", "z", "intensity", "ring", "t_rel")
# ring: the beam (0..channels-1) that made the point, -1 when unknown
# t_rel: seconds between the point's delivery and the newest delivery in the scan (<= 0)


@dataclass
class LidarScan:
    points: np.ndarray       # Nx6, columns LIDAR_COLUMNS; sensor frame: x forward, y right, z up
    timestamp: float = field(default_factory=time.time)
    frames: int = 1          # CARLA deliveries merged into this scan (1 = a single per-frame wedge)
    span_s: float = 0.0      # simulation seconds those deliveries cover (about 0.1 = one rotation)
    sim_time: Optional[float] = None   # simulation time of the newest delivery in the scan
    sensor_matrix: Optional[np.ndarray] = None   # 4x4 sensor->world pose at the newest delivery
    columns: tuple = LIDAR_COLUMNS

    def __post_init__(self):
        pts = self.points
        if pts is not None and getattr(pts, "ndim", 0) == 2 and pts.shape[0] and pts.shape[1] != len(self.columns):
            raise ValueError(f"LidarScan has {pts.shape[1]} columns, expected {len(self.columns)} {self.columns}")


def ring_ids(measurement, n: int) -> np.ndarray:
    """Ring (beam) id of each of the n points of a CARLA LiDAR measurement.
    Points arrive channel by channel and `get_point_count(channel)` gives the
    counts. -1 for every point when the counts do not add up or the API lacks
    them. Shared by the stack, the probe and the fixture recorder."""
    ring = np.full(n, -1.0, dtype=np.float32)
    try:
        channels = int(measurement.channels)
        counts = [int(measurement.get_point_count(ch)) for ch in range(channels)]
        if sum(counts) == n and n > 0:
            ring = np.repeat(np.arange(channels, dtype=np.float32), counts)
    except Exception:
        pass
    return ring


def decode_lidar(scan) -> np.ndarray:
    """CARLA LidarMeasurement -> Nx5 float32 [x, y, z, intensity, ring]."""
    pts = np.frombuffer(scan.raw_data, dtype=np.float32).reshape((-1, 4))
    n = pts.shape[0]
    out = np.empty((n, 5), dtype=np.float32)
    out[:, :4] = pts
    out[:, 4] = ring_ids(scan, n)
    return out


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

    def __init__(self, world, vehicle, full_sweep: Optional[bool] = None):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []

        # Perception fix 1: whole LiDAR sweeps instead of per-frame wedges.
        self.lidar_full_sweep = lidar_full_sweep_enabled() if full_sweep is None else bool(full_sweep)
        self._sweep = LidarSweepAccumulator(rotation_hz=LIDAR_ROTATION_HZ) if self.lidar_full_sweep else None
        self.lidar_sweep_errors = 0
        self._lidar_enabled = True

        # Latest data (thread-safe via GIL for simple reads)
        self.latest_camera: Optional[CameraFrame] = None
        # Surround views for the operator (observability only, not perception):
        # left / right / rear / top. Front stays in latest_camera.
        self.latest_frames: dict = {}
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

    @property
    def lidar_enabled(self) -> bool:
        return self._lidar_enabled

    @lidar_enabled.setter
    def lidar_enabled(self, value) -> None:
        value = bool(value)
        if value and not self._lidar_enabled and self._sweep is not None:
            self._sweep.reset()        # after a fault-injection drop, start a fresh sweep
        self._lidar_enabled = value

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

        # --- Surround view cameras (operator situational awareness) ---
        # Lower resolution + slower tick: cheap on the GPU, plenty for the dashboard.
        views = [
            ("left",  480, 360, 100, carla.Transform(carla.Location(x=0.0, y=-1.1, z=1.7),
                                                     carla.Rotation(yaw=-90, pitch=-15))),
            ("right", 480, 360, 100, carla.Transform(carla.Location(x=0.0, y=1.1, z=1.7),
                                                     carla.Rotation(yaw=90, pitch=-15))),
            ("rear",  480, 360, 90,  carla.Transform(carla.Location(x=-2.6, z=1.8),
                                                     carla.Rotation(yaw=180, pitch=-12))),
            # bird's-eye: floats above the van looking straight down (~45 m square)
            ("top",   420, 420, 90,  carla.Transform(carla.Location(x=0.0, z=22.0),
                                                     carla.Rotation(pitch=-90))),
        ]
        for view_name, w, h, fov, tf in views:
            bp = bp_lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(w))
            bp.set_attribute('image_size_y', str(h))
            bp.set_attribute('fov', str(fov))
            bp.set_attribute('sensor_tick', '0.15')   # ~7 Hz
            cam = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
            cam.listen(self._make_view_callback(view_name))
            self.sensors.append(cam)

        # --- LiDAR ---
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '150000')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('rotation_frequency', str(int(LIDAR_ROTATION_HZ)))
        # Full-sweep mode needs EVERY frame's wedge (sensor_tick 0.0); the
        # accumulator glues them. sensor_tick 0.1 kept one wedge in ten.
        lidar_bp.set_attribute('sensor_tick', '0.0' if self.lidar_full_sweep else '0.1')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar.listen(self._on_lidar)
        self.sensors.append(lidar)
        print(f"[CarlaSensorAdapter] LiDAR: {'full 360-degree sweeps (accumulated)' if self.lidar_full_sweep else 'raw per-frame deliveries (WARP_LIDAR_SWEEP=0)'}")

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

    def _make_view_callback(self, view_name):
        def _cb(image):
            if not self.camera_enabled:
                return
            # a COPY: raw_data is CARLA's reusable receive buffer, and the frame is
            # read later from another thread (the detector worker, the dashboard)
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4)).copy()
            self.latest_frames[view_name] = CameraFrame(
                image=array, width=image.width, height=image.height,
                fov=float(image.fov), timestamp=time.time()
            )
        return _cb

    def _on_camera(self, image):
        if not self.camera_enabled:
            return
        # a COPY: raw_data is CARLA's reusable receive buffer, and the detector
        # worker reads this frame from its own thread
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4)).copy()
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
        points = decode_lidar(scan)      # Nx5: x, y, z, intensity, ring (our own memory)
        frames, span = 1, 0.0
        sim_time = None
        matrix = None
        try:
            sim_time = float(scan.timestamp)
            matrix = np.asarray(scan.transform.get_matrix(), dtype=np.float64).reshape(4, 4)
        except Exception:
            pass
        if self._sweep is not None:
            try:
                if matrix is None or sim_time is None:
                    raise ValueError("delivery without a usable transform or timestamp")
                points = self._sweep.add(points, matrix, sim_time)     # Nx6: ..., t_rel appended
                frames, span = self._sweep.frames_in_sweep, self._sweep.span_s
            except Exception as e:          # never lose the raw delivery over a bookkeeping error
                self.lidar_sweep_errors += 1
                points = np.c_[points, np.zeros(points.shape[0], dtype=np.float32)]
                matrix = None                # not to be trusted either
                if self.lidar_sweep_errors in (1, 10, 100, 1000):
                    print(f"[CarlaSensorAdapter] LiDAR sweep accumulation failed ({self.lidar_sweep_errors}x): {e}")
        else:
            points = np.c_[points, np.zeros(points.shape[0], dtype=np.float32)]   # t_rel = 0: one delivery
        self.latest_lidar = LidarScan(points=points, timestamp=time.time(), frames=frames, span_s=span,
                                      sim_time=sim_time, sensor_matrix=matrix)
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
