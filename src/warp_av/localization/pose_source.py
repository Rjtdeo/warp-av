"""The one place the stack learns where the van is.

Until Planning V2 the van asked the simulator where it was in THREE independent places,
none of which knew about the others (measured 2026-09-16):

  * `localization.py` -- the official path, feeding planner, behaviour and control;
  * `carla_sensor_adapter.py` -- the LiDAR sweep, which needs the sensor's pose AT CAPTURE
    to glue per-frame wedges into one 360-degree scan without smearing;
  * `camera_lidar_perception.py` -- the occupancy grid's ego-motion, and where blobs are
    placed on it.

Three doors meant that replacing the first one would have left the other two reading truth,
and the second one shapes the point cloud itself -- so perception would still have been built
on the answer we were trying to stop using. This module is the single door.

Today the implementation behind it is still CARLA's own answer (`CarlaTruthPoseSource`), and
that is deliberate: this change is structural, not behavioural. Later an `EstimatedPoseSource`
built from IMU, GNSS, wheel speed and LiDAR odometry takes its place, and nothing above has to
move, because everything above already asks here.

WHY sensor_to_world TAKES A TIMESTAMP
-------------------------------------
A sweep is assembled from wedges captured over about a tenth of a second, and each wedge has
to be placed where the sensor actually was when that wedge was captured -- not where the van
is at the current tick. Using the newest pose would smear every standing object by however far
the van moved inside the rotation. So the interface asks for a pose AT A TIME, not the latest
one. `CarlaTruthPoseSource` answers from what the simulator stamped on the delivery; an
estimator answers by looking that time up in its own pose history and composing it with the
sensor's mount transform. The shape of the question is what matters here: it is the one that
an estimator can still answer honestly.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .localization import Pose, PoseCovariance


class EgoPoseSource:
    """Where the van is, and where a sensor was when it captured something.

    Duck-typed on purpose -- the stack's other collaborators are too, and an ABC here would
    buy nothing but an import.
    """

    def pose(self) -> Pose:
        """The van's pose NOW. Yaw in radians. Never raises: on failure return an unhealthy
        Pose, the way LocalizationSystem always has."""
        raise NotImplementedError

    def sensor_to_world(self, sim_time: float, captured=None) -> Optional[np.ndarray]:
        """The 4x4 sensor-to-world matrix at the moment `sim_time` was captured.

        `captured` is whatever the sensor delivery carried with it -- for CARLA that is the
        transform the simulator stamped on the scan. A truth source reads it; an estimator
        ignores it and looks `sim_time` up in its own history. Return None when the pose for
        that moment is not known, and the caller will fall back to a single un-de-skewed
        delivery rather than smear the sweep.
        """
        raise NotImplementedError


class CarlaTruthPoseSource(EgoPoseSource):
    """The simulator's own answer: exact, instantaneous, never wrong.

    This is not localization. It is the thing localization has to replace, kept behind the
    interface so that replacing it is a one-line change rather than a hunt through perception.

    It takes any object that answers `get_transform()` -- the CARLA actor live, and the
    stand-in adapters the replay harness and the unit tests build. `get_velocity()` is used
    when it exists and treated as a standstill when it does not, because the offline
    stand-ins have a transform but no velocity and perception only ever wanted x, y and yaw.
    """

    def __init__(self, actor, world=None):
        self._actor = actor
        # L3: the simulator's own clock, so the speed this source reports carries the time it
        # was taken rather than the time it was asked for. `get_snapshot` returns the last
        # tick the client already has, so this costs no round trip.
        self._world = world
        if self._world is None:
            try:
                self._world = actor.get_world()
            except Exception:
                self._world = None

    def pose(self) -> Pose:
        try:
            tf = self._actor.get_transform()
            speed = 0.0
            get_velocity = getattr(self._actor, "get_velocity", None)
            if get_velocity is not None:
                v = get_velocity()
                speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            return Pose(
                x=tf.location.x,
                y=tf.location.y,
                z=getattr(tf.location, "z", 0.0),
                yaw=math.radians(tf.rotation.yaw),
                speed=speed,
                cov=PoseCovariance.exact(),
                sim_time=self.sim_time(),
            )
        except Exception as e:                       # noqa: BLE001 - same contract as before
            return Pose(healthy=False, reason=f"POSE_SOURCE_ERROR: {e}",
                        confidence=0.0, cov=PoseCovariance.unknown())

    def sim_time(self) -> Optional[float]:
        """The simulator's clock now, or None where there is no simulator (replay, tests)."""
        try:
            return float(self._world.get_snapshot().timestamp.elapsed_seconds)
        except Exception:
            return None

    def sensor_to_world(self, sim_time: float, captured=None) -> Optional[np.ndarray]:
        """What CARLA stamped on the delivery, which IS the sensor's pose at capture.

        `sim_time` is unused here and that is the honest state of things: the simulator hands
        the answer over with the data, so there is nothing to look up. The argument exists for
        the estimator that will need it.
        """
        if captured is None:
            return None
        try:
            return np.asarray(captured.get_matrix(), dtype=np.float64).reshape(4, 4)
        except Exception:
            return None
