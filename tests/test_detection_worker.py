"""
Perception V2 day 1 (fix 5): the camera detector runs in its own thread; the
driving loop reads the newest result and never waits for it.
"""
import os
import threading
import time
import types

import numpy as np

from warp_av.perception.detection_worker import DetectionWorker, yolox_inline_from_env

PERC = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "perception", "camera_lidar_perception.py")
MAIN = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "main.py")


class Frames:
    """A camera that produces a new frame every `period` seconds."""

    def __init__(self, period=0.05):
        self.period = period
        self.t0 = time.time()
        self.image = np.zeros((6, 8, 4), dtype=np.uint8)

    def latest(self):
        k = int((time.time() - self.t0) / self.period)
        return types.SimpleNamespace(image=self.image, timestamp=self.t0 + k * self.period)


def test_env_flag():
    assert yolox_inline_from_env({}) is False
    assert yolox_inline_from_env({"WARP_YOLO_INLINE": "1"}) is True
    assert yolox_inline_from_env({"WARP_YOLO_INLINE": "off"}) is False


def test_worker_runs_on_new_frames_only_and_never_blocks_the_caller():
    seen = []
    lock = threading.Lock()

    def slow_detect(img):
        with lock:
            seen.append(img.shape)
        time.sleep(0.03)                          # a "slow model"
        return ["box"]

    frames = Frames(period=0.05)
    w = DetectionWorker(slow_detect, frames.latest, interval_s=0.05, name="t")
    assert w.latest() == ([], float("inf"))       # nothing yet
    w.start()
    t0 = time.perf_counter()
    for _ in range(20):                           # the caller keeps looping while the model works
        w.latest()
    assert time.perf_counter() - t0 < 0.05
    time.sleep(0.4)
    dets, age = w.latest(max_age_s=1.0)
    assert dets == ["box"] and age < 0.3
    assert 3 <= w.runs <= 10                      # ~0.4 s / (0.05 interval + 0.03 work)
    assert w.last_inference_ms >= 25
    assert all(shape == (6, 8, 3) for shape in seen)   # BGRA trimmed to 3 channels
    w.stop()
    assert not w.running


def test_worker_skips_a_frame_it_has_seen():
    calls = []
    frame = types.SimpleNamespace(image=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=123.0)
    w = DetectionWorker(lambda img: calls.append(1) or [], lambda: frame, interval_s=0.01)
    w.start()
    time.sleep(0.15)
    w.stop()
    assert len(calls) == 1                        # same timestamp every time: one run


def test_stale_result_is_withheld_and_errors_are_counted():
    frames = Frames(period=0.02)
    w = DetectionWorker(lambda img: ["x"], frames.latest, interval_s=0.01)
    w.start()
    time.sleep(0.1)
    w.stop()
    dets, age = w.latest(max_age_s=1.0)
    assert dets == ["x"]
    dets, age = w.latest(max_age_s=-1.0)          # anything at all is stale
    assert dets == [] and age >= 0.0

    def bad(img):
        raise RuntimeError("model exploded")

    w2 = DetectionWorker(bad, frames.latest, interval_s=0.01)
    w2.start()
    time.sleep(0.1)
    w2.stop()
    assert w2.errors >= 2 and w2.consecutive_errors == w2.errors and w2.latest()[0] == []
    # a success resets the streak
    flaky = {"n": 0}

    def sometimes(img):
        flaky["n"] += 1
        if flaky["n"] % 2:
            raise RuntimeError("x")
        return ["ok"]
    w3 = DetectionWorker(sometimes, Frames(period=0.02).latest, interval_s=0.01)
    w3.start()
    time.sleep(0.15)
    w3.stop()
    assert w3.errors >= 1 and w3.consecutive_errors <= 1


def test_perception_and_loop_wiring():
    src = open(PERC).read()
    assert "if self.yolox_inline:" in src
    assert "self._worker = DetectionWorker(self.detector.detect," in src
    assert "self._worker.latest(self.detection_max_age_s)" in src
    assert "if self._worker.consecutive_errors >= self.detector_fail_after:" in src
    assert "return dataclasses.replace(self._last_output, timestamp=now)" in src
    main = open(MAIN).read()
    assert "sleep_remainder(started, dt)" in main
    assert '"loop_hz": round(self._loop_hz, 1) if self._loop_hz else None' in main
    assert main.count("self.camera_lidar_perception.close()") >= 2    # ground-truth switch and shutdown


def test_lidar_scan_carries_sim_time_and_camera_frames_are_copies():
    from warp_av.adapters.carla_sensor_adapter import CarlaSensorAdapter, LidarScan
    import inspect
    src = inspect.getsource(CarlaSensorAdapter)
    assert src.count(".reshape((image.height, image.width, 4)).copy()") == 2
    assert LidarScan(points=np.zeros((0, 6), np.float32)).sim_time is None
    assert "sim_time=sim_time" in src


def test_sleep_remainder_only_sleeps_what_is_left():
    from warp_av.pacing import sleep_remainder
    t = time.monotonic() - 0.06                   # 60 ms of work already done in a 100 ms tick
    t0 = time.perf_counter()
    slept = sleep_remainder(t, 0.1)
    elapsed = time.perf_counter() - t0
    assert 0.02 <= slept <= 0.045 and elapsed < 0.09
    assert sleep_remainder(time.monotonic() - 0.5, 0.1) <= 0.005  # already late: only the tiny yield
