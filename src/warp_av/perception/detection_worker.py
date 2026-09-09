"""
Camera detection in its own thread (Perception V2, day 1 / fix 5).

The camera model (YOLOX on the CPU) takes 100-180 ms per picture. Run inside
the driving loop it held the van to 4-5 decisions a second. Here it runs in
a background thread on the newest camera frame, at most once per
`interval_s`, and the loop only ever reads the latest finished result.

Rules the thread keeps:
  * it never runs on the same frame twice;
  * a detector error is counted and logged (rate-limited), never raised;
  * `latest()` hands back an empty list once the result is older than
    `max_age_s`, so a stalled detector cannot leave stale boxes in play.

The detector object is only ever touched from this thread (OpenCV DNN nets
are not thread-safe), so callers must not run it inline at the same time.
No CARLA, no OpenCV imports here: testable with any callable.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional, Tuple


def yolox_inline_from_env(env=None) -> bool:
    """WARP_YOLO_INLINE=1 restores the old behaviour (detector inside the loop)."""
    env = os.environ if env is None else env
    return str(env.get("WARP_YOLO_INLINE", "0")).strip().lower() in ("1", "true", "on", "yes")


class DetectionWorker:
    LOG_EVERY_S = 10.0

    def __init__(self, detect_fn: Callable, frame_fn: Callable, interval_s: float = 0.25,
                 name: str = "detector", views=None):
        """`views`: {name: frame_fn}. Given them, the detector takes them in TURN -- one
        picture per pass -- and keeps the newest answer for each separately.

        One detector, several cameras. The van has five cameras and until day 13 only the
        front one was ever used to name anything; a person standing beside the van came out
        as an unnamed lump with a bin's worth of room. Running a second detector would cost
        another lot of computer time the van has not got. It does not need one: the laser has
        already FOUND everything all round, so all that is missing is the name, and a name
        does not change from second to second. So the same detector looks at a different
        camera each pass and every view's answer is held until that view comes round again.
        """
        self._detect = detect_fn            # image (H, W, 3) -> detections
        self._frame = frame_fn              # () -> object with .image and .timestamp, or None
        self._views = dict(views) if views else None
        self._order = list(self._views) if self._views else None
        self._turn = 0
        self.interval_s = float(interval_s)
        self.name = name
        self._lock = threading.Lock()
        self._result: Tuple[list, Optional[float], float] = ([], None, 0.0)   # detections, frame ts, published (monotonic)
        self._by_view = {}                  # view -> (detections, frame ts, published)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_inference_ms = 0.0
        self.runs = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.started_at = None              # monotonic, set by start()
        self._last_log = 0.0

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- the thread ------------------------------------------------------
    def _loop(self) -> None:
        last_frame_ts = None
        next_run = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_run:
                self._stop.wait(min(0.01, next_run - now))
                continue
            view = None
            if self._order:
                view = self._order[self._turn % len(self._order)]
                self._turn += 1
                frame = self._views[view]()
            else:
                frame = self._frame()
            ts = getattr(frame, "timestamp", None) if frame is not None else None
            if frame is None or (view is None and ts == last_frame_ts):
                self._stop.wait(0.01)                 # nothing new yet
                continue
            t0 = time.perf_counter()
            try:
                image = frame.image
                detections = list(self._detect(image[:, :, :3] if image.ndim == 3 and image.shape[2] > 3 else image))
                self.consecutive_errors = 0
            except Exception as e:                  # never kill the thread over one bad frame
                self.errors += 1
                self.consecutive_errors += 1
                detections = []
                if time.monotonic() - self._last_log >= self.LOG_EVERY_S:
                    self._last_log = time.monotonic()
                    print(f"[{self.name}] detection failed ({self.errors} so far): {e}")
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                published = time.monotonic()
                if view is None or view == "front":
                    self._result = (detections, ts, published)
                if view is not None:
                    self._by_view[view] = (detections, ts, published)
                self.last_inference_ms = ms
                self.runs += 1
            last_frame_ts = ts
            next_run = now + self.interval_s

    def latest_by_view(self, max_age_s: float = 1.0):
        """{view: (detections, age_s)} for every camera that has an answer worth using."""
        out = {}
        with self._lock:
            snapshot = dict(self._by_view)
        now = time.monotonic()
        for view, (detections, _, published) in snapshot.items():
            age = now - published
            out[view] = ([], age) if age > max_age_s else (detections, age)
        return out

    # ---- what the loop reads ---------------------------------------------
    def latest(self, max_age_s: float = 1.0) -> Tuple[List, float]:
        """(detections, age_s). Empty once the result is older than max_age_s
        or nothing has been produced yet (age = inf)."""
        with self._lock:
            detections, _, published = self._result
        if not published:
            return [], float("inf")
        age = time.monotonic() - published
        if age > max_age_s:
            return [], age
        return detections, age
