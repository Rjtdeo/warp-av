"""Loop pacing helper (Perception V2 day 1). Kept out of main.py so it can be
unit-tested without importing the whole stack."""
import time


def sleep_remainder(started: float, dt: float) -> float:
    """Sleep whatever is left of a tick period after the work. The loop used to
    sleep the whole period on top of the work (3.7-5 Hz instead of 10).
    `started` is a time.perf_counter() stamp -- monotonic, and fine on every machine:
    Windows' time.monotonic() steps in 15.6 ms lumps, which is 15% of a 10 Hz tick spent
    pacing by guesswork. Returns the seconds slept."""
    left = dt - (time.perf_counter() - started)
    if left > 0:
        time.sleep(left)
        return left
    time.sleep(0.002)          # an overrunning tick still yields to the sensor and API threads
    return 0.002
