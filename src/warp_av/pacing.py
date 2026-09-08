"""Loop pacing helper (Perception V2 day 1). Kept out of main.py so it can be
unit-tested without importing the whole stack."""
import time


def sleep_remainder(started: float, dt: float) -> float:
    """Sleep whatever is left of a tick period after the work. The loop used to
    sleep the whole period on top of the work (3.7-5 Hz instead of 10).
    `started` is a time.monotonic() stamp. Returns the seconds slept."""
    left = dt - (time.monotonic() - started)
    if left > 0:
        time.sleep(left)
        return left
    time.sleep(0.002)          # an overrunning tick still yields to the sensor and API threads
    return 0.002
