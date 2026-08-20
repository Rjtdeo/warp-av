"""
Troy fix #5 regression tests: no steering oscillation, no random brake taps.

Simulates a van (kinematic bicycle model: wheelbase 3.5 m, max wheel angle
~0.6 rad, 10 Hz — no CARLA needed) driving with VehicleController and the same
speed-scaled lookahead main.py uses, then measures the things Troy complained
about: weave after settling, overshoot, and brake usage while cruising.
"""
import math

from warp_av.control.controller import VehicleController

DT = 0.1
WHEELBASE = 3.5
MAX_WHEEL_ANGLE = 0.6   # rad at steering command 1.0


class SimVan:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, speed=0.0):
        self.x, self.y, self.yaw, self.speed = x, y, yaw, speed

    def step(self, cmd):
        # crude longitudinal model: ~4 m/s^2 full throttle, ~6 m/s^2 full brake, light drag
        accel = 4.0 * cmd.throttle - 6.0 * cmd.brake - 0.05 * self.speed
        self.speed = max(0.0, self.speed + accel * DT)
        wheel = cmd.steering * MAX_WHEEL_ANGLE
        self.yaw += (self.speed / WHEELBASE) * math.tan(wheel) * DT
        self.x += self.speed * math.cos(self.yaw) * DT
        self.y += self.speed * math.sin(self.yaw) * DT


def lookahead_point(path, x, y, speed, start_idx):
    """Same rule as main.py (aim 1.6 s ahead, clamped 5–13 m), but tracking
    progress along the path so curved/wrapping paths pick the right point."""
    la = max(5.0, min(13.0, 1.6 * speed))
    # advance the progress index to the nearest point in a local window
    best_i, best_d = start_idx, float("inf")
    for i in range(start_idx, min(start_idx + 60, len(path))):
        d = math.hypot(path[i][0] - x, path[i][1] - y)
        if d < best_d:
            best_i, best_d = i, d
    for i in range(best_i, len(path)):
        if math.hypot(path[i][0] - x, path[i][1] - y) >= la:
            return path[i], best_i
    return path[-1], best_i


def drive(van, path, desired_speed, seconds):
    ctrl = VehicleController()
    steers, brakes, ys, speeds = [], [], [], []
    idx = 0
    for _ in range(int(seconds / DT)):
        (tx, ty), idx = lookahead_point(path, van.x, van.y, van.speed, idx)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed,
                                   tx, ty, desired_speed, should_stop=False)
        van.step(cmd)
        steers.append(cmd.steering)
        brakes.append(cmd.brake)
        ys.append(van.y)
        speeds.append(van.speed)
    return steers, brakes, ys, speeds


def sign_changes(values, eps=0.02):
    signs = [1 if v > eps else -1 if v < -eps else 0 for v in values]
    signs = [s for s in signs if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def test_straight_road_offset_converges_without_weave():
    # Van starts 2 m left of a straight road, already at cruise speed.
    path = [(float(i), 0.0) for i in range(0, 400)]
    van = SimVan(x=0.0, y=2.0, yaw=0.0, speed=8.0)
    steers, brakes, ys, _ = drive(van, path, desired_speed=8.0, seconds=20)

    settle = int(8.0 / DT)                    # after 8 s it must be settled
    assert abs(ys[-1]) < 0.5, f"did not converge to the lane: final offset {ys[-1]:.2f} m"
    assert max(abs(y) for y in ys[settle:]) < 0.6, "still deviating after settling"
    weave = sign_changes(steers[settle:])
    assert weave <= 4, f"steering still oscillating after settling: {weave} sign flips in 12 s"


def test_no_brake_taps_while_cruising():
    # Straight road, correct lane, speed step 0 -> 8 m/s: accelerating and
    # holding cruise must not touch the brakes.
    path = [(float(i), 0.0) for i in range(0, 400)]
    van = SimVan(speed=0.0)
    _, brakes, _, speeds = drive(van, path, desired_speed=8.0, seconds=25)

    assert max(speeds) < 9.0, f"overshoot too big: {max(speeds):.2f} m/s"
    assert abs(speeds[-1] - 8.0) < 0.7, f"did not hold cruise: {speeds[-1]:.2f} m/s"
    brake_ticks = sum(1 for b in brakes if b > 0.01)
    assert brake_ticks == 0, f"random brake taps while cruising: {brake_ticks} ticks"


def test_curve_tracking_stays_in_lane():
    # 20 m radius curve at 5 m/s: stay within 1.5 m of the arc, no flip-flop storm.
    R = 20.0
    path = [(R * math.sin(t / 100.0), R - R * math.cos(t / 100.0)) for t in range(0, 700)]
    van = SimVan(speed=5.0)
    steers, _, _, _ = drive(van, path, desired_speed=5.0, seconds=18)
    # cross-track error against the circle centred at (0, R)
    err = abs(math.hypot(van.x - 0.0, van.y - R) - R)
    assert err < 1.5, f"left the curve: cross-track {err:.2f} m"
    assert sign_changes(steers[int(4 / DT):]) <= 6, "steering flip-flopping through the curve"


def test_stop_is_still_instant_and_resets_smoothing():
    ctrl = VehicleController()
    # build up steering state first
    for _ in range(5):
        ctrl.compute_command(0, 0, 0.0, 8.0, 10, 5, 8.0, should_stop=False)
    cmd = ctrl.compute_command(0, 0, 0.0, 8.0, 10, 5, 8.0, should_stop=True)
    assert cmd.brake == 1.0 and cmd.throttle == 0.0 and cmd.steering == 0.0
    assert ctrl._last_steer == 0.0, "smoothing state must reset on stop"


def test_low_speed_keeps_full_authority():
    # At parking speed the gain schedule must NOT weaken steering.
    ctrl = VehicleController()
    cmd = None
    for _ in range(10):   # let the filter converge
        cmd = ctrl.compute_command(0, 0, 0.0, 1.5, 3, 3, 2.0, should_stop=False)
    assert abs(cmd.steering) > 0.8, f"low-speed steering too weak: {cmd.steering:.2f}"


def test_normal_slowing_is_capped_but_stop_is_full_brake():
    ctrl = VehicleController()
    # cruise 8 m/s, target suddenly 3 m/s (slow zone entry): brake must be firm but capped
    cmd = ctrl.compute_command(0, 0, 0.0, 8.0, 20, 0, 3.0, should_stop=False)
    assert 0.0 < cmd.brake <= ctrl.SERVICE_BRAKE_CAP, f"service brake not capped: {cmd.brake}"
    # obstacle/e-stop path is untouched: full brake
    cmd = ctrl.compute_command(0, 0, 0.0, 8.0, 20, 0, 0.0, should_stop=True)
    assert cmd.brake == 1.0
