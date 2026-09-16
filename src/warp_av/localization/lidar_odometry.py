"""How far the van turned, measured against the world itself.

Phase L5. Everything the estimator has been told about heading so far comes from one of two
places, and neither of them can answer the question on its own:

    GNSS + motion model  ->  the direction the van TRAVELLED
    compass              ->  the direction the van POINTS, plus an unknown offset

L4.1 showed what happens when only those two exist: a bias state asked to reconcile them
converged onto compass-offset-minus-sideslip, roughly twice the offset actually injected,
and heading got worse. There was no third opinion to break the tie.

This is the third opinion. Aligning one LiDAR sweep to the next measures how the van moved
relative to the STANDING WORLD -- buildings, poles, kerbs, parked cars. It owes nothing to
the compass and nothing to the direction of travel.

WHY IT ONLY WORKS ON A TRUTH-FREE SWEEP. Until this phase the sweep was assembled using the
pose CARLA stamped on each delivery, so the cloud already encoded the van's exact motion.
Registering two such clouds would have recovered the simulator's own answer and called it a
measurement. adapters/lidar_sweep.py now assembles from gyro and wheel speed instead
(localization/ego_motion.py), which is what makes this honest.

WHY 2-D. Three degrees of freedom, x, y and yaw. A van on a road is planar over the tenth of
a second between sweeps, and solving for roll, pitch and z as well would hand three badly
observed directions somewhere to hide error -- they would absorb residual that belongs to yaw.

WHY POINT-TO-LINE, NOT POINT-TO-POINT. The first build here was plain point-to-point, and it
was wrong in a way worth writing down, because it is the same disease this project keeps
finding: it was confidently wrong. Two symptoms, one cause.

  * It UNDER-READ rotation against a long smooth wall. Nearest-neighbour correspondences slide
    freely along a featureless surface, so part of the turn simply goes unnoticed.
  * Driving down a corridor it read 0.42 m of a 1.00 m step and still called the fit good,
    because a point-to-point information matrix credits every correspondence with pinning BOTH
    axes. A point on a wall running alongside you pins nothing in the direction you are going.

Measuring the residual along the local surface NORMAL fixes both. A point may slide along the
wall it belongs to at no cost, which is the truth of the matter, and the information matrix
built from those same normals is then empty in exactly the directions the scene cannot see --
so a corridor now SAYS it cannot measure along-track motion instead of guessing at it.

In two dimensions the normals are cheap and stable: a local line fit over a handful of
neighbours, not a 3-D surface normal off 32 sparse beams, which really would be mostly noise.
NDT was not chosen because its score surface is harder to turn into an honest covariance.

IT DRIVES NOTHING. Shadow mode: the measurement is published for scoring and nothing reads it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:                                  # pragma: no cover - exercised on the rig
    cKDTree = None


class _Neighbours:
    """Nearest-neighbour lookup over a small 2-D cloud, with or without scipy.

    WHY THIS EXISTS. The first version imported scipy directly. It ran on the development
    machine, where scipy happens to be installed as somebody else's transitive dependency, and
    refused every single registration on the CARLA rig, where it is not -- reporting
    NO_CORRESPONDENCE, which reads as "the scene did not match" rather than "the maths never
    ran". A missing library should not be able to disguise itself as a geometric failure.

    So the cloud is searched with numpy when scipy is absent. The clouds here are small by
    construction -- a few hundred points after voxel thinning -- and a bucketed brute-force
    search over that is comfortably inside the 100 ms a 10 Hz sweep allows, so nothing is
    given up by not requiring the dependency.
    """

    def __init__(self, xy: np.ndarray):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.tree = cKDTree(self.xy) if cKDTree is not None else None

    def query(self, pts: np.ndarray, radius: float):
        """(distance, index) of the nearest point to each of `pts`; distance is inf and index
        out of range where nothing lies within `radius`."""
        if self.tree is not None:
            return self.tree.query(pts, k=1, distance_upper_bound=radius)
        n = self.xy.shape[0]
        if n == 0 or pts.shape[0] == 0:
            return np.full(pts.shape[0], np.inf), np.zeros(pts.shape[0], dtype=np.int64)
        best_d = np.full(pts.shape[0], np.inf)
        best_i = np.zeros(pts.shape[0], dtype=np.int64)
        step = max(1, int(2_000_000 // max(1, n)))          # cap the working matrix
        for a in range(0, pts.shape[0], step):
            blk = pts[a:a + step]
            d2 = ((blk[:, None, 0] - self.xy[None, :, 0]) ** 2
                  + (blk[:, None, 1] - self.xy[None, :, 1]) ** 2)
            j = np.argmin(d2, axis=1)
            best_d[a:a + step] = np.sqrt(d2[np.arange(blk.shape[0]), j])
            best_i[a:a + step] = j
        far = best_d > radius
        best_d[far] = np.inf
        best_i[far] = n                                      # out of range, like cKDTree
        return best_d, best_i

    def knn(self, k: int):
        """Indices of the k nearest points to every point in the cloud, itself included."""
        n = self.xy.shape[0]
        k = min(k, n)
        if self.tree is not None:
            return self.tree.query(self.xy, k=k)[1]
        out = np.zeros((n, k), dtype=np.int64)
        step = max(1, int(2_000_000 // max(1, n)))
        for a in range(0, n, step):
            blk = self.xy[a:a + step]
            d2 = ((blk[:, None, 0] - self.xy[None, :, 0]) ** 2
                  + (blk[:, None, 1] - self.xy[None, :, 1]) ** 2)
            out[a:a + step] = np.argpartition(d2, k - 1, axis=1)[:, :k]
        return out


# ---- what gets registered ----------------------------------------------------------------
#: Points nearer than this are mostly road and the van's own body; further than this the beams
#: are too sparse to correspond reliably between sweeps.
MIN_RANGE_M = 3.0
MAX_RANGE_M = 40.0
#: Height band above the sensor, in metres. Below the lower edge is road; above the upper edge
#: is mostly foliage and wires, which move in the wind and match badly.
MIN_Z_M = -2.0
MAX_Z_M = 1.0
#: One point per this many metres of grid, so a dense near wall cannot outvote everything else.
VOXEL_M = 0.35

# ---- the fit -----------------------------------------------------------------------------
MAX_ITERS = 25
#: A correspondence further apart than this is not the same bit of world.
MAX_CORRESPONDENCE_M = 2.0
#: Neighbours used for the local line fit that gives each target point its normal.
NORMAL_NEIGHBOURS = 8
#: A neighbourhood flatter than this is a line and its normal is trustworthy; rounder than
#: this (a bush, a corner, sparse clutter) and the normal is arbitrary, so the correspondence
#: falls back to a full point-to-point residual, which for a corner is the right cost anyway.
MIN_PLANARITY = 0.55
#: Huber threshold, in robust scales. Residuals inside it are scored normally; outside it
#: their influence grows linearly rather than quadratically, so a moving car bends the fit a
#: little instead of dominating it.
#:
#: This replaced a hard "keep the best 80 %" trim, which was WRONG in a way worth recording.
#: Trimming ranks by residual and throws away the tail -- but when the fit is still off, the
#: points that would CORRECT it are the ones with the largest residuals. Driving a corridor
#: with the estimate 0.5 m short, every point on the walls slid along them for free and scored
#: well, while the parked-car end faces that alone could see the error scored badly and were
#: discarded. The fit then settled at zero and claimed 4 mm of uncertainty, wrong by 0.49 m.
#: A robust weight keeps those points and lets them pull.
HUBER_K = 1.5
#: Floor under the robust scale, so a near-perfect fit does not make every small residual an
#: outlier and start chasing noise.
MIN_ROBUST_SCALE_M = 0.05
#: Converged once an iteration moves the answer less than this.
TOL_M = 1e-4
TOL_RAD = 1e-5

# ---- when to refuse ----------------------------------------------------------------------
MIN_POINTS = 150
MIN_CORRESPONDENCES = 80
MIN_INLIER_RATIO = 0.35
MAX_RMSE_M = 0.60
#: Yaw is only observable if the matched geometry is spread around the van. All of it in one
#: direction and a rotation looks like a translation (the aperture problem). This is the
#: smallest eigenvalue of the normalised information matrix, and it is the honest way to say
#: "there was not enough shape here to be sure".
MIN_YAW_INFORMATION = 0.05
#: Longer than this between sweeps and the pair is not worth registering.
MAX_DT_S = 0.5


@dataclass
class LidarOdometryMeasurement:
    """Relative motion between two sweeps. A MEASUREMENT, with its own opinion of itself."""
    sim_time: float                  # the newer sweep's timestamp
    dt: float
    delta_x: float = 0.0             # metres, in the older sweep's frame
    delta_y: float = 0.0
    delta_yaw: float = 0.0           # radians
    # --- quality, all of it measured, none of it assumed ---
    fitness: float = 0.0             # share of source points that found a correspondence
    inlier_ratio: float = 0.0        # share kept after trimming and gating
    rmse_m: float = float("inf")     # residual of the kept correspondences
    correspondences: int = 0
    iterations: int = 0
    converged: bool = False
    yaw_information: float = 0.0     # how much shape there was to pin the rotation
    valid: bool = False
    reason: str = "NOT_RUN"
    #: 3x3 over (dx, dy, dyaw) in m^2, m^2, rad^2. None when the fit was refused.
    cov: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def sigma_yaw_rad(self) -> float:
        if self.cov is None:
            return float("nan")
        v = float(self.cov[2, 2])
        return math.sqrt(v) if v > 0 else float("nan")

    @property
    def delta_yaw_deg(self) -> float:
        return math.degrees(self.delta_yaw)

    def as_dict(self) -> dict:
        return {"sim_time": round(self.sim_time, 4), "dt": round(self.dt, 4),
                "dx": round(self.delta_x, 4), "dy": round(self.delta_y, 4),
                "dyaw_deg": round(self.delta_yaw_deg, 4),
                "fitness": round(self.fitness, 3), "inlier_ratio": round(self.inlier_ratio, 3),
                "rmse_m": round(self.rmse_m, 4) if math.isfinite(self.rmse_m) else None,
                "correspondences": self.correspondences, "iterations": self.iterations,
                "converged": self.converged,
                "yaw_information": round(self.yaw_information, 4),
                "sigma_yaw_deg": (round(math.degrees(self.sigma_yaw_rad), 4)
                                  if self.cov is not None else None),
                "valid": self.valid, "reason": self.reason}


def _voxel_thin(xy: np.ndarray, cell: float) -> np.ndarray:
    """One point per cell. Keeps a dense nearby wall from outvoting the rest of the scene,
    and makes the cost of a sweep roughly independent of how close the van is to something."""
    if xy.shape[0] == 0:
        return xy
    keys = np.floor(xy / cell).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xy[np.sort(idx)]


def prepare(points: np.ndarray) -> np.ndarray:
    """Sweep -> the Nx2 the registration actually sees.

    Ground and canopy go by HEIGHT, not by any label: this has to work off geometry alone,
    because a real van has no list of which returns came from the road.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return np.zeros((0, 2))
    z = pts[:, 2]
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = (r >= MIN_RANGE_M) & (r <= MAX_RANGE_M) & (z >= MIN_Z_M) & (z <= MAX_Z_M)
    return _voxel_thin(pts[keep][:, :2], VOXEL_M)


def normals_2d(xy: np.ndarray, k: int = NORMAL_NEIGHBOURS):
    """A local line direction for every point, as (normal, planarity).

    `planarity` is 1 - lambda_min/lambda_max of the neighbourhood scatter: 1 for points strung
    out along a wall, 0 for a round blob where no direction is special. It is what decides
    whether a point gets a point-to-line residual or a point-to-point one.
    """
    n = xy.shape[0]
    if n < k + 1:
        return np.zeros((n, 2)), np.zeros(n)
    idx = _Neighbours(xy).knn(k)
    nb = xy[idx]                                     # (n, k, 2)
    nb = nb - nb.mean(axis=1, keepdims=True)
    # per-point 2x2 scatter, done as one batched product
    cxx = np.einsum("ij,ij->i", nb[:, :, 0], nb[:, :, 0])
    cyy = np.einsum("ij,ij->i", nb[:, :, 1], nb[:, :, 1])
    cxy = np.einsum("ij,ij->i", nb[:, :, 0], nb[:, :, 1])
    tr = cxx + cyy
    det = cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(0.0, tr ** 2 / 4.0 - det))
    lmax = tr / 2.0 + disc
    lmin = tr / 2.0 - disc
    # the normal is the eigenvector of the SMALLER eigenvalue
    nx = cxy
    ny = lmin - cxx
    norm = np.hypot(nx, ny)
    flat = norm < 1e-12
    nx = np.where(flat, 1.0, nx / np.where(flat, 1.0, norm))
    ny = np.where(flat, 0.0, ny / np.where(flat, 1.0, norm))
    planarity = np.where(lmax > 1e-12, 1.0 - lmin / np.maximum(lmax, 1e-12), 0.0)
    return np.c_[nx, ny], planarity


def _kabsch_2d(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """The rigid (yaw, dx, dy) taking `src` onto `dst`, in closed form."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    p, q = src - cs, dst - cd
    theta = math.atan2(float(np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])),
                       float(np.sum(p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1])))
    c, s = math.cos(theta), math.sin(theta)
    t = cd - np.array([c * cs[0] - s * cs[1], s * cs[0] + c * cs[1]])
    return theta, float(t[0]), float(t[1])


def _apply(xy: np.ndarray, theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    out = np.empty_like(xy)
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1] + tx
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1] + ty
    return out


def register(source: np.ndarray, target: np.ndarray,
             init: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    """Align `source` onto `target`, both Nx2. Returns the fit and everything about it.

    WHY THE ROBUST WEIGHT IS THE DYNAMIC FILTER. A van, a cyclist and a walking pedestrian do
    not move with the rest of the scene, so after a rigid fit their residuals stay large while
    everything else settles. Huber weighting pushes their influence down towards nothing
    WITHOUT needing to know what they are -- no labels, no truth, no classifier, and nothing
    that could reach for a CARLA actor list. What it cannot survive is a scene where the movers
    ARE the majority, and `inlier_ratio` and `rmse_m` are what say so when that happens.
    """
    out = {"theta": 0.0, "tx": 0.0, "ty": 0.0, "fitness": 0.0, "inlier_ratio": 0.0,
           "rmse": float("inf"), "correspondences": 0, "iterations": 0, "converged": False,
           "yaw_information": 0.0, "cov": None}
    if source.shape[0] < MIN_POINTS or target.shape[0] < MIN_POINTS:
        return out
    tree = _Neighbours(target)
    nrm, planarity = normals_2d(target)
    flat = planarity >= MIN_PLANARITY
    theta, tx, ty = init
    H = None
    sigma2 = None
    for it in range(MAX_ITERS):
        c, s_ = math.cos(theta), math.sin(theta)
        u = np.c_[c * source[:, 0] - s_ * source[:, 1],
                  s_ * source[:, 0] + c * source[:, 1]]          # R(theta) p
        moved = u + np.array([tx, ty])
        dist, idx = tree.query(moved, MAX_CORRESPONDENCE_M)
        ok = np.isfinite(dist)
        n_corr = int(ok.sum())
        out["fitness"] = n_corr / float(source.shape[0])
        if n_corr < MIN_CORRESPONDENCES:
            out["iterations"] = it
            return out
        sel = ok
        m = int(sel.sum())
        if m < MIN_CORRESPONDENCES:
            out["iterations"] = it
            return out
        q = target[np.clip(idx[sel], 0, target.shape[0] - 1)]
        d = moved[sel] - q                       # the full point-to-point offset
        us = u[sel]
        lever = np.c_[-us[:, 1], us[:, 0]]       # d(R p)/dtheta
        ci = np.clip(idx[sel], 0, target.shape[0] - 1)
        fsel = flat[ci]
        ns = nrm[ci]

        # A point on a wall is free to slide ALONG it: score only the across-wall part.
        # A corner has no meaningful normal, and for a corner the full offset IS the right
        # cost, so those correspondences keep both rows.
        rows_J, rows_r = [], []
        if fsel.any():
            n_f, d_f, l_f = ns[fsel], d[fsel], lever[fsel]
            rows_r.append(np.einsum("ij,ij->i", n_f, d_f))
            rows_J.append(np.c_[n_f[:, 0], n_f[:, 1], np.einsum("ij,ij->i", n_f, l_f)])
        if (~fsel).any():
            d_p, u_p = d[~fsel], us[~fsel]
            n_p = int(d_p.shape[0])
            J_p = np.zeros((2 * n_p, 3))
            J_p[0::2, 0] = 1.0
            J_p[1::2, 1] = 1.0
            J_p[0::2, 2] = -u_p[:, 1]
            J_p[1::2, 2] = u_p[:, 0]
            rows_J.append(J_p)
            rows_r.append(np.c_[d_p[:, 0], d_p[:, 1]].reshape(-1))
        J = np.concatenate(rows_J, axis=0)
        r = np.concatenate(rows_r, axis=0)
        # robust scale from the median absolute residual: a estimate of spread that a handful
        # of moving objects cannot inflate the way a mean square would
        scale = max(MIN_ROBUST_SCALE_M,
                    1.4826 * float(np.median(np.abs(r - np.median(r)))))
        a = np.abs(r) / scale
        wt = np.where(a <= HUBER_K, 1.0, HUBER_K / np.maximum(a, 1e-9))
        out["inlier_ratio"] = float(np.mean(a <= HUBER_K))
        Jw = J * wt[:, None]
        H = Jw.T @ J
        g = Jw.T @ r
        try:
            delta = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            out["iterations"] = it
            return out
        tx += float(delta[0])
        ty += float(delta[1])
        theta = math.atan2(math.sin(theta + float(delta[2])), math.cos(theta + float(delta[2])))
        sigma2 = max(1e-8, float(np.sum(wt * r ** 2) / max(1.0, float(np.sum(wt)))))
        out.update(theta=theta, tx=tx, ty=ty, correspondences=m,
                   rmse=float(math.sqrt(float(np.mean(np.sum(d ** 2, axis=1))))),
                   iterations=it + 1)
        if math.hypot(float(delta[0]), float(delta[1])) < TOL_M and abs(float(delta[2])) < TOL_RAD:
            out["converged"] = True
            break

    if H is None:
        return out
    # --- how well was this pinned down? ---
    # Built from the SAME normals as the fit, so a direction the scene cannot see is a
    # direction with no information in it. A corridor has every normal across the road, which
    # leaves the along-road column empty and the smallest eigenvalue near zero -- which is the
    # honest answer, and the one the point-to-point version could not give.
    lev2 = max(1e-9, float(np.mean(np.sum(us ** 2, axis=1))))
    n_rows = max(1, int(J.shape[0]))
    Hn = H / n_rows
    root = math.sqrt(lev2)
    Hn[0, 2] = Hn[2, 0] = Hn[0, 2] / root
    Hn[1, 2] = Hn[2, 1] = Hn[1, 2] / root
    Hn[2, 2] = Hn[2, 2] / lev2
    try:
        out["yaw_information"] = float(np.min(np.linalg.eigvalsh(Hn)))
    except np.linalg.LinAlgError:
        out["yaw_information"] = 0.0
    try:
        out["cov"] = (sigma2 or 1e-8) * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        out["cov"] = None
    return out


class LidarOdometry:
    """Feed it every sweep; get back how the van moved since the last one.

    Shadow only. Nothing that steers, plans or brakes reads a number this produces.
    """

    def __init__(self, keep_recent: int = 16):
        self._prev: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self.last: Optional[LidarOdometryMeasurement] = None
        # Every measurement since the last few reports, not just the newest. A sweep is
        # registered at about 10 Hz and the evidence recorder samples at 5, so reporting only
        # `last` would silently drop half of them -- and a scoring run that sees half the
        # measurements cannot say what the accumulated drift was.
        self.recent: deque = deque(maxlen=int(keep_recent))
        self.attempts = 0
        self.valid_count = 0
        self.refusals: dict = {}

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None

    def _refuse(self, m: LidarOdometryMeasurement, why: str) -> LidarOdometryMeasurement:
        m.valid = False
        m.reason = why
        self.refusals[why] = self.refusals.get(why, 0) + 1
        return m

    def update(self, points: np.ndarray, sim_time: float) -> Optional[LidarOdometryMeasurement]:
        """One sweep in. None until there is a previous sweep to register against."""
        cur = prepare(points)
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = cur, float(sim_time)
        if prev is None or prev_t is None:
            return None
        dt = float(sim_time) - prev_t
        m = LidarOdometryMeasurement(sim_time=float(sim_time), dt=dt)
        self.attempts += 1
        self.last = m
        self.recent.append(m)
        if not (0.0 < dt <= MAX_DT_S):
            return self._refuse(m, "BAD_DT")
        if cur.shape[0] < MIN_POINTS or prev.shape[0] < MIN_POINTS:
            return self._refuse(m, "TOO_FEW_POINTS")
        # register the NEWER sweep onto the OLDER one: the result is then the motion of the
        # van from the old frame to the new one, which is the direction the caller expects.
        fit = register(cur, prev)
        m.fitness = fit["fitness"]
        m.inlier_ratio = fit["inlier_ratio"]
        m.rmse_m = fit["rmse"]
        m.correspondences = fit["correspondences"]
        m.iterations = fit["iterations"]
        m.converged = fit["converged"]
        m.yaw_information = fit["yaw_information"]
        m.cov = fit["cov"]
        m.delta_yaw = float(fit["theta"])
        m.delta_x = float(fit["tx"])
        m.delta_y = float(fit["ty"])
        if m.correspondences < MIN_CORRESPONDENCES:
            return self._refuse(m, "NO_CORRESPONDENCE")
        if not m.converged:
            return self._refuse(m, "NO_CONVERGENCE")
        if m.inlier_ratio < MIN_INLIER_RATIO:
            return self._refuse(m, "LOW_INLIERS")
        if not math.isfinite(m.rmse_m) or m.rmse_m > MAX_RMSE_M:
            return self._refuse(m, "HIGH_RESIDUAL")
        if m.yaw_information < MIN_YAW_INFORMATION:
            return self._refuse(m, "WEAK_GEOMETRY")
        if m.cov is None or not np.isfinite(m.cov).all():
            return self._refuse(m, "NO_COVARIANCE")
        m.valid = True
        m.reason = "OK"
        self.valid_count += 1
        return m

    def state(self) -> dict:
        d = {"attempts": self.attempts, "valid": self.valid_count,
             "failure_rate": (round(1.0 - self.valid_count / self.attempts, 4)
                              if self.attempts else None),
             "refusals": dict(self.refusals)}
        if self.last is not None:
            d["last"] = self.last.as_dict()
            d["recent"] = [x.as_dict() for x in self.recent]
        return d
