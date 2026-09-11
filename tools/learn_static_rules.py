"""
Learn when the van may call a LiDAR blob STATIC -- a thing that will not move -- from the
answer key recorded by tools/record_static_truth.py (Planning V2, static vs dynamic).

Everything starts DYNAMIC. A blob is called static only when one of three plain rules fits:

  POLE       tall, thin, and off the road        (lamp posts, sign posts, traffic lights)
  STRUCTURE  much of it high up, well off road   (building fronts, walls, fences, trees)
  LOW        very low, and off the road          (kerb edges, low walls, planter rims)

The thresholds are learned, not guessed, and learned lopsided on purpose: a person or a
vehicle called static is the one mistake that is not allowed -- zero on the roads it learns
from, with a margin: every threshold must still let no person or vehicle through when it
is loosened by one step. A pole left "dynamic" only costs a needless caution.

It learns on some roads and is tested on others it never saw (grouped by road, because
neighbouring viewpoints see the same objects and a random split would let it study the
test), five times over with different road splits.

    python tools/learn_static_rules.py static_truth.npz
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np

# CARLA classes that must never be called static: they move, or can at any moment
MUST_STAY_DYNAMIC = ("pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle")
# ... and the ones that are sure never to move
STATIC_CLASSES = ("building", "wall", "fence", "pole", "traffic light", "traffic sign",
                  "vegetation", "static prop", "sidewalk/kerb", "terrain", "ground",
                  "bridge", "rail track", "guard rail")
# CARLA's "dynamic prop" in Town10HD is bins, food carts, tables, dumpsters: things people
# move, never road users. Counted separately, neither as a hit nor as a mistake.

# A blob is named by a majority vote of its points, which hides the worst case: a person
# leaning on a wall comes out as ONE blob that is mostly wall. So a blob holding this many
# points of a person or vehicle must stay dynamic whatever its majority says (65 such blobs
# in the second recording, most of them people beside poles, trees and shop fronts).
HOLDS_A_ROAD_USER_POINTS = 3

# Only things within reach of the road are labelled: further off than this, nothing can get
# onto the road in the next few seconds, and the van ignores it -- the same 12 m the
# prediction step already uses (planning/prediction.MAX_START_LATERAL_M). The first run
# showed why it has to be said out loud: two lorries parked in a yard 31-41 m from any road
# look exactly like a building corner to the laser.
IN_REACH_OF_ROAD_M = 12.0

# The LiDAR alone never calls anything static that touches a lane: POLE and STRUCTURE need
# the whole blob at least 0.25 m clear of every lane a vehicle can use. The first run
# showed why: a bus or lorry seen side-on from 20 m can be two points wide and 3 m tall, a
# perfect "pole" -- but it is always IN a lane. Only LOW may reach the lane edge (25 cm
# in), because a kerb sits on it by definition -- and no deeper, and no taller than 30 cm:
# a car half-hidden behind another shows as 3-5 points 0.3-0.4 m up, INSIDE the lane.
# POLE never goes below 2.3 m, whatever the data allows: the tallest person in the recording
# measured 1.92 m and the data alone settled on 2.0 m -- 8 cm of margin, and real people come
# taller than CARLA's. And a pole has points all the way up it: at least 30 % above 2 m. The
# fresh recording's one mistake was 4 points of a person and ONE point of the pole behind
# them -- "3.46 m tall" -- and this floor keeps 95 % of real poles.
GRID = {
    "pole":      {"min_height": np.arange(2.3, 3.21, 0.1), "max_long": np.arange(0.2, 0.81, 0.1),
                  "min_gap": np.arange(0.25, 2.01, 0.25), "min_frac_high": np.arange(0.30, 0.61, 0.05)},
    "structure": {"min_frac_high": np.arange(0.10, 0.71, 0.05), "min_gap": np.arange(0.25, 12.1, 0.75),
                  "min_long": np.arange(0.0, 5.1, 1.0)},
    "low":       {"max_height": np.arange(0.10, 0.31, 0.05), "min_gap": np.arange(-0.25, 2.01, 0.25)},
}
# Tried and dropped, both measured on this data:
#  * a SHORT rule (under ~0.6 m, off the road) for cones, bins and bollards: it called parked
#    bicycles and a motorbike on the pavement static, and caught 5 % of props. A bin and a
#    parked scooter look the same to the laser; telling them apart is the camera's job.
#  * a floor of 6 points before judging anything: it threw away far poles and building
#    fragments (static caught fell from 45 % to 28 %) and let the LOW rule loosen into the lane.
# Only the clustering's own minimum stays: a 2-point blob is never judged.
MIN_POINTS_TO_JUDGE = 3


def load(path):
    d = np.load(path)
    f = {k: d[k] for k in d.files}
    f["long"] = np.maximum(f["length"], f["width"])
    f["height"] = np.where(np.isfinite(f["height"]), f["height"], 0.0)
    f["frac_high"] = np.where(np.isfinite(f["frac_high"]), f["frac_high"], 0.0)
    # unknown distance from the road (no lane anywhere near) never counts as "off the road"
    f["road_gap"] = np.where(np.isfinite(f["road_gap"]), f["road_gap"], -99.0)
    # how many of its points belong to a person or vehicle (from the top-3 vote counts)
    f["road_user_points"] = np.array([sum(v for c, v in json.loads(js).items() if c in MUST_STAY_DYNAMIC)
                                      for js in f["tags_seen"]])
    return f


def rule_mask(f, kind, p):
    if kind == "pole":
        return ((f["height"] >= p["min_height"]) & (f["long"] <= p["max_long"]) & (f["road_gap"] >= p["min_gap"])
                & (f["frac_high"] >= p["min_frac_high"]))
    if kind == "structure":
        return ((f["frac_high"] >= p["min_frac_high"]) & (f["road_gap"] >= p["min_gap"])
                & (f["long"] >= p["min_long"]))
    if kind == "low":
        return (f["height"] > 0) & (f["height"] <= p["max_height"]) & (f["road_gap"] >= p["min_gap"])
    raise ValueError(kind)


def loosened(kind, p):
    """Every one-step loosening of a setting: the margin check. At the edge of the grid the
    step is taken anyway (one grid step past it), so an edge value never escapes the check."""
    g = GRID[kind]
    out = []
    for k, v in p.items():
        vals = list(g[k])
        step = float(vals[1] - vals[0])
        loosen_up = k.startswith("max")          # a max is loosened by raising it
        q = dict(p)
        q[k] = float(v + step if loosen_up else v - step)
        out.append(q)
    return out


def learn(f, rows, dyn, stat, already):
    """Greedy: the biggest rule first, each taking the static things the others missed."""
    chosen = {}
    covered = already.copy()
    for kind in ("structure", "pole", "low"):
        g = GRID[kind]
        best, best_key = None, None
        judged = rows & (f["n"] >= MIN_POINTS_TO_JUDGE)
        for combo in itertools.product(*g.values()):
            p = dict(zip(g.keys(), (float(x) for x in combo)))
            m = rule_mask(f, kind, p) & judged
            if (m & dyn).any():
                continue
            if any((rule_mask(f, kind, q) & judged & dyn).any() for q in loosened(kind, p)):
                continue                                # no margin: one step looser lets one through
            gain = int((m & stat & ~covered).sum())
            key = (gain, sum(p.values()) if kind != "low" else -sum(p.values()))
            if best is None or key > best_key:
                best, best_key = p, key
        if best is not None:
            chosen[kind] = best
            covered |= rule_mask(f, kind, best) & judged
    return chosen


def apply(f, rules):
    m = np.zeros(len(f["height"]), dtype=bool)
    for kind, p in rules.items():
        m |= rule_mask(f, kind, p)
    return m & (f["n"] >= MIN_POINTS_TO_JUDGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None, help="write the rules learned on ALL roads here (json)")
    ap.add_argument("--check", default=None,
                    help="a second, fresh recording: test the rules learned on ALL of the first on it")
    a = ap.parse_args()
    f = load(a.npz)
    reach = f["road_gap"] <= IN_REACH_OF_ROAD_M
    far = int((~reach).sum())
    far_dyn = int((~reach & np.isin(f["tag_name"], MUST_STAY_DYNAMIC)).sum())
    f = {k: v[reach] for k, v in f.items()}
    name = f["tag_name"]
    holds = f["road_user_points"] >= HOLDS_A_ROAD_USER_POINTS
    dyn = np.isin(name, MUST_STAY_DYNAMIC) | holds
    stat = np.isin(name, STATIC_CLASSES) & ~holds
    roads = np.unique(f["road_id"])
    print(f"{len(name)} blobs within {IN_REACH_OF_ROAD_M:.0f} m of a road, on {len(roads)} roads: "
          f"{stat.sum()} static things, {dyn.sum()} holding a person or vehicle "
          f"({int((holds & ~np.isin(name, MUST_STAY_DYNAMIC)).sum())} of them hidden in a static majority)")
    print(f"(left out: {far} blobs further off, {far_dyn} of them people or vehicles)\n")

    tot_hit = tot_stat = tot_bad = tot_dyn = 0
    per_class = {}
    for fold in range(a.folds):
        rng = np.random.RandomState(100 + fold)
        order = roads.copy()
        rng.shuffle(order)
        test_roads = set(order[: len(order) // 2])
        te = np.isin(f["road_id"], list(test_roads))
        tr = ~te
        rules = learn(f, tr, dyn, stat, np.zeros_like(tr))
        s = apply(f, rules)
        hit, n_st = int((s & stat & te).sum()), int((stat & te).sum())
        bad, n_dy = int((s & dyn & te).sum()), int((dyn & te).sum())
        tot_hit += hit; tot_stat += n_st; tot_bad += bad; tot_dyn += n_dy
        for c in set(name[te]):
            m = (name == c) & te
            h, n = per_class.get(c, (0, 0))
            per_class[c] = (h + int((s & m).sum()), n + int(m.sum()))
        print(f"split {fold + 1}: learned on {tr.sum()} blobs, tested on {te.sum()} from unseen roads -> "
              f"static caught {100.0 * hit / max(1, n_st):.0f}%, people/vehicles called static {bad}/{n_dy}")
        if bad:
            for i in np.where(s & dyn & te)[0]:
                print(f"    MISTAKE: {name[i]} n={f['n'][i]} h={f['height'][i]:.2f} long={f['long'][i]:.2f} "
                      f"frac>2m={f['frac_high'][i]:.2f} gap={f['road_gap'][i]:+.1f} {f['tags_seen'][i]}")

    print(f"\nALL {a.folds} SPLITS, on roads the rules never saw:")
    print(f"  static things called static : {tot_hit}/{tot_stat} ({100.0 * tot_hit / max(1, tot_stat):.0f}%)")
    print(f"  people/vehicles called static: {tot_bad}/{tot_dyn}")
    print("  by class (share called static):")
    for c, (h, n) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
        print(f"    {c:14s} {100.0 * h / max(1, n):4.0f}%  ({h}/{n})")

    rules = learn(f, np.ones(len(name), dtype=bool), dyn, stat, np.zeros(len(name), dtype=bool))
    print("\nrules learned on ALL roads (the ones the van would use):")
    for kind, p in rules.items():
        print(f"  {kind:10s} " + ", ".join(f"{k}={v:.2f}" for k, v in p.items()))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rules, fh, indent=2)
        print(f"-> {a.out}")
    if a.check:
        g = load(a.check)
        g = {k: v[g["road_gap"] <= IN_REACH_OF_ROAD_M] for k, v in g.items()}
        gname = g["tag_name"]
        gholds = g["road_user_points"] >= HOLDS_A_ROAD_USER_POINTS
        gdyn = np.isin(gname, MUST_STAY_DYNAMIC) | gholds
        gstat = np.isin(gname, STATIC_CLASSES) & ~gholds
        s = apply(g, rules)
        print(f"\nFRESH RECORDING ({a.check}), rules never tuned on it:")
        print(f"  static things called static : {int((s & gstat).sum())}/{int(gstat.sum())} "
              f"({100.0 * (s & gstat).sum() / max(1, gstat.sum()):.0f}%)")
        print(f"  people/vehicles called static: {int((s & gdyn).sum())}/{int(gdyn.sum())}")
        for i in np.where(s & gdyn)[0]:
            print(f"    MISTAKE: {gname[i]} n={g['n'][i]} h={g['height'][i]:.2f} long={g['long'][i]:.2f} "
                  f"frac>2m={g['frac_high'][i]:.2f} gap={g['road_gap'][i]:+.1f} {g['tags_seen'][i]}")
        for c in ("pole", "sidewalk/kerb", "building", "fence", "vegetation", "traffic sign", "static prop"):
            m = gname == c
            if m.sum():
                print(f"    {c:14s} {100.0 * (s & m).sum() / m.sum():4.0f}%  ({int((s & m).sum())}/{int(m.sum())})")


if __name__ == "__main__":
    sys.exit(main())
