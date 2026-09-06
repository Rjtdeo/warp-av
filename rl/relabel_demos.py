"""
Round 9: re-label recorded drives for the gear-aware brain (10 inputs).
    python rl\\relabel_demos.py rl\\demos\\demos.npz rl\\demos\\dagger_1.npz --suffix _g

The recordings were made with an unsigned speed. Direction of travel is
visible in the recorded positions: along the heading, the van moved forward
or backward between one step and the next. The speed is signed from that, and
a tenth input "last gear was reverse" is added: the previous step's applied
gear where known (the instructor's own drives: label == applied), otherwise
the previous step's direction of travel.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def relabel(obs, act, ep, labels_were_applied, still_mps=0.05):
    n = len(obs)
    out = np.zeros((n, obs.shape[1] + 1), dtype=np.float32)
    out[:, :obs.shape[1]] = obs
    ax, ay, herr, spd = obs[:, 0] * 15.0, obs[:, 1] * 6.0, obs[:, 2] * math.pi, obs[:, 3] * 5.0
    moving_back = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if ep[i] != ep[i + 1]:
            continue
        along = (ax[i + 1] - ax[i]) * math.cos(herr[i]) + (ay[i + 1] - ay[i]) * math.sin(herr[i])
        moving_back[i] = along < -0.5 * still_mps * 0.1 and spd[i] > still_mps
    # the last step of an episode: copy the one before it
    for i in range(1, n):
        if i == n - 1 or ep[i] != ep[i + 1]:
            if ep[i] == ep[i - 1]:
                moving_back[i] = moving_back[i - 1]
    out[:, 3] = np.where(moving_back, -np.abs(obs[:, 3]), np.abs(obs[:, 3]))
    prev_rev = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if ep[i] != ep[i - 1]:
            continue
        if labels_were_applied:
            prev_rev[i] = 1.0 if act[i - 1, 2] > 0.0 else 0.0
        else:
            prev_rev[i] = 1.0 if moving_back[i - 1] else 0.0
    out[:, obs.shape[1]] = prev_rev
    return out, moving_back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--suffix", default="_g")
    ap.add_argument("--student-driven", nargs="*", default=[],
                    help="files whose labels were NOT the applied actions (DAgger rounds)")
    a = ap.parse_args()
    for f in a.files:
        d = np.load(f)
        applied = os.path.basename(f) not in {os.path.basename(x) for x in a.student_driven}
        obs2, back = relabel(d["obs"], d["act"], d["ep"], labels_were_applied=applied)
        out = os.path.splitext(f)[0] + a.suffix + ".npz"
        np.savez_compressed(out, obs=obs2, act=d["act"], rew=d["rew"], ep=d["ep"], done=d["done"])
        print(f"[relabel] {os.path.basename(f)} -> {os.path.basename(out)}: {len(obs2)} steps, "
              f"{back.mean() * 100:.1f}% moving backward, obs {obs2.shape[1]}")


if __name__ == "__main__":
    main()
