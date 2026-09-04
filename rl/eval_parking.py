"""
Report card for the trained parking student.

    python rl\\eval_parking.py --episodes 30            # the real exam (full distance)
    python rl\\eval_parking.py --episodes 30 --p 0.45   # grade one easier level
    python rl\\eval_parking.py --episodes 30 --mixed    # old rounds-1-3 style grading

By default EVERY attempt starts at the full 16 m lane distance. Rounds 1-3
graded with the training mix instead, so roughly one attempt in six began with
the van already inside the box — those scores were not comparable to a real
exam and flattered the student.

Writes rl/REPORT_CARD.md and rl/eval_runs.csv so the raw result is in the repo.
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from rl.parking_env import CarlaParkingEnv
from rl.parking_math import Curriculum

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "parking_ppo.zip")
CARD = os.path.join(HERE, "REPORT_CARD.md")
RAW = os.path.join(HERE, "eval_runs.csv")


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.join(HERE, ".."),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _write_card(results, mode, episodes, seed):
    parked = [r for r in results if r.get("result") == "parked"]
    fails = {}
    for r in results:
        if r.get("result") != "parked":
            fails[r.get("result", "?")] = fails.get(r.get("result", "?"), 0) + 1

    lines = ["# RL parking report card", ""]
    lines.append(f"- when: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- code: {_git_rev()}")
    lines.append(f"- exam: {mode}, {episodes} attempts, seed {seed}")
    lines.append("")
    lines.append(f"**Parked fully inside: {len(parked)}/{len(results)}**")
    lines.append("")
    if parked:
        lines.append("| measure | mean | worst |")
        lines.append("|---|---|---|")
        lines.append(f"| width margin (m) | {np.mean([r['m_wid'] for r in parked]):.2f} "
                     f"| {min(r['m_wid'] for r in parked):.2f} |")
        lines.append(f"| length margin (m) | {np.mean([r['m_len'] for r in parked]):.2f} "
                     f"| {min(r['m_len'] for r in parked):.2f} |")
        lines.append(f"| heading off (deg) | {np.mean([abs(r['herr_deg']) for r in parked]):.1f} "
                     f"| {max(abs(r['herr_deg']) for r in parked):.1f} |")
        lines.append("")
    if fails:
        lines.append("Failures: " + ", ".join(f"{k} x{v}" for k, v in sorted(fails.items())))
        lines.append("")
    lines.append("Raw per-attempt rows: `rl/eval_runs.csv`.")
    with open(CARD, "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(RAW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "p", "start_dist_m", "result",
                    "m_len", "m_wid", "herr_deg", "hit", "ax", "ay", "speed"])
        for i, r in enumerate(results, 1):
            w.writerow([i, r.get("p"), r.get("start_dist"), r.get("result"),
                        r.get("m_len", ""), r.get("m_wid", ""), r.get("herr_deg", ""),
                        r.get("hit", ""), r.get("ax", ""), r.get("ay", ""), r.get("speed", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.0,
                    help="0.0 = full distance (the real exam), 1.0 = already in the box")
    ap.add_argument("--mixed", action="store_true",
                    help="draw difficulty at random like rounds 1-3 did")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    if not os.path.exists(MODEL):
        sys.exit(f"no trained model at {MODEL} — train one first")

    # Load the brain BEFORE touching CARLA. A half-written checkpoint (trainer
    # killed mid-save) makes PPO.load raise, and if the world were already in
    # synchronous mode by then nothing would ever tick it again — the exam would
    # take every other CARLA client down with it.
    model = PPO.load(MODEL)
    mode = "mixed difficulty" if a.mixed else f"full distance (p={a.p:.2f})"
    print(f"[eval] {mode}, {a.episodes} attempts")
    env = CarlaParkingEnv(seed=a.seed, exam_p=a.p)   # different bays than training
    results = []
    try:
        for ep in range(a.episodes):
            p = Curriculum.LEVELS[env.rng.randrange(len(Curriculum.LEVELS))] \
                if a.mixed else a.p
            obs, _ = env.reset(options={"p": p})
            done = False
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, done, _tr, info = env.step(action)
            results.append(info)
            detail = ""
            if info.get("result") == "parked":
                detail = (f"margins {info.get('m_len')}/{info.get('m_wid')} m, "
                          f"heading {info.get('herr_deg')} deg")
            elif info.get("result") == "collision":
                detail = (f"hit {info.get('hit')} at {info.get('ax')} m along / "
                          f"{info.get('ay')} m across, {info.get('herr_deg')} deg, "
                          f"{info.get('speed')} m/s")
            print(f"episode {ep + 1:3d}: p={info.get('p')} {info.get('result'):13s} {detail}")
    finally:
        env.close()

    parked = [r for r in results if r.get("result") == "parked"]
    print("\n================ REPORT CARD ================")
    print(f"exam: {mode}")
    print(f"parked fully inside: {len(parked)}/{len(results)}")
    if parked:
        print(f"width margin: mean {np.mean([r['m_wid'] for r in parked]):.2f} m "
              f"(min {min(r['m_wid'] for r in parked):.2f})")
        print(f"heading off: mean {np.mean([abs(r['herr_deg']) for r in parked]):.1f} deg")
    fails = {}
    for r in results:
        if r.get("result") != "parked":
            fails[r.get("result")] = fails.get(r.get("result"), 0) + 1
    if fails:
        print("failures:", fails)
    _write_card(results, mode, a.episodes, a.seed)
    print(f"\nwritten: {CARD}\n         {RAW}")


if __name__ == "__main__":
    main()
