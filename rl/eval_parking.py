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

    _write_rows(results)


def _write_rows(results):
    with open(RAW, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "p", "start_dist_m", "result",
                    "m_len", "m_wid", "herr_deg", "hit", "ax", "ay", "speed",
                    "spawn_yaw_off_deg", "spawn_lat_off_m", "neighbours"])
        for i, r in enumerate(results, 1):
            w.writerow([i, r.get("p"), r.get("start_dist"), r.get("result"),
                        r.get("m_len", ""), r.get("m_wid", ""), r.get("herr_deg", ""),
                        r.get("hit", ""), r.get("ax", ""), r.get("ay", ""), r.get("speed", ""),
                        r.get("spawn_yaw_off_deg", ""), r.get("spawn_lat_off_m", ""),
                        r.get("neighbours", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.0,
                    help="0.0 = full distance (the real exam), 1.0 = already in the box")
    ap.add_argument("--mixed", action="store_true",
                    help="draw difficulty at random like rounds 1-3 did")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lane-start", type=float, default=16.0,
                    help="metres back down the lane for a full-distance start (trained at 16)")
    ap.add_argument("--yaw-noise", type=float, default=3.0,
                    help="random heading error at the start, +/- degrees (trained at 3)")
    ap.add_argument("--lateral-noise", type=float, default=0.0,
                    help="random sideways offset of the start, +/- metres (trained at 0)")
    ap.add_argument("--obs-noise", type=str, default="",
                    help="noise on what the student SEES: pos_m,yaw_deg,speed e.g. 0.25,3,0.2")
    ap.add_argument("--obs-dropout", type=float, default=0.0,
                    help="chance per step that what the student sees freezes on last step's numbers")
    ap.add_argument("--obs-delay", type=int, default=0,
                    help="steps of lag (0.1 s each) between the world and what the student sees")
    ap.add_argument("--neighbour-p", type=float, default=0.0,
                    help="chance of a real parked car in the bay BEHIND the target (training uses 0.5)")
    ap.add_argument("--neighbour-ahead-p", type=float, default=0.0,
                    help="chance of one in the bay AHEAD (training uses 0.3)")
    ap.add_argument("--model", type=str, default="",
                    help="brain file to examine (default rl/models/parking_ppo.zip)")
    ap.add_argument("--no-feelers", action="store_true",
                    help="give the student the round-6 five-number view (needed for brains trained before round 7)")
    ap.add_argument("--tag", type=str, default="",
                    help="name this exam; rows go to rl/exams/<date>_<tag>.csv and the main report card is left alone")
    a = ap.parse_args()
    obs_noise = tuple(float(v) for v in a.obs_noise.split(",")) if a.obs_noise else None

    model_path = a.model or MODEL
    if not os.path.exists(model_path):
        sys.exit(f"no trained model at {model_path} — train one first")

    # Load the brain BEFORE touching CARLA. A half-written checkpoint (trainer
    # killed mid-save) makes PPO.load raise, and if the world were already in
    # synchronous mode by then nothing would ever tick it again — the exam would
    # take every other CARLA client down with it.
    model = PPO.load(model_path)
    print(f"[eval] brain: {model_path}")
    mode = "mixed difficulty" if a.mixed else f"full distance (p={a.p:.2f})"
    print(f"[eval] {mode}, {a.episodes} attempts")
    env = CarlaParkingEnv(seed=a.seed, exam_p=a.p, lane_start_m=a.lane_start,
                          yaw_noise_deg=a.yaw_noise, lateral_noise_m=a.lateral_noise,
                          obs_noise=obs_noise, obs_dropout=a.obs_dropout,
                          obs_delay=a.obs_delay, neighbour_p=a.neighbour_p,
                          neighbour_ahead_p=a.neighbour_ahead_p,
                          use_feelers=not a.no_feelers)   # different bays than training
    if a.tag:
        print(f"[eval] harder-exam settings: lane start {a.lane_start} m, yaw +/-{a.yaw_noise} deg, "
              f"lateral +/-{a.lateral_noise} m, obs noise {obs_noise}, "
              f"dropout {a.obs_dropout}, delay {a.obs_delay} steps, "
              f"neighbours behind {a.neighbour_p} / ahead {a.neighbour_ahead_p}")
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
    if a.tag:
        os.makedirs(os.path.join(HERE, "exams"), exist_ok=True)
        path = os.path.join(HERE, "exams",
                            f"{datetime.date.today().isoformat()}_{a.tag}.csv")
        global RAW
        RAW = path
        _write_rows(results)
        print(f"\nwritten: {path}")
    else:
        _write_card(results, mode, a.episodes, a.seed)
        print(f"\nwritten: {CARD}\n         {RAW}")


if __name__ == "__main__":
    main()
