"""
Report card for the trained parking student.

    python rl\\eval_parking.py --episodes 30

Runs the learned brain (no learning, no exploration) through N random-bay
parking attempts and prints: success rate, margins, headings, failures.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from rl.parking_env import CarlaParkingEnv

MODEL = os.path.join(os.path.dirname(__file__), "models", "parking_ppo.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    a = ap.parse_args()

    env = CarlaParkingEnv(seed=123)          # different bays than training
    model = PPO.load(MODEL)
    results = []
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset()
            done = False
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, done, _tr, info = env.step(action)
            results.append(info)
            print(f"episode {ep + 1:3d}: {info.get('result'):13s} "
                  + (f"margins {info.get('m_len')}/{info.get('m_wid')} m, "
                     f"heading {info.get('herr_deg')} deg"
                     if info.get("result") == "parked" else ""))
    finally:
        env.close()

    parked = [r for r in results if r.get("result") == "parked"]
    print("\n================ REPORT CARD ================")
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


if __name__ == "__main__":
    main()
