"""
Round 9, step 2: the teacher drives in the arena and every moment is recorded.

    python rl\\record_demos.py --episodes 3000 --out rl\\demos\\demos.npz

Needs CARLA; run INSTEAD of the trainer. Starts are spread 16-36 m back with
knocked-off-course perturbations (yaw, sideways), cars 1/2/3 bays back or one
ahead on most attempts. Records (observation, teacher action, reward) per step
and the result per episode; keeps every episode (the fine-tune's rewards say
which were good), and prints the teacher's own park rate by hazard - that is
the baseline the copied brain must match.
"""
import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from rl.parking_env import CarlaParkingEnv
from rl.teacher import teacher_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "demos", "demos.npz"))
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--lane-start", type=float, default=16.0)
    ap.add_argument("--lane-start-jitter", type=float, default=20.0, help="starts spread over [start, start+jitter]")
    ap.add_argument("--yaw-noise", type=float, default=10.0, help="knocked-off-course heading, +/- deg")
    ap.add_argument("--lateral-noise", type=float, default=0.5, help="knocked-off-course sideways, +/- m")
    ap.add_argument("--neighbour-p", type=float, default=0.6, help="chance of a car 1/2/3 bays back")
    ap.add_argument("--neighbour-ahead-p", type=float, default=0.3)
    a = ap.parse_args()

    env = CarlaParkingEnv(host=a.host, seed=a.seed, exam_p=0.0, lane_start_m=a.lane_start,
                          lane_start_jitter_m=a.lane_start_jitter, yaw_noise_deg=a.yaw_noise,
                          lateral_noise_m=a.lateral_noise, neighbour_p=a.neighbour_p,
                          neighbour_ahead_p=a.neighbour_ahead_p, neighbour_behind_bays=(1, 2, 3),
                          use_feelers=True, reverse=True)
    obs_all, act_all, rew_all, ep_all, done_all = [], [], [], [], []
    results = collections.Counter()
    by_hazard = collections.defaultdict(collections.Counter)
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                ax, ay, herr = env._slot_frame()
                _, _, _, speed = env._pose()
                if env._reversing:
                    speed = -speed
                steer, pedal, gear = teacher_action(ax, ay, herr, speed, env._feelers_now, reverse_ok=True)
                action = np.array([steer, pedal, 1.0 if gear > 0.5 else -1.0], dtype=np.float32)
                obs_all.append(np.asarray(obs, dtype=np.float32))
                act_all.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                rew_all.append(float(reward))
                ep_all.append(ep)
                done = terminated or truncated
                done_all.append(done)
            results[info.get("result", "?")] += 1
            by_hazard[info.get("hazard", "") or "none"][info.get("result", "?")] += 1
            if (ep + 1) % 25 == 0:
                print(f"[demos] {ep + 1} episodes, {len(obs_all)} steps, parked {results['parked']}/{ep + 1}, "
                      f"{time.time() - t0:.0f} s")
                np.savez_compressed(a.out, obs=np.stack(obs_all), act=np.stack(act_all),
                                    rew=np.array(rew_all, dtype=np.float32), ep=np.array(ep_all, dtype=np.int32),
                                    done=np.array(done_all, dtype=bool))
    finally:
        env.close()
    np.savez_compressed(a.out, obs=np.stack(obs_all), act=np.stack(act_all),
                        rew=np.array(rew_all, dtype=np.float32), ep=np.array(ep_all, dtype=np.int32),
                        done=np.array(done_all, dtype=bool))
    print(f"[demos] DONE: {len(ep_all)} steps from {a.episodes} episodes -> {a.out}")
    print(f"[demos] teacher results: {dict(results)}")
    for hz, c in sorted(by_hazard.items()):
        n = sum(c.values())
        print(f"[demos]   {hz:8s} parked {c['parked']}/{n} ({c['parked'] / n * 100:.0f}%)  {dict(c)}")


if __name__ == "__main__":
    main()
