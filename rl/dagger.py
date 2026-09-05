"""
Round 9, step 4: tidy the drift (DAgger). The copied brain drives; at every
step the teacher says what IT would have done; those corrections are added to
the demos and the brain is re-fitted. Two or three rounds remove the "small
error grows into a big one" problem of pure copying.

    python rl\\dagger.py --model rl\\models\\parking_ppo_round9_bc.zip --demos rl\\demos\\demos.npz --episodes 400

Needs CARLA; run INSTEAD of the trainer. Writes rl/demos/dagger_<k>.npz
(the visited states with the teacher's labels) and re-runs pretrain_bc.py
on demos + all dagger files, saving the new brain over --model.
"""
import argparse
import collections
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from rl.parking_env import CarlaParkingEnv
from rl.teacher import teacher_action

HERE = os.path.dirname(os.path.abspath(__file__))


def merge(paths, out):
    parts = [np.load(p) for p in paths]
    keys = ("obs", "act", "rew", "ep", "done")
    merged = {k: [] for k in keys}
    ep_offset = 0
    for d in parts:
        for k in keys:
            v = d[k]
            if k == "ep":
                v = v + ep_offset
            merged[k].append(v)
        ep_offset += int(d["ep"].max()) + 1 if len(d["ep"]) else 0
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in merged.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--beta", type=float, default=0.3,
                    help="chance per episode that the TEACHER drives (classic DAgger mixing)")
    a = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cpu")
    env = CarlaParkingEnv(host=a.host, seed=a.seed + a.round, exam_p=0.0, lane_start_m=16.0,
                          lane_start_jitter_m=20.0, yaw_noise_deg=10.0, lateral_noise_m=0.5,
                          neighbour_p=0.6, neighbour_ahead_p=0.3, neighbour_behind_bays=(1, 2, 3),
                          use_feelers=True, reverse=True)
    obs_all, act_all, rew_all, ep_all, done_all = [], [], [], [], []
    results = collections.Counter()
    by_hazard = collections.defaultdict(collections.Counter)
    rng = np.random.default_rng(a.seed + a.round)
    try:
        for ep in range(a.episodes):
            obs, _ = env.reset()
            teacher_drives = rng.random() < a.beta
            done = False
            while not done:
                ax, ay, herr = env._slot_frame()
                _, _, _, speed = env._pose()
                if env._reversing:
                    speed = -speed
                steer, pedal, gear = teacher_action(ax, ay, herr, speed, env._feelers_now, reverse_ok=True)
                label = np.array([steer, pedal, 1.0 if gear > 0.5 else -1.0], dtype=np.float32)
                obs_all.append(np.asarray(obs, dtype=np.float32))
                act_all.append(label)
                if teacher_drives:
                    action = label
                else:
                    action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rew_all.append(float(reward)); ep_all.append(ep)
                done = terminated or truncated
                done_all.append(done)
            who = "teacher" if teacher_drives else "student"
            results[(who, info.get("result", "?"))] += 1
            if not teacher_drives:
                by_hazard[info.get("hazard", "") or "none"][info.get("result", "?")] += 1
            if (ep + 1) % 25 == 0:
                print(f"[dagger {a.round}] {ep + 1} episodes, {len(obs_all)} labelled steps")
    finally:
        env.close()
    out = os.path.join(HERE, "demos", f"dagger_{a.round}.npz")
    np.savez_compressed(out, obs=np.stack(obs_all), act=np.stack(act_all),
                        rew=np.array(rew_all, dtype=np.float32), ep=np.array(ep_all, dtype=np.int32),
                        done=np.array(done_all, dtype=bool))
    print(f"[dagger {a.round}] student results by hazard (before this round's re-fit):")
    for hz, c in sorted(by_hazard.items()):
        n = sum(c.values())
        print(f"[dagger {a.round}]   {hz:8s} parked {c['parked']}/{n} ({c['parked'] / n * 100:.0f}%)")
    merged = merge([a.demos] + sorted(glob.glob(os.path.join(HERE, "demos", "dagger_*.npz"))),
                   os.path.join(HERE, "demos", "merged.npz"))
    print(f"[dagger {a.round}] re-fitting on {merged}")
    subprocess.check_call([sys.executable, os.path.join(HERE, "pretrain_bc.py"), "--demos", merged, "--out", a.model])


if __name__ == "__main__":
    main()
