"""
Round 9, step 3: the brain copies the teacher.

    python rl\\pretrain_bc.py --demos rl\\demos\\demos.npz --out rl\\models\\parking_ppo_round9_bc.zip

Fits the same 64x64 PPO policy (9 inputs, 3 controls) to the recorded
(observation -> teacher action) pairs by regression, and its value head to
the discounted returns of the recorded rewards, so the practice phase starts
with a sensible critic. No CARLA needed: a dummy environment supplies the
spaces. Saves in the PPO format so eval_parking.py, --resume and the van's
RLParker load it unchanged.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def discounted_returns(rew, done, gamma=0.99):
    out = np.zeros_like(rew)
    g = 0.0
    for i in range(len(rew) - 1, -1, -1):
        if done[i]:
            g = 0.0
        g = rew[i] + gamma * g
        out[i] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models", "parking_ppo_round9_bc.zip"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--parked-only", action="store_true", help="copy only episodes that ended parked")
    ap.add_argument("--log-std", type=float, default=-1.6, help="action noise the practice phase starts with")
    ap.add_argument("--mirror", action="store_true",
                    help="add the left-hand mirror image of every demo (the town has no left-hand bays)")
    a = ap.parse_args()

    import torch
    import gymnasium as gym
    from stable_baselines3 import PPO

    d = np.load(a.demos)
    obs, act, rew, ep, done = d["obs"], d["act"], d["rew"], d["ep"], d["done"]
    if a.parked_only:
        # an episode ended parked if its last reward was the big prize (>= 100)
        last = {}
        for i in range(len(ep)):
            if done[i]:
                last[int(ep[i])] = rew[i] >= 100.0
        keep = np.array([last.get(int(e), False) for e in ep])
        obs, act, rew, ep, done = obs[keep], act[keep], rew[keep], ep[keep], done[keep]
    ret = discounted_returns(rew, done)
    if a.mirror:
        from rl.parking_math import mirror_observation, mirror_action
        m_obs = np.array([mirror_observation(o) for o in obs], dtype=obs.dtype)
        m_act = np.array([mirror_action(x) for x in act], dtype=act.dtype)
        obs = np.concatenate([obs, m_obs]); act = np.concatenate([act, m_act])
        ret = np.concatenate([ret, ret]); ep = np.concatenate([ep, ep + ep.max() + 1])
        print(f"[bc] mirrored: {len(obs)} steps after adding the left-hand images")
    print(f"[bc] {len(obs)} steps, {len(set(ep.tolist()))} episodes, obs {obs.shape[1]}, act {act.shape[1]}")

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs.shape[1],), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(act.shape[1],), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs.shape[1], dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs.shape[1], dtype=np.float32), 0.0, True, False, {}

    model = PPO("MlpPolicy", Dummy(), verbose=0, seed=7, n_steps=1024, batch_size=256, learning_rate=3e-4)
    pol = model.policy
    X = torch.as_tensor(obs, dtype=torch.float32)
    Y = torch.as_tensor(np.clip(act, -1.0, 1.0), dtype=torch.float32)
    R = torch.as_tensor(ret, dtype=torch.float32).unsqueeze(1)
    R = (R - R.mean()) / (R.std() + 1e-6) * 30.0        # the critic learns the shape; PPO recalibrates the scale
    def heads(x):
        """(action mean, value) from the policy's networks."""
        feats = pol.extract_features(x)
        if isinstance(feats, tuple):                       # unshared extractors (not our case, but be safe)
            lat_pi, lat_vf = pol.mlp_extractor.forward_actor(feats[0]), pol.mlp_extractor.forward_critic(feats[1])
        else:
            lat_pi, lat_vf = pol.mlp_extractor(feats)
        return pol.action_net(lat_pi), pol.value_net(lat_vf)

    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    n = len(X)
    for epoch in range(a.epochs):
        perm = torch.randperm(n)
        tot_a = tot_v = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            mean, value = heads(X[idx])
            loss_a = torch.nn.functional.mse_loss(mean, Y[idx])
            loss_v = torch.nn.functional.mse_loss(value, R[idx])
            loss = loss_a + 0.05 * loss_v
            opt.zero_grad(); loss.backward(); opt.step()
            tot_a += loss_a.item() * len(idx); tot_v += loss_v.item() * len(idx)
        if epoch % 10 == 0 or epoch == a.epochs - 1:
            print(f"[bc] epoch {epoch:3d}  action mse {tot_a / n:.4f}  value mse {tot_v / n:.1f}")
    with torch.no_grad():
        pol.log_std.fill_(a.log_std)
    model.save(a.out)
    # how well does the copy agree with the teacher on its own demos?
    with torch.no_grad():
        mean, _ = heads(X)
        mean = mean.numpy()
    err = np.abs(np.clip(mean, -1, 1) - np.clip(act, -1, 1))
    gear_ok = np.mean((mean[:, 2] > 0.5) == (act[:, 2] > 0.5)) if act.shape[1] > 2 else float("nan")
    print(f"[bc] saved {a.out}; mean |steer err| {err[:, 0].mean():.3f}, |pedal err| {err[:, 1].mean():.3f}, "
          f"gear agreement {gear_ok * 100:.1f}%")


if __name__ == "__main__":
    main()
