"""
Warm-start a nine-number brain from a five-number one.

    python rl\\warm_start.py --from rl\\models\\parking_ppo_round6.zip

The round-6 brain already knows the pull-in from 16 m. Round 7 adds four
feeler inputs; training from zero relearned round 3's standstill in 1,200
attempts. Here the old network's weights are copied into a new one whose extra
input columns start at ZERO, so on day one it behaves exactly like round 6 and
the feelers are learned on top. Saves rl/models/parking_ppo.zip for --resume.
Needs CARLA (to build the environment); run INSTEAD of the trainer.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from stable_baselines3 import PPO

from rl.parking_env import CarlaParkingEnv

MODEL = os.path.join(os.path.dirname(__file__), "models", "parking_ppo.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    a = ap.parse_args()

    old = PPO.load(a.src)
    n_old = old.observation_space.shape[0]
    env = CarlaParkingEnv()
    try:
        n_new = env.observation_space.shape[0]
        assert n_new > n_old, f"new view has {n_new} numbers, old had {n_old}"
        new = PPO("MlpPolicy", env, verbose=1, seed=7,   # verbose is saved with the brain: keep the training table
                  n_steps=1024, batch_size=256, learning_rate=3e-4)
        osd, nsd = old.policy.state_dict(), new.policy.state_dict()
        copied, widened = 0, []
        with torch.no_grad():
            for k, v in nsd.items():
                if k not in osd:
                    continue
                o = osd[k]
                if o.shape == v.shape:
                    v.copy_(o); copied += 1
                elif o.dim() == 2 and v.dim() == 2 and v.shape[0] == o.shape[0] \
                        and v.shape[1] == n_new and o.shape[1] == n_old:
                    v.zero_(); v[:, :n_old] = o; widened.append(k)
                else:
                    raise RuntimeError(f"cannot map {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
        new.policy.load_state_dict(nsd)
        print(f"[warm] copied {copied} tensors, widened {len(widened)}: {widened}")

        # Proof: with the feelers reading 'nothing near' the new brain must act
        # exactly like the old one. (Feelers are the LAST inputs.)
        obs, _ = env.reset()
        obs5 = obs[:n_old].astype(np.float32)
        obs9 = np.concatenate([obs5, np.zeros(n_new - n_old, np.float32)])
        a_old, _ = old.predict(obs5, deterministic=True)
        a_new, _ = new.predict(obs9, deterministic=True)
        assert np.allclose(a_old, a_new, atol=1e-5), (a_old, a_new)
        print(f"[warm] check passed: same action {np.round(a_old, 3)} on the same view")
        new.save(MODEL)
        print(f"[warm] saved {MODEL}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
