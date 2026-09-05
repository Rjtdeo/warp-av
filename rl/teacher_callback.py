"""
Round 9, step 5: practise, but stay close to the teacher at first.

After every PPO rollout, a few regression steps pull the policy's mean action
toward the teacher's action on the states the brain actually visited, with a
weight that fades to zero over --teacher-fade steps. Waymo's "Imitation is
not enough" keeps an imitation term during RL for the same reason: rewards
alone let the policy wander into strange driving before they teach it
anything. Steps where the brain had chosen reverse are skipped (the brain's
observation carries an unsigned speed, so the teacher's label there is unsure).
"""
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from rl.teacher import teacher_from_obs


class TeacherCallback(BaseCallback):
    def __init__(self, weight=1.0, fade_steps=1_000_000, epochs=2, lr=3e-4, verbose=0):
        super().__init__(verbose)
        self.weight, self.fade_steps, self.epochs, self.lr = weight, fade_steps, epochs, lr
        self._start = None
        self._opt = None
        self.last_loss = None

    def _on_training_start(self):
        self._start = self.model.num_timesteps

    def _on_step(self):
        return True

    def _actor_mean(self, x):
        pol = self.model.policy
        feats = pol.extract_features(x)
        lat_pi = pol.mlp_extractor.forward_actor(feats[0] if isinstance(feats, tuple) else feats)
        return pol.action_net(lat_pi)

    def _on_rollout_end(self):
        done = self.model.num_timesteps - (self._start or 0)
        w = self.weight * max(0.0, 1.0 - done / float(self.fade_steps))
        if w <= 0.0:
            return
        buf = self.model.rollout_buffer
        obs = buf.observations.reshape(-1, buf.observations.shape[-1])
        acts = buf.actions.reshape(-1, buf.actions.shape[-1])
        forward = acts[:, 2] <= 0.5 if acts.shape[1] > 2 else np.ones(len(acts), dtype=bool)
        obs = obs[forward]
        if len(obs) < 32:
            return
        labels = np.array([teacher_from_obs(o) for o in obs], dtype=np.float32)
        labels[:, 2] = np.where(labels[:, 2] > 0.5, 1.0, -1.0)
        X = torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)
        Y = torch.as_tensor(labels, dtype=torch.float32, device=self.model.device)
        if self._opt is None:
            self._opt = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                idx = perm[i:i + 256]
                loss = w * torch.nn.functional.mse_loss(self._actor_mean(X[idx]), Y[idx])
                self._opt.zero_grad(); loss.backward(); self._opt.step()
        self.last_loss = float(loss.item())
        if self.verbose:
            print(f"[teacher] weight {w:.3f}, stay-close loss {self.last_loss:.4f} on {len(X)} forward steps")
