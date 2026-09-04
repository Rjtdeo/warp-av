"""
Overnight trainer for the RL parking student.

    python rl\\train_parking.py                    # fresh training (400k steps)
    python rl\\train_parking.py --resume           # continue from the checkpoint
    python rl\\train_parking.py --steps 100000
    python rl\\train_parking.py --resume --start-level 3

Difficulty is no longer a fixed random mix: a Curriculum starts at "already in
the box, just stop" and steps out towards the full 16 m lane start as the
recent success rate passes 55%. --start-level skips straight to a harder rung
when resuming a student that already mastered the easy ones.

Writes:
    rl/models/parking_ppo.zip         latest checkpoint (every ~10k steps)
    rl/train_log.csv                  per-episode: steps, reward, result, difficulty
Run INSTEAD of the main van program (it owns the simulator while training).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl.parking_env import CarlaParkingEnv
from rl.parking_math import Curriculum
from rl.train_log import COLUMNS, prepare_log

MODEL = os.path.join(os.path.dirname(__file__), "models", "parking_ppo.zip")
LOG = os.path.join(os.path.dirname(__file__), "train_log.csv")


class EpisodeLogger(BaseCallback):
    def __init__(self, fresh=True):
        super().__init__()
        self._t0 = time.time()
        self._episodes = 0
        self._last_save = 0
        # Round 4 widened this log from 5 columns to 7. The CARLA machine still
        # has the rounds 1-3 file, and appending wide rows under a narrow header
        # makes the WHOLE log unreadable (pandas raises, csv.DictReader silently
        # drops p and start_dist). Check the header, not just the file's
        # existence, and set a mismatched log aside instead of corrupting it.
        new, backup = prepare_log(LOG, fresh=fresh)
        if backup:
            print(f"[train] previous train_log.csv kept at {backup}; "
                  f"this run logs to a fresh file")
        self._f = open(LOG, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(COLUMNS)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._episodes += 1
                self._w.writerow([int(time.time() - self._t0), self.num_timesteps,
                                  self._episodes, round(ep["r"], 1),
                                  info.get("result", "?"),
                                  info.get("p", ""), info.get("start_dist", ""),
                                  info.get("neighbours", ""), info.get("stage", ""),
                                  info.get("reverse_steps", ""), info.get("hazard", "")])
                self._f.flush()
                if self._episodes % 25 == 0:
                    print(f"[train] {self._episodes} episodes, "
                          f"{self.num_timesteps} steps, last reward {ep['r']:.0f}, "
                          f"difficulty p={info.get('p')}")
        if self.num_timesteps - self._last_save >= 10000:
            self._last_save = self.num_timesteps
            self.model.save(MODEL)
            print(f"[train] checkpoint saved at {self.num_timesteps} steps")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-level", type=int, default=0,
                    help="curriculum rung to begin on (0 = easiest, 5 = full distance)")
    ap.add_argument("--reverse", action="store_true", help="third control: reverse gear (round 8)")
    ap.add_argument("--obstacles", action="store_true", help="hazards in stages (round 8)")
    ap.add_argument("--start-stage", type=int, default=0, help="obstacle stage to begin on (0-2)")
    ap.add_argument("--start-rung", type=int, default=0,
                    help="stage-1 rung to begin on: 0 = car 4 bays back, 1 = 3, 2 = 2")
    ap.add_argument("--explore-std", type=float, default=0.0,
                    help="on resume, raise steer/throttle action noise to at least this "
                         "(a trained brain barely explores; a new skill needs variety)")
    ap.add_argument("--lane-start", type=float, default=16.0,
                    help="metres back along the lane the far starts begin (22 for the car two bays back)")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    curriculum = Curriculum(start_level=a.start_level)
    from rl.parking_math import Stages
    env = CarlaParkingEnv(curriculum=curriculum, reverse=a.reverse, obstacles=a.obstacles,
                          lane_start_m=a.lane_start,
                          stages=Stages(start=a.start_stage, start_rung=a.start_rung)
                          if a.obstacles else None)
    if a.obstacles:
        print(f"[train] {env.stages.describe()}")
    print(f"[train] curriculum: {curriculum.describe()}")
    try:
        if a.resume and os.path.exists(MODEL):
            print(f"[train] resuming from {MODEL}")
            model = PPO.load(MODEL, env=env)
            if a.explore_std > 0:
                import math
                import torch
                with torch.no_grad():
                    ls = model.policy.log_std
                    before = [round(v, 3) for v in ls.exp().tolist()]
                    n = min(2, ls.shape[0])           # steer and throttle only; the gear keeps its own
                    ls[:n] = torch.maximum(ls[:n], torch.full_like(ls[:n], math.log(a.explore_std)))
                    after = [round(v, 3) for v in ls.exp().tolist()]
                print(f"[train] exploration: action noise {before} -> {after}")
        else:
            model = PPO("MlpPolicy", env, verbose=1, seed=7,
                        n_steps=1024, batch_size=256, learning_rate=3e-4)
        model.learn(total_timesteps=a.steps,
                    callback=EpisodeLogger(fresh=not a.resume),
                    reset_num_timesteps=not a.resume)
        model.save(MODEL)
        print(f"[train] DONE — model at {MODEL}")
        print(f"[train] curriculum ended at: {curriculum.describe()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
