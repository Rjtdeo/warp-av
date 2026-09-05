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
import glob
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
        self._last_snapshot = 0
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
            # Keep a rolling history too: round 8g's last five minutes went bad
            # (every attempt timing out) and the only copy of the brain was
            # the one saved after them. Every 100k steps, a dated snapshot;
            # the newest five are kept.
            if self.num_timesteps - self._last_snapshot >= 100000:
                self._last_snapshot = self.num_timesteps
                snap = MODEL.replace(".zip", f"_ckpt_{self.num_timesteps // 1000}k.zip")
                self.model.save(snap)
                snaps = sorted(glob.glob(MODEL.replace(".zip", "_ckpt_*k.zip")),
                               key=lambda f: int(f.rsplit("_ckpt_", 1)[1][:-5]))
                for old in snaps[:-5]:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
                print(f"[train] snapshot {os.path.basename(snap)} (keeping the newest five)")
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
    ap.add_argument("--max-stage", type=int, default=2,
                    help="highest obstacle stage the ladder may unlock (1 = never the car right behind)")
    ap.add_argument("--lr", type=float, default=None,
                    help="on resume, set the learning rate (e.g. 1.5e-4 late in training)")
    ap.add_argument("--start-rung", type=int, default=0,
                    help="stage-1 rung to begin on: 0 = car 4 bays back, 1 = 3, 2 = 2")
    ap.add_argument("--explore-std", type=float, default=0.0,
                    help="on resume, raise steer/throttle action noise to at least this "
                         "(a trained brain barely explores; a new skill needs variety)")
    ap.add_argument("--lane-start-jitter", type=float, default=0.0,
                    help="spread every far start over [start, start + this] metres (train only)")
    ap.add_argument("--teacher-weight", type=float, default=0.0,
                    help="round 9: pull toward the rule-based teacher after each rollout, fading to 0")
    ap.add_argument("--teacher-fade", type=int, default=1_000_000,
                    help="steps over which --teacher-weight fades to zero")
    ap.add_argument("--sides", choices=("right", "left", "both"), default="right",
                    help="which side of the road the practice bays are on")
    ap.add_argument("--lane-start", type=float, default=16.0,
                    help="metres back along the lane the far starts begin (22 for the car two bays back)")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    curriculum = Curriculum(start_level=a.start_level)
    from rl.parking_math import Stages
    env = CarlaParkingEnv(curriculum=curriculum, reverse=a.reverse, obstacles=a.obstacles,
                          lane_start_m=a.lane_start, lane_start_jitter_m=a.lane_start_jitter,
                          sides=("right", "left") if a.sides == "both" else (a.sides,),
                          stages=Stages(start=a.start_stage, start_rung=a.start_rung, max_level=a.max_stage)
                          if a.obstacles else None)
    if a.obstacles:
        print(f"[train] {env.stages.describe()}")
    print(f"[train] curriculum: {curriculum.describe()}")
    try:
        if a.resume and os.path.exists(MODEL):
            print(f"[train] resuming from {MODEL}")
            model = PPO.load(MODEL, env=env)
            if a.lr is not None:
                from stable_baselines3.common.utils import constant_fn
                model.learning_rate = a.lr
                model.lr_schedule = constant_fn(a.lr)
                print(f"[train] learning rate set to {a.lr}")
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
        callbacks = [EpisodeLogger(fresh=not a.resume)]
        if a.teacher_weight > 0:
            from rl.teacher_callback import TeacherCallback
            callbacks.append(TeacherCallback(weight=a.teacher_weight, fade_steps=a.teacher_fade, verbose=1))
            print(f"[train] staying close to the teacher: weight {a.teacher_weight}, fading over {a.teacher_fade} steps")
        model.learn(total_timesteps=a.steps,
                    callback=callbacks,
                    reset_num_timesteps=not a.resume)
        model.save(MODEL)
        print(f"[train] DONE — model at {MODEL}")
        print(f"[train] curriculum ended at: {curriculum.describe()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
