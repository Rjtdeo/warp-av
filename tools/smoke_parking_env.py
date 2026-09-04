"""
Ninety-second proof that the practice arena works before an overnight run.

    python tools\smoke_parking_env.py

Builds the arena with neighbours forced on, resets at the full distance and at
the easiest rung, drives straight ahead for a few seconds, and prints what the
student sees. Restores the simulator on exit. Run INSTEAD of the trainer.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from rl.parking_env import CarlaParkingEnv

env = CarlaParkingEnv(seed=5, exam_p=0.0, neighbour_p=1.0, neighbour_ahead_p=1.0)
try:
    print(f"observation has {env.observation_space.shape[0]} numbers")
    for p in (0.0, 0.0, 0.75):
        obs, _ = env.reset(options={"p": p})
        print(f"\np={p}: start {env._start_dist:.1f} m, neighbours spawned {len(env.neighbours)}, "
              f"feelers {np.round(obs[5:], 2)}")
        for k in range(50):
            obs, r, done, _, info = env.step(np.array([0.0, 0.6], dtype=np.float32))
            if k % 10 == 9 or done:
                tail = f" -> {info['result']} {info.get('hit', '')}" if done else ""
                print(f"   step {k + 1:2d}: feelers {np.round(obs[5:], 2)}  reward {r:6.1f}{tail}")
            if done:
                break
    print("\nsmoke test finished cleanly")
finally:
    env.close()
