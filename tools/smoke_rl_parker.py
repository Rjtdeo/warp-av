"""Load the learned parker with the real brain and take a few decisions.
No CARLA needed. Run on the box the stack runs on."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from warp_av.planning.rl_parker import RLParker

p = RLParker()
print("brain:", p.model_path, "exists:", os.path.exists(p.model_path))
slot = dict(x=100.0, y=50.0, yaw=0.0, length=7.0, width=2.5)
t = time.time(); out = p.act(84.0, 47.0, 0.0, 0.0, slot); print(f"first decision (loads the brain): {time.time()-t:.2f} s ->", {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items() if k != "reason"}, "|", out["reason"])
t = time.time()
for i in range(50):
    out = p.act(84.0 + i * 0.3, 47.0 + i * 0.06, 0.0, 2.0, slot)
print(f"50 more decisions: {(time.time()-t)*1000/50:.2f} ms each | last:", out["reason"])
