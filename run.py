#!/usr/bin/env python3
"""
Run Warp AV — simple entry point.
Usage: python3 run.py
"""
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# CARLA's PythonAPI extras (agents.navigation.*) — the planner needs them.
# Honor CARLA_PYTHONAPI if set, otherwise check the standard install spots,
# so the stack starts identically from any shell, SSH session, or scheduler.
_candidates = [os.environ.get("CARLA_PYTHONAPI"),
               r"C:\CARLA\WindowsNoEditor\PythonAPI\carla",
               os.path.expanduser("~/CARLA/PythonAPI/carla")]
for _c in _candidates:
    if _c and os.path.isdir(os.path.join(_c, "agents")):
        sys.path.append(_c)
        break

from warp_av.main import main

if __name__ == "__main__":
    main()
