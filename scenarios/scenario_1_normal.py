"""
Scenario 1: Normal Mission
Vehicle receives a destination and drives there successfully.
"""
import requests
import time

API = "http://localhost:5000"

# Get available destinations
points = requests.get(f"{API}/api/spawn_points").json()
dest = points[5]  # Pick a destination that's not too far
print(f"Sending vehicle to Point {dest['idx']}: ({dest['x']}, {dest['y']})")

# Start mission
resp = requests.post(f"{API}/api/mission/start", json={"x": dest['x'], "y": dest['y']})
print(f"Mission started: {resp.json()}")

# Monitor until complete
for i in range(300):  # 60 second timeout
    state = requests.get(f"{API}/api/state").json()
    if state.get('mission', {}).get('state') == 'idle' and i > 5:
        print("Mission completed!")
        break
    behavior = state.get('behavior', '?')
    speed = state.get('pose', {}).get('speed', 0)
    reason = state.get('behavior_reason', '')
    print(f"  [{i*0.2:.1f}s] {behavior} | {speed:.1f} m/s | {reason}")
    time.sleep(0.2)
else:
    print("Timeout — mission did not complete in 60s")
