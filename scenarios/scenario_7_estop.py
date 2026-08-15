"""
Scenario 7: Emergency Stop
Trigger E-STOP during a mission. Vehicle must stop immediately.
"""
import requests
import time

API = "http://localhost:5000"

# Start a mission
points = requests.get(f"{API}/api/spawn_points").json()
requests.post(f"{API}/api/mission/start", json={"x": points[5]['x'], "y": points[5]['y']})
print("Mission started, waiting for vehicle to reach speed...")
time.sleep(5)

state = requests.get(f"{API}/api/state").json()
print(f"Speed before E-STOP: {state['pose']['speed']:.1f} m/s")

# Trigger E-STOP
print("\n*** TRIGGERING EMERGENCY STOP ***")
requests.post(f"{API}/api/estop")

# Monitor response
for i in range(20):
    state = requests.get(f"{API}/api/state").json()
    speed = state.get('pose', {}).get('speed', 0)
    safety = state.get('safety', {}).get('state', '?')
    reason = state.get('safety', {}).get('reason', '')
    print(f"  [{i*0.2:.1f}s] speed={speed:.1f} m/s | safety={safety} | {reason}")
    if speed < 0.1:
        print("\n✓ Vehicle stopped!")
        break
    time.sleep(0.2)

# Clear E-STOP
print("\nClearing E-STOP...")
requests.post(f"{API}/api/estop/clear")
state = requests.get(f"{API}/api/state").json()
print(f"Safety state after clear: {state['safety']['state']}")
