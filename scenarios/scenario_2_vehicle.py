"""
Scenario 2: Vehicle Ahead
Spawns a stopped vehicle in the ego vehicle's path.
The system should detect it and stop.
"""
import carla
import requests
import time

API = "http://localhost:5000"

# Connect to CARLA directly to spawn obstacle vehicle
client = carla.Client("localhost", 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Get ego vehicle position from API
state = requests.get(f"{API}/api/state").json()
ego_x = state['pose']['x']
ego_y = state['pose']['y']
ego_yaw = state['pose']['yaw']

# Start a mission first
points = requests.get(f"{API}/api/spawn_points").json()
requests.post(f"{API}/api/mission/start", json={"x": points[5]['x'], "y": points[5]['y']})
print("Mission started, waiting 3s for vehicle to start moving...")
time.sleep(3)

# Get updated position
state = requests.get(f"{API}/api/state").json()
ego_x = state['pose']['x']
ego_y = state['pose']['y']

# Spawn a stopped car 20m ahead on the road
import math
yaw_rad = math.radians(state['pose']['yaw'])
block_x = ego_x + math.cos(yaw_rad) * 20
block_y = ego_y + math.sin(yaw_rad) * 20

bp = bp_lib.filter('vehicle.tesla.model3')[0]
spawn = carla.Transform(carla.Location(x=block_x, y=block_y, z=0.5))
blocker = world.spawn_actor(bp, spawn)
print(f"Spawned blocking vehicle at ({block_x:.1f}, {block_y:.1f})")

# Monitor response
for i in range(60):
    state = requests.get(f"{API}/api/state").json()
    behavior = state.get('behavior', '?')
    reason = state.get('behavior_reason', '')
    print(f"  [{i*0.5:.1f}s] {behavior} | {reason}")
    if 'VEHICLE' in reason.upper() or 'stopped_vehicle' in behavior:
        print("\n✓ Vehicle detected and stopped!")
        break
    time.sleep(0.5)

# Cleanup
blocker.destroy()
print("Blocker removed")
