import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Provide a minimal fake `carla` module so adapter/controller code imports without the simulator.
if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    class _VC:
        def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
            self.throttle, self.steer, self.brake, self.hand_brake, self.reverse = throttle, steer, brake, hand_brake, reverse
    class _Loc:
        def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
    class _Rot:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0): self.pitch, self.yaw, self.roll = pitch, yaw, roll
    class _Tr:
        def __init__(self, location=None, rotation=None): self.location, self.rotation = location or _Loc(), rotation or _Rot()
    carla.VehicleControl, carla.Location, carla.Rotation, carla.Transform = _VC, _Loc, _Rot, _Tr
    carla.Client = object
    sys.modules["carla"] = carla
