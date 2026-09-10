"""
A sandbox for measurement scripts that spawn their own van and sensors.

Use this instead of talking to CARLA directly whenever a script needs to put its own
Sprinter in the world, hang a camera or LiDAR off it, or step the simulator by hand.

    from scratch_world import ScratchWorld

    with ScratchWorld() as sim:
        van = sim.spawn(sim.blueprint("vehicle.mercedes.sprinter"), transform)
        cam = sim.camera(van, width=800, height=600, fov=90.0)
        sim.set_state(light, carla.TrafficLightState.Red)
        image = cam.frame()                 # a picture that shows THIS state, not the last one
    # everything the script made is gone; the world is as it was found

Three things it does that every one of the scripts it replaces got wrong at least once:

1. It REFUSES TO RUN WHILE THE STACK IS UP. The old cleanup was "destroy every sensor.*
   actor in the world", which is the running van's own camera and LiDAR. That put
   "camera not working -- driving slowly" into live runs on 2026-09-10 and capped the van
   at 2 m/s for fifteen seconds while nobody could see why. Lock-step mode does the same
   kind of harm more quietly: the stack stops receiving frames and looks dead. And a second
   Sprinter in the world is the leftover-van trap, which cost about an hour twice in one
   session. None of these can happen if the stack is not running, so that is the rule.

2. It cleans up ONLY WHAT IT SPAWNED, by actor id. Never by pattern.

3. Its camera hands back a frame that was RENDERED AFTER the last tick, matched by frame
   number. The first light-colour measurement read every colour one step out of order
   because it took whichever frame had arrived.

It also puts the world's settings and weather back exactly as they were, even on a crash.
"""
from __future__ import annotations

import json
import queue
import sys
import time
import urllib.request
from typing import Any, List, Optional


def stack_is_up(api: str = "http://127.0.0.1:5000", timeout: float = 2.0) -> bool:
    """Does the van's own API answer? If it does, the stack owns the world."""
    try:
        json.load(urllib.request.urlopen(api + "/api/state", timeout=timeout))
        return True
    except Exception:
        return False


class _Grabber:
    """A camera whose `frame()` is the picture rendered after the most recent tick."""

    def __init__(self, sim: "ScratchWorld", actor):
        self.actor = actor
        self._sim = sim
        self._q: "queue.Queue" = queue.Queue()
        actor.listen(self._q.put)

    def frame(self, settle_ticks: int = 3, timeout: float = 10.0):
        """Tick a few times so the render catches up with any state change, then return the
        image whose frame number is at least the last tick's. Never a stale one."""
        image = None
        for _ in range(max(1, settle_ticks)):
            frame_no = self._sim.tick()
            while True:
                image = self._q.get(timeout=timeout)
                if image.frame >= frame_no:
                    break
        return image

    def stop(self) -> None:
        try:
            self.actor.stop()
        except Exception:
            pass


class ScratchWorld:
    def __init__(self, host: str = "127.0.0.1", port: int = 2000,
                 api: str = "http://127.0.0.1:5000", sync: bool = True,
                 fixed_dt: float = 0.05, allow_running_stack: bool = False,
                 world: Any = None):
        self.host, self.port, self.api = host, port, api
        self.sync, self.fixed_dt = sync, fixed_dt
        self.allow_running_stack = allow_running_stack
        self.world = world                  # tests hand in a fake; real use connects
        self.client = None
        self._mine: List[Any] = []          # actors THIS script made, newest last
        self._grabbers: List[_Grabber] = []
        self._settings = None
        self._weather = None
        self._frozen: List[Any] = []

    # ---- lifecycle ---------------------------------------------------------------------
    def __enter__(self) -> "ScratchWorld":
        if not self.allow_running_stack and stack_is_up(self.api):
            sys.exit("the van's stack is running. This script spawns its own van and sensors "
                     "and steps the simulator by hand, which breaks the stack's camera, LiDAR "
                     "and clock. Stop the stack first (schtasks /End /TN WarpAVStack), or pass "
                     "allow_running_stack=True if you really know what you are doing.")
        if self.world is None:
            import carla
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(60.0)
            self.world = self.client.get_world()
        self._settings = self.world.get_settings()
        try:
            self._weather = self.world.get_weather()
        except Exception:
            self._weather = None
        if self.sync:
            s = self.world.get_settings()
            s.synchronous_mode = True
            s.fixed_delta_seconds = self.fixed_dt
            self.world.apply_settings(s)
        return self

    def __exit__(self, *exc) -> None:
        for g in self._grabbers:
            g.stop()
        # newest first, so a sensor goes before the van it is attached to
        for actor in reversed(self._mine):
            try:
                actor.destroy()
            except Exception:
                pass
        self._mine.clear()
        for light in self._frozen:
            try:
                light.freeze(False)
            except Exception:
                pass
        self._frozen.clear()
        try:
            if self._weather is not None:
                self.world.set_weather(self._weather)
        finally:
            if self._settings is not None:
                self.world.apply_settings(self._settings)
        return None

    # ---- the world ---------------------------------------------------------------------
    def tick(self) -> int:
        if self.sync:
            return int(self.world.tick())
        time.sleep(self.fixed_dt)
        return int(self.world.get_snapshot().frame)

    def blueprint(self, name: str):
        lib = self.world.get_blueprint_library()
        found = lib.filter(name)
        if not found:
            raise KeyError(f"no blueprint matches {name!r}")
        return found[0]

    def spawn(self, blueprint, transform, attach_to=None):
        """Spawn and remember. Raises if CARLA refuses (something is in the way)."""
        actor = self.world.spawn_actor(blueprint, transform, attach_to=attach_to)
        self._mine.append(actor)
        return actor

    def try_spawn(self, blueprint, transform, attach_to=None):
        """Spawn and remember, or None if something is in the way."""
        actor = self.world.try_spawn_actor(blueprint, transform, attach_to=attach_to)
        if actor is not None:
            self._mine.append(actor)
        return actor

    def camera(self, van, width: int = 800, height: int = 600, fov: float = 90.0,
               mount=(2.0, 0.0, 1.8), pitch_deg: float = -10.0) -> _Grabber:
        """The van's front camera, exactly as the sensor adapter mounts it."""
        import carla
        bp = self.blueprint("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        tf = carla.Transform(carla.Location(*mount), carla.Rotation(pitch=pitch_deg))
        actor = self.spawn(bp, tf, attach_to=van)
        g = _Grabber(self, actor)
        self._grabbers.append(g)
        return g

    def freeze_lights(self, lights) -> None:
        """Hold every light where it is, and let them all go again at exit."""
        for light in lights:
            light.freeze(True)
            self._frozen.append(light)

    def set_state(self, light, state) -> None:
        light.set_state(state)

    @property
    def spawned(self) -> List[Any]:
        return list(self._mine)
