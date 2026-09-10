"""The sandbox for measurement scripts must never touch what it did not make.

The habit it replaces was "destroy every sensor.* in the world" as cleanup. That is the
running van's own camera and LiDAR, and it put "camera not working -- driving slowly" into
live runs on 2026-09-10.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import scratch_world as sw                                      # noqa: E402


class FakeActor:
    def __init__(self, ident, parent=None):
        self.id = ident
        self.alive = True
        self.listening = False

    def destroy(self):
        self.alive = False

    def listen(self, cb):
        self.listening = True

    def stop(self):
        self.listening = False


class FakeSettings:
    def __init__(self):
        self.synchronous_mode = False
        self.fixed_delta_seconds = None


class FakeWorld:
    def __init__(self):
        self.settings = FakeSettings()
        self.weather = "sunny"
        self.actors = [FakeActor(1), FakeActor(2)]          # the stack's van and its camera
        self._next = 100
        self.frame = 0

    def get_settings(self):
        s = FakeSettings()
        s.synchronous_mode = self.settings.synchronous_mode
        s.fixed_delta_seconds = self.settings.fixed_delta_seconds
        return s

    def apply_settings(self, s):
        self.settings.synchronous_mode = s.synchronous_mode
        self.settings.fixed_delta_seconds = s.fixed_delta_seconds

    def get_weather(self):
        return self.weather

    def set_weather(self, w):
        self.weather = w

    def spawn_actor(self, bp, tf, attach_to=None):
        a = FakeActor(self._next)
        self._next += 1
        self.actors.append(a)
        return a

    def try_spawn_actor(self, bp, tf, attach_to=None):
        return self.spawn_actor(bp, tf, attach_to)

    def tick(self):
        self.frame += 1
        return self.frame


def test_it_refuses_to_run_while_the_stack_is_up(monkeypatch):
    monkeypatch.setattr(sw, "stack_is_up", lambda api="", timeout=2.0: True)
    with pytest.raises(SystemExit) as why:
        with sw.ScratchWorld(world=FakeWorld()):
            pass
    assert "stack is running" in str(why.value)


def test_it_destroys_only_what_it_made(monkeypatch):
    monkeypatch.setattr(sw, "stack_is_up", lambda api="", timeout=2.0: False)
    world = FakeWorld()
    theirs = list(world.actors)
    with sw.ScratchWorld(world=world) as sim:
        van = sim.spawn("bp", "tf")
        cam = sim.spawn("bp", "tf", attach_to=van)
        assert van.alive and cam.alive
    assert not van.alive and not cam.alive, "its own actors should be gone"
    assert all(a.alive for a in theirs), "it touched an actor that was not its own"


def test_it_puts_the_world_back_even_after_a_crash(monkeypatch):
    monkeypatch.setattr(sw, "stack_is_up", lambda api="", timeout=2.0: False)
    world = FakeWorld()
    world.weather = "rain"
    with pytest.raises(RuntimeError):
        with sw.ScratchWorld(world=world, fixed_dt=0.05) as sim:
            assert world.settings.synchronous_mode is True
            world.set_weather("fog")
            sim.spawn("bp", "tf")
            raise RuntimeError("the script fell over")
    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None
    assert world.weather == "rain"
    assert all(a.alive for a in world.actors[:2])


def test_a_frame_is_never_older_than_the_last_tick(monkeypatch):
    """The first light-colour measurement read every colour one step out of order because it
    took whichever picture had arrived. The grabber waits for one rendered after the tick."""
    monkeypatch.setattr(sw, "stack_is_up", lambda api="", timeout=2.0: False)
    world = FakeWorld()

    class Img:
        def __init__(self, frame):
            self.frame = frame

    with sw.ScratchWorld(world=world) as sim:
        cam_actor = sim.spawn("bp", "tf")
        g = sw._Grabber(sim, cam_actor)
        # the queue already holds two STALE pictures from before any tick
        g._q.put(Img(0))
        g._q.put(Img(0))
        # and, after the ticks the grabber will do, the fresh one arrives
        def tick_and_deliver():
            f = world.tick()
            g._q.put(Img(f))
            return f
        sim.tick = tick_and_deliver
        got = g.frame(settle_ticks=2)
    assert got.frame == world.frame, "handed back a picture from before the tick"
