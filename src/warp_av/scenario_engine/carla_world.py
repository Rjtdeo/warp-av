"""
CARLA-side helpers for the scenario runner: resolve spawn specs into transforms,
drive scenario actors (lead cars, walkers, props), watch collisions.

This is the ONLY scenario-engine module that imports `carla`.  The runner keeps
all timeline / trigger / evaluation logic simulator-agnostic so a future
"physical test range" backend can implement the same three calls:
    resolve_location(spec)  spawn(actor_spec)  step(dt)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import carla


class WorldHelper:
    def __init__(self, host: str, port: int, ego_hint_xy: Tuple[float, float] | None = None):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp = self.world.get_blueprint_library()
        self.ego = self._find_ego(ego_hint_xy)
        self.spawned: Dict[str, carla.Actor] = {}
        self.collisions: List[dict] = []
        self._collision_sensor = None
        self._tm = None
        self._route_xy: List[Tuple[float, float]] = []
        if self.ego is not None:
            self._attach_collision_sensor()

    # ------------------------------------------------------------------ ego
    def _find_ego(self, hint):
        """The autonomy stack does not tag its vehicle; pick the vehicle closest to the API pose."""
        vehicles = self.world.get_actors().filter("vehicle.*")
        if not vehicles:
            return None
        if hint is None:
            return vehicles[0]
        hx, hy = hint
        return min(vehicles, key=lambda v: (v.get_location().x - hx) ** 2 + (v.get_location().y - hy) ** 2)

    def _attach_collision_sensor(self):
        bp = self.bp.find("sensor.other.collision")
        self._collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, ev):
        other = ev.other_actor
        self.collisions.append({"t": time.time(), "other": other.type_id if other else "?",
                               "other_id": other.id if other else -1,
                               "impulse": math.sqrt(ev.normal_impulse.x ** 2 + ev.normal_impulse.y ** 2 + ev.normal_impulse.z ** 2)})

    def ego_xy_yaw(self):
        t = self.ego.get_transform()
        return t.location.x, t.location.y, math.radians(t.rotation.yaw)

    def town(self) -> str:
        return self.map.name.split("/")[-1]

    # ------------------------------------------------------------------ weather
    def set_weather(self, preset: str | None = None, **overrides):
        w = getattr(carla.WeatherParameters, preset) if preset else self.world.get_weather()
        for k, v in overrides.items():
            if hasattr(w, k):
                setattr(w, k, v)
        self.world.set_weather(w)

    def set_traffic_lights(self, state: str):
        """Force the traffic light nearest to ego (and its group) into a state. Used by traffic_control scenarios."""
        lights = self.world.get_actors().filter("traffic.traffic_light")
        if not lights:
            return False
        ex, ey, _ = self.ego_xy_yaw()
        tl = min(lights, key=lambda l: (l.get_location().x - ex) ** 2 + (l.get_location().y - ey) ** 2)
        mapping = {"red": carla.TrafficLightState.Red, "green": carla.TrafficLightState.Green,
                   "yellow": carla.TrafficLightState.Yellow}
        base = state.split("_then_")[0].replace("_on_approach", "").replace("all_", "").replace("_flash", "")
        tl.set_state(mapping.get(base, carla.TrafficLightState.Red))
        tl.freeze(True)
        return True

    # ------------------------------------------------------------------ geometry
    def set_route(self, xy: List[Tuple[float, float]]):
        self._route_xy = xy

    def _waypoint_at_route_distance(self, s: float, flags: dict) -> carla.Waypoint:
        """Waypoint `s` metres along the planned route (or along the lane ahead of ego if no route yet)."""
        if self._route_xy and s >= 0:
            acc = 0.0
            prev = self._route_xy[0]
            target = None
            for xy in self._route_xy[1:]:
                acc += math.dist(prev, xy)
                prev = xy
                if acc >= s:
                    target = xy
                    break
            if target is None:
                target = self._route_xy[-1]
            wp = self.map.get_waypoint(carla.Location(x=target[0], y=target[1], z=0.0))
        else:
            ex, ey, _ = self.ego_xy_yaw()
            wp = self.map.get_waypoint(carla.Location(x=ex, y=ey, z=0.0))
            remaining = abs(s)
            step = 2.0
            while remaining > 0:
                nxt = wp.next(step) if s >= 0 else wp.previous(step)
                if not nxt:
                    break
                wp = nxt[0]
                remaining -= step
        if flags.get("at_next_junction"):
            probe, n = wp, 0
            while not probe.is_junction and n < 200:
                nx = probe.next(2.0)
                if not nx:
                    break
                probe, n = nx[0], n + 1
            wp = probe
        if flags.get("after_curve"):
            probe, n, yaw0 = wp, 0, wp.transform.rotation.yaw
            while n < 200:
                nx = probe.next(2.0)
                if not nx:
                    break
                probe, n = nx[0], n + 1
                if abs((probe.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 30:
                    nx = probe.next(8.0)
                    probe = nx[0] if nx else probe
                    break
            wp = probe
        return wp

    def resolve_location(self, spec: dict) -> carla.Transform:
        mode = spec.get("mode")
        if mode == "route_ahead":
            wp = self._waypoint_at_route_distance(float(spec.get("distance_m", 0.0)), spec)
            tr = wp.transform
        elif mode == "at_destination":
            if not self._route_xy:
                raise RuntimeError("at_destination needs a planned route")
            total = sum(math.dist(a, b) for a, b in zip(self._route_xy, self._route_xy[1:]))
            wp = self._waypoint_at_route_distance(total + float(spec.get("offset_m", 0.0)), spec)
            tr = wp.transform
        elif mode == "absolute":
            if "spawn_point_index" in spec:
                sps = self.map.get_spawn_points()
                tr = sps[spec["spawn_point_index"] % len(sps)]
                if spec.get("sidewalk"):
                    loc = self.world.get_random_location_from_navigation()
                    if loc:
                        tr = carla.Transform(loc, tr.rotation)
            else:
                tr = carla.Transform(carla.Location(x=spec["x"], y=spec["y"], z=spec.get("z", 0.5)),
                                     carla.Rotation(yaw=spec.get("yaw_deg", 0.0)))
        elif mode == "relative_to_ego":
            ex, ey, eyaw = self.ego_xy_yaw()
            a, l = float(spec.get("ahead_m", 10.0)), float(spec.get("lateral_m", 0.0))
            x = ex + math.cos(eyaw) * a - math.sin(eyaw) * l
            y = ey + math.sin(eyaw) * a + math.cos(eyaw) * l
            tr = carla.Transform(carla.Location(x=x, y=y, z=0.5), carla.Rotation(yaw=math.degrees(eyaw)))
        elif mode == "ego_current":
            ex, ey, eyaw = self.ego_xy_yaw()
            return carla.Transform(carla.Location(x=ex, y=ey, z=0.0), carla.Rotation(yaw=math.degrees(eyaw)))
        elif mode == "spawn_point":
            sps = self.map.get_spawn_points()
            if spec["index"] >= len(sps):
                raise IndexError(f"spawn point {spec['index']} out of range ({len(sps)})")
            return sps[spec["index"]]
        elif mode == "off_map":
            return carla.Transform(carla.Location(x=9000.0, y=9000.0, z=0.0))
        else:
            raise ValueError(f"unknown location mode {mode}")

        # lateral offset + yaw offset
        lat = float(spec.get("lateral_m", 0.0))
        if lat:
            rv = tr.get_right_vector()
            tr = carla.Transform(carla.Location(x=tr.location.x + rv.x * lat, y=tr.location.y + rv.y * lat, z=tr.location.z),
                                 carla.Rotation(yaw=tr.rotation.yaw))
        yo = float(spec.get("yaw_offset_deg", 0.0))
        return carla.Transform(carla.Location(x=tr.location.x, y=tr.location.y, z=tr.location.z),
                               carla.Rotation(yaw=tr.rotation.yaw + yo))

    # ------------------------------------------------------------------ actors
    def spawn(self, spec: dict) -> Optional[carla.Actor]:
        tr = self.resolve_location(spec["spawn"])
        z_off = {"vehicle": 0.6, "pedestrian": 1.0, "prop": 0.3}[spec["type"]]
        tr = carla.Transform(carla.Location(x=tr.location.x, y=tr.location.y, z=tr.location.z + z_off), tr.rotation)
        bps = self.bp.filter(spec["blueprint"])
        if not bps:
            # fall back to a same-class blueprint so the scenario still runs
            fallback = {"vehicle": "vehicle.tesla.model3", "pedestrian": "walker.pedestrian.0001", "prop": "static.prop.trafficcone01"}[spec["type"]]
            bps = self.bp.filter(fallback)
        bp = bps[0]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"scenario_{spec['name']}")
        actor = self.world.try_spawn_actor(bp, tr)
        if actor is None:
            # nudge up and retry once (common for props on uneven ground)
            tr.location.z += 0.8
            actor = self.world.try_spawn_actor(bp, tr)
        if actor is None:
            return None
        self.spawned[spec["name"]] = actor
        return actor

    def destroy(self, name: str):
        a = self.spawned.pop(name, None)
        if a is not None:
            try:
                a.destroy()
            except Exception:
                pass

    def positions(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for n, a in list(self.spawned.items()):
            try:
                l = a.get_location()
                out[n] = (l.x, l.y)
            except Exception:
                pass
        return out

    def ego_distance_to(self, name: str) -> float:
        a = self.spawned.get(name)
        if a is None:
            return 999.0
        ex, ey, _ = self.ego_xy_yaw()
        l = a.get_location()
        return math.dist((ex, ey), (l.x, l.y))

    def traffic_manager(self):
        if self._tm is None:
            self._tm = self.client.get_trafficmanager()
        return self._tm

    def cleanup(self):
        for n in list(self.spawned):
            self.destroy(n)
        if self._collision_sensor is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Actor behaviours.  One controller per actor; step(dt) called every poll.
# ----------------------------------------------------------------------

class ActorController:
    def __init__(self, wh: WorldHelper, spec: dict):
        self.wh = wh
        self.spec = spec
        self.name = spec["name"]
        self.kind = spec["behavior"]["kind"]
        self.p = spec["behavior"]
        self.actor: Optional[carla.Actor] = None
        self.triggered = False          # trigger fired (behaviour started)
        self.trigger_time: Optional[float] = None
        self.t_phase = 0.0
        self.phase = 0
        self.travelled = 0.0
        self.start_xy = None
        self.done = False
        self.spawn_failed = False

    # --- lifecycle
    def spawn_now(self):
        if self.kind == "appear":
            return  # spawns on trigger
        self.actor = self.wh.spawn(self.spec)
        if self.actor is None:
            self.spawn_failed = True
            return
        if self.spec["type"] == "vehicle":
            if self.kind in ("constant_speed", "brake_hard", "cut_in", "cut_out", "oncoming") and "trigger" not in self.spec:
                self._start_constant(self.p.get("speed_mps", 5.0))
            elif self.kind == "autopilot":
                self.actor.set_autopilot(True, self.wh.traffic_manager().get_port())
            else:
                self.actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        elif self.spec["type"] == "pedestrian":
            if self.kind == "walk_along" and "trigger" not in self.spec:
                self._walk_along()
        if "trigger" not in self.spec:
            self.triggered = True
            self.trigger_time = time.time()

    def fire(self):
        """Trigger fired: start the behaviour."""
        if self.triggered:
            return
        self.triggered = True
        self.trigger_time = time.time()
        self.t_phase = 0.0
        self.phase = 0
        if self.kind == "appear":
            self.actor = self.wh.spawn(self.spec)
            if self.actor is None:
                self.spawn_failed = True
            return
        if self.actor is None:
            return
        if self.spec["type"] == "vehicle":
            if self.kind in ("constant_speed", "oncoming"):
                self._start_constant(self.p.get("speed_mps", 5.0))
            elif self.kind == "brake_hard":
                self._stop_constant()
                self.actor.apply_control(carla.VehicleControl(brake=1.0))
            elif self.kind in ("cut_in", "cut_out"):
                self._stop_constant()
            elif self.kind == "reverse":
                self.actor.apply_control(carla.VehicleControl(throttle=0.35, reverse=True))
        elif self.spec["type"] == "pedestrian":
            if self.kind in ("cross_road", "dart_out", "cross_and_stop"):
                self._walk_toward_lane()
            elif self.kind == "walk_along":
                self._walk_along()
        # remove_after handled in step()

    def step(self, dt: float):
        if self.done or self.actor is None:
            return
        self.t_phase += dt
        try:
            if self.kind == "remove_after" and self.triggered and self.t_phase >= float(self.p.get("delay_s", 5.0)):
                self.wh.destroy(self.name)
                self.actor = None
                self.done = True
            elif self.kind in ("cut_in", "cut_out") and self.triggered:
                self._step_cut()
            elif self.kind == "brake_hard" and not self.triggered and "trigger" in self.spec and self.t_phase < 0.2:
                self._start_constant(self.p.get("speed_mps", 5.0))   # drive until trigger
            elif self.kind in ("cross_road", "dart_out", "cross_and_stop") and self.triggered:
                self._step_walker()
            elif self.kind == "reverse" and self.triggered and self.t_phase > 6.0:
                self.actor.apply_control(carla.VehicleControl(brake=1.0, reverse=False))
        except Exception:
            self.done = True

    # --- vehicle helpers
    def _start_constant(self, v):
        try:
            self.actor.enable_constant_velocity(carla.Vector3D(float(v), 0.0, 0.0))
        except Exception:
            self.actor.apply_control(carla.VehicleControl(throttle=min(1.0, float(v) / 10.0)))

    def _stop_constant(self):
        try:
            self.actor.disable_constant_velocity()
        except Exception:
            pass

    def _step_cut(self):
        side = self.p.get("side", "left")
        into = 1.0 if (self.kind == "cut_in") == (side == "left") else -1.0   # +steer = right
        v = float(self.p.get("speed_mps", 6.0))
        thr = min(1.0, 0.35 + v / 20.0)
        if self.phase == 0 and self.t_phase < 0.9:
            self.actor.apply_control(carla.VehicleControl(throttle=thr, steer=0.35 * into))
        elif self.phase == 0:
            self.phase, self.t_phase = 1, 0.0
        elif self.phase == 1 and self.t_phase < 0.9:
            self.actor.apply_control(carla.VehicleControl(throttle=thr, steer=-0.35 * into))
        elif self.phase == 1:
            self.phase, self.t_phase = 2, 0.0
        elif self.phase == 2:
            if self.p.get("then_brake") or self.kind == "cut_out":
                self.actor.apply_control(carla.VehicleControl(brake=1.0))
            else:
                self._start_constant(v)
            self.done = True

    # --- walker helpers
    def _lane_dir(self):
        """Unit vector from the walker toward the lane centre (perpendicular to the lane)."""
        lat = float(self.spec["spawn"].get("lateral_m", 3.0))
        tr = self.actor.get_transform()
        wp = self.wh.map.get_waypoint(tr.location)
        rv = wp.transform.get_right_vector()
        sgn = -1.0 if lat > 0 else 1.0
        return carla.Vector3D(rv.x * sgn, rv.y * sgn, 0.0), abs(lat)

    def _walk_toward_lane(self):
        d, lat = self._lane_dir()
        self.start_xy = (self.actor.get_location().x, self.actor.get_location().y)
        self._cross_total = 2 * lat + 1.0
        self._dwell_point = lat
        self.actor.apply_control(carla.WalkerControl(direction=d, speed=float(self.p.get("speed_mps", 1.3))))

    def _walk_along(self):
        tr = self.actor.get_transform()
        wp = self.wh.map.get_waypoint(tr.location)
        fv = wp.transform.get_forward_vector()
        if self.p.get("direction") == "toward":
            fv = carla.Vector3D(-fv.x, -fv.y, 0.0)
        self.actor.apply_control(carla.WalkerControl(direction=fv, speed=float(self.p.get("speed_mps", 1.2))))

    def _step_walker(self):
        l = self.actor.get_location()
        self.travelled = math.dist(self.start_xy, (l.x, l.y))
        if self.kind == "cross_and_stop":
            dwell = float(self.p.get("dwell_s", 10.0))
            if self.phase == 0 and self.travelled >= self._dwell_point:
                self.actor.apply_control(carla.WalkerControl(speed=0.0))
                self.phase, self.t_phase = 1, 0.0
            elif self.phase == 1 and self.t_phase >= dwell:
                d, _ = self._lane_dir()
                self.actor.apply_control(carla.WalkerControl(direction=d, speed=float(self.p.get("speed_mps", 1.3))))
                self.phase = 2
            elif self.phase == 2 and self.travelled >= self._cross_total:
                self.actor.apply_control(carla.WalkerControl(speed=0.0))
                self.done = True
        elif self.travelled >= self._cross_total:
            self.actor.apply_control(carla.WalkerControl(speed=0.0))
            self.done = True
