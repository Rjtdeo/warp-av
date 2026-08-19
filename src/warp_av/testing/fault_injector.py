"""
Fault Injector — the single place that knows how to break the system on purpose.

Exposed via POST /api/test/inject {"component": ..., "action": ..., ...params}
and used by the scenario runner.  Every injection is logged as an event so the
mission log shows exactly when a fault was introduced.

Supported (component → actions):
    perception          disable enable freeze stale(age_s) latency(latency_s) crash
    localization        disable enable freeze stale(age_s) low_confidence(value, ramp_s)
                        noise(offset_m, mode=jump|drift, confidence) crash
    camera|lidar|gnss|imu   disable enable drop(duration_s)   (sensor adapter flags)
    controller          disable enable nan_command stale(age_s)
    planner             disable enable
    vehicle_connection  disable enable freeze(duration_s)
    tick_latency        latency(latency_s, mode=constant|spike, duration_s) enable
    api                 (accepted, recorded, no effect — API is not safety critical)
Anything else returns {"success": False, "reason": ...} so callers can tell
"unsupported" from "applied".
"""
import threading
import time


class FaultInjector:
    def __init__(self, system):
        self.sys = system
        self.active = {}          # component -> description (shown in /api/state)
        self.tick_latency_s = 0.0
        self._tick_latency_until = 0.0
        self._tick_latency_mode = "constant"

    # ------------------------------------------------------------------
    def inject(self, component: str, action: str, **params) -> dict:
        handler = getattr(self, f"_{component}", None)
        if handler is None:
            return {"success": False, "reason": f"unknown component {component}"}
        try:
            ok = handler(action, **params)
        except Exception as e:  # never let a test hook crash the loop thread
            return {"success": False, "reason": f"{type(e).__name__}: {e}"}
        if ok:
            desc = f"{action} {params}" if params else action
            if action == "enable":
                self.active.pop(component, None)
            else:
                self.active[component] = desc
            self.sys.logger.log_event("fault_injected", f"{component}: {desc}")
            return {"success": True, "component": component, "action": action}
        return {"success": False, "reason": f"{component} does not support {action}"}

    def extra_tick_delay(self) -> float:
        """Called by the main loop each tick."""
        if self.tick_latency_s <= 0:
            return 0.0
        if self._tick_latency_until and time.time() > self._tick_latency_until:
            self.tick_latency_s = 0.0
            self.active.pop("tick_latency", None)
            return 0.0
        if self._tick_latency_mode == "spike":
            d, self.tick_latency_s = self.tick_latency_s, 0.0
            self.active.pop("tick_latency", None)
            return d
        return self.tick_latency_s

    # ------------------------------------------------------------------
    def _perception(self, action, **p):
        if action == "disable":
            self.sys.perception.disable(); return True
        if action == "enable":
            self.sys.perception.enable(); return True
        fn = getattr(self.sys.perception, "inject_fault", None)
        return bool(fn and fn(action, **p))   # camera/LiDAR perception has no hooks yet

    def _localization(self, action, **p):
        if action == "disable":
            self.sys.localization.disable(); return True
        if action == "enable":
            self.sys.localization.enable(); return True
        return self.sys.localization.inject_fault(action, **p)

    def _sensor(self, name, action, **p):
        flag = f"{name}_enabled"
        sa = self.sys.sensor_adapter
        if action == "disable":
            setattr(sa, flag, False); return True
        if action == "enable":
            setattr(sa, flag, True); return True
        if action == "drop":
            setattr(sa, flag, False)
            dur = float(p.get("duration_s", 1.0))
            threading.Timer(dur, lambda: setattr(sa, flag, True)).start()
            return True
        if action in ("noise", "latency"):
            # Recorded only: ground-truth perception/localization do not consume raw sensors yet.
            return True
        return False

    def _camera(self, action, **p): return self._sensor("camera", action, **p)
    def _lidar(self, action, **p):  return self._sensor("lidar", action, **p)
    def _gnss(self, action, **p):   return self._sensor("gnss", action, **p)
    def _imu(self, action, **p):    return self._sensor("imu", action, **p)

    def _controller(self, action, **p):
        if action == "disable":
            self.sys.controller.disable(); return True
        if action == "enable":
            self.sys.controller.enable(); return True
        return self.sys.controller.inject_fault(action, **p)

    def _planner(self, action, **p):
        if action == "disable":
            self.sys.planner.disable(); return True
        if action == "enable":
            self.sys.planner.enable(); return True
        return False

    def _vehicle_connection(self, action, **p):
        va = self.sys.vehicle_adapter
        if action == "disable":
            va.simulate_connection_loss(True); return True
        if action == "enable":
            va.simulate_connection_loss(False); return True
        if action == "freeze":
            va.simulate_connection_loss(True)
            threading.Timer(float(p.get("duration_s", 2.0)), lambda: va.simulate_connection_loss(False)).start()
            return True
        return False

    def _tick_latency(self, action, **p):
        if action == "enable":
            self.tick_latency_s = 0.0; return True
        if action == "latency":
            self.tick_latency_s = float(p.get("latency_s", 0.2))
            self._tick_latency_mode = p.get("mode", "constant")
            dur = float(p.get("duration_s", 0))
            self._tick_latency_until = time.time() + dur if dur else 0.0
            return True
        return False

    def _api(self, action, **p):
        return True   # recorded only
