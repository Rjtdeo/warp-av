"""
Scenario runner.

Executes one catalog scenario against a *running* CARLA + Warp AV stack:

    1. set weather (CARLA)
    2. spawn untriggered actors (CARLA, positioned along ego's lane)
    3. start the mission through the operator API (POST /api/mission/start)
    4. fetch the planned route (GET /api/route) and re-anchor route-relative spawns
    5. poll GET /api/state at ~10 Hz, stepping actor behaviours and firing triggers
       (e-stop / pause / inject / ... go through the same HTTP API the console uses)
    6. stop at timeout or terminal condition, clean up actors
    7. compute metrics, evaluate, write scenarios/results/<id>.json

`--dry-run` performs steps 1–4 in "plan only" mode without CARLA or the API so
the catalog can be sanity-checked on any machine.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .evaluator import compute_metrics, evaluate

RESULTS_DIR = Path(__file__).resolve().parents[3] / "scenarios" / "results"
POLL_HZ = 10.0


class RunnerError(RuntimeError):
    pass


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path, **kw):
        r = requests.get(self.base + path, timeout=3, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path, body=None):
        r = requests.post(self.base + path, json=body or {}, timeout=5)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}


class ScenarioRunner:
    def __init__(self, api_url="http://localhost:5000", carla_host="localhost", carla_port=2000,
                 results_dir: Path = RESULTS_DIR, dry_run: bool = False, verbose: bool = True):
        self.api = Api(api_url)
        self.carla_host, self.carla_port = carla_host, carla_port
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def run(self, scenario: dict) -> dict:
        sid = scenario["id"]
        self.log(f"\n=== {sid} · {scenario['name']} [{scenario['category']}/{scenario['capability_status']}] ===")
        if self.dry_run:
            return self._dry_run(scenario)

        from .carla_world import WorldHelper, ActorController  # carla import isolated here

        t_start = time.time()
        meta = {"collisions": [], "route_xy": [], "trigger_time": None, "first_fault_time": None,
                "clear_time": None, "mission_completed": False, "event_log": [], "warnings": []}
        trace: List[dict] = []
        wh = None
        err = None
        try:
            st = self._state()
            wh = WorldHelper(self.carla_host, self.carla_port, (st["pose"]["x"], st["pose"]["y"]))
            if wh.ego is None:
                raise RunnerError("no ego vehicle found in CARLA — is the autonomy stack running?")
            town_now = wh.town()
            if town_now != scenario["odd"]["town"]:
                meta["warnings"].append(f"town mismatch: scenario wants {scenario['odd']['town']}, CARLA has {town_now}")
                self.log(f"  ! {meta['warnings'][-1]} (continuing)")

            # make sure we start from a clean autonomy state
            self.api.post("/api/estop/clear")
            self.api.post("/api/mission/stop")
            for comp in ("perception", "localization", "controller", "planner", "vehicle_connection", "camera", "lidar", "gnss", "imu", "tick_latency"):
                self.api.post("/api/test/inject", {"component": comp, "action": "enable"})
            if "cruise_speed_mps" in scenario["mission"]:
                self.api.post("/api/config/speed_limit", {"cruise_speed_mps": scenario["mission"]["cruise_speed_mps"]})
            time.sleep(0.5)

            # weather
            wh.set_weather(scenario["odd"]["weather"])

            # actors
            ctrls = [ActorController(wh, a) for a in scenario.get("actors", [])]
            pre_route_ok = all(a["spawn"].get("mode") != "at_destination" for a in scenario.get("actors", []))
            start_at = float(scenario["mission"].get("start_at_s", 0.0))

            def spawn_all():
                for c in ctrls:
                    if c.actor is None and not c.triggered and not c.spawn_failed:
                        c.spawn_now()
                        if c.spawn_failed:
                            meta["warnings"].append(f"spawn failed: {c.name}")
                            self.log(f"  ! spawn failed for actor {c.name}")

            # events
            events = [dict(e, _fired=False, _time=None) for e in scenario.get("events", [])]
            mission_started = False
            mission_start_time = None
            t0 = time.time()
            last_poll = t0
            dt = 1.0 / POLL_HZ
            timeout = float(scenario["timeout_s"])
            terminal_since = None
            route_loaded = False

            def start_mission(dest_spec):
                nonlocal mission_started, mission_start_time, route_loaded
                dest_tr = wh.resolve_location(dest_spec)
                code, resp = self.api.post("/api/mission/start", {"x": dest_tr.location.x, "y": dest_tr.location.y})
                meta["event_log"].append({"t": time.time() - t0, "event": "start_mission", "resp": resp, "code": code})
                self.log(f"  → mission start ({dest_tr.location.x:.1f}, {dest_tr.location.y:.1f}) -> {code} {resp}")
                mission_started = True
                mission_start_time = time.time()
                route_loaded = False
                if not resp.get("success"):
                    meta["warnings"].append(f"mission start refused: {resp}")

            if start_at <= 0:
                if pre_route_ok:
                    spawn_all()
                start_mission(scenario["mission"]["destination"])
                time.sleep(0.3)
                self._load_route(wh, meta)
                route_loaded = True
                spawn_all()

            # ------------------------------------------------ main poll loop
            while True:
                now = time.time()
                elapsed = now - t0
                if elapsed > timeout:
                    self.log("  timeout reached")
                    break
                if not mission_started and elapsed >= start_at:
                    spawn_all()
                    start_mission(scenario["mission"]["destination"])
                if mission_started and not route_loaded and now - mission_start_time > 0.3:
                    self._load_route(wh, meta)
                    route_loaded = True
                    spawn_all()

                st = self._state()
                ex, ey, _ = wh.ego_xy_yaw()
                sample = {"t": now, "state": st, "actors": wh.positions(), "ego": (ex, ey)}
                trace.append(sample)

                # actor triggers & behaviour stepping
                for c in ctrls:
                    if not c.triggered and "trigger" in c.spec and self._trigger_met(c.spec["trigger"], st, elapsed, wh, c, events, ctrls):
                        c.fire()
                        if meta["trigger_time"] is None:
                            meta["trigger_time"] = now
                        meta["event_log"].append({"t": elapsed, "actor_trigger": c.name})
                        self.log(f"  ⚡ actor {c.name} triggered at {elapsed:.1f}s")
                    c.step(dt)
                if meta["trigger_time"] is None and mission_started and all("trigger" not in c.spec for c in ctrls) and ctrls:
                    meta["trigger_time"] = mission_start_time

                # events
                for i, e in enumerate(events):
                    if e["_fired"]:
                        continue
                    if self._trigger_met(e["trigger"], st, elapsed, wh, None, events, ctrls):
                        e["_fired"], e["_time"] = True, now
                        self._fire_event(e, wh, meta, t0, start_mission)
                        if meta["trigger_time"] is None and e["action"] not in ("wait", "set_weather", "set_speed_limit"):
                            meta["trigger_time"] = now

                # terminal conditions: mission completed/failed/cancelled and stable for 2 s
                ms = st.get("mission", {}).get("state", "idle")
                if ms == "completed" or (mission_started and ms in ("idle", "failed", "cancelled") and now - mission_start_time > 3):
                    if ms == "completed":
                        meta["mission_completed"] = True
                    pending = any(not e["_fired"] for e in events)
                    if not pending:
                        terminal_since = terminal_since or now
                        if now - terminal_since > 2.0:
                            break
                else:
                    terminal_since = None
                if st.get("mission", {}).get("state") == "completed":
                    meta["mission_completed"] = True
                if "mission_complete" in (st.get("behavior") or ""):
                    meta["mission_completed"] = True

                # pace
                sleep = dt - (time.time() - now)
                if sleep > 0:
                    time.sleep(sleep)

            meta["collisions"] = list(wh.collisions)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.log("  RUNNER ERROR: " + err)
            self.log(traceback.format_exc())
        finally:
            try:
                self.api.post("/api/mission/stop")
                self.api.post("/api/estop/clear")
                for comp in ("perception", "localization", "controller", "planner", "vehicle_connection", "camera", "lidar", "gnss", "imu", "tick_latency"):
                    self.api.post("/api/test/inject", {"component": comp, "action": "enable"})
            except Exception:
                pass
            if wh is not None:
                wh.cleanup()

        meta["elapsed_s"] = time.time() - t_start
        metrics = compute_metrics(trace, meta)
        verdict = evaluate(scenario, metrics, runner_error=err)
        result = {
            "scenario_id": sid, "name": scenario["name"], "category": scenario["category"], "family": scenario["family"],
            "capability_status": scenario["capability_status"], "verdict": verdict["verdict"], "reason": verdict["reason"],
            "checks": verdict["checks"], "metrics": metrics, "warnings": meta["warnings"], "event_log": meta["event_log"],
            "collisions": meta["collisions"], "started_at": t_start, "elapsed_s": round(meta["elapsed_s"], 2),
            "trace_len": len(trace), "behaviors_timeline": self._timeline(trace, t_start),
        }
        (self.results_dir / f"{sid}.json").write_text(json.dumps(result, indent=1, default=str))
        (self.results_dir / f"{sid}.trace.jsonl").write_text("\n".join(json.dumps(s, default=str) for s in trace))
        self.log(f"  => {result['verdict']}: {result['reason']}")
        return result

    # ------------------------------------------------------------------ helpers
    def _state(self) -> dict:
        return self.api.get("/api/state")

    def _load_route(self, wh, meta):
        try:
            route = self.api.get("/api/route")
            xy = [(p["x"], p["y"]) for p in route]
            meta["route_xy"] = xy
            wh.set_route(xy)
            self.log(f"  route: {len(xy)} waypoints")
        except Exception as e:
            meta["warnings"].append(f"route fetch failed: {e}")

    def _trigger_met(self, tr: dict, st: dict, elapsed: float, wh, ctrl, events, ctrls) -> bool:
        base = 0.0
        if "after_event" in tr:
            ref = events[tr["after_event"]]
            if not ref["_fired"]:
                return False
            base = ref["_time"] - (time.time() - elapsed)
        conds = []
        if "at_s" in tr:
            conds.append(elapsed >= base + float(tr["at_s"]))
        elif "after_event" in tr:
            conds.append(True)
        if "ego_speed_gt" in tr:
            conds.append(float(st.get("pose", {}).get("speed", 0.0)) > float(tr["ego_speed_gt"]))
        if "ego_within_m" in tr:
            # distance from ego to this actor's spawn (or nearest scenario actor for events)
            if ctrl is not None and ctrl.actor is not None:
                d = wh.ego_distance_to(ctrl.name)
            elif ctrl is not None:
                try:
                    loc = wh.resolve_location(ctrl.spec["spawn"]).location
                    ex, ey, _ = wh.ego_xy_yaw()
                    d = math.dist((ex, ey), (loc.x, loc.y))
                except Exception:
                    d = 999.0
            else:
                ds = [wh.ego_distance_to(c.name) for c in ctrls if c.actor is not None]
                if ds:
                    d = min(ds)
                else:
                    dest = st.get("destination")
                    ex, ey, _ = wh.ego_xy_yaw()
                    d = math.dist((ex, ey), (dest["x"], dest["y"])) if dest else 999.0
            conds.append(d <= float(tr["ego_within_m"]))
        if "on_behavior" in tr:
            conds.append(st.get("behavior") == tr["on_behavior"])
        if "on_mission_state" in tr:
            conds.append(st.get("mission", {}).get("state") == tr["on_mission_state"])
        return bool(conds) and all(conds)

    def _fire_event(self, e: dict, wh, meta, t0, start_mission):
        act, p = e["action"], e.get("params", {}) or {}
        t_rel = round(time.time() - t0, 2)
        resp = None
        try:
            if act == "estop":
                resp = self.api.post("/api/estop")
            elif act == "estop_clear":
                resp = self.api.post("/api/estop/clear")
                meta["clear_time"] = time.time()
            elif act == "pause":
                resp = self.api.post("/api/mission/pause")
            elif act == "resume":
                resp = self.api.post("/api/mission/resume")
                meta["clear_time"] = time.time()
            elif act == "stop_mission":
                resp = self.api.post("/api/mission/stop")
            elif act in ("start_mission", "change_destination"):
                start_mission(p["destination"])
                resp = "see start_mission"
            elif act == "set_speed_limit":
                resp = self.api.post("/api/config/speed_limit", {"cruise_speed_mps": p.get("cruise_speed_mps", 8.0)})
            elif act == "inject":
                body = dict(p)
                resp = self.api.post("/api/test/inject", body)
                if body.get("action") not in ("enable",) and meta["first_fault_time"] is None:
                    meta["first_fault_time"] = time.time()
                if body.get("action") == "enable":
                    meta["clear_time"] = time.time()
            elif act == "set_weather":
                wh.set_weather(p.get("preset"), **{k: v for k, v in p.items() if k != "preset"})
                resp = "ok"
            elif act == "wait":
                if "traffic_light" in p:
                    resp = wh.set_traffic_lights(p["traffic_light"])
                else:
                    resp = "ok"
        except Exception as ex:
            resp = f"ERROR {ex}"
            meta["warnings"].append(f"event {act} failed: {ex}")
        meta["event_log"].append({"t": t_rel, "event": act, "params": p, "resp": resp})
        self.log(f"  ⚡ event {act} {p if p else ''} at {t_rel}s -> {resp}")

    def _timeline(self, trace, t_start):
        out, last = [], None
        for s in trace:
            key = (s["state"].get("behavior"), s["state"].get("safety", {}).get("state"), s["state"].get("mission", {}).get("state"))
            if key != last:
                out.append({"t": round(s["t"] - t_start, 2), "behavior": key[0], "safety": key[1], "mission": key[2],
                            "reason": s["state"].get("behavior_reason"), "speed": s["state"].get("pose", {}).get("speed")})
                last = key
        return out

    # ------------------------------------------------------------------ dry run
    def _dry_run(self, scenario: dict) -> dict:
        plan = {
            "weather": scenario["odd"]["weather"], "town": scenario["odd"]["town"],
            "mission": scenario["mission"],
            "actors": [{"name": a["name"], "type": a["type"], "blueprint": a["blueprint"], "spawn": a["spawn"],
                        "behavior": a["behavior"]["kind"], "trigger": a.get("trigger", "at_start")} for a in scenario["actors"]],
            "events": [{"trigger": e["trigger"], "action": e["action"], "params": e.get("params", {})} for e in scenario["events"]],
            "pass_criteria": scenario["pass_criteria"], "timeout_s": scenario["timeout_s"],
        }
        self.log(json.dumps(plan, indent=1))
        return {"scenario_id": scenario["id"], "verdict": "DRY_RUN", "plan": plan}
