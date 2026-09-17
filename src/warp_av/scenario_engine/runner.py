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
    7. compute metrics, evaluate, write scenarios/results/runs/<run_id>/<id>.json (+ .trace.jsonl,
       run_manifest.json, SUMMARY.md, summary.csv) -- one folder per run, nothing overwritten

`--dry-run` performs steps 1–4 in "plan only" mode without CARLA or the API so
the catalog can be sanity-checked on any machine.
"""
from __future__ import annotations

import json
import math
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .evaluator import compute_metrics, evaluate
from .evidence import build_evidence, write_run_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "scenarios" / "results"
RUNS_DIR = RESULTS_DIR / "runs"
POLL_HZ = 10.0
ENABLE_COMPONENTS = ("perception", "localization", "controller", "planner", "vehicle_connection",
                     "camera", "lidar", "gnss", "imu", "tick_latency")


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git"] + args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def repo_git_sha() -> Optional[str]:
    return _git(["rev-parse", "HEAD"])


def repo_git_dirty() -> Optional[bool]:
    out = _git(["status", "--porcelain", "--untracked-files=no"])
    return None if out is None else bool(out)


def parse_reset_spec(text: Optional[str]) -> Optional[dict]:
    """'spawn:12' -> spawn point 12; 'lane:x,y' -> the lane under x,y, facing along it (how every
    earlier live run placed the van); 'x,y,yaw' -> that pose; '' / None -> no reset."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("spawn:"):
        return {"mode": "spawn_point", "index": int(t[len("spawn:"):])}
    if t.startswith("lane:"):
        x, y = (float(v) for v in t[len("lane:"):].split(","))
        return {"mode": "lane", "x": x, "y": y}
    parts = [float(v) for v in t.split(",")]
    if len(parts) == 2:
        parts.append(0.0)
    if len(parts) != 3:
        raise ValueError(f"reset spec must be 'spawn:N' or 'x,y,yaw_deg', got {text!r}")
    return {"mode": "absolute", "x": parts[0], "y": parts[1], "yaw_deg": parts[2]}


class RunnerError(RuntimeError):
    pass


class Api:
    def __init__(self, base: str):
        # On Windows "localhost" resolves to ::1 first; the stack listens on IPv4 only, and the
        # failed IPv6 attempt cost ~2 s on EVERY call (first V1 run: 0.5 Hz polling, 39 s of
        # setup per scenario). The number is what the loopback address is for.
        self.base = base.rstrip("/").replace("//localhost:", "//127.0.0.1:").replace("//localhost/", "//127.0.0.1/")

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
                 results_dir: Path = RESULTS_DIR, dry_run: bool = False, verbose: bool = True,
                 run_id: Optional[str] = None, seed: Optional[int] = None, reset_ego: Optional[dict] = None):
        self.api = Api(api_url)
        self.api_url = api_url
        self.carla_host, self.carla_port = carla_host, carla_port
        self.results_dir = Path(results_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        # V1 evidence: every run gets its own folder, named by time and code, and nothing in it
        # is ever overwritten. Folders are made on first write, so --dry-run touches no disk.
        self.git_sha = repo_git_sha()
        self.git_dirty = repo_git_dirty()
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + (self.git_sha or "nogit")[:7]
        self.run_dir = self.results_dir / "runs" / self.run_id
        self.seed = seed
        self.reset_ego = reset_ego
        self.planned_ids: Optional[List[str]] = None
        self._manifest: Optional[dict] = None

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
                "clear_time": None, "mission_completed": False, "event_log": [], "warnings": [],
                # V1 evidence
                "run_id": self.run_id, "git_sha_runner": self.git_sha, "git_dirty": self.git_dirty,
                "seed": self.seed, "map": None, "map_expected": scenario["odd"]["town"], "carla_version": None,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "start_xy": None, "goal_xy": None,
                "stack": None, "start_check": None, "ego_reset": None,
                "mission_ids": [], "mission_records": [], "phase_times": {}}
        phase = meta["phase_times"]
        t_run0 = time.time()
        trace: List[dict] = []
        wh = None
        err = None
        try:
            st = self._state()
            wh = WorldHelper(self.carla_host, self.carla_port, (st["pose"]["x"], st["pose"]["y"]))
            if wh.ego is None:
                raise RunnerError("no ego vehicle found in CARLA — is the autonomy stack running?")
            town_now = wh.town()
            meta["map"] = town_now
            meta["carla_version"] = wh.versions()
            if town_now != scenario["odd"]["town"]:
                meta["warnings"].append(f"town mismatch: scenario wants {scenario['odd']['town']}, CARLA has {town_now}")
                self.log(f"  ! {meta['warnings'][-1]} (continuing)")

            # make sure we start from a clean autonomy state
            phase["world_ready"] = round(time.time() - t_run0, 2)
            self.api.post("/api/estop/clear")
            self.api.post("/api/mission/stop")
            if self.reset_ego is not None:
                # V1: the same start for every scenario, so a run does not inherit where the last one ended
                try:
                    off = wh.reset_ego(self.reset_ego)
                    meta["ego_reset"] = {"spec": self.reset_ego, "settled_off_m": round(off, 2)}
                    self.log(f"  ego reset to {self.reset_ego} (came to rest {off:.2f} m from it)")
                    if off > 2.0:
                        meta["warnings"].append(f"ego reset came to rest {off:.1f} m from the requested point")
                except Exception as e_reset:
                    meta["warnings"].append(f"ego reset failed: {e_reset}")
                    self.log(f"  ! {meta['warnings'][-1]}")
            phase["ego_reset_done"] = round(time.time() - t_run0, 2)
            for comp in ENABLE_COMPONENTS:
                self.api.post("/api/test/inject", {"component": comp, "action": "enable"})
            if "cruise_speed_mps" in scenario["mission"]:
                self.api.post("/api/config/speed_limit", {"cruise_speed_mps": scenario["mission"]["cruise_speed_mps"]})
            time.sleep(0.5)
            st0 = self._state()
            phase["clean_start_done"] = round(time.time() - t_run0, 2)
            meta["start_check"] = self._start_check(st0)
            if not meta["start_check"]["clean"]:
                meta["warnings"].append("unclean start: " + "; ".join(meta["start_check"]["problems"]))
                self.log(f"  ! {meta['warnings'][-1]}")
            meta["stack"] = {"git_sha": st0.get("version"), "perception_mode": st0.get("perception_mode"),
                             "noise_profile": (st0.get("ekf") or {}).get("noise_profile"),
                             "cruise_speed_mps": st0.get("cruise_speed_mps"), "loop_hz_at_start": st0.get("loop_hz"),
                             "uptime_s": st0.get("uptime_s")}
            if self.git_sha and st0.get("version") and not self.git_sha.startswith(str(st0.get("version"))):
                meta["warnings"].append(f"stack runs {st0.get('version')} but this checkout is {self.git_sha[:7]}: stale stack?")
                self.log(f"  ! {meta['warnings'][-1]}")

            # weather
            wh.set_weather(scenario["odd"]["weather"])

            # actors
            ctrls = [ActorController(wh, a) for a in scenario.get("actors", [])]
            if self.seed is not None:
                # the only randomness the engine can pin: CARLA's traffic manager (autopilot actors)
                try:
                    wh.traffic_manager().set_random_device_seed(int(self.seed))
                except Exception as e_seed:
                    meta["warnings"].append(f"seed not applied to the traffic manager: {e_seed}")
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
                if meta["start_xy"] is None:
                    sx, sy, _ = wh.ego_xy_yaw()
                    meta["start_xy"] = (round(sx, 2), round(sy, 2))
                meta["goal_xy"] = (round(dest_tr.location.x, 2), round(dest_tr.location.y, 2))
                phase.setdefault("first_mission_post", round(time.time() - t_run0, 2))
                code, resp = self.api.post("/api/mission/start", {"x": dest_tr.location.x, "y": dest_tr.location.y})
                meta["event_log"].append({"t": time.time() - t0, "event": "start_mission", "resp": resp, "code": code})
                phase.setdefault("first_mission_started", round(time.time() - t_run0, 2))
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

                phase.setdefault("first_poll", round(now - t_run0, 2))
                t_a = time.perf_counter()
                st = self._state()
                t_b = time.perf_counter()
                ex, ey, eyaw = wh.ego_xy_yaw()
                sample = {"t": now, "state": st, "actors": wh.positions(), "ego": (ex, ey)}
                try:
                    # V1 evidence: what CARLA knows at the same instant. Never fatal.
                    sample["ego_yaw"] = round(eyaw, 4)
                    sample["ego_speed"] = round(wh.ego_speed(), 3)
                    sample["ego_box"] = wh.ego_box()
                    sample["boxes"] = wh.actor_boxes()
                    sample["nearest_any"] = wh.nearest_other_actor()
                    sample["light"] = wh.ego_light()
                except Exception as e_ev:
                    sample["evidence_error"] = f"{type(e_ev).__name__}: {e_ev}"
                t_c = time.perf_counter()
                sample["poll_ms"] = {"api": round((t_b - t_a) * 1000, 1), "truth": round((t_c - t_b) * 1000, 1)}
                trace.append(sample)
                mid = (st.get("mission") or {}).get("mission_id")
                if mid and mid not in meta["mission_ids"]:
                    meta["mission_ids"].append(mid)

                # actor triggers & behaviour stepping
                t_d = time.perf_counter()
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
                sample["poll_ms"]["actors"] = round((time.perf_counter() - t_d) * 1000, 1)

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
                for comp in ENABLE_COMPONENTS:
                    self.api.post("/api/test/inject", {"component": comp, "action": "enable"})
            except Exception:
                pass
            try:
                # The stack drops a finished mission from /api/state on the same tick it completes,
                # so "completed" is never visible there (first V1 run, WAV-0010). The mission's own
                # record in /api/history is the only place the outcome survives.
                hist = self.api.get("/api/history")
                meta["mission_records"] = [h for h in hist if h.get("mission_id") in meta["mission_ids"]]
            except Exception as e_h:
                meta["warnings"].append(f"mission history not read: {e_h}")
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
            "run_id": self.run_id, "runner_error": err,
            "meta": {k: meta.get(k) for k in ("run_id", "git_sha_runner", "git_dirty", "seed", "map", "map_expected",
                                              "carla_version", "started_at", "start_xy", "goal_xy", "stack",
                                              "start_check", "ego_reset", "mission_ids", "mission_records",
                                              "phase_times")},
        }
        self._persist(sid, result, trace)
        self.log(f"  => {result['verdict']}: {result['reason']}")
        return result

    # ------------------------------------------------------------------ V1 evidence: persistence
    def _start_check(self, st: dict) -> dict:
        """Is the van actually clean before the scenario starts? Faults cleared, safety ok, no
        mission, standing still. Recorded, never enforced: an unclean start is itself evidence."""
        problems = []
        faults = st.get("active_faults") or {}
        if faults:
            problems.append(f"faults still active: {sorted(faults)}")
        safety = (st.get("safety") or {}).get("state")
        if safety not in (None, "ok"):
            problems.append(f"safety state {safety}: {(st.get('safety') or {}).get('reason')}")
        mstate = (st.get("mission") or {}).get("state", "idle")
        if mstate != "idle":
            problems.append(f"mission state {mstate}")
        speed = float((st.get("pose") or {}).get("speed") or 0.0)
        if speed > 0.3:
            problems.append(f"still moving at {speed:.1f} m/s")
        if st.get("last_tick_error"):
            problems.append(f"last tick error: {st.get('last_tick_error')}")
        return {"clean": not problems, "problems": problems,
                "stack_collisions_before": (st.get("collision") or {}).get("count"),
                "autonomy_state": st.get("autonomy_state")}

    def _unique(self, stem: str, suffix: str) -> Path:
        p = self.run_dir / f"{stem}{suffix}"
        n = 2
        while p.exists():
            p = self.run_dir / f"{stem}__{n}{suffix}"
            n += 1
        return p

    def _write_manifest(self, result: dict):
        if self._manifest is not None:
            return
        meta = result.get("meta") or {}
        self._manifest = {
            "run_id": self.run_id, "started_at": meta.get("started_at"),
            "git_sha_runner": self.git_sha, "git_dirty": self.git_dirty,
            "git_sha_stack": (meta.get("stack") or {}).get("git_sha"),
            "map": meta.get("map"), "carla_server": (meta.get("carla_version") or {}).get("server"),
            "carla_client": (meta.get("carla_version") or {}).get("client"),
            "seed": self.seed, "reset_ego": self.reset_ego, "api_url": self.api_url,
            "carla_host": self.carla_host, "carla_port": self.carla_port, "poll_hz": POLL_HZ,
            "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "planned_ids": self.planned_ids,
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(self._manifest, indent=1, default=str), encoding="utf-8")

    def _persist(self, sid: str, result: dict, trace: List[dict]) -> dict:
        """Write <sid>.json + <sid>.trace.jsonl into the run folder, never over an existing file,
        then rebuild the run's SUMMARY.md / summary.csv."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(result)
        res_path = self._unique(sid, ".json")
        stem = res_path.name[:-len(".json")]
        trace_path = self.run_dir / f"{stem}.trace.jsonl"
        result["result_file"] = str(res_path)
        result["trace_file"] = str(trace_path)
        result["evidence"] = build_evidence(result)
        trace_text = "\n".join(json.dumps(s, default=str) for s in trace)
        trace_path.write_text(trace_text, encoding="utf-8")
        res_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        try:
            write_run_summary(self.run_dir)
        except Exception as e_sum:
            self.log(f"  ! summary not rebuilt: {e_sum}")
        return {"result": res_path, "trace": trace_path}

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
