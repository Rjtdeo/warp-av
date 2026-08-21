#!/usr/bin/env python3
"""
Mission sweep: run hundreds of end-to-end missions against the LIVE stack
through the same operator API the dashboard uses, exercising every feature
(parking slots, dense traffic, red lights, jaywalkers, cut-ins, weather,
component faults) and grading each run honestly.

Runs from any machine that can reach the stack's API — designed to drive the
Windows CARLA machine from the Mac overnight:

    python3 tools/mission_sweep.py --shakedown          # 3 quick proving runs
    python3 tools/mission_sweep.py --full               # the whole 300-run plan
    python3 tools/mission_sweep.py --full --no-resume   # ignore previous results

Everything is written as it happens:
    <out>/run_NNN.json      verdict + metrics + failure diagnosis per run
    <out>/trace_NNN.jsonl   5 Hz state samples (replayable evidence)
    <out>/frames/*.jpg      front-camera snapshots at parking / hazards
    <out>/SCOREBOARD.md     live scoreboard, rewritten after every run

If the stack or CARLA dies, the built-in watchdog restarts them over SSH and
the sweep resumes by itself. Nothing here needs a human until the report.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
API = "http://192.168.1.101:5000"
SSH_HOST = "Rajat@192.168.1.101"
SSH_KEY = os.path.expanduser("~/.ssh/warp_av_ed25519")
WIN_REPO = r"C:\Users\Rajat\Desktop\warp-av"
WIN_PY = r"C:\Users\Rajat\AppData\Local\Programs\Python\Python310\python.exe"
CARLA_EXE = r"C:\CARLA\WindowsNoEditor\CarlaUE4.exe"

SAMPLE_DT = 0.2          # state poll period (5 Hz)
DEST_MIN_M = 110.0       # crow-fly destination window
DEST_MAX_M = 380.0
CRUISE = 8.0

# Behaviours in which standing still is legitimate (queues, lights, holds)
JUSTIFIED_STOPS = {
    "stopped_red_light", "waiting_at_junction", "stopped_vehicle",
    "stopped_pedestrian", "stopped_obstacle", "stopped_blocked",
    "following_vehicle", "parking", "stopped_safety", "stopped_estop",
    "approaching_destination",
}
STUCK_S_JUSTIFIED = 180.0   # even a queue/red light should clear within this
STUCK_S_OTHER = 75.0        # stationary with no valid reason

VAN_HALF_LEN = 2.35         # remote inside-the-box check (van footprint)
VAN_HALF_WID = 0.95


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------
def _http(path, payload=None, timeout=5.0, method=None):
    url = API + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method or ("POST" if payload is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def api_get(path, timeout=5.0, retries=1):
    for i in range(retries + 1):
        try:
            return _http(path, timeout=timeout)
        except Exception:
            if i == retries:
                raise
            time.sleep(0.4)


def api_post(path, payload=None, timeout=8.0, retries=1):
    for i in range(retries + 1):
        try:
            return _http(path, payload if payload is not None else {}, timeout=timeout)
        except Exception:
            if i == retries:
                raise
            time.sleep(0.6)


def fetch_frame(out_path, view="front"):
    try:
        req = urllib.request.Request(f"{API}/api/camera/frame?view={view}")
        with urllib.request.urlopen(req, timeout=4.0) as r:
            blob = r.read()
        if blob and len(blob) > 1000:
            with open(out_path, "wb") as f:
                f.write(blob)
            return True
    except Exception:
        pass
    return False


# ----------------------------------------------------------------------
# SSH watchdog: revive CARLA / the stack when they die
# ----------------------------------------------------------------------
def ssh(cmd, timeout=30):
    try:
        r = subprocess.run(
            ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             SSH_HOST, cmd],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 99, str(e)


def carla_running():
    rc, out = ssh('Get-Process CarlaUE4-Win64-Shipping -ErrorAction SilentlyContinue | '
                  'Select-Object -ExpandProperty Id')
    return rc == 0 and out.strip() != ""


def start_carla():
    ssh(f"powershell -NoProfile -ExecutionPolicy Bypass -File {WIN_REPO}\\tools\\win\\start_carla.ps1")


def kill_stack():
    # NB: the Name filter keeps this from matching (and killing) the SSH
    # shell that carries this very command line.
    ssh('Get-CimInstance Win32_Process | Where-Object {($_.Name -eq "python.exe") -and '
        '(($_.CommandLine -like "*warp_av.main*") -or ($_.CommandLine -like "*run.py*"))} '
        '| ForEach-Object {Stop-Process -Id $_.ProcessId -Force}')


def start_stack():
    ssh(f"powershell -NoProfile -ExecutionPolicy Bypass -File {WIN_REPO}\\tools\\win\\start_stack.ps1")


def api_alive(timeout=4.0):
    try:
        s = api_get("/api/state", timeout=timeout, retries=0)
        return isinstance(s, dict) and "tick" in s
    except Exception:
        return False


def ensure_stack_up(log, max_wait_s=900):
    """Blocks until /api/state answers. Restarts stack, then CARLA+stack."""
    if api_alive():
        return True
    log("STACK DOWN — watchdog engaging")
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < max_wait_s:
        attempt += 1
        if not carla_running():
            log(f"watchdog: CARLA is dead — restarting it (attempt {attempt})")
            start_carla()
            time.sleep(80)          # UE4 takes a while to open the world
        log(f"watchdog: restarting the stack (attempt {attempt})")
        kill_stack()
        time.sleep(3)
        start_stack()
        for _ in range(30):
            time.sleep(2)
            if api_alive():
                log("watchdog: stack is back")
                time.sleep(4)       # let sensors settle
                return True
    log("watchdog: could NOT revive the stack — will keep waiting")
    while not api_alive():
        time.sleep(60)
    return True


# ----------------------------------------------------------------------
# The 300-run plan (deterministic)
# ----------------------------------------------------------------------
def build_plan():
    rng = random.Random(20260820)
    runs = []

    def spec(kind, weather="ClearNoon", dense=False, parked=0,
             take_chosen=False, fill_all=False):
        return {"kind": kind, "weather": weather, "dense": dense,
                "parked": parked, "take_chosen": take_chosen, "fill_all": fill_all}

    # ---- Phase 1: every feature, clear day (200) ----
    quiet = []
    for i in range(65):
        s = spec("baseline")
        if i < 8:
            s["parked"] = 4          # occupied bays on approach
        elif i < 11:
            s["parked"] = 2
            s["take_chosen"] = True  # force the re-scan retarget
        elif i < 13:
            s["fill_all"] = True     # every slot taken — honest "no free slot"
        quiet.append(s)
    quiet += [spec("red_light") for _ in range(30)]
    quiet += [spec("jaywalker") for _ in range(22)]
    quiet += [spec("cutin") for _ in range(22)]
    quiet += [spec("fault") for _ in range(6)]
    rng.shuffle(quiet)
    runs += quiet                                            # 145

    dense = [spec("baseline", dense=True) for _ in range(55)]
    for i in range(6):
        dense[i]["parked"] = 4
    rng.shuffle(dense)
    runs += dense                                            # 200

    # ---- Phase 2: weather (100) ----
    def weather_block(presets, n, n_dense, hazards):
        block = []
        for i in range(n):
            w = presets[i % len(presets)]
            block.append(spec("baseline", weather=w, dense=(i < n_dense)))
        for j, hz in enumerate(hazards):
            block[j]["kind"] = hz
        rng.shuffle(block)
        return block

    runs += weather_block(["SoftRainNoon"] * 2 + ["MidRainyNoon"] * 3 + ["HardRainNoon"] * 3,
                          30, 10, ["red_light"] * 3 + ["jaywalker"] * 3)
    runs += weather_block(["ClearSunset", "WetSunset"],
                          20, 6, ["cutin"] * 2)
    runs += weather_block(["ClearNight"] * 3 + ["WetNight"] * 2,
                          30, 10, ["red_light"] * 4 + ["jaywalker"] * 4)
    runs += weather_block(["HardRainNight"],
                          20, 5, ["red_light"] * 2 + ["jaywalker"] * 2)

    for i, r in enumerate(runs):
        r["run"] = i + 1
    return runs


def shakedown_plan():
    return [
        {"run": 1, "kind": "baseline", "weather": "ClearNoon", "dense": False,
         "parked": 0, "take_chosen": False, "fill_all": False},
        {"run": 2, "kind": "red_light", "weather": "ClearNoon", "dense": False,
         "parked": 0, "take_chosen": False, "fill_all": False},
        {"run": 3, "kind": "baseline", "weather": "ClearNight", "dense": True,
         "parked": 2, "take_chosen": True, "fill_all": False},
    ]


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def van_in_slot(x, y, yaw_deg, slot, tol=0.10):
    """All four van corners inside the slot rectangle (slot frame math)."""
    syaw = slot.get("yaw", 0.0)      # radians (planner convention)
    hl, hw = slot.get("length", 7.0) / 2.0, slot.get("width", 2.5) / 2.0
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    worst_dx = worst_dy = 0.0
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        px = x + sx * c * VAN_HALF_LEN - sy * s * VAN_HALF_WID
        py = y + sx * s * VAN_HALF_LEN + sy * c * VAN_HALF_WID
        dx, dy = px - slot["x"], py - slot["y"]
        cs, ss = math.cos(-syaw), math.sin(-syaw)
        lx = dx * cs - dy * ss
        ly = dx * ss + dy * cs
        worst_dx = max(worst_dx, abs(lx))
        worst_dy = max(worst_dy, abs(ly))
    inside = worst_dx <= hl + tol and worst_dy <= hw + tol
    return inside, round(hl - worst_dx, 2), round(hw - worst_dy, 2)


def ang_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def mission_ended_ok(mission_id):
    """True + entry if this mission finished 'completed' in /api/history."""
    try:
        h = api_get("/api/history", timeout=5.0)
        if isinstance(h, list):
            for m in reversed(h):
                if m.get("mission_id") == mission_id:
                    return m.get("state") == "completed", m
    except Exception:
        pass
    return False, None


# ----------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------
class RunAbort(Exception):
    pass


def pick_destination(rng, points, pose):
    cands = [p for p in points
             if DEST_MIN_M <= math.hypot(p["x"] - pose["x"], p["y"] - pose["y"]) <= DEST_MAX_M]
    rng.shuffle(cands)
    for p in cands[:8]:
        try:
            pv = api_post("/api/route/preview", {"x": p["x"], "y": p["y"]})
        except Exception:
            continue
        if pv.get("success") and 100.0 <= pv.get("distance_m", 0) <= 750.0:
            return p, pv["distance_m"]
    return None, None


def reset_world(spec, log):
    api_post("/api/estop/clear")
    api_post("/api/mission/stop")
    api_post("/api/scenario/clear")
    api_post("/api/test/park_cars", {"clear": True})
    try:
        api_post("/api/test/inject", {"component": "perception", "action": "enable"})
    except Exception:
        pass
    st = api_get("/api/state")
    have_traffic = st.get("traffic", {}).get("vehicles", 0)
    if spec["dense"] and have_traffic < 8:
        api_post("/api/traffic/clear", {"all": True})
        time.sleep(1.5)
        r = api_post("/api/traffic/spawn", {"cars": 15, "walkers": 12, "cyclists": 4},
                     timeout=60)
        log(f"  traffic ON: {r.get('vehicles', '?')} vehicles")
        time.sleep(2)
    elif not spec["dense"] and have_traffic > 0:
        api_post("/api/traffic/clear", {"all": True})
        time.sleep(1.5)
    w = api_post("/api/weather", {"preset": spec["weather"]})
    if not w.get("success"):
        log(f"  WARNING weather '{spec['weather']}' rejected: {w.get('reason')}")
    time.sleep(1.0)


def execute_run(spec, rng, points, out_dir, log):
    rid = spec["run"]
    res = {"run": rid, **{k: spec[k] for k in
                          ("kind", "weather", "dense", "parked", "take_chosen", "fill_all")}}
    trace_path = os.path.join(out_dir, f"trace_{rid:03d}.jsonl")
    frames_dir = os.path.join(out_dir, "frames")

    reset_world(spec, log)
    st0 = api_get("/api/state")
    res["collision_base"] = (st0.get("collision") or {}).get("count", 0)
    res["start_pose"] = st0.get("pose")
    res["version"] = st0.get("version")

    # Runs that occupy bays need a destination that actually HAS slot boxes;
    # retry a couple of destinations if the route ends bay-less.
    needs_slot = spec["take_chosen"] or spec["parked"] > 0 or spec["fill_all"]
    dest = dist = None
    slots_seen, chosen_seen, spot_kind, mission_id = 0, False, None, None
    for attempt in range(3):
        pose_now = (api_get("/api/state").get("pose")) or st0["pose"]
        dest, dist = pick_destination(rng, points, pose_now)
        if dest is None:
            break
        ok = api_post("/api/mission/start", {"x": dest["x"], "y": dest["y"]})
        if not ok.get("success"):
            dest = None
            continue
        # Slot data feed (what the dashboard draws) — settle a few seconds
        slots_seen, chosen_seen, spot_kind, mission_id = 0, False, None, None
        for _ in range(12):
            time.sleep(0.5)
            st = api_get("/api/state")
            slots = st.get("parking_slots") or []
            sp = st.get("parking_spot") or {}
            slots_seen = max(slots_seen, len(slots))
            chosen_seen = chosen_seen or any(s.get("chosen") for s in slots)
            spot_kind = sp.get("kind") or spot_kind
            mission_id = (st.get("mission") or {}).get("mission_id") or mission_id
            if chosen_seen and mission_id:
                break
        if needs_slot and not chosen_seen and attempt < 2:
            api_post("/api/mission/stop")
            time.sleep(1.5)
            continue
        break
    if dest is None:
        res["verdict"] = "SKIP"
        res["why"] = "no reachable destination in range from current position"
        return res
    res["destination"] = dest
    res["route_m"] = dist
    res["mission_id"] = mission_id
    res["chosen_slot_index_start"] = (api_get("/api/state").get("parking_spot") or {}).get("slot_index")

    if spec["parked"] or spec["fill_all"]:
        pk = api_post("/api/test/park_cars",
                      {"count": spec["parked"], "fill_all": spec["fill_all"],
                       "take_chosen": spec["take_chosen"]}, timeout=20)
        res["parked_spawned"] = pk.get("parked", 0)
        log(f"  parked {pk.get('parked', 0)} cars in the bay"
            + (" (chosen slot taken)" if spec["take_chosen"] else ""))

    timeout_s = min(480.0 if spec["dense"] else 420.0,
                    max(160.0, dist / (3.0 if spec["dense"] else 4.5) + 120.0))

    # ---- poll loop ----
    hazard = {"armed": spec["kind"] in ("red_light", "jaywalker", "cutin", "fault"),
              "fired": False, "t_fired": None, "cleared": False, "result": {}}
    reasons = []
    behaviors = set()
    min_clear = 1e9
    min_clear_moving = 1e9
    min_any_object = 1e9
    saw_executing = False
    stop_started = None
    settled_speeds = []
    steer_prev = 0.0
    steer_flips = 0
    moving_samples = 0
    brake_anom = 0
    max_steer = 0.0
    complete_at = None
    t_start = time.time()
    last_reason = None
    fail = None
    trace = open(trace_path, "w")

    try:
        while True:
            t = time.time() - t_start
            try:
                st = api_get("/api/state", timeout=4.0, retries=0)
            except Exception:
                if not api_alive():
                    trace.close()
                    res["verdict"] = "ERROR"
                    res["why"] = "stack went down mid-run"
                    res["elapsed_s"] = round(t, 1)
                    raise RunAbort()
                time.sleep(SAMPLE_DT)
                continue

            pose = st.get("pose") or {}
            beh = st.get("behavior", "")
            reason = st.get("behavior_reason", "")
            cmd = st.get("command") or {}
            per = st.get("perception") or {}
            tl = st.get("traffic_light") or {}
            jm = st.get("junction_ahead_m")
            speed = pose.get("speed", 0.0)
            closest = per.get("closest_distance", 999.0)
            mission_state = (st.get("mission") or {}).get("state", "?")

            behaviors.add(beh)
            if mission_state == "executing":
                saw_executing = True
            if reason != last_reason:
                reasons.append([round(t, 1), reason])
                last_reason = reason
            if closest < min_clear:
                min_clear = closest
            if speed > 0.5 and closest < min_clear_moving:
                min_clear_moving = closest
            objs = per.get("objects") or []
            if objs and speed > 0.5:
                min_any_object = min(min_any_object,
                                     min(o.get("distance", 999.0) for o in objs))
            cur_slots = st.get("parking_slots") or []
            slots_seen = max(slots_seen, len(cur_slots))
            chosen_seen = chosen_seen or any(s.get("chosen") for s in cur_slots)
            spot_kind = (st.get("parking_spot") or {}).get("kind") or spot_kind

            # smoothness bookkeeping
            steer = cmd.get("steer", 0.0)
            max_steer = max(max_steer, abs(steer))
            if speed > 2.0:
                moving_samples += 1
                if steer * steer_prev < 0 and abs(steer - steer_prev) > 0.05:
                    steer_flips += 1
                steer_prev = steer
                if (cmd.get("brake", 0) > 0.3 and speed > 5.0 and closest > 20.0
                        and tl.get("state") not in ("red", "yellow")
                        and (jm is None or jm > 25.0)
                        and not (hazard["fired"] and not hazard["cleared"])):
                    brake_anom += 1
            if (beh == "following_route" and abs(speed - CRUISE) < 1.5
                    and not (hazard["fired"] and not hazard["cleared"])):
                settled_speeds.append(speed)

            # stuck detection
            if speed < 0.15 and mission_state == "executing":
                if stop_started is None:
                    stop_started = (t, beh, reason)
                dur = t - stop_started[0]
                limit = STUCK_S_JUSTIFIED if beh in JUSTIFIED_STOPS else STUCK_S_OTHER
                if dur > limit:
                    fail = (f"STUCK {dur:.0f}s stationary — behavior '{beh}', "
                            f"reason: {reason!r}")
                    break
            else:
                stop_started = None

            # hazard state machines
            if hazard["armed"] and not hazard["fired"]:
                if spec["kind"] == "red_light" and jm is not None and speed > 3.0 \
                        and t > hazard.get("cooldown_until", 0.0) \
                        and (jm < 35.0 or tl.get("state") in ("green", "yellow", "red")):
                    api_post("/api/scenario/spawn", {"type": "red_light"})
                    hazard.update(fired=True, t_fired=t)
                    hazard["result"]["trigger_junction_m"] = round(jm, 1)
                    fetch_frame(os.path.join(frames_dir, f"run_{rid:03d}_hazard.jpg"))
                elif spec["kind"] == "jaywalker" and speed > 4.0 and (jm is None or jm > 30.0):
                    api_post("/api/scenario/spawn", {"type": "jaywalker"})
                    hazard.update(fired=True, t_fired=t)
                    fetch_frame(os.path.join(frames_dir, f"run_{rid:03d}_hazard.jpg"))
                elif spec["kind"] == "cutin" and speed > 4.5:
                    api_post("/api/scenario/spawn", {"type": "cutin"})
                    hazard.update(fired=True, t_fired=t)
                elif spec["kind"] == "fault" and t > 25.0 and speed > 3.0:
                    api_post("/api/test/inject", {"component": "perception", "action": "disable"})
                    hazard.update(fired=True, t_fired=t)

            elif hazard["fired"] and not hazard["cleared"]:
                hr = hazard["result"]
                dt_h = t - hazard["t_fired"]
                hr["min_closest"] = round(min(hr.get("min_closest", 1e9), closest), 1)
                hr["min_speed"] = round(min(hr.get("min_speed", 1e9), speed), 2)

                if spec["kind"] == "red_light":
                    if tl.get("state") in ("red", "yellow"):
                        hr["light_seen"] = True
                    if beh == "stopped_red_light" and speed < 0.3 and "hold_junction_m" not in hr:
                        hr["hold_junction_m"] = jm
                        hr["hold_stop_line_m"] = tl.get("stop_line_m")
                        fetch_frame(os.path.join(frames_dir, f"run_{rid:03d}_hold.jpg"))
                    if "hold_junction_m" in hr and dt_h > 10.0:
                        api_post("/api/scenario/clear")
                        hazard["cleared"] = True
                        hazard["t_cleared"] = t
                    elif not hr.get("light_seen") and dt_h > 12.0:
                        api_post("/api/scenario/clear")
                        hazard["attempts"] = hazard.get("attempts", 0) + 1
                        if hazard["attempts"] < 3:
                            # Unsignalized junction — re-arm for the next one.
                            hazard.update(fired=False, t_fired=None)
                            hazard["cooldown_until"] = t + 20.0
                            hr.pop("trigger_junction_m", None)
                        else:
                            hr["no_light_on_route"] = True
                            hazard["cleared"] = True
                            hazard["t_cleared"] = t
                    elif hr.get("light_seen") and "hold_junction_m" not in hr \
                            and jm is not None and jm < 2.0 and speed > 2.0:
                        fail = "RAN THE RED LIGHT — entered the junction at speed"
                        break
                elif spec["kind"] == "jaywalker":
                    if beh == "stopped_pedestrian":
                        hr["stopped_for_ped"] = True
                    if dt_h > 25.0 or (hr.get("stopped_for_ped") and beh not in
                                       ("stopped_pedestrian",) and speed > 2.0 and dt_h > 8.0):
                        api_post("/api/scenario/clear")
                        hazard["cleared"] = True
                        hazard["t_cleared"] = t
                elif spec["kind"] == "cutin":
                    if beh in ("following_vehicle", "stopped_vehicle"):
                        hr["reacted"] = True
                    if dt_h > 20.0:
                        api_post("/api/scenario/clear")
                        hazard["cleared"] = True
                        hazard["t_cleared"] = t
                elif spec["kind"] == "fault":
                    if st.get("safety", {}).get("state") != "ok":
                        hr["safety_reacted_s"] = hr.get("safety_reacted_s", round(dt_h, 1))
                    if dt_h > 6.0:
                        api_post("/api/test/inject", {"component": "perception", "action": "enable"})
                        hazard["cleared"] = True
                        hazard["t_cleared"] = t

            elif hazard.get("cleared") and "resumed" not in hazard["result"]:
                if speed > 2.0:
                    hazard["result"]["resumed"] = True
                    hazard["result"]["resume_after_s"] = round(t - hazard["t_cleared"], 1)

            trace.write(json.dumps({
                "t": round(t, 1), "x": pose.get("x"), "y": pose.get("y"),
                "yaw": pose.get("yaw"), "v": speed, "beh": beh,
                "st": cmd.get("steer"), "th": cmd.get("throttle"), "br": cmd.get("brake"),
                "cl": closest, "tl": tl.get("state"), "jm": jm,
                "col": (st.get("collision") or {}).get("count"),
                "ms": mission_state,
            }) + "\n")

            # end conditions
            if mission_state == "completed" or beh == "mission_complete":
                complete_at = t
                break
            if mission_state in ("failed", "cancelled"):
                fail = f"mission ended '{mission_state}' without completing"
                break
            if mission_state == "idle" and saw_executing:
                # Completion clears current_mission within the same tick, so a
                # finished mission reads 'idle' here — history has the truth.
                hist_ok, hist = mission_ended_ok(mission_id)
                if hist_ok:
                    complete_at = t
                    res["history_entry"] = hist
                else:
                    fail = ("mission ended 'idle' without a completed history entry "
                            f"({(hist or {}).get('state')}: {(hist or {}).get('reason_ended')})")
                break
            if t > timeout_s:
                fail = (f"TIMEOUT after {timeout_s:.0f}s (route {dist:.0f} m) — "
                        f"last behavior '{beh}', reason: {reason!r}")
                break

            # collision fast-exit
            colls = (st.get("collision") or {}).get("count", 0) - res["collision_base"]
            if colls > 0:
                fail = f"COLLISION: {(st.get('collision') or {}).get('last')}"
                break

            time.sleep(SAMPLE_DT)
    finally:
        try:
            trace.close()
        except Exception:
            pass

    # ---- post-run verdict ----
    st = api_get("/api/state")
    res["elapsed_s"] = round(time.time() - t_start, 1)
    res["behaviors_seen"] = sorted(behaviors)
    res["reason_changes"] = len(reasons)
    res["reasons_tail"] = reasons[-6:]
    res["slots_found"] = slots_seen
    res["slot_chosen"] = chosen_seen
    res["park_target_kind"] = spot_kind
    res["min_clearance_m"] = round(min_clear_moving, 1) if min_clear_moving < 1e9 else None
    res["min_object_m"] = round(min_any_object, 1) if min_any_object < 1e9 else None
    res["collisions"] = (st.get("collision") or {}).get("count", 0) - res["collision_base"]
    res["collision_last"] = (st.get("collision") or {}).get("last") if res["collisions"] else None
    res["last_tick_error"] = st.get("last_tick_error") or ""
    mins = res["elapsed_s"] / 60.0
    res["steer_flips_per_s"] = round(steer_flips / max(1.0, moving_samples * SAMPLE_DT), 3)
    res["max_abs_steer"] = round(max_steer, 2)
    res["brake_anomaly_ticks"] = brake_anom
    if len(settled_speeds) > 25:
        m = sum(settled_speeds) / len(settled_speeds)
        res["cruise_wobble"] = round(math.sqrt(
            sum((v - m) ** 2 for v in settled_speeds) / len(settled_speeds)), 2)
    res["hazard"] = hazard["result"] if hazard["armed"] else None
    if hazard["armed"] and not hazard["fired"]:
        res["hazard"] = {"not_triggered": True}

    # parking check
    sp = st.get("parking_spot") or {}
    slots = st.get("parking_slots") or []
    res["chosen_slot_index_end"] = sp.get("slot_index")
    res["rescan_retargeted"] = (spec["take_chosen"] and
                                res.get("chosen_slot_index_start") is not None and
                                sp.get("slot_index") != res.get("chosen_slot_index_start"))
    if complete_at is not None:
        pose = st.get("pose") or {}
        res["completed"] = True
        if sp.get("kind") == "slot" and sp.get("slot_index") is not None:
            slot = next((s for i, s in enumerate(slots) if i == sp["slot_index"]), None)
            if slot:
                inside, m_len, m_wid = van_in_slot(pose.get("x", 0), pose.get("y", 0),
                                                   pose.get("yaw", 0), slot)
                res["parked_inside_box"] = inside
                res["park_margin_len_m"] = m_len
                res["park_margin_wid_m"] = m_wid
                res["park_heading_off_deg"] = round(
                    ang_diff_deg(pose.get("yaw", 0.0), math.degrees(slot.get("yaw", 0.0))), 1)
        fetch_frame(os.path.join(out_dir, "frames", f"run_{rid:03d}_parked.jpg"))
    else:
        res["completed"] = False

    # verdict
    why = []
    if res["collisions"]:
        why.append(f"collision with {(res['collision_last'] or {}).get('with')}")
    if fail:
        why.append(fail)
    if res["last_tick_error"]:
        why.append(f"tick error: {res['last_tick_error'][:120]}")
    if complete_at is not None and res.get("park_target_kind") == "slot" \
            and res.get("parked_inside_box") is False and not spec["fill_all"]:
        why.append(f"finished OUTSIDE the slot box (margins {res.get('park_margin_len_m')}/"
                   f"{res.get('park_margin_wid_m')} m, heading off {res.get('park_heading_off_deg')}°)")
    if hazard["armed"] and hazard["fired"]:
        hr = hazard["result"]
        if spec["kind"] == "red_light" and hr.get("light_seen") and "hold_junction_m" not in hr:
            why.append("red light seen but no full stop recorded")
        if spec["kind"] == "jaywalker" and not hr.get("stopped_for_ped") \
                and hr.get("min_closest", 99) < 6.0:
            why.append(f"jaywalker got within {hr.get('min_closest')} m without a pedestrian stop")
        if spec["kind"] == "fault" and "safety_reacted_s" not in hr:
            why.append("perception was disabled but safety never left 'ok'")
    if spec["take_chosen"] and res["completed"] \
            and res.get("chosen_slot_index_start") is not None \
            and not res.get("rescan_retargeted"):
        why.append("chosen slot was occupied but the van never retargeted to another")
    if spec["take_chosen"] and res.get("chosen_slot_index_start") is None:
        res["take_chosen_note"] = "route had no slot to occupy — case not exercised"

    if spec["fill_all"] and res["collisions"] == 0 and not res["last_tick_error"]:
        # Every slot occupied: the HONEST outcome is refusing to park by
        # force. Safe non-completion (or a safe stop) is the pass here.
        res["verdict"] = "PASS"
        res["why"] = ("all slots occupied — van held off safely (expected); "
                      + (fail or "completed elsewhere"))
        return res

    if res["collisions"] or fail or res["last_tick_error"]:
        res["verdict"] = "ERROR" if "stack went down" in (fail or "") else "FAIL"
    elif why:
        res["verdict"] = "FAIL"
    elif hazard["armed"] and (not hazard["fired"] or hazard["result"].get("no_light_on_route")):
        res["verdict"] = "GAP"
        why.append("hazard could not be exercised on this route (no trigger window)")
    elif not res["completed"]:
        res["verdict"] = "FAIL"
        why.append("mission did not complete")
    else:
        res["verdict"] = "PASS"
    res["why"] = "; ".join(why) if why else "ok"
    return res


# ----------------------------------------------------------------------
# Scoreboard
# ----------------------------------------------------------------------
def write_scoreboard(out_dir, results, plan_total, incidents):
    by_v = {}
    for r in results:
        by_v[r["verdict"]] = by_v.get(r["verdict"], 0) + 1
    lines = ["# Mission Sweep — live scoreboard", ""]
    lines.append(f"**{len(results)} / {plan_total} runs done** — "
                 + " · ".join(f"{k}: {v}" for k, v in sorted(by_v.items())))
    coll = sum(r.get("collisions", 0) for r in results)
    parked_ok = sum(1 for r in results if r.get("parked_inside_box"))
    parked_checked = sum(1 for r in results if r.get("parked_inside_box") is not None)
    slots_ok = sum(1 for r in results if r.get("slots_found", 0) > 0)
    lines.append(f"")
    lines.append(f"- Collisions: **{coll}**")
    lines.append(f"- Parking slots offered: {slots_ok}/{len(results)} runs")
    lines.append(f"- Fully inside the box: {parked_ok}/{parked_checked} slot-parkings")
    if incidents:
        lines.append(f"- Incidents: " + "; ".join(incidents[-5:]))
    lines += ["", "| run | kind | weather | dense | verdict | why |",
              "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['kind']} | {r['weather']} | "
                     f"{'Y' if r['dense'] else ''} | {r['verdict']} | "
                     f"{str(r.get('why', ''))[:110]} |")
    with open(os.path.join(out_dir, "SCOREBOARD.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--expect-version", default=None)
    a = ap.parse_args()

    plan = shakedown_plan() if a.shakedown else build_plan()
    out_dir = a.out or ("sweep_out/shakedown" if a.shakedown else "sweep_out/full")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    logf = open(os.path.join(out_dir, "sweep.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    incidents = []
    results = []
    done = set()
    if not a.no_resume:
        for r in sorted(os.listdir(out_dir)):
            if r.startswith("run_") and r.endswith(".json"):
                with open(os.path.join(out_dir, r)) as f:
                    old = json.load(f)
                results.append(old)
                done.add(old["run"])
        if done:
            log(f"resuming: {len(done)} runs already on disk")

    log(f"mission sweep starting — {len(plan)} planned runs, output: {out_dir}")
    ensure_stack_up(log)
    st = api_get("/api/state")
    log(f"stack version {st.get('version')} · uptime {st.get('uptime_s')}s")
    if a.expect_version and st.get("version") != a.expect_version:
        log(f"WARNING: expected version {a.expect_version}, stack runs {st.get('version')}")

    # kill any leftover traffic-spawner scripts holding actors
    ssh('Get-CimInstance Win32_Process | Where-Object {($_.Name -eq "python.exe") -and '
        '($_.CommandLine -like "*spawn_traffic*")} '
        '| ForEach-Object {Stop-Process -Id $_.ProcessId -Force}')
    api_post("/api/traffic/clear", {"all": True})

    rng = random.Random(99)
    points = api_get("/api/spawn_points")
    log(f"{len(points)} destination points available")

    for spec in plan:
        if spec["run"] in done:
            continue
        log(f"run {spec['run']}/{len(plan)}: {spec['kind']}"
            + (f" +dense" if spec["dense"] else "")
            + (f" +parked{spec['parked']}" if spec["parked"] else "")
            + (" +fill_all" if spec["fill_all"] else "")
            + f" [{spec['weather']}]")
        try:
            res = execute_run(spec, rng, points, out_dir, log)
        except RunAbort:
            res = {"run": spec["run"], **spec, "verdict": "ERROR",
                   "why": "stack went down mid-run"}
            incidents.append(f"run {spec['run']}: stack down — watchdog restarted it")
            ensure_stack_up(log)
        except Exception as e:
            res = {"run": spec["run"], **spec, "verdict": "ERROR",
                   "why": f"harness exception: {e}"}
            log(f"  HARNESS ERROR: {e}")
            ensure_stack_up(log)
        results.append(res)
        with open(os.path.join(out_dir, f"run_{spec['run']:03d}.json"), "w") as f:
            json.dump(res, f, indent=1)
        write_scoreboard(out_dir, results, len(plan), incidents)
        log(f"  -> {res['verdict']}: {str(res.get('why'))[:140]}")
        time.sleep(2.0)

    # leave the world tidy
    try:
        api_post("/api/scenario/clear")
        api_post("/api/test/park_cars", {"clear": True})
        api_post("/api/traffic/clear", {"all": True})
        api_post("/api/weather", {"preset": "ClearNoon"})
    except Exception:
        pass
    log("sweep complete")
    n = {v: sum(1 for r in results if r["verdict"] == v)
         for v in ("PASS", "FAIL", "GAP", "ERROR", "SKIP")}
    log(f"final: {n}")


if __name__ == "__main__":
    main()
