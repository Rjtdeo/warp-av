#!/usr/bin/env python3
"""
Live check of static vs dynamic (Planning V2): drive a mission past the hard cases and
score every label the van gives against where CARLA says every person and vehicle is.

    python tools/check_static_live.py --dest 160 --follow 150 --cars 8 --walkers 10
    python tools/check_static_live.py --show --dest 450 --follow 420     # labels in CARLA

With --show every thing the van tracks gets a label in the CARLA window, above it: what the
van thinks it is (STATIC pole / wall / kerb, checking, CAN MOVE) -- and a red WRONG if a
static label ever sits on a real person or vehicle. The labels are CARLA text, which the
simulator draws for the spectator only: checked 2026-09-10, 21 test labels in front of the
van did not appear in its own camera, while a single debug POINT did (so no points or lines
are drawn here -- they would be in the pictures the van reads).

  1. Finds the stack's van by its reported pose; refuses if a mission is running or the
     stack is not in camera + LiDAR mode.
  2. Starts a mission, reads the route the van planned, and puts the hard cases found in
     the recordings along it (tools/record_static_truth.py): people standing right beside
     lamp posts and trees, a person against the building line, a car parked beside a lamp
     post and a lorry pulled over (on a shoulder or parking lane only, so the route stays
     open), cones and a bin on the pavement -- plus people walking about and cars driving.
  3. Follows the van with the spectator camera, so it can be watched in the CARLA window.
  4. Twice a second reads the van's objects and CARLA's people and vehicles (actors and the
     map's own parked cars). An object labelled STATIC whose footprint touches a person or
     a vehicle is a MISTAKE. Objects with no person or vehicle within 3 m give the share of
     things called static.
  5. Stops the mission, removes only what it placed, prints the scorecard.

Read-only towards the stack: it never touches the van, its sensors or the world settings.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import carla  # noqa: E402
from warp_av.planning.footprint import _boxes_touch  # noqa: E402

API = "http://127.0.0.1:5000"
OBJECT_MARGIN_M = 0.4        # the LiDAR sees a thing's near face: its footprint is a lower bound
NEAR_USER_M = 3.0            # "no person or vehicle near" for the share-caught figure
# A static object HOLDS a person or vehicle -- the mistake that is not allowed -- when its
# footprint covers a real part of theirs AND it reaches their height, so their points could
# be in it. Merely touching is not that: a lamp post a car drives past, a kerb a person
# stands on. Those are reported separately, as "static thing right beside a road user".
ROUTE_CLEARANCE_M = 3.2      # a parked vehicle's nearest corner at least this far off the route line:
                             # 1.9 m still let a shoulder-parked car hold the van (KNOWN_ISSUES)
HOLDS_COVER_SHARE = 0.25
HOLDS_HEIGHT_SHARE = 0.6
ROAD_USER_MAP_LABELS = ("Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Rider", "Pedestrians")
BIG = ("vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv",
       "vehicle.mitsubishi.fusorosa", "vehicle.ford.ambulance")


BLUE, LIGHT_BLUE, ORANGE, GREY, WHITE, RED = ((40, 120, 255), (140, 190, 255), (255, 140, 0),
                                              (170, 170, 170), (255, 255, 255), (255, 0, 0))
HARD_CASES = ("person beside a lamp post", "person beside a tree", "person against the building line",
              "car parked beside a lamp post", "parked lorry", "parked car", "prop on the pavement")


def label_for(o):
    """What the van thinks, in a few words, and the colour to say it in."""
    kind = o.get("type") or "?"
    word = {"pedestrian": "person", "vehicle": "vehicle", "cyclist": "cyclist"}.get(kind)
    why = o.get("motion_why") or ""
    if o.get("motion_class") == "static":
        return f"STATIC {({'structure': 'wall'}).get(o.get('static_rule'), o.get('static_rule'))}", BLUE
    if word:
        return f"CAN MOVE: {word}", ORANGE
    if why == "moving":
        return "CAN MOVE: moving", ORANGE
    if why.startswith("fits"):
        rule = why.split()[1] if len(why.split()) > 1 else ""
        return f"checking: {({'structure': 'wall'}).get(rule, rule)}?", LIGHT_BLUE
    return "can move", GREY


def get(path):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:    # planning a route can take a while
        return json.loads(r.read().decode())


def find_van(world, st):
    px, py = st["pose"]["x"], st["pose"]["y"]
    best, best_d = None, 3.0
    for v in world.get_actors().filter("vehicle.*"):
        loc = v.get_location()
        d = math.hypot(loc.x - px, loc.y - py)
        if d < best_d:
            best, best_d = v, d
    return best


def box_of_actor(a):
    tf, bb = a.get_transform(), a.bounding_box
    yaw = math.radians(tf.rotation.yaw)
    c = tf.transform(bb.location)
    return (c.x, c.y, yaw, bb.extent.x, bb.extent.y, 2.0 * bb.extent.z)


def _corners(cx, cy, heading, hl, hw):
    c, s = math.cos(heading), math.sin(heading)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))]


def _area(poly):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])))


def _clip(subject, clipper):
    """Sutherland-Hodgman: the part of `subject` inside the convex `clipper` (both CCW)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def cross(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = subject
    for a, b in zip(clipper, clipper[1:] + clipper[:1]):
        if not out:
            break
        inp, out = out, []
        for p, q in zip(inp, inp[1:] + inp[:1]):
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cross(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cross(p, q, a, b))
    return out


def covered_share(obj_box, user_box):
    """How much of a person's or vehicle's footprint an object's footprint covers (0-1)."""
    ox, oy, oh, ohl, ohw = obj_box
    ux, uy, uh, uhl, uhw = user_box[:5]
    user = _corners(ux, uy, uh, uhl, uhw)
    inter = _clip(user, _corners(ox, oy, oh, ohl, ohw))
    return (_area(inter) / max(1e-6, _area(user))) if len(inter) >= 3 else 0.0


def map_road_users(world):
    out = []
    for name in ROAD_USER_MAP_LABELS:
        label = getattr(carla.CityObjectLabel, name, None)
        if label is None:
            continue
        for o in world.get_environment_objects(label):
            bb = o.bounding_box
            out.append((f"map:{name.lower()}", (bb.location.x, bb.location.y,
                                                math.radians(bb.rotation.yaw), bb.extent.x, bb.extent.y,
                                                2.0 * bb.extent.z)))
    return out


def route_points(timeout_s=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = get("/api/route")
            if isinstance(r, list) and len(r) >= 2:
                return [(p["x"], p["y"]) for p in r]
        except Exception:
            pass
        time.sleep(0.2)
    return []


class Route:
    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.total = self.cum[-1]

    def at(self, arc):
        arc = max(0.0, min(self.total - 0.01, arc))
        for i in range(len(self.pts) - 1):
            if self.cum[i + 1] >= arc:
                (x0, y0), (x1, y1) = self.pts[i], self.pts[i + 1]
                seg = self.cum[i + 1] - self.cum[i]
                t = (arc - self.cum[i]) / seg if seg > 0 else 0.0
                return x0 + t * (x1 - x0), y0 + t * (y1 - y0), math.atan2(y1 - y0, x1 - x0)
        return self.pts[-1][0], self.pts[-1][1], 0.0

    def project(self, x, y):
        """(arc, lateral distance) of a point."""
        best = (0.0, float("inf"))
        for i in range(len(self.pts) - 1):
            (x0, y0), (x1, y1) = self.pts[i], self.pts[i + 1]
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy or 1e-9
            t = max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
            d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
            if d < best[1]:
                best = (self.cum[i] + t * math.sqrt(L2), d)
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=float, default=160.0, help="destination this far ahead along the lane")
    ap.add_argument("--follow", type=float, default=150.0, help="give up after this many seconds")
    ap.add_argument("--cars", type=int, default=8)
    ap.add_argument("--walkers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true", help="label everything in the CARLA window")
    ap.add_argument("--period", type=float, default=None, help="seconds between readings")
    ap.add_argument("--give-up-stuck", type=float, default=45.0,
                    help="end the run after the van has been held this long in one place")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    period = a.period or (0.25 if a.show else 0.5)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    bl = world.get_blueprint_library()
    st = get("/api/state")
    if st.get("perception_mode") != "camera_lidar":
        sys.exit(f"stack is in {st.get('perception_mode')} mode: this checks camera + LiDAR labels")
    if st.get("mission", {}).get("state") not in (None, "idle", "completed", "failed", "cancelled"):
        sys.exit(f"stack is busy: mission {st['mission']['state']}")
    van = find_van(world, st)
    if van is None:
        sys.exit("could not find the stack's van at its reported pose")

    placed, controllers = [], []
    labels = {}                                    # actor id -> what we put there

    def spawn(bp, tf, label):
        act = world.try_spawn_actor(bp, tf)
        if act is None:
            tf.location.z += 0.4
            act = world.try_spawn_actor(bp, tf)
        if act is not None:
            placed.append(act)
            labels[act.id] = label
        return act

    def leaves_route_open(act):
        """A parked vehicle must leave the van's path clear: the van is 2 m wide, and it
        swings out a little in turns. Run 3 lost its drive to an ambulance on a shoulder
        that stuck into the lane -- the van was right to stop, the test was wrong to put
        it there."""
        x, y, h, hl, hw, _ = box_of_actor(act)
        worst = min(route.project(cx, cy)[1] for cx, cy in _corners(x, y, h, hl, hw))
        return worst >= ROUTE_CLEARANCE_M

    walker_bps = list(bl.filter("walker.pedestrian.*"))
    four = [b for b in bl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
            and "sprinter" not in b.id]
    ctrl_bp = bl.find("controller.ai.walker")

    tf = van.get_transform()
    wp = cmap.get_waypoint(tf.location)
    dest = wp.next(a.dest)[0].transform.location
    spectator = world.get_spectator()

    def chase():
        t = van.get_transform()
        yaw = math.radians(t.rotation.yaw)
        back, up, pitch = (13.0, 7.0, -22.0) if a.show else (10.0, 5.0, -18.0)
        cam = carla.Location(x=t.location.x - back * math.cos(yaw), y=t.location.y - back * math.sin(yaw),
                             z=t.location.z + up)
        spectator.set_transform(carla.Transform(cam, carla.Rotation(pitch=pitch, yaw=t.rotation.yaw)))

    import threading
    following = threading.Event()

    def keep_chasing():                     # 20 times a second, so the view is smooth
        while not following.is_set():
            try:
                chase()
            except Exception:
                pass
            time.sleep(0.05)

    samples = []
    mistakes = {}
    beside = {}
    try:
        r = post("/api/mission/start", {"x": dest.x, "y": dest.y})
        print("mission start:", r, flush=True)
        route = Route(route_points())
        if route.total < 20.0:
            sys.exit("no usable route from the stack")
        print(f"route {route.total:.0f} m", flush=True)

        # ---- the hard cases, along the route ------------------------------------------
        poles = []
        for o in world.get_environment_objects(carla.CityObjectLabel.Poles):
            bb = o.bounding_box
            arc, lat = route.project(bb.location.x, bb.location.y)
            if 12.0 < arc < route.total - 10.0 and lat < 9.0:
                poles.append((arc, lat, bb.location.x, bb.location.y, max(bb.extent.x, bb.extent.y)))
        rng.shuffle(poles)
        trees = []
        for o in world.get_environment_objects(carla.CityObjectLabel.Vegetation):
            bb = o.bounding_box
            if bb.extent.z < 1.5 or max(bb.extent.x, bb.extent.y) > 3.0:
                continue
            arc, lat = route.project(bb.location.x, bb.location.y)
            if 12.0 < arc < route.total - 10.0 and lat < 9.0:
                trees.append((arc, lat, bb.location.x, bb.location.y, max(bb.extent.x, bb.extent.y)))
        rng.shuffle(trees)

        def ground_z(x, y):
            g = cmap.get_waypoint(carla.Location(x, y, tf.location.z), project_to_road=True,
                                  lane_type=carla.LaneType.Sidewalk)
            return (g.transform.location.z if g is not None else tf.location.z)

        def glue_person(x, y, reach, label):
            for _ in range(6):
                ang = rng.uniform(0, 2 * math.pi)
                r = reach + rng.uniform(0.35, 0.6)
                px, py = x + r * math.cos(ang), y + r * math.sin(ang)
                _arc, lat = route.project(px, py)
                if lat < 2.8:
                    continue                    # not in the van's lane: this is about the pavement
                act = spawn(rng.choice(walker_bps), carla.Transform(
                    carla.Location(px, py, ground_z(px, py) + 1.0), carla.Rotation(yaw=rng.uniform(0, 360))), label)
                if act is not None:
                    return act
            return None

        for (arc, lat, x, y, reach) in poles[:4]:
            glue_person(x, y, reach, "person beside a lamp post")
        for (arc, lat, x, y, reach) in trees[:2]:
            glue_person(x, y, reach, "person beside a tree")

        # the building line: the far edge of the pavement, part-way along
        for arc in (35.0, 70.0):
            x, y, h = route.at(arc)
            w = cmap.get_waypoint(carla.Location(x, y, tf.location.z))
            side = None
            for step in ("get_right_lane", "get_left_lane"):
                cur = w
                for _ in range(4):
                    cur = getattr(cur, step)() if cur is not None else None
                    if cur is not None and cur.lane_type == carla.LaneType.Sidewalk:
                        side = cur
                        break
                if side is not None:
                    break
            if side is not None:
                t = side.transform
                yaw = math.radians(t.rotation.yaw)
                off = side.lane_width / 2.0 - 0.35
                for sgn in (1, -1):
                    px, py = t.location.x - math.sin(yaw) * off * sgn, t.location.y + math.cos(yaw) * off * sgn
                    if route.project(px, py)[1] > side.lane_width:
                        spawn(rng.choice(walker_bps), carla.Transform(
                            carla.Location(px, py, t.location.z + 1.0), carla.Rotation(yaw=rng.uniform(0, 360))),
                            "person against the building line")
                        break
                # cones and a bin on the same pavement: static things, for the share caught
                for k, name in enumerate(("static.prop.trafficcone01", "static.prop.bin",
                                          "static.prop.trafficcone01")):
                    tt = side.next(3.0 + 2.5 * k)
                    if tt:
                        l = tt[0].transform.location
                        act = spawn(bl.find(name), carla.Transform(carla.Location(l.x, l.y, l.z + 0.05)),
                                    "prop on the pavement")
                        if act is not None:
                            act.set_simulate_physics(False)

        # parked vehicles, only where they cannot block the route: shoulder or parking lanes
        parked = 0
        for arc in (25.0, 45.0, 60.0, 85.0, 110.0):
            if parked >= 3 or arc > route.total - 15.0:
                break
            x, y, h = route.at(arc)
            w = cmap.get_waypoint(carla.Location(x, y, tf.location.z))
            lane = None
            for step in ("get_right_lane", "get_left_lane"):
                cur = w
                for _ in range(3):
                    cur = getattr(cur, step)() if cur is not None else None
                    if cur is not None and cur.lane_type in (carla.LaneType.Shoulder, carla.LaneType.Parking):
                        lane = cur
                        break
                if lane is not None:
                    break
            if lane is None:
                continue
            t = lane.transform
            bp = rng.choice(list(b for n in BIG for b in bl.filter(n)) if parked == 1 else four)
            act = spawn(bp, carla.Transform(carla.Location(t.location.x, t.location.y, t.location.z + 0.4),
                                            t.rotation), "parked lorry" if parked == 1 else "parked car")
            if act is not None and not leaves_route_open(act):
                placed.remove(act)
                labels.pop(act.id, None)
                act.destroy()
                act = None
            if act is not None:
                parked += 1
        # a car parked right beside a lamp post, on a shoulder or parking lane
        for (arc, lat, x, y, reach) in poles[4:12]:
            g = cmap.get_waypoint(carla.Location(x, y, tf.location.z), project_to_road=True,
                                  lane_type=carla.LaneType.Shoulder | carla.LaneType.Parking)
            if g is None or math.hypot(g.transform.location.x - x, g.transform.location.y - y) > 2.5:
                continue
            t = g.transform
            act = spawn(rng.choice(four), carla.Transform(carla.Location(t.location.x, t.location.y,
                                                                         t.location.z + 0.4), t.rotation),
                        "car parked beside a lamp post")
            if act is not None and not leaves_route_open(act):
                placed.remove(act)
                labels.pop(act.id, None)
                act.destroy()
                act = None
            if act is not None:
                break

        # people walking about near the route, and some traffic
        world.set_pedestrians_cross_factor(0.0)       # walking about, not into the van's path
        for _ in range(a.walkers * 6):
            if sum(1 for v in labels.values() if v == "walking person") >= a.walkers:
                break
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            arc, lat = route.project(loc.x, loc.y)
            if not (8.0 < arc < route.total and 3.0 < lat < 15.0):
                continue
            wk = spawn(rng.choice(walker_bps), carla.Transform(loc + carla.Location(z=1.0)), "walking person")
            if wk is None:
                continue
            c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=wk)
            controllers.append(c)
            c.start()
            d = world.get_random_location_from_navigation()
            if d is not None:
                c.go_to_location(d)
            c.set_max_speed(rng.uniform(1.0, 1.6))
        tm = client.get_trafficmanager()
        pts = [p for p in cmap.get_spawn_points()
               if 40.0 < math.hypot(p.location.x - tf.location.x, p.location.y - tf.location.y) < 150.0]
        rng.shuffle(pts)
        cars = 0
        for p in pts:
            if cars >= a.cars:
                break
            v = spawn(rng.choice(four), p, "driving car")
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                cars += 1
        print("placed:", dict(collections.Counter(labels.values())), flush=True)

        # ---- follow and score --------------------------------------------------------
        map_users = map_road_users(world)
        threading.Thread(target=keep_chasing, daemon=True).start()
        t0 = time.time()
        held_since, held_at = None, None
        wrong_total = 0
        while time.time() - t0 < a.follow:
            try:
                st = get("/api/state")
            except Exception:
                time.sleep(period)
                continue
            vz = van.get_location().z               # labels ride on the road as it rises and falls
            users = list(map_users)
            for act in list(world.get_actors().filter("walker.pedestrian.*")) + \
                    [v for v in world.get_actors().filter("vehicle.*") if v.id != van.id]:
                users.append((f"{act.type_id}#{act.id}", box_of_actor(act)))
            pyaw = math.radians(st["pose"]["yaw"])
            objs = st.get("perception", {}).get("objects", []) or []
            plan = st.get("planner") or {}
            rec = {"t": round(time.time() - t0, 1), "speed": st["pose"]["speed"],
                   "behavior": st.get("behavior"), "objects": 0, "static": 0,
                   "clear_static": 0, "clear": 0, "predicted": st.get("predicted_conflict"),
                   "planner": {k: plan.get(k) for k in ("reason", "blocker_id", "blocker_kind",
                                                        "blocker_distance_m", "blocker_lateral_m")},
                   "why": {}}
            by_id = {o.get("id"): o for o in objs}
            for o in objs:
                if o.get("x") is None or o.get("distance", 99) > 40.0:
                    continue
                rec["objects"] += 1
                heading = pyaw + math.radians(o.get("yaw_deg", 0.0))
                hl = max(o.get("length_m", 0.0), 0.2) / 2.0 + OBJECT_MARGIN_M
                hw = max(o.get("width_m", 0.0), 0.2) / 2.0 + OBJECT_MARGIN_M
                touching = [(n, ub) for n, ub in users
                            if _boxes_touch(o["x"], o["y"], heading, hl, hw, *ub[:5])]
                near = touching or [n for n, ub in users
                                    if _boxes_touch(o["x"], o["y"], heading, hl + NEAR_USER_M,
                                                    hw + NEAR_USER_M, *ub[:5])]
                is_static = o.get("motion_class") == "static"
                rec["static"] += is_static
                if a.show and o.get("distance", 99) <= 30.0:
                    text, rgb = label_for(o)
                    if rgb != GREY or o.get("distance", 99) <= 12.0:
                        world.debug.draw_string(
                            carla.Location(o["x"], o["y"], vz + (o.get("height_m") or 0.0) + 0.6),
                            text, draw_shadow=True, color=carla.Color(*rgb), life_time=period + 0.12)
                if not near:
                    rec["clear"] += 1
                    rec["clear_static"] += is_static
                    w = o.get("motion_why") or "?"
                    rec["why"][w] = rec["why"].get(w, 0) + 1
                if not (is_static and touching):
                    continue
                tight = (o["x"], o["y"], heading, max(o.get("length_m", 0.0), 0.2) / 2.0 + 0.1,
                         max(o.get("width_m", 0.0), 0.2) / 2.0 + 0.1)
                for n, ub in touching:
                    share = covered_share(tight, ub)
                    tall_enough = (o.get("height_m") or 0.0) >= HOLDS_HEIGHT_SHARE * ub[5]
                    # is the person or vehicle ALSO tracked as a thing of its own, and as what?
                    own = [f"{b.get('id')}:{b.get('motion_class')}" for b in objs
                           if b is not o and b.get("x") is not None
                           and _boxes_touch(b["x"], b["y"], 0.0, 0.3, 0.3, ub[0], ub[1], ub[2],
                                            ub[3] + 0.3, ub[4] + 0.3)]
                    entry = {"t": rec["t"], "object": o["id"], "rule": o.get("static_rule"), "user": n,
                             "covers": round(share, 2), "object_h": o.get("height_m"),
                             "user_h": round(ub[5], 2), "size": [o.get("length_m"), o.get("width_m")],
                             "user_tracked_as": own, "distance": o.get("distance")}
                    key = (o["id"], n)
                    if share >= HOLDS_COVER_SHARE and tall_enough:
                        if a.show:
                            world.debug.draw_string(
                                carla.Location(o["x"], o["y"], vz + (o.get("height_m") or 0.0) + 1.2),
                                "WRONG: static on a person/vehicle", draw_shadow=True,
                                color=carla.Color(*RED), life_time=period + 0.12)
                        if key not in mistakes:
                            mistakes[key] = entry
                            print(f"  MISTAKE: {entry}", flush=True)
                    elif key not in beside:
                        beside[key] = entry
            if a.show:
                vt = van.get_transform()
                for act in placed:
                    lab = labels.get(act.id)
                    if lab not in HARD_CASES:
                        continue
                    try:
                        loc = act.get_location()
                    except Exception:
                        continue
                    if math.hypot(loc.x - vt.location.x, loc.y - vt.location.y) <= 30.0:
                        world.debug.draw_string(carla.Location(loc.x, loc.y, loc.z + 2.6),
                                                f"test: {lab}", draw_shadow=True,
                                                color=carla.Color(*WHITE), life_time=period + 0.12)
                shown = [o for o in objs if o.get("x") is not None and o.get("distance", 99) <= 30.0]
                n_static = sum(1 for o in shown if o.get("motion_class") == "static")
                world.debug.draw_string(
                    carla.Location(vt.location.x, vt.location.y, vt.location.z + 3.6),
                    f"STATIC {n_static}  |  can move {len(shown) - n_static}  |  WRONG {len(mistakes)}"
                    f"  |  {st.get('behavior')} {st['pose']['speed']:.1f} m/s",
                    draw_shadow=True, color=carla.Color(*WHITE), life_time=period + 0.12)
            blk = rec["planner"].get("blocker_id")
            if blk is not None and blk in by_id:
                b = by_id[blk]
                rec["blocker"] = {"motion_class": b.get("motion_class"), "why": b.get("motion_why"),
                                  "type": b.get("type"), "size": [b.get("length_m"), b.get("width_m"),
                                                                  b.get("height_m")],
                                  "at": [b.get("x"), b.get("y")],
                                  "touches": [n for n, ub in users
                                              if _boxes_touch(b["x"], b["y"],
                                                              pyaw + math.radians(b.get("yaw_deg", 0.0)),
                                                              max(b.get("length_m", 0.0), 0.2) / 2 + 0.6,
                                                              max(b.get("width_m", 0.0), 0.2) / 2 + 0.6,
                                                              *ub[:5])]}
            samples.append(rec)
            if len(samples) % 10 == 0 and rec["planner"].get("reason") not in (None, "clear"):
                print(f"         planner: {rec['planner']} blocker: {rec.get('blocker')}", flush=True)
            if len(samples) % 10 == 0:
                print(f"  t={rec['t']:5.1f}s speed {rec['speed']:4.1f} {rec['behavior']:<14s} "
                      f"objects {rec['objects']:3d} static {rec['static']:3d} "
                      f"(clear of people/vehicles: {rec['clear_static']}/{rec['clear']})", flush=True)
            m = st.get("mission", {}).get("state")
            if m in ("completed", "failed", "cancelled") and time.time() - t0 > 5:
                print(f"mission {m}", flush=True)
                break
            here = (st["pose"]["x"], st["pose"]["y"])
            if held_at is None or math.hypot(here[0] - held_at[0], here[1] - held_at[1]) > 1.0:
                held_since, held_at = time.time(), here
            elif time.time() - held_since > a.give_up_stuck:
                print(f"van held in one place for {a.give_up_stuck:.0f} s ({st.get('behavior')}: "
                      f"{rec['planner']}) -- ending the run", flush=True)
                break
            time.sleep(period)
    finally:
        try:
            following.set()
        except Exception:
            pass
        try:
            post("/api/mission/stop")
        except Exception:
            pass
        for c in controllers:
            try:
                c.stop()
            except Exception:
                pass
        for act in reversed(controllers + placed):
            try:
                act.destroy()
            except Exception:
                pass
        print(f"removed the {len(placed)} things placed", flush=True)

    # ---- scorecard ------------------------------------------------------------------
    n = len(samples)
    clear = sum(s["clear"] for s in samples)
    clear_static = sum(s["clear_static"] for s in samples)
    moving = sum(1 for s in samples if s["speed"] > 0.5)
    predicted = sum(1 for s in samples if s["predicted"])
    beh = collections.Counter(s["behavior"] for s in samples)
    print("\nSCORECARD")
    print(f"  samples: {n} over {samples[-1]['t'] if samples else 0:.0f} s")
    print(f"  people/vehicles held inside a STATIC object: {len(mistakes)} (must be 0)")
    for k, v in mistakes.items():
        print(f"    {v}")
    print(f"  static things right beside a person or vehicle (fine, listed to check): {len(beside)}")
    for k, v in list(beside.items())[:12]:
        print(f"    {v}")
    print(f"  objects with no person or vehicle within 3 m, called static: "
          f"{clear_static}/{clear} ({100.0 * clear_static / max(1, clear):.0f}%)")
    print(f"  van moving on {100.0 * moving / max(1, n):.0f}% of samples; behaviours {dict(beh)}")
    print(f"  crossing predictions fired on {predicted} samples")
    why = collections.Counter()
    for s_ in samples:
        why.update(s_["why"])
    print("  why the objects clear of people/vehicles are what they are:")
    for w, k in why.most_common():
        print(f"    {k:6d}  {w}")
    blockers = collections.Counter((s_["planner"].get("reason"), (s_.get("blocker") or {}).get("motion_class"),
                                    tuple((s_.get("blocker") or {}).get("touches") or ()))
                                   for s_ in samples if s_["planner"].get("reason") not in (None, "clear"))
    print("  what stopped the van (planner reason, blocker's label, person/vehicle it touches):")
    for k, v in blockers.most_common(8):
        print(f"    {v:4d}  {k}")
    if a.out:
        Path(a.out).write_text(json.dumps({"samples": samples, "mistakes": list(mistakes.values()),
                                           "beside": list(beside.values())}, indent=1))


if __name__ == "__main__":
    main()
