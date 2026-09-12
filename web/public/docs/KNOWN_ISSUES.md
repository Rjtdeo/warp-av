# Known Issues and Current Limitations

Warp AV is a **CARLA research prototype**. It is not a production autonomous-driving
system and has never driven a real vehicle.

This file lists what is actually wrong or missing **today**, verified against the code
rather than against older documentation. Anything already solved has moved to
[Recently resolved](#recently-resolved) at the bottom or been deleted.

Priorities:

| | |
|---|---|
| **P0** | Blocks any real-vehicle work. Mostly simulator dependencies with no on-board replacement yet. |
| **P1** | Perception or behaviour robustness. The van drives, but these limit where and how well. |
| **P2** | Capability not yet attempted. |

---

## P0 — Real-vehicle blockers

### Localization is CARLA ground truth

**Current behaviour.** `localization/localization.py` reads `vehicle.get_transform()`.
GNSS and IMU sensors are attached in `carla_sensor_adapter.py` and their readings are
published, but nothing fuses them. There is no EKF, no wheel odometry, no uncertainty
estimate that anything downstream consumes.

**Why it matters.** Every route-relative calculation in the stack — the route corridor,
traffic-light distance-along-route, kerb offsets, parking-slot geometry — assumes the pose
is exact. On a real van, pose error is the dominant error source, and none of that code
has a tolerance for it.

**Planned resolution.** RTK GNSS + IMU + wheel odometry fusion behind the existing
`LocalizationSystem` interface, publishing a covariance the planner actually reads.

**Validation required.** Re-run the mission sweep with injected pose noise before and
after, and show which behaviours degrade.

---

### LiDAR sweep de-skew uses the simulator's pose

**Current behaviour.** `adapters/lidar_sweep.py` glues CARLA's per-frame wedges into one
360° sweep by transforming each wedge into the world frame using **the sensor transform
CARLA reports at capture**. That transform is exact and instantaneous.

**Why it matters.** On real hardware, motion compensation needs a time-aligned pose
estimate per wedge, which is a harder problem and a noisier input. Sweep quality —
and therefore clustering and object geometry — will not transfer unchanged.

**Planned resolution.** Feed de-skew from the localization estimate rather than the
simulator, once that estimate exists.

---

### Traffic-light geometry comes from the simulator

**Current behaviour.** Two separate map dependencies, both from CARLA:

* `SignalMap.from_world()` reads every light's stop waypoints and the `(road_id, lane_id)`
  pairs each light governs, and works out each lane's stop LINE: the earliest of the light's
  OpenDRIVE position, the lane's junction entry and its first zebra
  (`traffic_lights.stop_line_for_lane`). CARLA's stop waypoint itself is the centre of the
  light's trigger box, 1-7.8 m short of the paint on Town10HD, and is not used as the line.
* `LampMap.from_world()` reads `traffic_light.get_light_boxes()` — the **exact 3D position
  and size of each lamp housing**. The camera colour reader projects that box into the
  image and classifies the pixels inside it.

The colour itself is read from camera pixels. The *place to look* is not.

**Why it matters.** This is map-guided classification, not autonomous signal discovery.
A real van needs either an HD map carrying the same lamp priors, or a detector that finds
the signal head in the image itself.

**Planned resolution.** HD-map lamp priors where a map exists; an image-space signal-head
detector where one does not. The van's own YOLOX already reports COCO class 9
(traffic light) under about 40 m, which is a starting point but adds no range over the
projection.

**Validation required.** Re-run the light accuracy measurement with the ROI taken from a
detector instead of the map, and compare on the same lights and distances.

---

### CARLA traffic-light state is still reachable

**Current behaviour.** `carla_state_source()` in `perception/traffic_lights.py` reads the
light's true colour from the simulator. It is used when:

* the stack is in `ground_truth` perception mode; **or**
* `WARP_CAMERA_LIGHTS=0` is set, which exists for side-by-side comparison.

In the default `camera_lidar` mode the colour comes from the camera. The choice is made
per read (`main.py`, `light_colour()`), so switching perception mode at runtime switches
the light source with it. When no light is matched to the route, camera mode asks the MAP
which light governs the lane the van is in and the CAMERA its colour
(`TrafficLightLookahead.on_lane`); it no longer asks the simulator.

**Why it matters.** This path must not be mistaken for the camera path when reading
results. Any measurement should state which source was live.

---

### No drive-by-wire or real-sensor validation

**Current behaviour.** The vehicle interface talks to CARLA. `vehicle_interface.py`
defines the boundary a real DBW gateway would sit behind, and the adapter already rejects
non-finite or stale (>0.5 s) commands, which is what a physical gateway must do. Nothing
beyond that has been exercised against hardware.

**Also simulator-only:** the collision sensor (test instrumentation only), scenario
ground-truth labels used for scoring, and CARLA map topology used for route planning.

---

## P1 — Perception robustness

### One detector is shared across four cameras

**Current behaviour.** A single YOLOX-S instance runs in one background thread
(`perception/detection_worker.py`) at `inference_interval = 0.25 s`, taking the cameras
**in turn**: front → left → right → rear. That is roughly **4 detections per second in
total, so about 1 Hz per camera**, against camera frame rates of 10 Hz (front) and
~7 Hz (sides and rear).

**Why it matters.** Semantic labels on the side and rear views can be up to a second old.
Detections older than `detection_max_age_s = 1.0` are dropped rather than used stale, so
the failure mode is *missing* labels rather than wrong ones — but a fast crossing object
on a side camera may never be labelled at all.

**Planned resolution.** Either a cheaper detector, GPU inference, or a scheduler that
weights the front camera and the direction of travel instead of plain round-robin.

---

### Camera detections cannot create objects on their own

**Current behaviour.** Fusion runs in one direction. LiDAR clusters are formed first, then
each cluster is projected into whichever cameras could see it and takes a class from any
detection box it lands in (`camera_lidar_perception.py`, `_name_with_camera`). A camera
detection with **no** LiDAR cluster behind it produces nothing.

**Why it matters.** Anything the LiDAR misses is invisible regardless of how clearly the
camera sees it — beyond LiDAR range (50 m), below the near-field beam floor, or too thin
to cluster.

**Planned resolution.** Camera-only object hypotheses with a range estimate from box
geometry, held at lower confidence until LiDAR confirms them.

---

### Camera and LiDAR are not tightly time-synchronized

**Current behaviour.** Each sensor callback stamps its own arrival with `time.time()`.
Fusion uses the most recent frame and the most recent sweep, with a freshness cap, not a
matched pair of timestamps.

**Why it matters.** At 8 m/s a 100 ms mismatch is 0.8 m of projection error, which is
enough to put a cluster in the wrong detection box at range.

**Planned resolution.** Hardware-triggered or PTP-synchronized sensor clocks on the real
van; timestamp-matched fusion in software.

---

### Detector classes are generic COCO, not a Warp ontology

**Current behaviour.** YOLOX-S trained on COCO, run through OpenCV DNN at confidence 0.40.
Only a few classes are used: person (0), vehicle (car 2, motorcycle 3, bus 5, truck 7),
and bicycle/motorcycle (1, 3) at a lower 0.15 threshold **solely** to decide whether a
detected person is riding something. A cyclist is inferred, never detected directly —
either a person's box overlapping a bike's by 35%, or a person travelling faster than
4 m/s having covered at least 3 m.

Everything else — cones, bollards, pallets, kerb furniture, roll cages, open vehicle
doors — is an unnamed `obstacle` with measured geometry and no semantics.

**Why it matters.** A delivery van cares about exactly the objects COCO does not have.

**Planned resolution.** A Warp-specific detector. Not started; no training data collected.

---

### Sensor health checks cover the front camera only

**Current behaviour.** `_check_the_picture()` in `carla_sensor_adapter.py` is called from
`_on_camera`, which is the **front** camera. It checks brightness, contrast, frame-to-frame
change, and per-tile stuck patches (rain drops), plus LiDAR beam count and point-count
collapse. The report reaches `SafetySupervisor.update(health=...)` and produces a speed cap
or a stop.

The left, right, rear and top views are stored (`latest_frames`) and used for detection,
but **no health check runs on them**. A blinded side camera silently stops contributing
labels and nothing reports it.

The stuck-patch rule only counts a patch that is frozen solid (`TILE_FROZEN_BELOW`, under
0.10 change between frames). In CARLA a drop on the glass freezes its patch at exactly 0.00
and plain sky never does; a real camera's sensor noise never reads 0.00, so this number has
to be measured again on real hardware before it means anything there.

**Planned resolution.** Run the same picture checks on every perception camera and name
the failing view in the report.

---

### Near-field LiDAR blind region

**Current behaviour.** One roof LiDAR at z = 2.5 m, `lower_fov = -30°`. The lowest beam
reaches the ground about 4.3 m from the van; closer than that, low objects are below the
beam fan.

**Why it matters.** A kerb-height object directly in front of the bumper can be outside
the sensor's view. The occupancy grid's blind-pocket rule caps speed when an unseen
person-sized pocket is near, which mitigates but does not remove this.

**Planned resolution.** Evaluate a lower `lower_fov` (-40/-45°) and, if that is not enough,
a low front camera. Not yet measured.

---

### Occupancy and free space do not plan trajectories

**Current behaviour.** The ego-centric grid (`perception/occupancy.py`, 0.25 m cells) holds
FREE / OCCUPIED / UNKNOWN plus a separate ROAD-surface layer, with temporal memory (free
believed ~0.5 s, blocked longer) and whole-square ego-motion compensation.

Its influence on driving is a **speed cap** (`blind_spot_ahead()` → `blind_spot_m` in
`behavior.py`), the parking-spot check (`planning/parking_check.py`), and a **one-way second
opinion** on things standing in the way: when every square of the ground the van's body is
about to cover has been SEEN empty, a body the corridor check drew there is not believed
(`planner.nothing_is_standing_there`, `OccupancyGrid.strip_ahead`, used in `main.py`). It
validates, slows and unblocks; it does not generate paths, and it can never STOP the van on
its own — occupied or unseen space ahead is not yet a reason to stop, and the decision is
taken outside the planner's counted reasons, so `BLOCKED_OCCUPANCY` and `UNKNOWN_SPACE` are
still never produced. Planning still works off the route polyline and a corridor test.

**This is not "occupancy-based motion planning" and should not be described as such.**

**Planned resolution.** Use the grid as the primary free-space representation for
trajectory generation, which also requires a real planner rather than a corridor test.

---

### Tracking uncertainty is measured but not used

**Current behaviour.** `perception/tracking.py` runs a constant-velocity Kalman filter per
track with a real covariance matrix `P`, persistent IDs, velocity, a moving/stationary
decision, and a median size estimate. Confidence and a `size_uncertain` flag reach the world
model.

Nothing in `planning/` or `behavior/` reads the covariance. Decisions use point estimates.

**Note:** `size_uncertain` was measured on a real pedestrian (88% of frames) and a real car
(100%) as well as on phantom tracks (100%) — it flags almost everything and is **not**
usable as a filter as it stands.

---

### Kerb sensing is geometric, not road understanding

**Current behaviour.** `perception/road_edges.py` fits a left and a right kerb line from
LiDAR points, each with a confidence flag and a `lateral_at(x)` query. Used for parking-spot
placement and to stop kerb fragments being reported as obstacles.

**Not implemented:** lane-marking perception, drivable-area segmentation, junction
geometry, road-type semantics. Roads without a kerb produce no edge at all. Lane keeping
comes from the map route, not from vision.

---

### Prediction is constant velocity

**Current behaviour.** `planning/prediction.py` projects each moving object forward at its
current velocity, 3.5 s ahead in 0.5 s steps, and warns when a projected position lands in
the route corridor at a time the van will be there. Gates: the object must close on the
path at ≥0.5 m/s, start within 12 m of the route, be a plausible road user by size, and not
claim a speed its shape cannot support.

**Not implemented:** intent, multimodality, interaction, map-aware manoeuvre priors.
A vehicle indicating a turn is treated exactly like one going straight.

---

### Compute limits the decision rate

**Current behaviour.** The main loop runs at roughly **9–10 Hz** measured
(`/api/state.loop_hz`). YOLOX on CPU is the largest single cost, and its thread count is
capped deliberately (`WARP_DETECTOR_THREADS`) because letting it take every core slowed
the ground filter — which never touches the camera — by 2×.

**Why it matters.** Control quality and reaction distance both scale with loop rate.
A multi-rate design (fast control, slower perception) is the usual answer and does not
exist yet.

---

## P2 — Not yet attempted

* **Radar.** No radar sensor, no radar fusion. Nothing in the stack expects one.
* **Sensor calibration.** Camera intrinsics come from the simulator's FOV and image size;
  extrinsics are the mount transforms in `camera_model.VIEW_MOUNTS`. No calibration
  procedure, no calibration validation, no allowance for drift.
* **Re-planning around a blockage.** The van stops and waits. There is a blocked-route
  timeout but no alternative route.
* **Geofence / ODD enforcement.** 40 `odd_boundary` scenarios define the contract; nothing
  enforces it.
* **Signs.** Stop, give-way and speed-limit signs are not read.
* **Reverse manoeuvres.** Forward pull-over only. A taken parking slot cannot be recovered
  from by reversing.
* **Multi-fault reporting.** `SafetySupervisor` reports only the first failed check, so a
  second simultaneous fault is invisible (`cf_double_failure`).
* **Recovery policy.** A component coming back auto-resumes motion with no hysteresis
  (`cf_recover`, `cf_flapping`).
* **Wall-clock dependence.** `time.time()` throughout, so an NTP or GNSS clock step trips
  every staleness check at once.

---

## Behaviour gaps that are not perception

These are real and observed; they are listed here so they are not mistaken for sensing
faults.

* **Off-route blindness.** `planner.filter_to_route_corridor` reports 999 m clear when the
  van is more than about 2.2 m from its planned route. Measured: with a barrel at 6 m the
  van detects it at every lateral offset and still reports a clear path. This caused a
  mailbox strike on a long route.
* **The behaviour is a list of rules, and it says which one answered.** Every decision
  carries one reason code from a closed set (`behavior/transitions.py`, 24 of them), the 18
  rules are asked in a written order (`BehaviorSystem.RULES`), and every change of state or
  reason -- plus the moves the van makes, the go-around and the parking choices -- is kept in
  order and published (`behavior_changes`, the mission log, and the operator page). What is
  NOT done: the rules are still a first-match-wins list rather than a state machine with
  allowed transitions, so nothing forbids a change (parking -> junction, say) that should
  never happen; and a rule cannot say "I nearly fired", so what came SECOND is never
  recorded. Traffic lights deliberately stay with the behaviour rather than becoming the
  planner's `blocked_signal` (see planning/instrumentation.py).
* **A pass needs a lane to borrow.** The go-around plans the path round whatever is standing
  in the lane and slides the van's body along it before taking it (`planner.plan_overtake` +
  `pull_in_blocker`, with moving traffic judged by `planner.overtake_blocker`, which also
  looks behind). Since 2026-09-11 that is not only a car: `planner.pass_refused` says a
  person or a rider is waited for and never driven round, something moving is followed, and
  anything the camera has not named is passed only while the camera is working -- the thing
  is dead ahead in its view, so "not called a person" is a judgement, and with a stale or
  missing camera it is not. Measured that day: a 0.45 m barrel passed with 1.97 m to spare, a
  0.65 m box with 2.05 m, a person standing in the lane waited for until the test ended.
  What is still missing is the narrow version: the van always swings a full lane, so on a
  single-lane road, or with the next lane occupied, a barrel still stops it. Squeezing past
  inside the lane on a measured gap is not built.
* **Steering wander.** Full-lock steering for about 7% of one long run. The controller is a
  tuned pure-pursuit with a centreline correction term; a Stanley controller or MPC is the
  longer-term answer.
* **Junction give-way is radius-based.** No lane-level right-of-way, so a vehicle driving
  *away* on the crossing road still counts as a conflict. Creep-on-timeout after 12 s is a
  pragmatic policy that needs review before real roads.
* **Passing parked cars close by depends on a good look at them.** The swept-path check now
  judges a stationary thing by a rectangle fitted to its points (`tracking.fit_rectangle`),
  kept on the map frame, and for a thing standing still the most complete view of it so far.
  Measured on a car passed at 0.51 m: heading 0.5 deg off at the median, 2.7 deg at the 90th
  percentile, 17 deg at worst; when only a car's rear is visible the fit is the bumper (7 of
  268 readings). The near side comes out 0.04-0.19 m further from the van than CARLA's box
  (mirrors) -- covered by the 0.10 m pad. These figures are CARLA's; a real LiDAR must be
  measured again.
* **Parking into a strip depends on how the thing beside it was measured.** The pull-in is
  gentle and ends straight (see Recently resolved). Twice on 2026-09-11 a free strip was
  refused by street furniture beside it: a lamp post about 1 m beyond the strip, and a line
  of kerbside furniture measured 6.3 x 0.84 m whose fitted rectangle (8.0 x 2.3 m, named a
  vehicle by the camera) reached into the way in -- the second failed a mission 3.6 m short
  of the spot. Two causes were fixed the same day (a fitted rectangle may no longer claim
  more ground than the points it came from; the heading allowance is capped by a thing's own
  shape), and the free-space second opinion above now carries the van past a body drawn on
  ground the laser has seen empty: the same run then parked 0.2 m from the spot with no part
  of the van in the driving lane. That is ONE proven run, not a reliability claim. When a
  spot is refused the fallback can still be a stop at the kerb edge of the DRIVING lane,
  which blocks the lane. Along the slot the van can finish near its ends (0.02 m to spare
  front/back once). `ROAD_BOUNDARY` exists as a planner reason but is not produced yet.

---

## Operational and tooling limitations

Real, but not perception or safety faults.

* **Parking is forward-only.** No reverse or parallel manoeuvre, so a slot that has been
  taken cannot be recovered from by backing out. Slot completion requires the whole van
  inside the box, parallel to the lane within 6°.
* **The map cannot see painted kerb markings or mid-block driveways**, so a software-derived
  parking slot can still land on one. Hand-annotated no-parking zones per street are the
  clean fix.
* **`get_next_waypoint` scans the whole route** to find the closest point before searching
  forward — O(n) per tick. Fine at 200 waypoints, noticeable on a 2 km route.
* **The operator console polls** `/api/state` every 250 ms rather than receiving pushes.
  Works, adds latency.
* **CARLA host/port, vehicle type and sensor configuration are partly hardcoded** in the
  adapters rather than fully driven by config.
* **Pedestrian-versus-obstacle tie-breaking at equal range depends on iteration order.**
* **Stopping the stack's scheduled task does not stop the stack.** `schtasks /End` ends
  `start_stack.bat` but not the `python run.py` it started; the next `schtasks /Run` then
  fails quietly while the old stack keeps running old code. Stop the python process itself.
  A stopped or killed stack also leaves its van's sensors in the world -- CARLA does not
  remove a sensor with the thing it rides on (81 sensors, 45 of them cameras still rendering,
  found 2026-09-10). `tools/clear_leftover_vans.py --remove --sensors` clears them
  (`--stack-down` when the stack is stopped on purpose).

### Scenario runner

* Does not switch CARLA towns; a town mismatch is recorded as a warning only.
* Cut-in and cut-out are open-loop steering pulses, not lane-accurate. Weave is ignored.
* CARLA has no animal or pallet props; a bin and a plant pot stand in.
* `noise` and `latency` faults on raw sensors are accepted and logged but inert until
  perception consumes raw sensors directly.
* The ego vehicle is found as "the vehicle nearest the API pose" because the adapter does
  not set a `role_name`.

---

## Testing limitations

* **722 offline tests** run without CARLA (`python -m pytest tests/ -q`) and cover
  perception maths, tracking, occupancy, prediction, planner geometry, behaviour priority,
  safety, and the traffic-light lookahead.
* **28 live checks** (`tools/full_check.py`) place real objects in CARLA and read the answer
  back through the van's own interfaces. These need a running stack and are not in CI.
* The replay harness (`perception/replay.py`) scores recorded fixtures offline, but uses a
  **stand-in detector** — the real camera model does not run there, so camera behaviour is
  not covered by replay.
* The 1000-scenario catalog is validated structurally and by dry-run. It has **not** been
  executed end-to-end.
* Offline fixtures were all recorded on flat roads. Ground-filter behaviour on hills is
  therefore not covered, which is why one kerb fix is deferred pending a hill recording.
* Measurement scripts that spawn their own van and sensors must use
  `tools/scratch_world.py`, which refuses to run while the stack is up. Earlier scripts
  destroyed every `sensor.*` actor as cleanup — including the running van's own camera and
  LiDAR — which silently corrupted live measurements.

---

## Recently resolved

Kept short deliberately; this is not a history file.

* **Perception no longer reads the simulator's actor list by default.** `camera_lidar` is
  the boot default; `ground_truth` remains an automatic fallback so the stack still starts
  without `models/yolox_s.onnx`.
* **Four cameras feed perception**, not one.
* **Camera–LiDAR fusion exists** (geometric projection through a calibrated camera model).
* **Object tracking exists** — persistent IDs, velocity, moving/stationary — so car
  following works in camera mode.
* **Sensor health is wired to the safety supervisor** and produces degraded-speed and stop
  behaviour, including picture-quality faults (dark, frozen, covered, rain-blocked tiles).
* **The van passes a car parked beside its path.** It waited 65 s beside one with 0.96 m
  of real room (WAV-0228), and over 2 minutes behind another. Two measuring faults, checked
  against CARLA's truth: the car's box was drawn round the AVERAGE of its points -- which a
  car seen from its corner piles on the van's side, 0.9 m from the true centre -- and its
  heading came from the spread of those points, which an L of side-and-end bends (9-20 deg
  off), kept in the van's frame from an older sighting. Now the planner judges a fitted
  rectangle, on the map frame, from the best view of a thing standing still; and a parked
  vehicle that only the safety margin is in the way of is passed slowly (PASS_CLEARANCE_M
  0.15 m instead of 0.30 m) rather than waited behind. Live, parked car beside a straight
  lane: 0.81 m gap passed with no stop, 0.51 m passed slowly, 0.15 m waited and then
  overtook (2.8 m). WAV-0228 now parks; a 657 m mission with cars parked 1.9 m from the route
  completed with one 1.5 s hesitation. People, cyclists, unknown things and static posts
  keep the full margin.
* **The van no longer waits for ever at a kerb when pulling in.** 4 of 5 pedestrian
  scenarios ended with it parked in front of a kerb piece until the run timed out. Three
  causes, each fixed and measured: (1) it chose bays one slot long -- its body leaves the
  lane 13.1 m before the slot's centre, over bare kerb -- so a slot now needs
  `APPROACH_BAY_BEHIND_M` (14 m) of free, straight bay behind it, and the kerbside pull-over
  the same; (2) a kerb 0.5 m from the parked van's side met 0.55 m of padding, so a
  kerb-height thing (<= 0.20 m, still, unnamed) now needs tyre clearance only (0.10 m);
  (3) whatever still blocks the way in near the spot and will not move ends the mission there
  after 6 s. After: 4 of 5 completed, 0 contacts; the fifth waited behind a car parked 2.6 m
  from the path (the parked-car item above). (2026-09-11: (3) finished the mission 39 degrees
  across the lane and called it parked -- it now chooses another spot before the pull-in
  starts, and fails the mission, saying why, once already turning in. The bay needed behind a
  slot is now 17 m, for the gentler ramp below.)
* **Plain sky no longer reads as "something on the lens".** A clean front camera was called
  broken for 18-75 % of daytime driving (the sky, and the far end of the road, hardly change
  while driving), and each time the safety supervisor slowed the van to 2 m/s and then
  stopped it -- it failed every pedestrian scenario before the pedestrian was reached. A patch
  now also has to be frozen solid. Measured on 3,512 front-camera frames in 8 drives: clean
  lens 0 %, five drops still caught on 99 % of the driving, one drop 1 % (was 87 %); live, five
  drops were caught in 0.8 s and a clean 15 s drive gave 0 of 55 alarms.
* **Fault injection works in camera mode** (`camera_covered`, `camera_blanked`,
  `camera_frozen`, `camera_drops`, `lidar_dead_beams`).
* **Traffic lights are read from the camera in camera mode**, using map-provided lamp
  geometry; the simulator's colour is retained only for `ground_truth` mode and for
  side-by-side comparison.
* **Traffic-light lookahead is route-based and lane-matched**, replacing a proximity query
  that first reported a red light 2 m away.
* **Parking-slot occupancy is re-checked on approach**, not only once when the slot is
  chosen -- and in camera mode it comes from the van's own LiDAR free-space map
  (`planning/parking_check.py`), not the simulator's list of cars. A spot it has not seen is
  not free: the van confirms the spot and the strip it will sweep before it turns in.
* **Red lights: the van stops at the lane's stop line, with its front bumper.** Measured
  against CARLA on 2026-09-11 it had crossed the stop line on red at light 11: it held at the
  zebra (2.2-3.9 m PAST the painted bar on Town10HD), measured from the middle of the van (the
  bumper ended 0.35 m past even that), and a predicted crosser returned "slowing" before the
  light was looked at. The painted bars were measured from overhead pictures at all 15 lights
  (`tools/measure_stop_bars.py` -> `tools/data/`), and every drive is now scored against the
  paint (`tools/record_mission_truth.py`). Yellow is decided once, as it changes: go on only if
  the van cannot stop before the line at 2.5 m/s2. Three live runs of an 11-light route: 0 red
  runs, the bumper 1.9-2.7 m short of the paint.
* **Parking ends straight, beside the lane it came in on.** The pull-in ramp is capped at 15
  degrees, lies in one lane after a 10 m settle, and the spot moves up to 80 m past the pin --
  across a junction if need be -- rather than squeeze it; it used to be squeezed into the
  road left, and moved the van 6.5 m sideways in 6 m (57 degrees). The distance to the
  destination is measured by road: a route round a block passed 22.7 m from its spot with
  about 250 m still to drive, and the van slowed and checked for it on the wrong street.
  Live, 11-light route: parked inside the slot, 0.4 m from the spot, 2 degrees off, no part in
  the driving lane.
* **A bus shelter is no longer called a vehicle.** A camera "vehicle" name is dropped for a
  blob at least 2.3 m tall, 3 m long and 0.3 m clear of every lane
  (`motion_class.vehicle_name_implausible`); on CARLA's answer key no real vehicle meets that,
  even one step looser. Dropping a name never makes a thing static.
