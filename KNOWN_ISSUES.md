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

* `SignalMap.from_world()` reads every light's stop waypoints, giving stop-line positions
  and the `(road_id, lane_id)` pairs each light governs.
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
the light source with it.

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

Its only influence on driving is `blind_spot_ahead()` → `blind_spot_m` → a **speed cap** in
`behavior.py`. It validates and slows; it does not generate paths. Planning still works off
the route polyline and a corridor test.

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
* **Steering wander.** Full-lock steering for about 7% of one long run. The controller is a
  tuned pure-pursuit with a centreline correction term; a Stanley controller or MPC is the
  longer-term answer.
* **Junction give-way is radius-based.** No lane-level right-of-way, so a vehicle driving
  *away* on the crossing road still counts as a conflict. Creep-on-timeout after 12 s is a
  pragmatic policy that needs review before real roads.
* **Yellow is always a stop.** The distance to the line is now known, so dilemma-zone
  handling is possible, but is not implemented.

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
* **Fault injection works in camera mode** (`camera_covered`, `camera_blanked`,
  `camera_frozen`, `camera_drops`, `lidar_dead_beams`).
* **Traffic lights are read from the camera in camera mode**, using map-provided lamp
  geometry; the simulator's colour is retained only for `ground_truth` mode and for
  side-by-side comparison.
* **Traffic-light lookahead is route-based and lane-matched**, replacing a proximity query
  that first reported a red light 2 m away.
* **Parking-slot occupancy is re-checked on approach**, not only once when the slot is
  chosen.
