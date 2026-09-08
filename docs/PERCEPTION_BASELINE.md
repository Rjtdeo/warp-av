# Perception baseline (camera + LiDAR), 2026-09-07

Measurement only. No change to the van's code was made for this document.

## Why

On 2026-09-07 the van hit a barrel placed in its lane (scenario WAV-0294) in
camera + LiDAR mode. It did not see the barrel at 7.5 m and first reported
it at 3.3 m from the van's centre, where the bumper is already touching.
The question: at what distance does the pipeline see small road objects,
and at which stage are they lost?

## How it was measured

`tools/perception_probe.py`, run on the CARLA machine with the stack idle:

* Town03, van parked, camera + LiDAR mode. The probe attaches a second
  camera and LiDAR to the van, identical to the stack's own (800x600 fov 90
  at x=2.0 z=1.8; 32 channels, 150k points/s, 50 m, 10 Hz, sensor_tick 0.1,
  at z=2.5), and runs the SAME `CameraLidarPerception` class on them.
* One object at a time on the lane centre ahead, at 20, 15, 12, 10, 8, 6,
  4 m. 20 frames per distance. The live stack's own object list was read
  over `/api/state` at the same time (cross-check).
* Stages recorded: raw LiDAR points on the object; points left after the
  height cut; points left after the 3x downsample; cluster found; camera
  box on the object; class after fusion; tracked object reported; what the
  live stack reported.

Raw data: `docs/perception_baseline/probe_2026-09-07_town03_static.csv` and
`probe_2026-09-07_console.txt`.

## Results

Fraction of frames in which the object was reported as a tracked object by
the pipeline (probe) / by the live stack. `-` = never.

| Object  | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m | 4 m |
|---------|------|------|------|------|-----|-----|-----|
| barrel  | - / - | - / - | - / - | - / - | 0.80 / - | 0.85 / 0.50 | 0.80 / 0.55 |
| cone    | - / - | 0.80 / - | - / - | 0.25 / - | 0.80 / 0.25 | 0.90 / 1.00 | 0.85 / 0.70 |
| planter | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| car     | 0.90 / 0.70 | 0.75 / 1.00 | 0.95 / 0.65 | 0.85 / 1.00 | 0.80 / 0.60 | 0.90 / 1.00 | 0.90 / 0.70 |
| person  | - / - | - / - | - / - | 0.70 / 0.10 | 0.90 / 1.00 | 0.85 / 0.50 | 0.80 / 1.00 |

Where the points go (barrel, mean per frame):

| Stage                         | 20 m | 15 m | 12 m | 10 m | 8 m  | 6 m  | 4 m  |
|-------------------------------|------|------|------|------|------|------|------|
| raw LiDAR points on it        | 1.1  | 5.2  | 4.2  | 7.2  | 15.7 | 20.6 | 42.5 |
| after the height cut          | 0.6  | 1.8  | 1.9  | 1.9  | 5.0  | 6.7  | 15.3 |
| after keeping every 3rd point | 0.2  | 0.8  | 0.6  | 0.7  | 1.5  | 2.4  | 5.2  |
| cluster found (needs >= 3)    | 0    | 0    | 0    | 0    | 0.25 | 0.50 | 0.80 |

Planter: 7 to 245 raw points at every distance, 0 after the height cut.
Never seen, at any distance, by the probe or by the live stack.

Type given to the object: barrel, cone, planter: `obstacle` (correct enough).
Car: `obstacle` at 20 m, `vehicle` from 15 m, by the shape rule (blob wider
than 0.9 m with 6+ points), not by the camera. Person: always `obstacle`,
never `pedestrian`. The camera saw "car" only at 6 to 10 m and "person" only
at 6 m. Nothing for barrel, cone, barrier (once "airplane" for the barrier).

## What one LiDAR scan really contains

The pipeline works on the latest scan only. Measured on the van:

| LiDAR setting            | deliveries/s | points per delivery | 10-degree sectors covered (of 36) |
|--------------------------|--------------|---------------------|-----------------------------------|
| sensor_tick 0.1 (stack)  | 9.7          | 400 to 5,500 (mean 2,200 to 4,000) | 4 to 29, varies every scan |
| sensor_tick 0.0          | 49.3         | 350 to 4,900        | 3 to 13                           |

A full sweep would be 15,000 points and all 36 sectors. CARLA runs
asynchronously at about 58 frames/s here, so each delivery is a partial
sweep pointing in a different direction. In 3 of 10 scans not one point was
ahead of the van.

## Timing

* YOLOX on CPU: 136 to 180 ms per inference, run inside the control loop
  every 0.25 s.
* Control loop: designed for 10 Hz, measured 3.7 Hz (14,464 ticks in
  4,135 s of uptime).
* One perception update in the probe: 60 to 105 ms.

## Findings, ranked by effect

1. **Partial LiDAR sweeps.** The van never gets a full 360-degree scan;
   each scan is a wedge in a changing direction. Few points, and objects
   blink. Sensor set-up, not perception code.
2. **The height cut removes low objects.** Points below about 0.35 m above
   the road are dropped (`minimum_lidar_z = -2.15` with the LiDAR at 2.5 m).
   A planter never survives. Kerb-height objects in general do not.
3. **Small objects cannot reach 3 points.** After the cut and the 3x
   downsample (`xy = pts[mask][::3]`), a barrel has under 1 point at 10 m.
   `cluster_points(min_points=3)` needs 3. Result: barrels and cones appear
   at 6 to 8 m, part of the time.
4. **Fusion bug: camera labels almost never attach.** `CameraDetection.box`
   is `(x_left, y_top, width, height)`, but the fusion step in
   `CameraLidarPerception.update()` reads `bx1, bx2 = det.box[0], det.box[2]`
   and tests `bx1 - 12 <= u <= bx2 + 12`, i.e. it uses the WIDTH as the
   right edge. A label attaches only by coincidence. This is why a person
   is never a `pedestrian` in camera + LiDAR mode. One-line fix: `bx2 =
   det.box[0] + det.box[2]`.
5. **Camera model is weak for this job.** COCO classes only (no cone,
   barrel, barrier, planter); with the 0.40 threshold it saw a car only
   inside 10 m and a person only at 6 m.
6. **Loop rate.** Inline YOLOX makes the whole van run at 3.7 Hz. At 2 m/s
   that is 0.5 m of travel per decision.

## Targets before moving on

* barrel, cone, barrier, planter on the lane centre: reported by 15 m in
  at least 90% of frames, no gaps longer than 0.3 s, from a moving van.
* person: reported as `pedestrian` by 12 m.
* control loop back at 10 Hz in camera + LiDAR mode.
* WAV-0294 (barrel in lane) passes in camera + LiDAR mode.

## Fix candidates, in the order to try (none done yet)

1. Full sweeps: accumulate LiDAR deliveries into one 360-degree scan per
   0.1 s (or run CARLA synchronous with a fixed 0.1 s step).
2. Height cut down to about 0.10 to 0.15 m with a real ground filter.
3. No 3x downsample; minimum points scaled with range.
4. Fusion: right edge = x + width.
5. YOLOX in a background thread; the loop uses the latest result.
6. Later: a detector that knows road props, or LiDAR-shape classes for them.

Re-measure with the same probe after each step:

    venv\Scripts\python.exe tools\perception_probe.py --frames 20 --distances 20,15,12,10,8,6,4 --objects barrel,cone,planter,car,person


---

# Fix 1 result: whole LiDAR sweeps (2026-09-07 evening, commit d51661d)

What changed: `src/warp_av/adapters/lidar_sweep.py` glues CARLA's per-frame
LiDAR wedges into one 360-degree sweep per rotation (de-skewed with the
sensor pose, seam de-duplicated, the van's own body returns dropped first).
The adapter now asks CARLA for every frame (sensor_tick 0.0) in sweep mode.
Flag `WARP_LIDAR_SWEEP` (default on; `0` = old behaviour). Nothing in
perception itself changed.

Checks before deployment: 17 unit tests; a four-lens adversarial review
(geometry, timing/threading, downstream consumers, tests) whose two
actionable findings, self-returns smeared behind a moving van and seam
double-counting, were fixed before the code left the Mac; on the CARLA
machine a barrel placed 10 m ahead and 2 m right of the parked van landed
0.16 m from its true world position through the accumulator's matrices
(frame convention confirmed). Cost: about 1 ms per delivery.

## What one scan contains now

| | points per scan | 10-degree sectors covered (of 36) |
|---|---|---|
| before (one wedge per 0.1 s) | 3,000 to 5,700 | 15 to 28 |
| after (accumulated sweep)    | 7,100 to 7,600 | 36, every scan |

## Probe A/B, same pipeline, wedges vs sweeps

Fraction of the 2 s window in which the object was reported (probe pipeline,
`docs/perception_baseline/probe_2026-09-07_fix1_sweep_{off,on}.csv`):

| Object | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m | 4 m |
|---|---|---|---|---|---|---|---|
| car, wedges | 0.90 | 0.80 | 0.85 | 0.80 | 0.80 | 0.95 | n/a |
| car, sweeps | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | n/a |
| person, wedges | - | - | 0.65 | 0.55 | 0.80 | 0.85 | 0.90 |
| person, sweeps | - | 0.75 | 0.90 | 0.90 | 0.95 | 0.95 | 0.95 |
| cone, wedges | - | - | - | 0.60 | 0.75 | 0.85 | 0.85 |
| cone, sweeps | - | - | - | 0.35 | 0.85 | 0.95 | 0.95 |
| barrel, wedges | - | - | - | - | 0.85 | 0.90 | 0.95 |
| barrel, sweeps | - | - | - | - | - | 0.80 | 0.95 |
| planter, either | - | - | - | - | - | - | - |

Cluster found per frame (the LiDAR stage on its own, before the tracker's
memory): car at 20 m 0.35 -> 0.95, at 15 m 0.50 -> 1.00; person at 10 m
0.25 -> 0.80, at 8 m 0.40 -> 1.00; cone at 6 m 0.40 -> 0.95, at 4 m
0.60 -> 1.00; barrel at 6 m 0.65 -> 0.75, at 4 m 0.80 -> 0.95. The gain is
steadiness: objects that have enough points are now seen in every frame
instead of blinking. Objects that do not have enough points (small props
beyond about 8 m, planters at any distance) are unchanged: that is the
height cut and the 3x downsample, fixes 2 and 3.

With the probe's camera check corrected (box right edge = x + width), the
camera turns out to see far more than the fusion uses: a person 100% of
frames at 4 to 20 m, a car 100% at 6 to 20 m, cone and barrel at 12 to 20 m
as "fire hydrant", the planter at 4 to 6 m as "bench". None of it reaches
the object list because of the fusion bug (finding 4).

## Drive test: WAV-0294, barrel in the lane, camera + LiDAR

| | before fix 1 (15:20) | after fix 1 (17:27) |
|---|---|---|
| contacts with the barrel | 892 | 0 |
| closest approach (centre to centre) | 3.13 m (bumper on the barrel) | 5.39 m |
| first report of the barrel | 3.3 m, blinking | 5.3 m, steady for 45 s |
| runner verdict | FAIL (collision) | FAIL (stopped 6.4 s after the trigger, limit 6.0) |

The safety criterion (no contact) now passes. The remaining FAIL is the
runner's stop-time rule, 0.4 s over, with the van cruising at 2 to 3 m/s and
the barrel first seen at 5.3 m: earlier detection (fixes 2 and 3) is what
moves that number.

## Watch items carried forward

* Facades and hedges clipped by the 55 m range circle can now become stable
  tracks carrying the van's own speed and the `vehicle` shape label (a
  reviewer's unverified finding). Check the object list while driving; a
  "structure" label for very long clusters is the likely answer.
* The control loop still runs at about 3.5 Hz (YOLOX inline, finding 6).


---

# Fix 2 result: road removal by local patches (2026-09-07 night, commits 6426a3c + 0dd6aa2)

What changed: `src/warp_av/perception/ground_filter.py` replaces the flat
35 cm line with local patches: 1.5 m tiles, the lowest believable point of a
tile is the road there, car-covered tiles borrow their neighbours' typical
(median) height, empty tiles take the corrected neighbours and then a road
plane fitted through all believable tiles, never more than 3 m down; every
point is measured against the highest ground of its four nearest tiles;
keep above 12 cm, below 4 m. Far out (sparse rings) a tile with same-height
mates across the road is a road ring, otherwise its lowest ring is a car
bottom and the plane is borrowed.

Kerbs became visible, so a road-edge pass runs BEFORE the main clustering:
long, low blobs beside the lane lose their off-lane points (a kerb strip can
no longer glue itself to a lamp post and drag its centroid into the lane's
wide-body band, a reviewer's finding); a long, low thing in the lane stays
an obstacle. Clusters now carry `height` and `length`; the parker's feelers
ignore blobs under 0.30 m (the brain never saw kerbs). Switch:
`WARP_GROUND_FILTER=flat` restores the old line.

Checks before deployment: 19 unit tests (hills to 12 %, dips, embankment,
bridge deck, car-covered tiles, ditch inside a tile, far rings, blind circle,
kerb strip + lamp post, NaN input); two independent reviews (maths/geometry
and downstream behaviour); their findings fixed and turned into tests.

## Scored against CARLA's labelled LiDAR (whole sweeps, 90 of them)

`tools/ground_filter_score.py`, van parked in Town03, five objects placed
6-22 m ahead; raw output in `docs/perception_baseline/ground_filter_score_2026-09-07.txt`.

| | old flat line | patches |
|---|---|---|
| road points deleted (want > 99 %) | 99.9 % | 99.9 % |
| object points kept, all (want > 95 %) | 73.6 % | 84.6 % |
| object points kept, 0-10 m | 80.2 % | 92.7 % |
| object points kept, 10-20 m | 60.5 % | 77.1 % |
| object points kept, 20-35 m | 94.4 % | 94.4 % |
| pedestrian points kept | 89.2 % | 94.8 % |
| props (static) kept | 51.0 % | 65.2 % |
| cost per sweep on the CARLA machine | 0.02 ms | 20-25 ms |

Beyond 35 m the two rules differ by about two points per sweep (a whole
scan holds only ~8 object points that far out): noise, not a regression.
Sidewalk surfaces: 0 % kept by both, as wanted.

## Probe A/B, sweeps on, flat vs patches

Fraction of frames the LiDAR stage found the object (cluster), then the
tracked object; `docs/perception_baseline/probe_2026-09-07_fix2_{flat,patches}.csv`.

| Object | 15 m | 12 m | 10 m | 8 m | 6 m | 4 m |
|---|---|---|---|---|---|---|
| barrel, flat | 0 | 0 | 0.10 | 0.35 | 1.00 | 1.00 |
| barrel, patches | 0 | 0 | 0.40 | 0.60 | 1.00 | 1.00 |
| cone, flat | 0.20 | 0.20 | 0.65 | 1.00 | 1.00 | 1.00 |
| cone, patches | 0 | 0.30 | 0.90 | 1.00 | 1.00 | 1.00 |
| planter, flat | - | - | - | - | - | - |
| planter, patches | - | - | 0.45 | 1.00 | 0.95 | 0.90 |
| person, flat | 0.35 | 0.20 | 0.80 | 1.00 | 1.00 | 1.00 |
| person, patches | 0.20 | 0.40 | 0.75 | 1.00 | 1.00 | 1.00 |
| car, either | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |

Tracked (with the tracker's memory): barrel at 10 m 0.45 -> 0.80; cone at
12 m 0.70 -> 0.90; planter never -> 0.70 at 10 m, 0.95 at 8 m. The live
stack, running patches, reported the planter in every frame from 10 m in.

Points that survive the road cut, barrel: 15 m 2.1 -> 2.5, 12 m 2.6 -> 4.8,
10 m 5.2 -> 7.5, 8 m 6.7 -> 9.6. After the 3x downsample a barrel at 12-15 m
still has under 3 points: that is fix 3.

## Drive test: WAV-0294, barrel in the lane, camera + LiDAR

| | before | fix 1 | fix 1 + 2 |
|---|---|---|---|
| contacts | 892 | 0 | 0 |
| closest approach (centre to centre) | 3.13 m | 5.39 m | 6.02 m |
| first report of the barrel | 3.3 m | 5.3 m | 5.9 m |
| runner verdict | FAIL (collision) | FAIL (stop 6.4 s after trigger, limit 6.0) | FAIL (6.5 s) |

No contact, a longer margin, and the runner's stop-timing rule still 0.5 s
over: the van meets the barrel 2 s after it appears and detects it at 6-8 m
while moving (10 m parked). Earlier detection is fix 3.

## Things learned, carried forward

* The planter is reported as `vehicle`: the shape rule (blob wider than
  0.9 m with 6+ points) knows no height. Fix 4 should add height to it.
* The filter costs 20-25 ms per sweep on the CARLA machine (2-5 ms on the
  Mac). Fine at today's 3.7 Hz; worth a look when the loop reaches 10 Hz.
* Far field (> 35 m): a lone far object's lowest ring can still pass for
  ground when nothing road-like is near; ~2 points per sweep.


---

# Fix 3 result: every point kept, far blobs with 2 points (2026-09-07 night, commit f07af93)

What changed: the pipeline no longer keeps one LiDAR point in three
(`WARP_LIDAR_THIN=3` restores it). Beyond 15 m a blob may count with 2
points (flagged weak); the tracker needs three weak sightings in a row (or
one strong and two weak) before it reports it, and forgets before it
associates so "in a row" holds. Review fixes folded in: the cluster cap
keeps blobs ahead in the van's corridor first (cap 120, was 60 nearest), the
"car-sized blob" rule wants 12 raw points and 0.5 m of height, low 2-point
blobs beside the lane are dropped as kerb crumbs, and the road-edge rule
requires the blob to run along the road, so a 5 m planter trough lying
across a bend is no longer mistaken for a kerb.

## Probe, standard set, old thinning vs every point (van on a bend in Town03)

Fraction of frames the LiDAR stage found the object (cluster);
`docs/perception_baseline/probe_2026-09-07_fix3_{before,after}_std.csv`.

| Object | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m |
|---|---|---|---|---|---|---|
| barrel | 0.10 -> 0.45 | 0.30 -> 0.80 | 0.35 -> 0.95 | 0.50 -> 0.90 | 0.70 -> 1.00 | 1.00 |
| cone | 0.35 -> 0.80 | 0.40 -> 0.95 | 0.50 -> 1.00 | 0.85 -> 1.00 | 1.00 | 1.00 |
| planter (5 m trough) | - | - | 0.20 -> 0.95 | 0.45 -> 1.00 | 0.40 -> 0.95 | 0 -> 0.95 |
| person | 0.20 -> 0.90 | 0.60 -> 1.00 | 0.65 -> 1.00 | 0.90 -> 1.00 | 1.00 | 1.00 |
| car | 0.95 -> 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| barrier (after only) | | 0.50 | 0.70 | 1.00 | 1.00 | 1.00 |

Points on a barrel that reach the clusterer: 15 m 1.3 -> 3.1, 12 m 2.1 -> 6.1,
10 m 2.3 -> 6.9. The planter's jump at 6-8 m is the along-the-road rule: on
this bend the trough sat beside the van's axis and the old rule called it a
kerb.

## Probe, "random things", old thinning vs every point

Eight props that are not cones: `probe_2026-09-07_fix3_random_{before,after}.csv`.

| Object | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m |
|---|---|---|---|---|---|---|
| trash bag | 0 -> 0.07 | 0 -> 0.53 | 0 -> 0.47 | 0.07 -> 0.60 | 0.07 -> 1.00 | 0.47 -> 1.00 |
| shopping trolley | 0 -> 0.33 | 0.13 -> 0.47 | 0 -> 0.33 | 0.13 -> 0.73 | 0.33 -> 1.00 | 0.80 -> 1.00 |
| bin | 0.20 -> 0.60 | 0.67 -> 0.87 | 0.60 -> 1.00 | 0.53 -> 1.00 | 1.00 | 1.00 |
| bench | 0.13 -> 0.80 | 0.07 -> 0.67 | 0 -> 0.80 | 0.27 -> 0.93 | 0.67 -> 1.00 | 1.00 |
| dirt pile | - | - | 0 -> 0.33 | 0 -> 0.53 | 0 -> 0.60 | 0.60 -> 1.00 |
| chain barrier | - | 0.20 -> 0.40 | 0.13 -> 0.27 | 0 -> 0.87 | 0.27 -> 1.00 | 0.27 -> 1.00 |
| iron plank (5 cm) | - | - | - | - | - | - |
| suitcase | 0 -> 0.07 | 0.20 -> 0.67 | 0.07 -> 0.87 | 0.13 -> 0.87 | 0.47 -> 1.00 | 0.87 -> 1.00 |

The plank is 5 cm high: below the 12 cm floor by design (the road's own
bumps live there). Everything else is now seen by 10-12 m, most by 15 m.

## Live stack after deployment

`/api/state` perception_runtime: thinning step 1, 26 clusters (cap 120 not
reached), ground filter 18 ms, YOLOX 95-170 ms; loop rate 5.1 Hz in that
sample (was 3.7).

## Drive test: WAV-0294, barrel in the lane, camera + LiDAR

| | before | fix 1 | fix 1+2 | fix 1+2+3 |
|---|---|---|---|---|
| contacts | 892 | 0 | 0 | 0 |
| first report of the barrel | 3.3 m | 5.3 m | 5.9 m | 15.0 m |
| closest approach (centre to centre) | 3.13 m | 5.39 m | 6.02 m | 7.57 m |
| runner verdict | FAIL (collision) | FAIL (stop 6.4 s after trigger) | FAIL (6.5 s) | FAIL (8.5 s) |

The van now sees the barrel at 15 m, slows to 2 m/s there, and stops
7.4 m short. The runner's "stopped within 6 s of the trigger" rule fails for
the opposite reason to before: a van that slows early takes longer to come
to rest. The safety criteria (no contact, distance, stopped_obstacle) pass.
That timing rule belongs to the scenario catalog, not to perception; worth
revisiting when the catalog is next touched.

## Carried forward

* Naming: a bench, a chain barrier or a cone close by can be called
  `vehicle` by the shape rule; the camera's names never attach (fix 4).
* Weak 2-point tracks 15-20 m ahead in the lane would make the van slow to
  2-3 m/s if they were phantoms; on flat roads the road filter leaves no
  such points (score: 0 of 153k at 10-35 m), on grades this is unmeasured.
* The loop is still bound by YOLOX in the main thread (fix 5).


---

# Perception V2, day 1 (fix 5): detector off the loop, loop paced (2026-09-07 late, commit 0e1f902)

What changed: YOLOX runs in its own thread (`perception/detection_worker.py`)
on the newest camera frame, at most every 0.25 s; the loop reads the newest
finished result and treats one older than 1 s as none. `run()` sleeps only
the remainder of the 100 ms period. Review fixes folded in: camera frames
are now copies (they were views into CARLA's reusable buffer, read from a
second thread); the worker stops on a switch to ground truth and at
shutdown; three consecutive detector errors or a stalled first result make
perception unhealthy (an inline error did the same before); the tracker
runs only when the LiDAR sweep has advanced (`LidarScan.sim_time`), so a
10 Hz loop cannot observe one rotation twice. Switch: `WARP_YOLO_INLINE=1`.

## The number that had to move

| | before | after |
|---|---|---|
| decisions per second (30 s average, idle in camera mode) | 3.56 | 9.16 (EMA 9.5) |
| work per tick | not measured | 35 ms |
| YOLOX per inference | 120-150 ms | 104-108 ms, in the background |
| detector result age seen by the loop | up to 250 ms + inference | 60-90 ms |

## The numbers that had to stay

Probe, standard set, straight road (`docs/perception_baseline/probe_2026-09-07_day1_std.csv`),
cluster fraction / live-stack fraction:

| Object | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m |
|---|---|---|---|---|---|---|
| barrel | 0.15 / 1.00 | 0.50 / 1.00 | 0.85 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| cone | 0.75 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| planter | 0 / 0 | 0.45 / 1.00 | 0.85 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| car | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| person | 0.95 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

The live stack, now sampling its tracker at 9 Hz, reports every object in
every frame from 15 m in, and the barrel from 20 m: the tracker's memory
(1.2 s) now spans ~11 ticks instead of 4, so a blob missing from one sweep
no longer drops out of the object list.

## Drive test: WAV-0294, barrel in the lane

| | fix 1+2+3 (3.7 Hz) | day 1 (9 Hz) |
|---|---|---|
| contacts | 0 | 0 |
| first report of the barrel | 15.0 m | 16.4 m |
| closest approach | 7.57 m | 7.71 m |
| runner verdict | FAIL (stop-timing rule, 8.5 s) | FAIL (same rule, 8.55 s) |

## Carried forward

* The controller's per-tick constants (steering low-pass, slew limits) now
  act at their design rate, 10 Hz, instead of 3.7-5 Hz: the van pulls away
  and steers more briskly. The parking brain and the sensor tests must be
  re-run before their old results are trusted (planned).
* Camera boxes still lag the sweep they are matched to by up to ~0.5 s
  (was ~0.37 s); harmless at 4 m/s, and second-order to the match bug (fix 4).


---

# Perception V2, day 2: ring id, age and sensor pose on every LiDAR point; first replay fixture (2026-09-08, commit 4c68436)

A foundation day: no decision changes. Every LiDAR point now carries its
beam (ring) id, recovered from CARLA's per-channel counts before the sweep
gluing (`decode_lidar` / `ring_ids` in the sensor adapter), and its age
inside the sweep (`t_rel`, appended by the accumulator). `LidarScan` names
its six columns (x, y, z, intensity, ring, t_rel), keeps the sensor pose of
the newest delivery, and refuses a wrong column count. Every consumer reads
the first three columns only (verified file by file). The probe now mounts
its camera like the stack (pitch -10) and builds identical scans.

Live check of the ring recovery: 32 rings, one elevation each, 10.0 down to
-30.0 degrees, spread under 0.02 degrees within a ring.

## Numbers that had to stay (fresh boot, day-1 code vs day-2 code)

| | before | after |
|---|---|---|
| decisions per second | 9.3 | 9.2 |
| work per tick | 29 ms | 40 ms (the probe was running alongside for part of the window) |
| points per scan | 7,274 | 7,240 |
| sweep errors | 0 | 0 |

Probe, cluster fraction, `docs/perception_baseline/probe_2026-09-08_day2_{before,after}.csv`:

| Object | 20 m | 15 m | 12 m | 10 m | 8 m | 6 m |
|---|---|---|---|---|---|---|
| barrel | 0.20 -> 0.35 | 0.40 -> 0.55 | 0.95 -> 1.00 | 1.00 | 1.00 | 1.00 |
| cone | 0.60 -> 0.80 | 0.75 -> 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| planter | 0 | 0.55 -> 0.45 | 0.75 -> 0.75 | 1.00 | 1.00 | 1.00 |
| car | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| person | 0.95 -> 0.90 | 0.95 -> 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Within run-to-run noise; the far barrel/cone rows moved up a little.

## The first replay fixture

`tests/fixtures/perception/town03_straight_a/` (3.8 MB), recorded with
`tools/record_fixture.py` from the parked van with a barrel (9 m), planter
(13 m), cone (16 m), car (22 m), pedestrian (6 m) and bin (12 m) placed:

* `deliveries.npz`: 50 deliveries of a plain LiDAR configured exactly like
  the van's own, with CARLA's drop-off: 95,921 points, about 7,360 per
  rotation (the live stack reports 7,240), columns x, y, z, intensity, ring,
  plus each delivery's simulation time and sensor pose;
* `labels.npz`: the same seconds from the labelled LiDAR (no drop-off, about
  twice as dense): the answer key, never fed to the pipeline;
* 12 front-camera JPEGs with timestamps, `objects.json` with the true pose
  and box of every placed object, `meta.json`.

`tests/test_fixture_town03.py` glues the deliveries with the real
accumulator (36 of 36 sectors), runs the real ground filter and clusterer,
and requires a cluster within reach of every placed object. It passes on
the Mac with no CARLA and no camera model. Day 3 turns this into the scored
harness.

# Perception V2, day 3: the scored replay harness (2026-09-08)

Day 2 gave us recordings. Day 3 turns them into a scorecard that runs on any
machine in five seconds, with no CARLA and no camera model:

```
python3 tools/replay_score.py
```

`src/warp_av/perception/replay.py` replays a fixture's LiDAR deliveries
through the real sweep accumulator and the real `CameraLidarPerception.update()`
and compares every report with the answer key. It feeds the pipeline exactly
as the van does: hand over the accumulator's sweep after every delivery and
let the pipeline's own 0.08 s rule decide when to look again. Two things are
deliberately different: a stand-in camera model is injected through the new
`detector=` argument (the van still builds YOLOX), and the tracker is given
the sweep's simulation time instead of the wall clock, so a replay is
repeatable and its drop-outs happen at the van's pace.

What it measures:

* **recall** per placed object: share of updates (after the tracker's
  warm-up) in which something was reported within reach of it
  (reach = max(1.5 m, the object's half-diagonal + 0.5 m));
* **cluster recall**: the same one stage earlier, before the tracker;
* **lidar pts**: how many points the dense labelled LiDAR got back from the
  object. Under 3 means the sensor could not see it at all: the row is
  printed but not scored;
* **position error**: median distance from the report to the object's
  origin. The pipeline reports the visible part of a thing, so a 5 m planter
  or a car legitimately shows 1–2 m;
* every **other report**, classified with the labelled scan: *solid* (at
  least 3 non-ground labelled points within 1.5 m: walls, poles, parked
  cars), *kerb* (only sidewalk points), or *phantom* (nothing but
  road/terrain/emptiness: a ground leftover or a far ghost). Phantoms are
  counted overall, within 20 m, and in the lane;
* **ground removal** on the labelled sweep, as `tools/ground_filter_score.py`;
* sweep **coverage** (sectors of the 360° each fed sweep held) and **ms per
  update**.

Each placed object is credited with at most one report, and each report with
at most one object, smallest object first. Without that rule the 5 m
planter's wide reach was credited with the bin's report standing 2.4 m away,
and the scorecard read 1.00 for an object the pipeline never saw.

`tools/replay_score.py --set-baseline` writes `expected.json` next to each
fixture (today's numbers minus a margin: recall −0.10, phantoms per update
+1.0, road deleted −0.02, object kept −0.04). `tests/test_replay_harness.py`
fails when a change drops below them. Other switches: `--ground flat`,
`--thin 3`, `--fixture NAME`, `--json out.json`.

## Four fixtures

Recorded with `tools/record_fixture.py --at x,y` (new: moves the parked van
there, facing along the lane, and refuses a spot that is not on a road), 2.5
seconds each, the same six objects placed along the lane: walker 6 m / +1.3,
barrel 9 m, bin 12 m / −1.6, planter 13 m / +0.6, cone 16 m / −0.5, car 22 m.

| fixture | where | deliveries | points per rotation |
|---|---|---|---|
| `town03_straight_a` | straight road, (−6.5, −79.1) | 63 | 7,373 |
| `town03_bend_a` | a 49° bend, road 23, (−122.0, 133.2) | 47 | 6,843 |
| `town03_junction_a` | straight, junction 15–20 m ahead, (124.7, −194.4) | 89 | 7,230 |
| `town03_in_junction_a` | standing inside the central junction, (−17.6, 9.9) | 68 | 7,114 |

Each is about 3.3 MB: 2.5 s of plain LiDAR (the pipeline's input, with
CARLA's 45 % drop-off), 0.4 s of labelled LiDAR (the answer key, one
rotation is all the scorer uses), ten camera frames, the true pose and box
of every object, and the van's pose.

Getting these right took three attempts, and the scorecard caught both
mistakes: the first pair had the objects placed along the *opposite* lane
(the recorder took a compass heading, not the van's own lane); the second
pair was too short to score and had the pedestrian floating 1 m above the
road (the "lift props that sink into the road" rule also lifted a walker,
whose origin is its waist, not its feet). The recorder now places along the
van's own lane, prints each object's true "ahead / right of the nose"
position, leaves walkers alone, and warns when the van stands in a junction.

## The scorecard today (`docs/perception_baseline/day3_replay_scorecard.txt`)

Recall per object, then cluster recall in brackets where it differs:

| object | straight | bend | junction approach | inside junction |
|---|---|---|---|---|
| walker 5.5–6.1 m | 1.00 | 1.00 | 1.00 | 1.00 |
| barrel 8.2–9.1 m | 1.00 | 1.00 | 1.00 | 0.91 |
| bin 10.5–12.3 m | 1.00 | 1.00 | 1.00 | 0.91 |
| **planter 11.7–13.0 m** | **0.90** (0.29) | **0.52** (0.05) | **0.45** (0.09) | **0.09** (0.04) |
| cone 14.1–16.1 m | 0.95 | 1.00 (0.86) | 1.00 | 0.91 (0.39) |
| car 19.2–22.2 m | 1.00 | 1.00 | 1.00 | 1.00 |
| updates scored | 24 | 24 | 25 | 26 |
| phantoms per update (within 20 m) | 0.6 (0.0) | 2.0 (0.0) | 3.0 (0.0) | 10.5 (2.4) |
| phantoms in the lane | 0 | 0 | 0 | 0 |
| road deleted / object points kept | 99.9 % / 88 % | 100 % / 83 % | 100 % / 78 % | 99.7 % / 90 % |
| sweep coverage | 35/36 | 35/36 | 35/36 | 35/36 |
| ms per update (Mac) | 9.5 | 9.2 | 10.1 | 10.3 |

Position error: 0.10–0.24 m for the small things; 1.2–2.1 m for the car
(reported at its near face) and for the planter when it is seen at all.

## Live cross-check

Straight after recording, the probe placed the same objects at the same
distances from the junction-approach spot with the live stack
(`logs/perception_probe_20260908_115020.csv`):

| object, distance | live van (tracked) | offline scorecard |
|---|---|---|
| person 6 m | 0.97 | 1.00 |
| barrel 9 m | 0.97 | 1.00 |
| cone 16 m | 0.93 | 1.00 |
| car 22 m | 0.97 | 1.00 |
| planter 13 m | 0.97 (cluster 0.47) | 0.45 (cluster 0.09) |
| planter 16 m | 0.00 (cluster 0.00) | – |
| planter 22 m | 0.00 (cluster 0.00) | – |

Five of six objects agree within a few percent. The planter is weak in both,
and both agree it is the one object the pipeline cannot rely on.

## What the scorecard found on its first day

1. **The planter is nearly invisible, and this is a real hole.** It is a
   12.5 cm high trough; the ground filter keeps points more than 12 cm above
   the road. In the answer key its points sit 9–17 cm above the road, so
   most of them are cut as ground and 2–4 survive per rotation. Inside the
   junction it is seen in 1 update of 11; on a bend, half the time; on a
   straight road, 9 times in 10 only because the tracker holds it between
   sightings (the clusterer finds it in 3 of 10). The live probe says the
   same thing more bluntly: at 16 m and 22 m the live van finds no cluster
   at all. Anything flatter than a kerb is currently below the van's notice.
2. **Ground leftovers inside junctions.** Standing in the central junction,
   the pipeline reports 10.5 "obstacles" per update that sit on road or
   terrain points only, 2.4 of them within 20 m, where the road meets grass.
   None is in the lane, so the van does not brake for them, but the
   occupancy grid (day 9) would inherit them. On plain roads it is 0.6–3.0
   per update and none within 20 m.
3. **Naming is unreliable without the camera.** The barrel on the bend is
   called "vehicle" in 21 of 21 updates, the bin in 20 of 21; the car on the
   straight is "obstacle" in 4 of 21. The shape rule alone cannot do this.
   Day 5 (calibrated camera fusion) owns it.
4. **The cone is fine now.** Day 2 measured it at 0.50 on a bend; with these
   recordings it is 0.91–1.00 everywhere, though the clusterer alone drops
   to 0.39 inside the junction.
5. **The patches road filter earns its place**
   (`day3_replay_scorecard_flat_ab.txt`): against the old flat cut it holds
   3.0 phantoms per update instead of 9.8 on the junction approach, and
   keeps 78–90 % of object points instead of 74–83 %. It costs 8 ms per
   update instead of 1.4 ms.

## What stayed the same

382 tests pass (355 before day 3, 27 new). No pipeline behaviour changed:
the only van-code edits are the injectable detector (default unchanged) and
an optional CARLA import in the adapter (identical when CARLA is installed).

# Perception V2, day 4: how big is it? (2026-09-08)

Until today the van reported a thing as a point: "an obstacle, 12.3 m ahead,
0.2 m to the right". It never said how big it was. Day 4 measures the
footprint, and then uses the measurement to fix the day-3 finding that flat
objects disappear.

## What the van now reports

`DetectedObject` carries four new numbers, all in metres and degrees:

| field | meaning |
|---|---|
| `length_m` | the long side of the footprint the LiDAR has seen |
| `width_m` | the short side |
| `height_m` | how tall, measured above the local road |
| `yaw_deg` | which way the long side points, in the van's frame |

A cluster gets them from the 2D spread of its own points (the same principal
axis the road-edge rule already used), and the track keeps the largest view
of each side, because the LiDAR only ever catches part of a thing and a
bigger view is the truer one. `0.0` means "not measured", which is what the
camera-only and ground-truth modes still report.

The replay harness scores them against the dense labelled scan of each
object, which is the fair answer key: it asks how much of the *visible*
object the van recovered, not how big the object is in the map.

## Two settings changed, both measured first

| setting | was | now | why |
|---|---|---|---|
| clustering grid | 1.0 m | **0.8 m** | at 1.0 m a barrel and the planter beside it became one 4.4 m blob |
| ground cut | 12 cm | **8 cm** | a 12.5 cm planter had almost all its points thrown away as road |

Across the four fixtures, replaying the same recordings through the same
pipeline with only these two settings different:

| | day 3 settings | day 4 settings |
|---|---|---|
| mean recall | 0.89 | **0.90** |
| objects seen in 9 updates out of 10 | 18 of 24 | **20 of 24** |
| blobs covering two objects at once, per update | 0.11 | **0.03** |
| median size error | 0.42 m | **0.26 m** |
| phantoms per update | 4.3 | 4.8 |
| phantoms **in the lane** per update | 0.00 | **0.00** |
| object points kept by the road filter | 85 % | **90 %** |
| ms per update | 9.9 | 10.0 |

The extra 0.5 phantoms per update all sit off the lane, mostly in junctions
where the road meets grass. Nothing new appears in the van's path.

## Live check on CARLA

The probe places one object at a time in front of the parked van and counts
how often the live stack reports it (`logs/perception_probe_20260908_123004.csv`
against day 3's `..._115020.csv`):

| object, distance | day 3 cluster / tracked | day 4 cluster / tracked |
|---|---|---|
| planter 22 m | 0.00 / 0.00 | 0.00 / 0.00 |
| **planter 16 m** | **0.00 / 0.00** | **0.87 / 0.93** |
| **planter 13 m** | **0.47 / 0.97** | **0.97 / 0.97** |
| planter 9 m | 1.00 / 0.97 | 1.00 / 0.97 |
| cone 16 m | 0.93 / 0.93 | 1.00 / 0.97 |
| cone 13 m | 0.93 / 0.93 | 1.00 / 0.97 |
| barrel 22 m | 0.73 / 0.90 | 0.57 / 0.70 |
| barrel 9 m | 1.00 / 0.97 | 1.00 / 0.97 |
| car, person, all ranges | 0.97–1.00 | 0.97–1.00 |

Points surviving the road filter on the planter: 1.0 → 5.4 at 16 m, 3.3 →
8.7 at 13 m, 13.4 → 28.8 at 9 m. The one loss is the barrel at 22 m, where
a 0.8 m grid splits its two or three far points into separate blobs.

Drive test, barrel standing in the lane (`scenarios/run_scenario.py
WAV-0294`): **PASS**, no contact, stopped 8.03 m short of it, and the stop
came 0.31 s after the trigger (the rule allows 6 s). The van began slowing
at **19.0 m**, against 16.4 m on day 3.

## What the size numbers look like

Height is measured well: 0.77 m against 0.77 for the barrel, 1.83 against
1.83 for the pedestrian, 1.56 against 1.56 for the car, on the bend fixture.
Length and width are lower bounds, because the LiDAR sees one face: the car
reads 4.40 x 1.48 m where the visible outline is 3.89 x 1.48. Heading is
good when two faces are visible (2° on the car) and meaningless for round
things like a barrel or a cone, where the long side is whichever way the
noise falls.

Known limit, recorded in the tests: the track keeps the largest view it has
ever had, so if one frame glues an object to a taller neighbour, that
inflated size sticks until the track is dropped. The tests only check
heights for objects the van holds in at least half its updates.

## What is still open

* The planter at 20 m and beyond is still invisible: at that range the road
  filter has nothing left to keep.
* Junctions still produce about 11 phantom reports per update, none in the
  lane.
* Naming without the camera is still wrong (a bin called "vehicle"); day 5.

401 tests pass, 9 of them new for footprints.

## Day 4 live validation (2026-09-08, watched in CARLA)

`tools/live_obstacle_demo.py --sequence planter,barrel,cone,bin --drop-ahead 24
--dest 200`: the van drives a real mission, each object is dropped 24 m ahead
of it while it moves, and is removed once the van has seen it and stopped.

| object | first seen | stopped short by |
|---|---|---|
| planter (12.5 cm tall) | 17.3 m | 12.9 m |
| barrel | 19.2 m | 8.0 m |
| cone | 18.8 m | 13.8 m |
| bin | 17.4 m | 14.7 m |

No contact, no false stop. On day 3 the same planter was not detected at all
beyond about 13 m, and the barrel drive test triggered at 16.4 m.

A separate live run printed the size the van measures as it closes in on a
planter: first sighting at 18.2 m as `0.68 x 0.05 x 0.17 m`, growing to
`2.34 x 0.67 x 0.18 m` at the stop, against a real `4.95 x 0.86 x 0.12 m`.
Height is right to a few centimetres; length and width grow as more of the
object comes into view, as expected from a sensor that sees one face.

Two operational notes from this session, both of them my own mistakes rather
than the van's: restarting the stack while an old process still holds a van
leaves a second Sprinter parked on the road, which the van correctly reports
as a vehicle blocking its path; and a scenario or demo that is interrupted
can leave props behind. Clear both before a live run.

# Perception V2, day 5: what is it? (2026-09-08)

Day 4 told the van how big a thing is. Day 5 tells it what the thing is. The
van has had a camera and a working person detector all along; the answer
never reached the LiDAR blob it belonged to.

## Three faults, all in the joining step

1. **The detection box was read wrong.** The detector returns
   `(left, top, width, height)`. The fusion read it as
   `(left, top, right, bottom)`, so for a person 100 px wide starting at
   column 400 it asked whether anything sat between column 400 and column
   100. Nothing ever does. Every camera label was thrown away, which is why
   the probe showed `cam 1.00 [personx30] fused -`: the camera saw a person
   in all thirty frames and the van still called it an obstacle.
2. **The geometry was a guess.** A blob's column came from
   `width/2 + (width/2) * y / x`, which assumes the camera sits exactly
   where the LiDAR sits and looks dead level. It sits 2 m further forward,
   0.7 m lower, and is tilted ten degrees down. The row was never worked out
   at all, so a kerb and a lamp post at the same bearing were the same thing.
3. **The nearest blob inside a box took the name.** A cone standing in front
   of a car falls inside the car's box, so the cone became a "vehicle" and
   the car behind it became an "obstacle". The scorecard found this one
   after the first two were fixed.

## What replaced them

`src/warp_av/perception/camera_model.py` does the real chain: LiDAR frame ->
vehicle frame -> camera frame -> pixels, using the mounts and the tilt from
the sensor adapter. A blob is aimed at the middle of its own body, and its
ground point (the road directly beneath it) is projected too.

A box then claims a blob when three things hold: the blob's middle lands
inside the box, the box's height suits the blob's size and range (within a
factor of 2.5), and among the candidates the blob whose ground point sits
nearest the box's bottom edge wins. The bottom edge is where a thing touches
the road, which is what separates a cone at 14 m from a car at 20 m.

The old "any car-sized blob is a car" rule stays where the front camera
cannot look (the sides, the back) and whenever the picture is stale. Inside
the camera's view a blob must be plainly car-sized (extent 1.6 m, 25 points,
0.7 m tall) before shape alone may overrule a camera that looked straight at
it and saw no car. That is what stops a bin from being called a vehicle.

## Measured offline: does the name reach the right thing?

The replay harness gained a scripted perfect camera: it boxes every person
and car exactly where the geometry says they must appear, and ignores props,
which is what the real model does since COCO has no barrel or cone class.
This tests the joining step, not the detector.

```
python3 tools/replay_score.py --camera perfect
```

| | before day 5 | after |
|---|---|---|
| person called a pedestrian | 0.00 | **1.00** |
| car called a vehicle | 0.00 | **1.00** |
| prop called an obstacle | 1.00 | **1.00** |

With the camera missing every second frame the numbers do not move. With no
camera at all a person is only an obstacle, which is honest: nothing else can
tell a person from a post.

## Measured live on CARLA

Probe, one object at a time in front of the parked van
(`logs/perception_probe_20260908_131205.csv`):

| object | 22 m | 16 m | 12 m | 8 m |
|---|---|---|---|---|
| person | pedestrian | pedestrian | pedestrian | pedestrian |
| car | vehicle | vehicle | vehicle | vehicle |
| barrel | obstacle | obstacle | obstacle | obstacle |
| cone | obstacle | obstacle | obstacle | obstacle |

Straight through the stack's own API, with the van parked on a clear road:

| placed | reported as | measured size |
|---|---|---|
| person 8 m | pedestrian | 0.54 x 0.23 x 1.72 m |
| barrel 12 m | obstacle | 0.35 x 0.08 x 0.80 m |
| car 20 m | vehicle | 1.79 x 1.04 x 1.18 m |

Driving, with each thing dropped 24 m ahead of the moving van
(`tools/live_obstacle_demo.py --sequence person,car,barrel`):

| dropped | first seen | stopped short by |
|---|---|---|
| person | 24.6 m | 12.2 m |
| car | 20.7 m, named a vehicle | 9.6 m |
| barrel | 20.3 m | 8.0 m |

## Proof the geometry itself is right

A separate check placed a person, a car and a barrel at 8, 12, 16 and 22 m
and three lateral offsets, ran the real detector on the real camera picture,
and compared the projection with the box the detector drew. In all fifteen
placements where the detector found anything, the projected point landed
inside its box, usually within one pixel of the centre; the widest miss was
13 px on a car at 8 m, where the box centre is the car's body and the
projection aims at the middle of the visible face. The thirteen placements
with no box are all barrels, which the model has no class for.

## Also today

`tools/clear_leftover_vans.py` removes Sprinters the stack is not driving.
Each stack restart spawns a fresh van, and an old process's van stays parked
on the road, where the new van correctly reports it as a vehicle blocking the
path. That cost two live runs today before I noticed.

423 tests pass, 21 of them new for the camera model and the fusion.

## Still open

* The planter beyond 20 m stays invisible, and junctions still produce about
  eleven phantom reports per update, none in the lane (day 4's list).
* Nothing the camera sees but the LiDAR misses is reported yet: a
  camera-only candidate has no distance. That needs the ground plane, and it
  belongs with the occupancy grid.
* The barrel is called a fire hydrant by the detector at some ranges. It is
  ignored, since neither is a class the van acts on, but a cone in a
  construction zone deserves better than "obstacle" one day.

# Perception V2, day 6: is it moving, or is it parked? (2026-09-08)

Day 5 told the van what a thing is. Day 6 tells it whether the thing is
going anywhere. This matters because the planner treats the two differently:
you wait for a moving car and you drive around a parked one.

## What was wrong

Speed came from subtracting two sightings and smoothing the result. The
middle of a LiDAR blob jumps about between turns, because a different part
of the object comes back each time, so the subtraction turned that jitter
into speed. Measured on the four recordings, where the van and every object
are parked:

* objects were reported as **moving in 41 % of sightings**;
* the fastest imaginary speed was **6.7 m/s** (24 km/h), on a bin.

## What replaced it

Each track now carries a small motion filter that holds where a thing is and
how fast it is going, together with how sure it is of each. A sighting nudges
the estimate rather than replacing it, by an amount that depends on how noisy
that sighting is: a blob 30 m away barely moves the answer, a solid one at
8 m moves it a lot.

On top of the filter, four rules decide "parked" or "moving":

1. the filtered speed must pass 1.2 m/s to become moving, and drop under
   0.6 m/s to become parked again, so it cannot flicker on the boundary;
2. the thing must actually have gone somewhere: 0.7 m over the last 1.5 s;
3. that travel must be in one direction. A blob whose middle hops between two
   parts of the same object wanders a long way and gets nowhere, and this
   rule tells the difference;
4. none of it counts until the fourth sighting: two jumpy sightings prove
   nothing.

A new track also takes its speed straight from its first two positions,
because waiting for the filter to work it out lets its guess fall behind a
fast car, and the track then breaks and restarts every frame.

Two settings changed with it, both measured: the association distance from
2.6 m to **2.0 m** (a bin and a planter 2.4 m apart were swapping tracks,
which looks exactly like movement), and the allowance for how briskly a road
user may change speed, set to **2.5 m/s²**, ordinary braking. Too small and
the van takes 1.5 s to believe a car has stopped; too large and parked things
start to jitter again.

`DetectedObject` gained `stationary`, and a parked thing reports its speed as
exactly 0.0. The state endpoint publishes both.

## Measured offline

Four recordings in which the van and every object are parked, so the right
answer is "never moving":

| | day 5 | day 6 |
|---|---|---|
| wrongly called moving | 41 % of sightings | **2.2 %** |
| fastest imaginary speed | 6.7 m/s | **2.6 m/s** |
| objects given a new number | 4 | **1** |
| recall | 0.91 | 0.90 |
| position error | 0.21 m | 0.23 m |
| ghosts per update, off the lane | 4.8 | 6.2 |
| ghosts **in the lane** | 0 | **0** |

A thing driven past at a steady speed, with the same blob jitter on top:

|真 speed | measured | called moving after | track numbers |
|---|---|---|---|
| 1.5 m/s | 1.51 m/s | 0.4 s | 1 |
| 5 m/s | 5.00 m/s | 0.2 s | 1 |
| 15 m/s | 15.00 m/s | 0.2 s | 1 |

The tighter association gate costs about 1.4 extra ghosts per update in
junctions, where blobs wander over the road edge; none of them is in the
van's lane, and none changes what the van does today. The alternative, asking
for a third sighting before a track is reported, removed a fifth of them but
also cost recall on the small far objects, so I kept recall.

## Measured live on CARLA

A Tesla driven down the lane towards the parked van at a steady 6 m/s, with a
barrel parked 3 m off to the side:

| car's distance | its real speed | the van's answer | the barrel |
|---|---|---|---|
| 38.0 m | 6.2 m/s | 5.3 m/s, moving, vehicle | parked, 0.0 |
| 27.5 m | 5.8 m/s | 7.0 m/s, moving, vehicle | parked, 0.0 |
| 22.6 m | 6.0 m/s | 6.0 m/s, moving, vehicle | parked, 0.0 |
| 17.7 m | 6.2 m/s | 5.7 m/s, moving, vehicle | parked, 0.0 |
| 13.3 m | 6.1 m/s | 5.9 m/s, moving, vehicle | parked, 0.0 |
| 8.3 m | 6.0 m/s | 5.5 m/s, moving, vehicle | parked, 0.0 |

436 tests pass, 13 of them new for motion.

## Still open

* Junctions produce about six ghost reports per update, none in the lane.
* The planter beyond 20 m is still invisible.
* Nothing the camera sees but the LiDAR misses is reported.

# Perception V2, day 7: one sheet of what the van knows (2026-09-08)

Days 1 to 6 each ended with a hunt for who else might be reading the thing I
had just changed. The behaviour layer read three loose fields off the
perception output, the web page rebuilt map coordinates itself, the planner
wrote its answers back into perception's own object, and the parking learner
read raw blobs. Day 7 puts one sheet in between.

## The sheet

`src/warp_av/world_model.py` holds `WorldModel`, built once a tick from the
perception output and the van's pose:

* every thing the van is following, with its place **in both frames** (metres
  ahead of and right of the van, and where it stands on the map), its
  measured size, what it is called, whether it is moving, how sure the van
  is, and how old the sighting is;
* `path`: what is in the way, how far, what kind, how fast, and which object
  it is;
* whether the eyes are working, and why not when they are not;
* the traffic light, and when all of this was true.

It answers the questions the rest of the stack actually asks:
`in_corridor()`, `nearest_ahead()`, `moving_objects()`, `parked_objects()`,
`of_kind()`, `crossing_vehicles()`, `by_id()`. It decides nothing. Nothing in
it brakes, steers or plans.

Wired in three places, all behaviour-neutral:

* the state endpoint's object list is now written from the sheet, and a test
  pins that every number matches the hand-written maths it replaced;
* `GET /api/world` serves the whole sheet;
* the behaviour layer's junction check asks `crossing_vehicles()` instead of
  walking perception's list itself, and a test runs both ways over the same
  seven vehicles and requires the same answer.

## What the sheet found on its first day

Publishing `stationary` for **every** object, rather than only the six placed
ones the scorecard tracks, showed that day 6's result was too kind. On the
four recordings, where nothing moves at all:

| | before day 7 | after |
|---|---|---|
| things wrongly called moving, whole scene | 25.8 per update | **13.4** |
| ... of those, within 25 m where the van acts | — | **1.8** |
| fastest imaginary speed | 249.8 m/s | **30.0 m/s** |

Day 6 measured only the six placed objects, all inside 22 m, and reported
2.2 %. That number was true and is unchanged. It simply did not cover the far
background, where a wall or a hedge is cut differently on every turn and its
middle wanders metres.

Four rules were added, each measured:

1. **Nothing does 900 km/h.** A jump faster than any road user is two
   different things being confused, not a measurement. The old tracker had
   this check and I dropped it on day 6 when I replaced the update step; it
   is back, both when a track starts and after every correction.
2. **Scenery does not drive.** A blob longer than a bus is a wall or a hedge
   and is never called moving.
3. **A far thing must travel further** before it counts, 0.15 m per metre of
   range, but never more than a walking person covers in the window, or a
   pedestrian at 25 m would be dismissed as scenery.
4. **The travel must be nearly a straight line** (0.92, up from 0.5). A car
   rounding a bend still scores about 0.96; a blob whose middle shuffles
   between two parts of the same object scores far less.

Everything that really moves is still caught:

| what | called moving after |
|---|---|
| a person walking at 25 m | 1.6 s |
| a car crawling at 2 m/s, 12 m | 0.9 s |
| a car at 8 m/s, 20 m | 0.4 s |
| a car at 6 m/s, 40 m | 0.5 s |
| a car rounding a bend | 0.4 s |

## Nothing about the driving changed

That was the day's constraint, and it holds:

* 454 tests pass, 16 of them new;
* the barrel drive test passes with no contact, a 7.42 m clearance and a stop
  0.33 s after the trigger (day 4: no contact, 8.03 m, 0.31 s, at a different
  spawn point);
* the live probe still reports a barrel at 22, 16, 12 and 8 m, and a person
  as a pedestrian at all four.

## Still open

* About 12 far ghosts per update are still called moving in junctions, all
  beyond 25 m. They are scenery whose blob wanders; the honest fix is an
  occupancy grid that models surfaces rather than treating every blob as an
  object (day 9).
* The planter beyond 20 m is invisible, and nothing the camera sees but the
  LiDAR misses is reported.

## Day 7 follow-up: how big is it, when things really do differ? (2026-09-08)

The live run of the world sheet showed a person reported as 8.6 m long and 4 m
tall. The cause was a rule written on day 4: a track keeps the largest view of
itself the van has ever had. That is right for a car, which reveals more of
itself as you approach, and wrong the moment one frame glues an object to a
wall, because the inflated size then sticks for the life of the track.

The obvious fix, clamping to a standard person size, would be worse: people are
not a standard size. A child, an adult, someone pushing a pram and someone
wheeling a bicycle are all people, and all need different room.

What replaced it:

* **the middle of the recent sightings**, twelve of them, rather than the
  largest ever. One merged frame is outvoted; a size that several frames agree
  on still comes through, which is how a person with a bicycle keeps their
  length;
* **the largest view only until there are three sightings to vote on**, since
  before that there is no majority;
* **limits that reject the impossible and never rewrite the real**: a thing the
  camera calls a person may be up to 2.5 m long, 1.8 m wide and 2.4 m tall, and
  a vehicle up to 14 m. A sighting past that is a merge, so it is thrown away,
  and the object is marked unsure rather than silently reported wrong;
* **`size_uncertain`**, true when recent sightings disagree by more than 60 %,
  so a reader knows the size is a guess;
* **`clearance_radius_m`**, the room to leave: what was measured, but never less
  than the kind of thing deserves. A small child gets a person's 0.6 m margin
  whatever the laser saw of them.

Measured on the four recordings, against the dense labelled scan:

| | old rule | new rule |
|---|---|---|
| median size error | 0.30 m | **0.06 m** |
| worst | 2.39 m, on the bin | 1.22 m, on the cone |

Live, three different walkers placed 10 m ahead at two different spots:

| who | real | the van says | room to leave |
|---|---|---|---|
| a slim adult | 0.38 x 0.38 x 1.86 | 0.40 x 0.12 x 1.75 | 0.60 m |
| a child | 0.50 x 0.50 x 1.10 | 0.16 x 0.05 x 1.06 | 0.60 m |
| a broader adult | 0.38 x 0.38 x 1.86 | 0.54 x 0.15 x 1.75 | 0.60 m |

The 8.6 m person is gone, and the van now tells a child from an adult by
height, 1.06 m against 1.75 m. The measured width stays much thinner than the
real body because the laser only sees the front surface, which is exactly why
the clearance is a floor rather than the measurement.

461 tests pass, 7 of them new for sizes.

### The unsure flag, fixed (2026-09-08)

Live, the flag fired on almost everything, including a person and a car whose
readings were perfectly steady. It compared the swing between sightings with a
share of the object's own size, and any natural swing is a big share of
something 0.4 m across, so every pedestrian came out unsure and the warning was
noise.

It is now an absolute swing, and it grows with range because the LiDAR's points
spread out: `0.15 m + 0.02 m per metre`. That comes from the recordings, where a
walker at 6 m swings 0.05 to 0.17 m, a barrel at 9 m 0.15 m, a bin at 12 m
0.27 m and a car at 22 m 0.32 m, while a blob merging with its neighbour swings
1.1 to 2.7 m.

On the four recordings the flag now agrees with the measured swing on **22 of
22** objects: the walker, the barrel, the bin and the car on the straight road
read steady, while the cone, the planter and the objects that merge inside
junctions are flagged. 463 tests pass.

### Someone on a bicycle (2026-09-08)

A live run called a cyclist a pedestrian, 0.65 m long against a real 1.66 m
body. The camera model has no cyclist class: it reports a person and a bicycle
in the same place. So the van now looks for that pair. When a person's box sits
at least 35 % over a bicycle's or a motorbike's, that person is riding.

`ObjectType.CYCLIST` joins pedestrian in `VULNERABLE_TYPES`, which means the van
always stops for one and never starts the blocked-road clock. Without that, a
cyclist waiting three seconds in front of the van would be written off as a
blocked road and the van would try to drive around them. A cyclist may be up to
2.8 m long, and is given 0.8 m of room even when badly measured, between a
pedestrian's 0.6 and a car's 1.2.

Riders are matched before bystanders. Around the motorbike the camera reported
two people and one bicycle: one person barely overlapping it and one sitting
squarely on it. Whichever is handled first takes the blob, so the rider has to
go first or the cyclist ends up named a pedestrian again.

Live, each thing placed 10 m ahead:

| what | the van calls it | room to leave |
|---|---|---|
| a person on foot | pedestrian | 0.60 m |
| a bicycle with a rider | **cyclist** | 0.80 m |
| a motorbike with a rider | **cyclist** | 0.80 m |
| a bicycle with nobody on it | pedestrian, then cyclist | 0.60-0.80 m |

**Two limits worth knowing.** The camera model detects a bicycle only some of
the time in this simulator: asked directly, it reported a person and no bicycle
for a ridden bike, and read an empty bicycle as a person. And the LiDAR sees
only the rider, not the frame, so a cyclist's measured length is about 0.66 m,
the same as a person's. Naming therefore rests on the camera alone. An
unattended bicycle called a cyclist is wrong but safe: it earns more room and
right of way, not less.

471 tests pass.

# Perception V2, day 8: when an eye goes blind (2026-09-08)

Measured first, on the live van, changing nothing. Each sense was switched off
in turn while the van drove:

| switched off | what the van did | what it said |
|---|---|---|
| the camera | full stop in 2 s | "Safety supervisor commanded stop" |
| the LiDAR | full stop in 2 s | "Safety supervisor commanded stop" |
| the GPS | **nothing at all**, carried on at 4 m/s | nothing |

Three problems in that. The van could not say which eye had gone dark. A lost
camera stopped it dead, although the LiDAR still finds everything solid and
what a camera loses is the name on a thing, not the thing. And a sense the
safety supervisor had never been told about could fail in silence.

## What replaced it

`src/warp_av/sensor_health.py` gathers every sense into one report and states
what losing each one means:

| sense | losing it means |
|---|---|
| LiDAR, position, controller, the van itself | **stop**: it cannot drive without them |
| camera, object detection, motion sensing | **slow**: crawl at 2 m/s, and stop if it lasts more than 10 s |
| GPS | **note**: report it; the position check above covers a lost fix |

The report goes to the safety supervisor, which gained a `degraded` state and a
speed cap. The cap can only ever slow the van down, never speed it up, and a
cap of zero is a stop. The behaviour layer honours it and says so in its
reason. The whole report is published at `/api/state` under `sensors`.

The pipeline itself changed to make "slow" honest: a missing or stale camera no
longer makes the whole of perception unhealthy. The van runs on the laser
alone, names nothing, and marks the tick `degraded`. Without the LiDAR it still
reports unhealthy and stops.

## Two faults my own first attempt had, both found live

* **A switched-off sensor was treated as deliberately off, not broken.** That
  is exactly how a fault is injected in this stack, so a switched-off LiDAR
  came out as "all senses working" while the van drove on. A van that cannot
  see does not care why. Not working is now not working.
* **The van was told it was crawling while it was actually frozen**, because
  perception still reported unhealthy underneath. That is what led to the
  laser-only path above.

## The same test, after

| switched off | what the van does | what it says |
|---|---|---|
| the camera | keeps driving, held to 1.7–2.1 m/s | "camera not working — driving slowly" |
| the LiDAR | stops within 1 s | "lidar not working — stopping" |
| the GPS | carries on; position is still good | "gps not working" |

And the whole cycle, live: driving at 3.4 m/s, the laser fails, the van stops
within a second and the mission pauses, the laser comes back and the senses
read healthy again, the mission **stays paused** until the operator presses
resume, and then the van drives on. A safety stop waiting for a human is the
existing design and is worth keeping; day 8 only made sure the van says why it
stopped and gets going again cleanly afterwards.

493 tests pass, 22 of them new.

## Still open

* A degraded crawl is a fixed 2 m/s for 10 s. A real van would pull over
  rather than stop in a live lane, which is planning work, not perception.
* The far-field ghosts and the invisible planter beyond 20 m are unchanged.

# Perception V2, day 9: a map of free space (2026-09-08)

Until today the van kept a list of things. A list answers "stop for that" and
cannot answer two other questions it has to answer:

* **where can I go?** Steering around anything needs the gap, not the object.
* **what is that long thing?** A wall is one surface. Chopping it into blobs
  invents objects that were never there, which is where about thirteen
  imaginary moving objects per update were coming from.

`src/warp_av/perception/occupancy.py` puts a grid on the van, 60 m across in
squares of 25 cm, and marks every square as one of three things:

| | meaning |
|---|---|
| **free** | a laser beam went through it and carried on |
| **blocked** | a beam stopped there |
| **not seen** | no beam has been through it, so nothing is claimed |

**Not seen is kept apart from free on purpose.** Behind a parked van is
unknown, not empty. A planner that treats the two the same drives into
whatever is hiding there.

Filling it in is one pass over the sweep. Every return becomes a bearing and a
range; for each narrow slice of bearing the nearest solid return is where the
world stops; everything nearer was passed through, so it is free; everything
beyond is hidden. The grid is built from the same sweep and the same road
decision the object list uses, so the two can never disagree.

## Two mistakes worth recording

**A thin thing can fall between two slices**, and the slice beside it would
sweep free straight past it. Every slice is now also held back by the nearest
wall its neighbours found. Getting this right took three attempts:

* holding a slice back by its neighbours' *reach* wipes the map out, because
  with slices this narrow most hold no points at all and a reach of zero
  spreads everywhere;
* holding it back by the neighbours' *wall* alone lets a slice that saw
  nothing inherit a far wall and claim more space than it ever saw, which made
  the numbers worse rather than better;
* the smaller of the two is the only version that can lose free space and
  never invent it.

**Asking how far the van can go from the middle of the van** returns nothing.
The squares under the van itself can never be seen, because its own returns
are thrown away. The walk now starts at the nose.

## Scored against the dense labelled scan

| recording | road called free | things blocked | **things wrongly called free** | hidden | ms |
|---|---|---|---|---|---|
| straight road | 96.0 % | 72.0 % | 1.66 % | 26.3 % | 2.2 |
| bend | 97.0 % | 72.8 % | 3.15 % | 24.1 % | 3.2 |
| junction approach | 95.9 % | 64.3 % | 3.25 % | 32.4 % | 2.1 |
| inside a junction | 92.8 % | 70.0 % | 2.87 % | 27.1 % | 1.9 |
| **overall** | **95.4 %** | **69.8 %** | **2.73 %** | 27.5 % | **2.4** |

The middle column looks low until you read the last: about a quarter of every
solid thing is its own hidden interior. The laser sees the front face of a car,
not the inside of it, and the grid correctly refuses to claim anything about
the rest. Blocked plus hidden is 97 %.

The column that matters for safety is the share of real things the map calls
free: **2.73 %**, from 6.76 % before the shadow rule. Almost all of what is
left is the flat planter, whose points the road filter still removes.

## Live

Parked on a clear road: free ahead 15.0 m. A car placed 9 m ahead: free ahead
drops to 6.5 m and a wedge of "not seen" opens up behind it. A barrier added
2.2 m to its right: free ahead 5.2 m and the free space to the right falls
from 13.2 m to 5.5 m. `GET /api/grid` returns the numbers and a small picture
of the map; the world sheet carries the summary.

506 tests pass, 13 of them new.

## Still open

* The map is built fresh from each turn of the laser. It does not yet
  accumulate over time, so what the van saw a second ago is forgotten.
* Nothing steers by it yet. That is the planner's to use, and this was the
  piece it was missing.

# Perception V2, day 10: where the road stops (2026-09-08)

Day 9's map says where there is space. It does not say where the van may put a
wheel. A laser beam travels along a pavement perfectly well, so a pavement came
back as free. Free is not drivable.

## What the measurement said before any code was written

The textbook answer is to find the kerb: a step of ten or fifteen centimetres
running alongside the road. Measured on the four recordings, using the labelled
scan to say exactly where the pavement is:

| recording | how high the pavement sits | kerb-band returns, left / right |
|---|---|---|
| straight road | −0.02 to 0.15 m | 8 / 6 |
| bend | 0.00 to 0.25 m | 7 / 33 |
| junction approach | 0.00 to 0.31 m | 4 / 50 |
| inside a junction | 0.00 to 0.15 m | 59 / 46 |

In this town a pavement is often level with the road, and a kerb can leave the
van as few as **six laser returns in twenty metres**. That is thin evidence, and
it changed the plan: the kerb is worth reporting when it is really there, and
must be declined the rest of the time, so the weight has to fall on something
better supported.

## What was built

**Two signals, and each is only asked what it can answer.**

*The kerb line*, in `src/warp_av/perception/road_edges.py`. Points in the kerb
band, 5 to 35 cm above the local road, are split left and right of the van and
a straight line is fitted through each, three times over, dropping the points
that disagree each round, because a parked car's wheels and a driveway leave
returns in the same band. A line is reported only with at least twelve points
over four metres, more than half of them agreeing, running within 25° of the
road. Otherwise the van says it does not know.

*The ground layer*, added to the day-9 grid. Every square now carries a second
mark: whether the laser actually landed on ground the van could roll on, as
opposed to space a beam merely passed through. It is filled along each bearing
out to the furthest ground return and never past the nearest solid thing,
because marking only the squares the returns land in draws thin arcs: the laser
touches the ground in rings, and the gap between two rings is still ground.
`drivable_at(x, y)` is free **and** ground, and inside a kerb line when there is
a confident one.

## Measured

The kerb line, against the near edge of the labelled pavement:

| recording | side | van says | truth | error |
|---|---|---|---|---|
| straight road | right | 3.83 m | 3.73 m | **0.10 m** |
| bend | right | 6.11 m | 7.12 m | 1.01 m, and reported as unsure |
| bend | left | −9.63 m | −7.44 m | 2.19 m |
| junction approach, inside a junction | both | not found | | correctly declined |

Live, driving a 200 m route for twenty-four seconds, the right-hand kerb was
held steadily at 4.4 to 4.75 m from the van, within two degrees of straight,
over 12 to 19 metres of road at a time. The left-hand kerb was almost never
found, which on that street is the right answer.

## The limit, stated plainly

**The van cannot reliably tell carriageway from pavement here.** The ground
layer covers both, because both are flat ground and this town's pavement is
often level with the road. What the van has gained is real but narrower than it
sounds:

* it separates ground it could roll on from free space it merely looked
  through, which day 9 could not do;
* where a kerb is physically there and the laser can see it, it holds the line
  to about ten centimetres and knows which side of it a square lies on;
* and where the kerb is not visible, it says so instead of inventing one.

520 tests pass, 14 of them new.
