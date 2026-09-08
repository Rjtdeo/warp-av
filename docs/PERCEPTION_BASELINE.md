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
