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
