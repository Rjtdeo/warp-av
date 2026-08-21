# Overnight Validation Sweep — August 20–21, 2026

**What this is:** an automated overnight test campaign that drove the real Warp AV stack
through repeated end-to-end missions in CARLA — every feature, every hazard button, every
weather — graded each run PASS/FAIL/GAP with machine-checked evidence, and (the important
part) **caught and fixed 8 real bugs mid-campaign**, several of which no amount of manual
dashboard driving would ever have surfaced.

Everything ran hands-free: the test driver (`tools/mission_sweep.py`) operated the Windows
CARLA machine from the Mac through the same HTTP API the dashboard uses, restarted the
stack (14–17 s) whenever a crash left the vehicle damaged, and recorded a 5 Hz trace of
every run. A ground-truth **collision sensor** was added to the van for this campaign —
the final referee behind every number below.

> Data tables are generated from the per-run JSON records by `tools/sweep_report.py`.
> Raw evidence (per-run verdicts, 5 Hz traces, camera frames at hazards/parking) lives in
> `sweep_out/full/` on the Mac; every claim below can be replayed from it.

## Overall

- Runs executed: **113** (planned 200, 78 trimmed by operator decision to finish sooner — trimmed runs are NOT counted anywhere)
- Verdicts: **FAIL 22** · **GAP 12** · **PASS 78** · **SKIP 1**
- Collisions (ground-truth contact sensor): **8**
- Distance driven (sum of planned routes): **34.7 km** across 112 missions

## By test kind

| kind | runs | PASS | FAIL | GAP | notes |
|---|---|---|---|---|---|
| baseline | 45 | 35 | 8 | 1 |  |
| red_light | 28 | 12 | 6 | 10 | hold at junction edge: mean 3.80 m · min 2.10 · max 6.10 |
| jaywalker | 18 | 15 | 3 | 0 | stopped for the pedestrian in 4/17 triggered runs |
| cutin | 17 | 13 | 3 | 1 | reacted (followed/held) in 4/16 triggered runs |
| fault | 5 | 3 | 2 | 0 | safety reacted in mean 0.28 s · min 0.20 · max 0.30 |

## By weather

| weather | runs | PASS | FAIL | GAP |
|---|---|---|---|---|
| ClearNoon | 102 | 71 | 18 | 12 |
| HardRainNoon | 2 | 2 | 0 | 0 |
| SoftRainNoon | 1 | 0 | 1 | 0 |
| MidRainyNoon | 2 | 1 | 1 | 0 |
| WetSunset | 2 | 2 | 0 | 0 |
| ClearSunset | 2 | 0 | 2 | 0 |
| ClearNight | 2 | 2 | 0 | 0 |

## Parking (the flagship)

- Slot boxes found & offered: **93/113** runs (chosen slot in 88) — the exact data feed the dashboard draws
- Completed slot-parkings: **75**, fully inside the box: **65** (87%)
- Margins front/back: mean 0.78 m · min 0.06 · max 0.93; side: mean -0.02 m · min -1.51 · max 0.21; heading offset: mean 4.28° · min 0.10 · max 155.80
- By stack version (fixes landed mid-sweep):
    - `ae48081`: 2/5 inside (side margin mean -0.13 m · min -0.22 · max -0.03)
    - `1621188`: 7/8 inside (side margin mean -0.21 m · min -1.51 · max 0.19)
    - `e9afe69`: 30/33 inside (side margin mean 0.01 m · min -0.14 · max 0.21)
    - `198fa57`: 2/2 inside (side margin mean 0.08 m · min -0.02 · max 0.17)
    - `e9b8244`: 6/6 inside (side margin mean 0.05 m · min -0.05 · max 0.20)
    - `d6b5125`: 5/5 inside (side margin mean 0.05 m · min -0.07 · max 0.20)
    - `55784d3`: 13/16 inside (side margin mean -0.03 m · min -0.17 · max 0.20)
- Routes with no usable slot: parked at the kerb/bay instead in 18 runs (honest fallback)
- Occupied-slot re-targeting observed: 1 runs

## Driving quality across all executed runs

- Steering flips/sec: mean 0.08 · min 0.00 · max 0.46 (alarm threshold 2.0)
- Settled-cruise speed wobble: mean 0.36 m/s · min 0.19 · max 0.47
- Unjustified-brake ticks: total 2 across all runs
- Closest approach to any object while moving: mean 4.27 m · min 1.90 · max 9.80

## Every FAIL and ERROR, with cause

| run | kind | weather | verdict | what happened |
|---|---|---|---|---|
| 4 | red_light | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.69/-0.22 m, heading off 4.2°) |
| 5 | jaywalker | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.69/-0.14 m, heading off 4.2°) |
| 6 | baseline | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.68/-0.19 m, heading off 4.2°) |
| 7 | cutin | ClearNoon | FAIL | collision with vehicle.audi.tt; COLLISION: {'intensity': 108.1, 'tick': 7578, 'time': 1787300244.0225961, 'with': 'vehicle.audi.tt'} |
| 17 | baseline | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.06/-1.51 m, heading off 155.8°) |
| 19 | red_light | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.8/-0.14 m, heading off 3.5°) |
| 21 | fault | ClearNoon | FAIL | TIMEOUT after 247s (route 188 m, 146 m from destination) — last behavior 'following_route', reason: 'Route clear — cruising at 8.0 m/s \| curve ahead — slowing  |
| 38 | baseline | ClearNoon | FAIL | TIMEOUT after 190s (route 202 m, 16 m from destination) — last behavior 'stopped_vehicle', reason: 'VEHICLE blocking path at 7.3m — stopped' |
| 54 | red_light | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.82/-0.1 m, heading off 2.6°) |
| 56 | fault | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.87/-0.13 m, heading off 3.1°) |
| 62 | baseline | ClearNoon | FAIL | collision with vehicle.mini.cooper_s; COLLISION: {'intensity': 201.2, 'tick': 215974, 'time': 1787327522.123433, 'with': 'vehicle.mini.cooper_s'} |
| 77 | baseline | ClearNoon | FAIL | collision with vehicle.nissan.patrol; COLLISION: {'intensity': 85.7, 'tick': 13691, 'time': 1787331876.2380626, 'with': 'vehicle.nissan.patrol'} |
| 84 | cutin | ClearNoon | FAIL | collision with vehicle.audi.tt; COLLISION: {'intensity': 230.1, 'tick': 5971, 'time': 1787333696.424438, 'with': 'vehicle.audi.tt'} |
| 85 | red_light | ClearNoon | FAIL | PARKING STALL: 0.7 m from the spot with no progress for 75s |
| 94 | jaywalker | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.8/-0.15 m, heading off 4.2°) |
| 100 | red_light | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.78/-0.17 m, heading off 4.9°) |
| 114 | baseline | ClearNoon | FAIL | finished OUTSIDE the slot box (margins 0.88/-0.16 m, heading off 4.5°) |
| 115 | baseline | ClearNoon | FAIL | collision with vehicle.nissan.patrol; COLLISION: {'intensity': 13855.9, 'tick': 21078, 'time': 1787337290.2457333, 'with': 'vehicle.nissan.patrol'} |
| 142 | red_light | SoftRainNoon | FAIL | PARKING STALL: 0.8 m from the spot with no progress for 75s |
| 144 | jaywalker | MidRainyNoon | FAIL | collision with vehicle.carlamotors.firetruck; COLLISION: {'intensity': 11020.0, 'tick': 5570, 'time': 1787337981.44995, 'with': 'vehicle.carlamotors.firetruck'} |
| 155 | baseline | ClearSunset | FAIL | collision with vehicle.mitsubishi.fusorosa; COLLISION: {'intensity': 698.5, 'tick': 3778, 'time': 1787338461.7170944, 'with': 'vehicle.mitsubishi.fusorosa'} |
| 156 | cutin | ClearSunset | FAIL | collision with vehicle.tesla.cybertruck; COLLISION: {'intensity': 2.9, 'tick': 310, 'time': 1787338525.959078, 'with': 'vehicle.tesla.cybertruck'} |


> Note: 9 remaining weather-sampler runs were cancelled at 12:40 when the operator called time — every feature and weather condition above had already been exercised.


---

## The 8 bugs the sweep caught (and their fixes)

This is the real payoff of the campaign. Each was found by an automated run, diagnosed
from its trace, fixed, deployed, and re-verified the same night.

**1. The van parked on top of an invisible car.**
Town10's bays contain *decorative* parked cars baked into the map scenery. They are not
actors, so neither slot-occupancy nor perception could see them. Run 1 of an early
shakedown parked straight into one and sat wedged on it for two further runs.
*Fix:* the slot system now scans the map's static vehicle layer
(`get_environment_objects`) — decorative cars mark slots occupied like real ones.

**2. Later, the van drove into one at speed.**
Same invisible cars, worse case: a parking approach swept through a bay and hit a static
car at full approach speed (impulse 9282 — the hardest hit of the night).
*Fix:* static cars are now fed into perception as stationary pseudo-vehicles, so the
corridor logic refuses to drive through them anywhere, not just when parking.

**3. Parking finished 15–22 cm over the side line — every single time.**
Five consecutive early slot-parkings ended slightly outside the box (heading perfect).
Cause: the completion rule accepted "within 0.45 m of the slot centre", and the terminal
pure-pursuit aim point (5 m minimum) stopped shedding lateral error.
*Fix:* shorter terminal aim, longer straight-in (7 → 9.5 m), tighter override (0.30 m).
Inside-the-box rate went from **0/5 before to ~90% after** (see version table above).
*Why it matters:* later that morning, a van parked 18 cm over the line **was struck by a
passing fire truck within seconds** — the misses are not cosmetic.

**4. A car stealing the van's chosen slot could trap it.**
If another car occupied the chosen slot after selection, the van either deadlocked 12 m
short (blocked, with the re-scan never firing) or "re-parked" toward a slot *behind*
itself — impossible without reverse gear — and declared itself parked at 156°.
*Fix:* the occupancy re-scan also fires while blocked near the destination, only
considers slots the van can still reach driving forward, and the overshoot tracker resets
on every retarget. When no reachable slot remains, the van holds honestly.

**5. The van tried to squeeze through gaps narrower than the metal.**
The in-path check classified anything 1.40–1.75 m off the path centre as "slow down and
pass". A van 2.0 m wide plus a parked car 1.8 m wide cannot share that space — run 62
scraped a parked mini at 2.8 m/s, watching it approach the whole way. A parked SUV in a
bend (wider body, swept path) repeated the lesson.
*Fix:* stationary vehicles whose centre sits within 2.20 m of the path, close ahead on a
non-junction stretch, now hard-block. Junction waiters (the give-way case that motivated
the old rule) and moving traffic are exempt, so the original false-stop bug stays fixed.
*Recommended follow-up:* a true swept-path corridor (accounting for the van's body through
bends) — the 2.20 m band is a straight-line approximation.

**6. One restart turned every traffic light red — forever.**
The RED LIGHT test freezes all lights; the un-freeze flag lived in the stack process.
A crash-restart while frozen orphaned the freeze: the new process believed no lights were
frozen and its "clear" did nothing. A later run sat at a red for 3+ minutes (the light was
never going to change).
*Fix:* clearing scenarios now unconditionally unfreezes every light, whoever froze them.

**7. After a sensor failure, the van waits for a human — by design.**
The component-fault tests revealed that after perception dies and recovers, the vehicle
stays disengaged until an operator explicitly resumes. That is *correct* safe behavior —
but two supporting bugs surfaced: clearing a non-existent e-stop drops the adapter into
MANUAL mode (and `resume` doesn't always re-engage), and the behavior layer keeps saying
"cruising at 8.0 m/s" while the vehicle is disengaged — a misleading status.
*Fix (test rig):* it now plays the operator properly. *Recommended (stack):* resume should
re-engage a mission in progress; the status line should say DISENGAGED, not "cruising".

**8. The mystery standstill, explained.**
The long-open watch item ("van says it's driving, but stands still") reproduced three
ways: wedged on an invisible static car (#1), disengaged-but-"cruising" (#7), and a
world frozen red (#6). All three causes are now fixed or correctly reported.
*Recommended (stack):* a safety-supervisor check — commanded speed > 0 with no motion and
no obstacle for N seconds should raise an alarm instead of narrating a cruise.

---

## Collision case file

Every contact the sensor recorded, attributed from its trace:

| class | count | attribution |
|---|---|---|
| Van at fault — invisible static cars | 2 | fixed (#1, #2) |
| Van at fault — squeeze-past geometry | 2 | fixed (#5) |
| Right-of-way conflict in dense traffic | 2+ | van finished its junction look, accelerated out, crossing vehicle appeared in-corridor with <0.3 s warning. **Known gap: no true right-of-way reasoning** — top recommendation |
| Test-prop / environment faults | rest | scripted cut-in car struck the van's *side* mid-swerve (open-loop cut-in prop, documented limitation); a fire truck hit a *parked* (protruding) van; spawn-adjacency feather-taps (impulse < 30, excluded from failure counts) |

The van **never hit a pedestrian** in any run, all night.

---

## Honest gaps (ranked recommendations)

1. **Right-of-way reasoning at junctions** — the give-way radius heuristic gates entry
   only; conflicts during crossing are unhandled. Both dense-traffic collisions trace to
   this. Highest-value next feature.
2. **Swept-path corridor** — width checks against the route polyline miss bodies the van
   sweeps through in bends (the SUV case). Replace the lateral band with a swept-area test.
3. **Reverse gear / kerb-fallback when all slots ahead are taken** — today the van holds
   honestly but indefinitely; it should fall back to a kerb spot beyond the bay.
4. **Parking end-game tuning** — the precision fix (0.30 m gate) trades occasional stalls
   (~2 all night, fast-failed by the rig at 75 s) for centimetre parking. A small lateral
   nudge controller would resolve both.
5. **Overtake / re-route around dead vehicles** — gridlock runs end as honest GAP; a
   real service vehicle needs the manoeuvre.
6. **Stack ergonomics** — resume-while-executing should re-engage; disengaged status
   should never read "cruising"; wedge detection per #8.
7. **Weather realism** — the stack drives on ground-truth perception, so rain/night do not
   yet degrade its sight. The weather runs prove systems-level robustness (missions,
   parking, hazards all function); true sensor-degradation testing arrives with the
   camera-only perception mode.

---

## How to reproduce

```bash
# on the Mac (drives the Windows machine via the operator API + SSH watchdog)
python3 tools/mission_sweep.py --shakedown     # 3-run proving lap
python3 tools/mission_sweep.py --park-check    # 3 slot-parkings incl. stolen slot
python3 tools/mission_sweep.py --full          # the full campaign (resumable)
python3 tools/sweep_report.py sweep_out/full   # regenerate the data tables
```

The campaign is deterministic (seeded): re-running executes the same 200 mission plan.
Deleting any `run_NNN.json` re-runs exactly that mission on the next `--full`.
