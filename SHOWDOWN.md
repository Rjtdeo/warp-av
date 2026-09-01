# Camera vs. Truth — the first measured showdown (Aug 31, 2026)

Same 20-mission exam (hazard-heavy: forced red lights, jaywalkers, cut-ins,
occupied bays, dense traffic), taken twice by the same van — once reading
perfect simulator data, once driving on its own camera + LiDAR.

| | PASS | known-gap | collisions | parked fully inside |
|---|---|---|---|---|
| **Ground truth** | 7/20 | 6 | 2 | 10/10 |
| **Camera + LiDAR** | 7/20 | 4 | 3 | 8/10 |

**Headline: the two-day-old camera stack matched perfect vision on missions
passed.** It paid in quality, not quantity: one extra collision, two
parking-precision misses (~0.1 m over the side line), and three slow-motion
timeouts near destinations (the known roadside-clutter stutter).

**The camera to-do list this measurement produces:**
1. Stutter/phantom-block near bays — 3 timeouts (cluster stability on
   hedges/kerbside clutter)
2. Three collisions to trace (early-tick contact suggests late detection of
   parked cars in narrow spots)
3. Parking precision tail (side margins -0.11/-0.14)

Note on absolute scores: 7/20 in BOTH modes reflects the exam's difficulty
(12 of 20 missions are deliberate hazards) — the number that matters is the
GAP between the columns, and today it is close to zero. Raw evidence:
`sweep_out/showdown_truth/`, `sweep_out/showdown_camera/` (Mac).
