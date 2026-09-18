# Early-slow reach fixtures (2026-09-18)

`bend_approach_ticks.json`: two recorded approaches to the parked Mustang at Town10's north-east bend, tick by tick
as the stack saw them (rig batch endpoint_after): the pose the planner used, the speed, the intended path it had
planned the tick before, the Mustang's track as perception published it, and what the planner and controller did.
`high_speed_WAV-0888`: 6.6-7.0 m/s, the hold at a 5.35 m body gap. `five_mps_BEND5`: 4.3-4.7 m/s. `flicker_WAV-0001_after`: the first live batch on the speed-aware reach (stack eca1c1d, before the hold hysteresis): held at 15.6 m, dropped for two ticks when the reach shrank with the speed, held again at 12 m. The route is the
recorded one in ../endpoint/recorded_routes.json. The truth box of the Mustang is scoring only.
