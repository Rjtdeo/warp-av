# Mission endpoint fixtures (2026-09-18)

`recorded_routes.json`: the routes the stack drove on the rig (runner `meta.route`, batch rejoin_after2) for three
pins on the x = 109.5 road of Town10HD, with the pin and the start. Points carry x, y, yaw, is_junction, road_id,
lane_id as `/api/route` published them. The first `plan_route_points` of each are what `plan_route` returned for that
pin (checked against the map on the rig with scratch/endpoint/replay_spot.py); the rest are the extension past the
pin as it was cut to the chosen stop. The tests rebuild the road from these points (a 2.0 m shoulder to the right
everywhere, as the rig reported) and reproduce the stop the van really chose.
