# Waymo's research, read for the van (2026-09-04)

Source: https://waymo.com/research/ — 98 papers, 2019–2026. By topic: Perception 61, Behaviour
Prediction 22, Simulation 13, Planning 8, General ML 7, End-to-End Driving 2. Plain-English digest
for a small CARLA cargo-van stack: one van, a rule-based planner, a learned parker, a lidar bay
finder, scripted traffic. Full abstracts were read for 18 candidates; four were read in full.

## The six that change what we do

### 1. ChauffeurNet — "imitate the best, synthesise the worst" (2019, Planning)
**What they did.** Trained a driving model by copying 30 million examples of good human driving and
found copying alone was not enough: the model drifted and hit parked cars. Fix: take the recorded
drives and deliberately knock the car off course in the middle (jitter the position by up to 0.5 m
and the heading by up to 60°), fit a smooth path back, and train on those recoveries too, at a tenth
of the weight of real examples. Add losses that directly punish overlap with other cars and leaving
the road, and "imitation dropout": for half the examples, switch the copying loss off so the safety
losses dominate. In their "nudge around a parked car" test, models without this collided about
half the time; the full model passed 90%.
**Why it matters to us.** That test is our problem. Our brain learned by trial only and picked up the
"swing into the parking lane early" habit; a copied teacher with deliberate off-course starts would
have taught the recovery from day one.
**What we do.** Round 9 records a rule-based teacher's drives *with perturbed starts* (we already
have yaw and lateral offset knobs) and keeps our collision and lane penalties in the fine-tune.

### 2. Imitation Is Not Enough — BC-SAC (2022/23, Planning)
**What they did.** Copying (behaviour cloning) plus reinforcement learning at the same time, on 100k
miles of real driving. The reward is deliberately simple and never positive: distance to the
nearest other car minus a 1 m safety offset, and distance to the road edge minus 1 m, both clipped
at zero. They trained on the hardest 10% of scenarios (a classifier predicts which are hard) and
reported results sliced by difficulty. Result: 38% fewer safety events on the hardest bucket than
copying alone, and the lowest spread of performance across difficulty levels.
**Why it matters to us.** Confirms the round-9 recipe and tells us to keep an imitation term during
the practice phase, not only before it. Their distance-shaped proximity reward is the shape we
already use. Reporting by difficulty slice is what our four exams already do.
**What we do.** In round 9 the practice phase keeps a "stay close to the teacher" term that fades
over training. Our teacher is a program, so it can label any state the learner visits: we run
DAgger-style rounds (learner drives, teacher labels, retrain) before any reward-based practice.

### 3. Zero-shot curricula from a difficulty model (2022, Planning)
**What they did.** Predicted which driving segments are hard, trained mostly on those. 10% of the
data, chosen well, matched training on everything; prioritising hard segments cut collisions 15%.
But over-aggressive focus on hard cases made the model worse at easy ones.
**Why it matters to us.** Our stage ladder and curriculum already do this by hand. Their warning
matches what we found: keep a share of easy attempts (our 25% revision draws and the empty-bay mix).
**What we do.** Keep the ladder; when training on the hardest rung, keep easy cases in the mix.

### 4. Improving Agent Behaviors with RL Fine-tuning (2024, Simulation) — the paper Rajat sent
**What they did.** Pre-train a traffic model by copying recorded drives, then fine-tune it by letting
it drive and scoring it: stay close to what the human did, minus a collision penalty. Collision score
0.67 → 0.88 (1.0 perfect) while staying human-like. A moderate collision weight was best; a large one
made everything worse.
**What we do.** The round-9 order: copy first, then practise. Keep the crash penalty moderate
(ours: 80, was 200).

### 5. Rate-Informed Discovery via Bayesian Adaptive Multifidelity Sampling — BAMS (2024, General ML)
**What they did.** Instead of testing on uniformly random scenarios, steer the test budget toward
regions where the system looks weak, while still estimating the overall failure rate. Found 10× more
issues than random testing with tighter rate estimates.
**Why it matters to us.** Our 1000-scenario scoreboard samples uniformly.
**What we do.** Bucket the scoreboard by difficulty (weather × traffic × parked cars × start
offset) and re-spend runs around buckets that failed. A simple adaptive mode for mission_sweep.

### 6. Pseudo-labelling and offboard 3D auto-labelling (2021, Perception)
**What they did.** A strong, slow "teacher" detector labels unlabeled lidar data offline; a small,
fast "student" trained on those labels beats a model trained on 3–10× more human labels.
**Why it matters to us.** In CARLA the ground truth is a free, perfect offboard teacher. Our
camera/lidar perception still names a car by a rule ("a car-sized blob is a car").
**What we do.** Record lidar clusters during drives with ground-truth labels attached, train a tiny
cluster classifier (vehicle / pedestrian / other), then test with ground truth switched off. This is
exactly "train with the truth, test with the sensors", the idea Rajat proposed on day one.

## Worth keeping in the back pocket
- **Motion-inspired unsupervised perception (2022)** and **Scalable scene flow (2021)**: tell moving
  from parked objects by comparing consecutive lidar sweeps, no labels needed. Warning from the flow
  paper: heavy down-sampling hurts flow (our perception down-samples 3× for clustering; fine for
  clustering, not for flow).
- **CausalAgents (2022)**: test robustness by deleting cars that should not matter. Cheap exam for
  us: a decoy car far from the bay must not change the parking manoeuvre.
- **Waymax, Sim Agents Challenge, Symphony, SceneDiffuser**: reactive traffic for testing planners.
  Our sweep uses scripted traffic; reactive traffic is a later upgrade.
- **Attentional Bottleneck (2020)**: make the model show what it attends to. Our parker's "reason"
  strings (nearest feeler, metres to go) already give that in words.

## Not for us, and why
EMMA, S4-Driver, Scaling Laws, SceneDiffuser++, Drive&Gen, SceneCrafter, Sensor2Sensor, Block-NeRF,
GINA-3D, the 3D-detector architecture papers (SWFormer, PVTransformer, RSN, StarNet, LEF, MoDAR, …),
the motion-forecasting families (MultiPath(++), Wayformer, MotionLM, Scene Transformer, TNT,
VectorNet, StopNet, Occupancy Flow, JFP, MotionDiffuser, KEMP), human-pose and segmentation work,
HDMapGen. They need Waymo-scale data and GPU fleets, or solve problems (forecasting 128 agents, city
simulation, camera-only 3D detection) that a one-van CARLA stack does not have.

## Round 9 plan, in order
1. **Teacher** (`rl/teacher.py`, pure maths, tested): hold the lane until 12 m before the bay or
   until past any car beside the van; then a smooth S-curve into the slot; stop inside. With a car
   right behind: pull past, reverse in. Outputs in the brain's action space (steer, pedal, gear).
2. **Record demos** (`rl/record_demos.py`): teacher drives in the arena from spread starts (16–36 m)
   with perturbed starts (yaw ±10°, lateral ±0.5 m), all hazard types; a few thousand drives.
3. **Copy** (`rl/pretrain_bc.py`): fit the same 64×64 brain to the demos by regression; save in the
   PPO format so everything downstream (eval, RLParker) works unchanged.
4. **DAgger rounds**: learner drives, teacher labels every visited state, retrain; two or three
   rounds to remove drift.
5. **Practise**: PPO with today's rewards plus a fading stay-close-to-teacher term.
6. **Exams**: the four hazard exams plus a decoy-car exam; compare with round 8 and with the teacher.

Expected: hours rather than days, because the brain starts with the lane-hold and the late turn-in
instead of discovering them by crashing.


## Beyond Waymo (added 2026-09-05)

- **Nuro, CIMRL (2024, arXiv 2406.08878).** Imitation gives the motion prior, reinforcement learning improves
  closed-loop behaviour on top of it, and explicit safety constraints keep the RL part from proposing dangerous
  actions; state-of-the-art in closed-loop simulation and on real-world benchmarks. Same shape as our round 9
  (teacher prior, fading pull toward it during practice, the lane-hold and proximity charges plus the van
  software's stop-override as the constraints). Their emphasis on the long tail supports keeping the hard
  hazard cases in every batch.
- **Wayve, GAIA-3 (Dec 2025).** A generative world model used to evaluate and validate the driving AI across
  vehicles and scenarios. Not for us: it needs fleet-scale video. The idea we borrow is the discipline: evaluate
  closed-loop, on scenarios the model did not train on, before trusting a change.
- **Practice notes from the same reading, applied tonight.** Test the instructor in the real simulator before
  recording thousands of drives (first contact: 12/25; the paper model had said 13/13). Measure the real
  vehicle (wheelbase 3.66 m, lock 70°, the reported pose is the box centre 1.8 m ahead of the rear axle) and
  put those numbers in the model instead of guessing. Filter junk starts before they become demonstrations.
