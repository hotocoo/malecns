# Training-loop audit

Findings and changes for the 24 checkpoints requested for the Monaco / W11
training run. Each item: what was there, what was wrong or missing, what was
done, and the test that pins it. Tests live in `tests/test_audit.py` unless
noted. Measurements are from an Apple M-series GPU (MPS) with the 6.24M-edge
graph, batch 64.

| # | item | status | key change |
|---|------|--------|-----------|
| 1 | reward function | fixed | lap bonus paid once per *new* lap; progress reward is a pure potential; teleport-sized progress jumps zeroed |
| 2 | episode cap | done | curriculum 1500 -> 2000 -> 3000 steps (24 -> 48 s); evaluation default 3000 |
| 3 | collision | improved | body half-width 1.0 m against the clearance field; crash reason recorded |
| 4 | forward/reverse mapping | changed | pedal in [-1, 1]: throttle / brake, no reverse gear; speed clamped >= 0 |
| 5 | steering sign | verified, fixed order | steer > 0 = left (CCW); lidar ray 0 now looks left to match the left-eye visual group |
| 6 | track progress | verified | nearest-centerline index field; wrap-safe; path-independent (test) |
| 7 | wall proximity | improved | 150 m range, 96 samples with power spacing (0.3 m near the body), walls = clearance <= 0 |
| 8 | visual field | improved | proximity + looming (positive proximity change) per ray group; loom gain is an evolved parameter |
| 9 | visual -> neuron wiring | checked | 9,201 LC/LPLC/MeTu projection neurons; no hex coordinates in v1.0, so left/right by soma side, within-eye order fixed but arbitrary |
| 10 | neuron activation health | measured, tooling | brain 7 Hz mean, VPN 29 Hz, DN 13 Hz, no runaway; `diagnose.py` flags silent / saturated / disconnected cells |
| 11 | motor neuron -> vehicle | fixed | readout: DN rates low-passed, population mean removed, random projection to 64 channels; temporal high-pass removed (it erased steady-state steering); init scaled so tanh is not pinned |
| 12 | throttle dynamics | replaced | W11 model: 750 kW power limit, traction limit that grows with downforce, aero drag (CdA 1.6 m^2), 795 kg |
| 13 | steering dynamics | replaced | bicycle model + friction circle: lateral demand served first, braking/traction from the remainder, understeer beyond grip; 80 ms actuator lag |
| 14 | timestep | unified | `src/defaults.py`: 2 ms brain step x 8 substeps = 16 ms control step everywhere; evaluator/viewer read it from the checkpoint (was 5 vs 8) |
| 15 | latency | measured | stimulus switch to half-way motor response: 7 control steps = 112 ms with motor_tau 0.12; now 0.2 (~80 ms), synaptic delay 2 ms |
| 16 | stuck detection | added | < 5 m of net progress over 4 s ends the episode as `stuck` |
| 17 | reward-hacking tests | added | finish-line oscillation, idling, circling, reverse, teleport jump, wall hugging |
| 18 | evolution fitness/selection | fixed | tie-aware rank normalisation (all-equal fitness gives a zero gradient); periodic deterministic evaluation of the mean on every start; `best.pt` |
| 19 | genome diversity | telemetry | fitness std, fraction of parameters at bounds, momentum norm logged per generation |
| 20 | deterministic evaluation | added | seeded device-local generators for sensory noise and perturbations; same seed -> identical episode (test) |
| 21 | curriculum | added | road x1.6 / 1500 steps -> x1.25 / 2000 -> x1.0 / 3000, advancing on rolling mean lap fraction; stage saved in the checkpoint |
| 22 | debug telemetry | added | per generation: laps, steps alive, speed, lateral g, steer, crash/reverse/stuck counts, DN Hz, visual Hz, body-steps/s, eval fields |
| 23 | long-horizon evaluation | added | `evaluate.py`: all 6 starts, 3000+ steps, lap time, endings, JSON table, trajectory plot, run recording |
| 24 | stress tracks | added | mirrored Monaco, 0.8x road width, procedural loops at 1.6x harmonic amplitude, 8 m road loop (`--suite`) |
| 25 | readout reset on resume | fixed | a checkpoint without `agent_cfg` was treated as a readout mismatch and its trained `w_out`/`b_out` thrown away (generation 112 of the first Monaco run collapsed from laps 1.0 to -21); missing metadata is now trusted, `best.pt` is only overwritten by a better evaluation |
| 26 | flat-landscape random walk | fixed | rank normalisation ties members within one step of time tax; sigma grows while a generation is uninformative and shrinks back after (`--sigma-max/--sigma-grow/--sigma-shrink`) |
| 27 | pinned tanh / bang-bang steering | fixed | readout is `g_out * cos(pattern, w_out)`: unit-length channel vector (5 Hz floor), unit readout direction, bounded gain; the motor pre-activation cannot exceed `g_out` whatever the firing level or the readout norm |
| 28 | collision body | fixed | oriented rectangle (nose to tail, both sides) against a bilinearly sampled clearance field; the nose no longer pokes through the barrier at an angle; a crashed car keeps its last legal pose |
| 29 | objective rotating with the start point | fixed | every member scored on all six starts each generation (`run_training.sh` default `STARTS_PER_GEN=6`); one start per generation made the objective swing between +20 and -18 |
| 30 | output population | changed | readout over descending *and* VNC motor neurons (`readout_roles`), completing the brain -> DN -> ventral cord -> muscle path |
| 31 | where did they die | telemetry | per generation: end-of-episode histogram around the lap, fitness/laps/reasons per start, motor pre-activation, grad/mu norms, timestamps; the viewer logs every episode ending to `logs/viewer_episodes.jsonl` |
| 32 | viewer hardcoding | removed | every constant the page used (vehicle, scenery, camera, colours, thresholds) lives in `src/viewer_config.py`, served by `/api/config`, and the page builds its panels from the server's frame schema |
| 33 | flat world, buildings on the track | fixed | real elevation: `fetch_terrain.py` pulls Copernicus EU-DEM (25 m) around any `--geojson` circuit; `terrain.py` derives the road profile (surveyed surface smoothed over 30 m, tunnel as a straight grade between portals probed 40 m outside each mouth, quay floor 1.2 m), a ground heightfield carved under the lowest nearby road and standing on the tunnel roof, building bases from the land under each footprint (over the tunnel: the roof). The Fairmont / Auditorium block no longer sits on the road inside the tunnel; portal headwalls close the hill face |
| 34 | fixed-horizon episodes | changed | no step cap by default (`--episode-steps -1`): episodes end by crash, reverse, stuck (pace floor 3 m/s) or `finished` (`CarConfig.max_laps`, one lap); `defaults.EPISODE_HARD_CAP` is a safety ceiling; `evaluate.py --steps 0` likewise |
| 35 | viewer episode list / fly panel | fixed | `/api/training` always carries the viewer's own episode endings whichever trainer log is inspected; the first-person fly panel is drawn every frame (it was only redrawn on resize) |
| 36 | flat physics on a hill | fixed | `Track.height`/`Track.grade` from the same survey the viewer draws (`terrain.road_profile_for_circuit`); `_drive` adds `-g * sin(grade)` along the road (`CarConfig.road_grade`); Monaco climbs 51 m, steepest 16 % |
| 37 | invented building heights | fixed | `fetch_heights.py`: OSM -> Overture Maps (OSM + Microsoft ML) -> median of surveyed neighbours, `height_source` recorded; the page no longer jitters storeys per footprint |
| 38 | dead bodies on the GPU | fixed | `train.rollout` drops finished bodies from the brain, the environment and the exploit monitor every poll (`COMPACT_BELOW` 0.75): membrane/current/adaptation gathered per body, the spike ring re-packed bit by bit, results scattered back in body order; a crashed car no longer costs a kernel word for the rest of the generation (`tests/test_compaction.py`: bit-identical to the uncompacted run) |
| 39 | zero generations per hour | fixed | uncapped 3,072-body generations took 40 min (one slow survivor kept the whole batch stepping); `run_training.sh` now caps at 6,000 steps (96 s) with 64 x 4 x 6 bodies, and `Brain.reset` no longer allocates 4.6 GB of dense spike history on the Metal path |
| 40 | one ES step destroyed the calibrated driver | fixed | generation 1 moved the mean by 26 % of its norm (grad norm 37, lr 0.05, sigma 0.05): 1.3 laps -> 8 km/h crawl. `--max-step-frac` trust region (3 % per generation), sigma 0.02 / max 0.06, lr 0.02, and the deterministic mean evaluation with revert guard runs every generation (`EVAL_EVERY=1`) |
| 41 | honest driving flagged as CRITICAL exploit | fixed | the monitor's expected reward omitted the low-speed steering term (`stall`) and the finish step: 2,205 of 3,072 cars "accounting" exploiters, generation update discarded |
| 42 | viewer died mid-run | fixed | `CarEnv.reset(mask)` wrote in place into inference tensors (`RuntimeError: Inplace update to inference tensor outside InferenceMode`) whenever a non-followed car ended; masked reset is functional now. `/api/track` read GPU tensors from HTTP threads (Metal command-buffer assertion); host copies are taken on the simulation thread. The simulation thread is supervised and restarts the episode on any exception |
| 43 | laggy, bouncing car | fixed | EventSource has no backpressure: the page fell seconds behind the stream; frames are coalesced to the display rate. Road height was the nearest centreline sample (stair steps forty times a second): interpolated along the segment, cars pitch with the grade, poses dead-reckon at the measured simulation rate between frames. Side panels stack label/canvas/legend in a column (the fleet rows and raster legend were clipped) |
| 44 | every evaluation crashed, training stuck at generation 0 | fixed | `evaluate_mean` built its environment from `agent.brain.batch`, which after a compacted generation was a leftover (859) while the expanded island means had `4 x (859 // 4) = 856` rows: `RuntimeError: The size of tensor a (859) must match the size of tensor b (856)` in `agent.act` on every generation's evaluation since compaction landed, the supervisor restarted from the same `es.pt`, nothing was ever saved (`logs/supervisor.log`). Evaluation now runs one body per (island, start) via `make_eval_env` and `rollout` resets the brain to the environment's size (`Brain.reset(batch)`); `tests/test_rollout_stub.py` drives 384 bodies through compaction and then a 24-body rollout |
| 45 | evaluation cost a whole generation | fixed | the full-batch evaluation drove 64 identical copies of every (island, start) pair under shared noise: 9,216,000 body-steps per evaluation, as much as the generation itself. 24 bodies now (144,000 body-steps, 64x fewer); `EVAL_EVERY=1` no longer halves the generations per hour |
| 46 | dying paid | fixed | the time tax was charged only while alive: a car holding 36 km/h for 12,000 steps scored -75, one crashing on step 400 scored -28, so the rank-based ES was paid for early crashes below 45 km/h (`bench_reward.py` B: crasher -20.98 vs survivor -40.60 under the old rule). `CarConfig.episode_steps` carries the step budget into the environment (`make_env`, `make_eval_env`, `evaluate.py`, the viewer via the checkpoint's `car_cfg`); an ended car is charged as standing still for the unused steps (time tax + full pace + alignment penalty per step, `crash_cost` in `last_terms`, mirrored by the exploit monitor). Now crasher -287 vs survivor -150 |
| 47 | progress paid in 1.6 m lumps | fixed | `Track.progress_at` returned the nearest centerline sample: 86 % of control steps at 54 km/h paid nothing and the rest paid 0.17, which also read as `idle_reward` to the monitor (147 cars flagged). Nearest sample plus the signed offset along its tangent: every moving step pays, same lap total (`tests/test_reward_shaping.py`) |
| 48 | slow on straights cost nothing | added | `Track.speed_ref` (`speed_profile`): the fastest this vehicle's grip circle, power, drag and brakes can pass each centerline sample (cornering speed closed form, backward braking pass, forward traction pass, road grade included). Below `pace_margin` (0.9) of it a quadratic deficit costs up to `pace_penalty` (0.04) per step; zero at pace. Monaco reference lap from the model: 1:25.9 on the centerline (real pole 1:10.2; the racing line is wider). Viewer gauges `pace`, `speed_ref`, `reward_pace` |
| 49 | steering paid as a fixed fee | added | `align_penalty` (0.03) x (1 - cos(heading error to a look-ahead point max(15 m, 1 s of travel) ahead)): graded, zero when pointed down the road, twice the penalty facing backwards, so steering 0.8 of what a corner needs is paid between full and none. Viewer gauges `align_deg`, `reward_align` |
| 50 | wall margin kept the apex out of reach | changed | `wall_margin` 1.5 m -> 0.75 m from the body edge: an 11 m road left 4.5 m of clearance at the centre and the old margin taxed the outer 3 m of usable road; 0.75 m still gives a gradient before the crash cliff |
| 51 | looming channel was always zero in training | fixed | `rollout` called `agent.sensory_rates` a second time per step for telemetry, which overwrote `prev_proximity` with the post-step view; the next `act` then saw zero change. Telemetry reads `agent.last_rates_hz` from the step's own `act` |
| 52 | ended cars kept moving | fixed | a crashed/stuck car's pose, speed and lap count went on changing (and its `done_reason` could flip back to alive when it bounced onto the road); `CarEnv.step` freezes ended cars (reward 0, done stays set) until `reset(mask)` |
| 53 | two trainers on one checkpoint | operational | two `run_training.sh` supervisors (PIDs 48329 and 48461 at the time) ran two trainers writing the same `checkpoints/es.pt` and `logs/train.jsonl` and sharing one GPU; kill one before restarting |

## Details

### 1. Reward

Before: `completed = floor(laps + delta) - floor(laps)`, clamped at zero. A car
crossing the finish line forwards, reversing over it, and crossing again earned
the lap bonus every time. Now the bonus is paid only when `floor(laps)` exceeds
the best lap count the car has ever reached (`best_laps`).

Progress reward is `delta_progress * progress_per_m * lap_length`, so the sum
along any path equals `progress_per_m` times net metres advanced; it cannot be
farmed by oscillation. Per-step `delta` larger than three steps of top-speed
travel is a teleport (the nearest centerline point snapped to another section,
which happens on the step a car leaves the road where two sections of the
circuit run close together) and counts as zero.

Balance at full scale: 1 lap = 329 reward + 100 bonus; time tax 0.02/step
(60 over a 3000-step episode) so slow driving nets negative; wall term ramps
inside 1.5 m of the body edge (max 0.03/step); crash -20 once, and dead cars
earn nothing further (`rollout` masks them).

### 3, 7. Collision and lidar

The rasterised `clearance` field (metres to the nearest road edge, negative off
road) is the single source of truth: a car crashes when `clearance - 1.0 m <=
0`; lidar rays hit where `clearance <= 0`. Cells are 0.54 m on the 2048^2 grid
covering the 1.1 km bounding box. Top speed covers 1.5 m per 16 ms step, well
under the 11 m road, so a step cannot tunnel through a barrier.

### 5, 9. Ray order and eye order

`car_env` rays were ordered right to left (angle -fov/2 first) while the agent
ordered its visual groups left eye first, so ray 0 (car's right) drove the
left-eye group. Harmless for the optimiser but wrong for interpretation and for
the viewer's mapping; rays now run left to right. The viewer's lidar drawing
was flipped to match.

### 11. Readout

`baseline_tau = 0.02` subtracted a slow running mean per descending neuron:
a temporal high-pass with a 0.8 s time constant. Measured with a constant
left-wall stimulus the projected common signal decayed from 2.0 to 0.39 over
2 s, i.e. the controller could not hold a steady steering offset through a
constant-radius corner. Left-vs-right separation of the raw rates was twice as
large and stable. Replaced with population-mean subtraction (instantaneous
common-mode removal). Projected channels have magnitude ~3 at scale 50; with
`w_out` initialised at 0.5 over 64 channels the motor pre-activation had std
~12 and `tanh` was pinned from generation 0 (a flat landscape, then drift,
which is the saturation failure seen at generation ~2100). Now scale 16 and
init 0.1: motor std ~0.8.

### 12, 13. Vehicle (Mercedes-AMG F1 W11)

Public figures used: 795 kg with fuel, ~750 kW, 3.70 m wheelbase, 2.0 m width,
drag area 1.6 m^2 in Monaco trim, mechanical grip 1.8 g rising with
downforce to a 4.5 g cap, launch traction 1.15 g. Model checks: 0-100 km/h in
2.6 s, 0-200 in 5.8 s, top speed 326 km/h, 300 -> 100 km/h braking in 81 m,
full-lock radius 10 m at 50 km/h (the Fairmont hairpin centerline is 8.5 m
after 8 m of smoothing; the road is 11 m wide). The friction circle serves
lateral demand first and gives braking/traction the remainder, so the car must
slow for corners and understeers rather than spins when it does not.

### 14, 15. Timestep and latency

`train.py` used 8 substeps, `evaluate.py` 5: a policy trained at a 16 ms
control period was evaluated at 10 ms. Both, and the viewer, now share
`defaults.py`, and the evaluator/viewer take `dt_ms` and `substeps` from the
checkpoint. Measured latency: 7 control steps (112 ms) from a stimulus switch
to half-way motor response, set by `motor_tau`; now 0.2 (about 80 ms, in the
range of fly visuomotor latency).

### 18, 19. Evolution

`argsort` ranks on tied fitness are an arbitrary permutation: with every
member scoring the same (all crashed at the same step, or all idle), the
"gradient" was full-size noise, and with momentum 0.9 it walked the mean into
saturation. Ranks are now averaged within ties, so all-equal fitness gives a
zero update. Sampling is seeded per generation from `--seed`. Every
`--eval-every` (10) generations the mean is run deterministically on all
start points; the best such score is saved as `checkpoints/best.pt`, so the
viewer and evaluator can use a checkpoint that was selected rather than the
latest noisy mean. `--starts-per-gen K` widens the brain batch to
`popsize x K` and scores every member on K start points at once.

### 21. Curriculum

Stage 0: road x1.6 (17.6 m), 1500 steps. Stage 1: x1.25, 2000 steps. Stage 2:
x1.0 (11 m), 3000 steps. Advance when the rolling mean lap fraction over 25
generations passes 0.12 then 0.20. The stage is checkpointed;
`viewer.py --follow-curriculum` draws the stage's road width.

### Throughput (headless training)

`train.py` never renders or paces. History: torch path 830 body-steps/s at
batch 64; first Metal kernel ~2,600 at batch 128; current kernel 10,300 at
batch 128 and 12,000 at 256 (M4 Max, realistic activity). Where the time
went, measured by ablation at batch 128: the original kernel's 5.3-6 ms per
substep was not the state update (2.3 ms fp32, 1.3 ms fp16) but the synapse
gather (4.1 ms), and of that almost all was a few hundred hub neurons with
over 1,024 inputs each walked serially by one thread (rows over 1,024 inputs
alone: 3.35 ms; rows up to 64 inputs: 0.36 ms). Fixes and their effect: SIMD
group per hub row with unrolled accumulators (gather 4.1 -> 1.2 ms), rows
sorted by in-degree (-> 0.8 ms), fp16 state (membrane 2.3 -> 1.3 ms), exact
skip of all-zero neuron-words (~25% of the network under drive), sensory
draws and output-spike counting inside the kernel (-6 ms of torch ops per
control step). The environment step is 1.5 ms. `torch.inference_mode` wraps
rollouts; the only host sync is a liveness poll every 25 steps.

### Diagnostics

`src/diagnose.py` runs 16 controlled scenarios as parallel bodies with shared
Poisson draws (open road; wall left/right at 5/15/40/100 m; wall ahead at
20/60/120 m; braking; accelerating; left and right bends) and reports role
rates, left/right eye lateralisation, per-DN directional sensitivity
(wall-left minus wall-right), silent / saturated / structurally disconnected
neurons, the steering and pedal commands per scenario with PASS/WARN verdicts,
and on a Monaco run the correlation of every DN with steering and pedal plus
the avoidance gain (steer versus left-right proximity). On the connectome
alone (untrained readout) 413 of 1,314 DNs already differ by > 2 Hz between a
wall 5 m to the left and 5 m to the right, so the lateral information reaches
the descending population; whether the readout uses it is what the verdicts
track over training.

### Exploit detector

`src/exploits.py` runs inside every training generation and every evaluation
(`ExploitMonitor`), with on-device accumulators and no host sync until the
report. Per car it tests fourteen signatures: reward above the distance bound
(`net_progress x progress_per_m x lap_length + laps x bonus`), lap-bonus
farming, back-and-forth oscillation (net/gross progress), progress-field
teleports, reverse driving, spinning, wall riding, wall phasing (alive while
the body is off the road, or a step whose straight path crosses off-road
cells), position jumps, speed violations, positive reward while stationary,
reward-accounting mismatches against the per-term breakdown the environment
exposes, early lap bonuses, and creeping just above the stuck threshold. Counts
and a severity land in each `train.jsonl` record under `exploits`, the trainer
prints a line when anything fires, and `evaluate.py` prints a per-scenario
line. `python3 src/exploits.py --log logs/train.jsonl` scans a whole run for
population-level signatures (fitness rising without lap fraction, parameters
piling onto bounds, survival without driving, reverse laps).

The viewer's barrier face now stands exactly at the physics wall (the road
edge where lidar rays stop and where a body edge touching it is a crash), and
the kerb is the outer 1.2 m of the road, so nothing the physics allows looks
like passing through a wall.

## Open items

- Within-eye retinotopy: v1.0 annotations carry no optic-lobe hex coordinates
  for the projection neurons, so the 4-5 groups per eye are body-id ordered.
- The 3D car body is the three.js Ferrari 458 scaled to W11 length; the
  dynamics are the W11. Drop a W11 glTF in `web/assets/` and point
  `drive.js` at it for the matching body.
- Water in the harbour: the OSM coastline is fetched but not yet turned into a
  sea polygon; quays and breakwaters are rendered.


### Throughput vs population (2026-09-15)

Brain-only Metal step on the M4 Max (viewer sharing the GPU), 9,000 driven
cells, 40 substeps: batch 256: 0.87 ms/substep (36,800 body-steps/s); 1,024:
2.16 ms (59,300); 3,072: 5.87 ms (65,400); 6,144: 10.2 ms (75,300); 12,288:
23.1 ms (66,400). The kernel is memory-bound past ~3k bodies: doubling the
`run_training.sh` default (4 islands x 128 x 6 starts = 3,072) to 6,144 buys
about 15 %, beyond that nothing. Uncapped episodes make a generation last as
long as its best car (a clean lap at 60 km/h is ~12,500 steps), which is why
the pace floor and the one-lap finish exist.

### Metal kernel (2026-09-14)

`brain.py` on MPS now runs `metal_lif.py`: one fused kernel per step with
bit-packed spikes (see README "Speed"). Zero spike mismatches against the torch
reference over 60 steps at batch 40 (`tests/test_metal.py`); |du| <= 6e-5 mV.
Batch 64, M4 Max, measured with another trainer sharing the GPU: brain step
7.61 -> 1.99 ms, `agent.act` (8 substeps) 69.5 -> 18.3 ms; batch 1 (viewer)
0.63 ms per brain step. Long-row threshold 64 chosen by sweep (16: 1.86 ms,
32: 1.93, 64: 1.36, 128: 1.41, 256: 1.86).

### Checkpoint migration

`ConnectomeAgent.migrate_state` maps a saved `mu`/`momentum` onto the current
`param_shapes` by block name; checkpoints without `param_shapes` use the known
141-parameter legacy layout (no `loom_gain`). Trainer, evaluator and viewer all
load through it, so a layout change keeps every trained block.

### Viewer wiring fixes

* Steer gauge drew positive (left) steer to the right; now grows leftwards with
  an L/R readout. Minimap, 3D car yaw and lidar were already consistent with
  `car_env` (steer > 0 = counter-clockwise).
* Viewer defaults to `--follow-curriculum`: it drove on the 11 m road while the
  trainer was on the x1.6 stage, so the same policy crashed far more on screen.
* Checkpoint note and DN ranking refresh when the generation changes; episode
  end reason (CRASH / REVERSE / STUCK) is shown through the reset; curriculum
  stage boundaries and deterministic evaluations are drawn on the curve.
* Harbour water: OSM coastline (water on its right-hand side) rasterised to
  water/land rectangles (`water_and_land`), quay walls along the coastline,
  cells within road half-width + 2 m forced to land (nearest water 8 m from
  the Monaco centerline).
33,eye encoding,"lidar was `1 - d/150`: a 4 m vs 7 m wall offset was invisible to the descending neurons (population d' 0.6); now road-relative per ray (`eye_encoding=\"road\"`: log distance against what the ray sees from the middle of an 11 m road, 0.4/octave), same offset d' 33; front distance decoded at R2 0.95 either way",fixed
34,readout calibration,"`calibrate.py`: a linear lidar teacher (`teacher.py`, `data/teacher_linear_robust.json`, searched with a 6-step actuation delay and filtered command noise) drives while the brain watches; ridge from the 2,129 output cells to steer/pedal becomes projection channels 0/1 (`readout_norm=\"channel\"`, stats stored under `readout` in the checkpoint); 3 DAgger rounds; held-out R2 steer 0.66-0.79, pedal 0.69-0.89",added
35,first lap,"brain alone on full-scale Monaco, 16000 steps (256 s): 12/12 cars > 1.4 laps, zero crashes, 65 km/h mean (`evaluate.py --steps 16000`); the earlier ES run reached 0.087 laps in 795 generations",done
36,episode length,"curriculum 2000 -> 3000 -> 6000 steps (a lap takes 9,000-14,000 at learning pace); lap completion is measured by `evaluate.py --steps 16000`",changed
37,first-person view,"viewer: `web/firstperson.js` draws what each eye group is told (wall column per ray, brightness = road-relative proximity, white = looming), horizon roll from lateral g, pitch from the pedal, barrier glow, and bars for speed sense, output-population rate, motor intent, pedals, reward; frame carries `proximity`, `loom`, `dn_hz`",added
38,viewer fleet,"`viewer.py --cars N --follow k`: N cars (one brain body each) spread round the lap, car k driven by ES island k mod islands; every car on the 3D circuit and minimap, a per-car brain activity strip (`fleet-canvas`, click a row to follow), `/api/follow?car=k`",added
39,driver and onboard camera,"`web/flyrig.js`: the fly seated in the followed car (head yaws into the turn, body rolls with lateral g, wings beat with throttle, legs on wheel and pedals, eyes glow with visual drive); `cockpit view` button = roll-hoop camera with the car in view; dims in DriveStyleConfig",added
40,checkpoint interface safety,"train.py builds the agent from the checkpoint's `agent_cfg` and refuses to run if a calibrated readout would be reset; viewer rebuilds its agent when the checkpoint interface changes (both had silently reset the readout and overwritten/shown a non-driving policy)",fixed
41,verified-improvement guard,"ES mean evaluated every `--eval-every` generations; a drop of more than `--eval-tolerance` below the last accepted evaluation reverts mu, clears momentum and shrinks sigma; best.pt promoted on lap time once every start completes a lap (`best_lap_steps`)",added
42,lap proof,"`verify_laps.py` (16,000 steps, every start, exit 0 only if all complete with no crash) appends to logs/laps.jsonl; `run_verify_loop.sh` re-verifies best.pt on change and es.pt periodically",added
43,exploit veto,"only CRITICAL exploit severity vetoes an ES update; HIGH `idle_reward` (slow crawl after a spin) was discarding whole generations and widening sigma",changed
44,reward configurability,"`train.py --car FIELD=VALUE` overrides any CarConfig field (reward terms, vehicle, sensing); `speed_penalty` default 0 (it charged a fast straight as much as the time tax); accounting exploit check verifies reward = sum of logged terms every step",changed
45,map audit,"dataset centerline (f1-circuits, 160 GPS points, 8 m smoothing) vs OSM streets: median offset 0.6 m; deviations only where the F1 layout leaves the streets (Nouvelle Chicane on the quay, Piscine); no self-overlap (13.5 m closest approach at the hairpin legs), raster edge exact at +-5.5 m, progress monotonic; OSM has 2 lanes and no width tags everywhere on the circuit so no per-segment width data exists (`logs/map_check.png`)",verified
