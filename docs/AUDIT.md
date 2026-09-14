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

`train.py` never renders or paces. The brain state moved to an `(n, batch)`
layout (the shape the sparse matmul produces), the membrane update was fused
to 10 full-width passes (from ~19), and the refractory counter is skipped when
`t_ref` rounds to one step. Batch 64 on MPS: brain step 14.3 -> 8.2 ms,
control step (8 substeps + environment) 122 -> 77 ms, 830 body-steps/s; a
3000-step generation takes ~4 minutes with 64 bodies. `torch.inference_mode`
wraps rollouts; the only host sync is a liveness poll every 25 steps.

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
