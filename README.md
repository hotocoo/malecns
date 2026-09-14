# malecns: a fly connectome drives Monaco

The complete male *Drosophila* central nervous system (MaleCNS v1.0, HHMI
Janelia FlyEM, CC-BY 4.0) downloaded, turned into a signed spiking network, and
wired into a Formula 1 car it has to keep on the Circuit de Monaco. Training
runs headless and continuously, resumes from checkpoints, and can be watched
live in the browser, neuron by neuron, from a separate process.

- 166,700 annotated neurons, 6.24M connections (>= 5 synapses), 89.9M synapses
- leaky integrate-and-fire dynamics with the Shiu et al. 2024 parameter set
- real neurotransmitter identity sets every connection's sign
- vision enters at the visual projection neurons; control is read from the
  1,314 descending neurons and the 815 ventral-cord motor neurons they drive,
  the whole brain -> descending -> VNC -> muscle path
- the circuit is the real Monaco centerline at full scale (3.29 km after
  corner smoothing, 11 m road), the car a Mercedes-AMG F1 W11 model
- the training loop was audited item by item; see [docs/AUDIT.md](docs/AUDIT.md)

Read [docs/RESEARCH.md](docs/RESEARCH.md) for the dataset background and the
literature this follows.

## Quick start

```bash
pip install -r requirements.txt
python3 src/download.py                          # ~1.15 GB from storage.googleapis.com
python3 src/build_graph.py --min-weight 5 --out data/graph_w5
python3 src/fetch_scenery.py                     # optional: OSM buildings/tunnel for the viewer
./run_training.sh                                # headless ES training, resumable
./run_viewer.sh                                  # live view, http://127.0.0.1:8765
python3 src/evaluate.py --suite --plot run.png   # long-horizon + stress tracks
python3 src/diagnose.py --plot logs/diag.png     # does vision produce avoidance?
python3 src/exploits.py --log logs/train.jsonl   # reward-hacking scan of a run
python3 -m pytest tests -q                       # 58 tests
```

## Watching it

`./run_viewer.sh` serves a page that shows the current checkpoint driving in
real time while training keeps running elsewhere:

- chase-camera 3D view of the circuit with the real Monaco buildings
  (OpenStreetMap footprints and heights), the Boulevard Louis II tunnel where
  the road is mapped underground, triple Armco barriers with posts, the nine
  lidar rays, and a top-down minimap
- the whole brain as a rotatable point cloud, one point per neuron at its
  measured soma position, lit as it spikes
- a spike raster over a 600-neuron sample banded by role, population firing
  rates, the descending neurons with the most steering influence, and the
  training curve

The trainer writes `checkpoints/es.pt` (latest mean) and `checkpoints/best.pt`
(best deterministic evaluation); the viewer re-reads its checkpoint whenever it
changes. Playback is paced to real time; `--speed 0` runs uncapped.

For watching without costing the trainer any GPU, record a run and replay it:

```bash
python3 src/evaluate.py --checkpoint checkpoints/best.pt --record logs/run.npz
python3 src/viewer.py --replay logs/run.npz
```

## Speed

On Apple GPUs the brain step is one fused Metal kernel (`src/metal_lif.py`,
via `torch.mps.compile_shader`). Every one of the 166,700 neurons and 6.24M
connections is stepped every 2 ms; what changed is only how:

- spikes are bit masks (one bit per body) and a 21 KB per-neuron "any body
  spiked" bitmap lets the synapse loop skip silent presynaptic cells;
- one thread owns one (neuron, 32-body word) and walks the row's synapses
  once for all 32 bodies; hub neurons (over 256 inputs, up to 6,660) get a
  whole SIMD group striding over their synapses; rows are sorted by in-degree
  so the 32 threads of a group finish together; the accumulator loops are
  fully unrolled (a dynamically indexed register array spilled to memory and
  cost more than the synapses);
- state is stored as fp16 by default (`--precision fp32` for the bit-exact
  reference), the refractory array is gone at a one-step refractory period
  (the previous spike bit is the flag), and a neuron-word whose 32 bodies sit
  at exactly zero state with no input this step is skipped (exact: zero in,
  zero out);
- the sensory Poisson kicks are drawn inside the kernel from a counter hash
  of (seed, substep, cell), and output-neuron spikes are counted into a
  small buffer as they fire, so no random tensors or spike gathers cross the
  bus per substep.

The torch path in `brain.py` is the reference and is used on CPU/CUDA, when a
per-neuron gain is set, or with `MALECNS_NO_METAL=1`; `tests/test_metal.py`
checks the fp32 kernel reproduces it spike for spike (0 of 853M differ at
batch 128) and the fp16 kernel statistically.

Measured on an M4 Max, population 128, realistic activity: brain substep
5.3-6.0 -> 1.3-1.7 ms, control step (8 substeps + car) 82 -> 12.4 ms, 1,560 ->
10,300 body-steps/s (12,000 at 256 bodies). The remaining cost is the
memory traffic of the fp16 state itself. Training is never paced or
rendered; only the viewer sleeps to real time (`--speed 0` runs it uncapped
too).

Checkpoints from older parameter layouts (141 parameters, before the looming
channel) are migrated block by block on load: trained blocks are kept, new
blocks start from their initial values, so a layout change never restarts
training from scratch.

## How it works

```
lidar rays (9, 180 deg, 150 m) ──► proximity + looming ──► Poisson spike kicks
                                                                │
                                                 visual projection neurons (9,201)
                                                                │
                                                     MaleCNS connectome
                                              166,700 LIF neurons, signed by
                                            predicted neurotransmitter identity
                                                                │
                              descending neurons (1,314) ──► steering, throttle/brake
```

The connectome is never rewired. Evolution strategies searches only the
parameters a fly would get from development and neuromodulation:

| parameter | size | meaning |
|---|---|---|
| `ray_gain` | 9 | drive strength per visual direction |
| `loom_gain` | 1 | extra drive from an expanding edge (proximity increasing) |
| `bias_hz` | 1 | tonic drive on the visual sheet |
| `speed_gain` | 1 | proprioceptive speed drive to ascending neurons |
| `w_out` | 64 x 2 | readout *direction* over a fixed random mix of output-neuron rates |
| `g_out` | 2 | readout gain: the motor pre-activation lies within +-`g_out` |
| `b_out` | 2 | readout bias |
| `dn_gain` | 1,314 | per-DN excitability, off by default (`learn_dn_gain`) |

144 parameters against 6.24M fixed synaptic weights. Output-neuron rates
(descending + motor) are low-pass filtered, the population mean is removed, a
fixed random projection mixes them into 64 channels, the channel vector is
scaled to unit length (with a floor at 5 Hz of activity so silence is not
amplified), and `g_out * cos(pattern, w_out) + b_out` goes through `tanh`,
which gives steering (positive = left)
and the pedal (positive = throttle, negative = brake).

Gradient descent is not an option: spikes are non-differentiable and the
substrate is fixed, so antithetic ES with tie-aware rank normalisation is
used. Each generation runs one episode for the whole population in parallel on
one batched brain, on all six start points around the lap at once (`--starts-per-gen`).
Members whose fitness differs by less than one step of time tax are tied, and
the perturbation scale widens while a generation shows no differences at all
and shrinks back once it does, so a flat landscape never becomes a random
walk. A
curriculum starts on a 1.6x wide road with 1500-step episodes and tightens to
the real 11 m and 3000 steps as the mean lap fraction improves. Every ten
generations the mean is evaluated deterministically on all starts and the best
result is kept.

### Vehicle

A single-track model with a friction-circle grip envelope, parameterised from
public W11 figures: 795 kg, ~750 kW, 3.70 m wheelbase, 2.0 m wide, drag area
1.6 m^2, grip 1.8 g rising with downforce to 4.5 g, 80 ms steering lag.
Checks: 0-100 km/h 2.6 s, top speed 326 km/h, full lock 10 m radius at
50 km/h. Lateral demand is served first; braking and traction get what the
grip circle has left, so the car must slow for corners.

### Episode ends

Leaving the road (body half-width 1.0 m), net reverse progress, or less than
5 m of progress in 4 s (stuck). The lap bonus is paid once per newly completed
lap; progress reward is a potential, so nothing is earned by oscillating. An
exploit detector runs in every generation and flags reward above the distance
bound, bonus farming, oscillation, teleports, wall phasing, spinning and more
(see [docs/AUDIT.md](docs/AUDIT.md)).

## Layout

```
src/defaults.py        timestep constants shared by every entry point
src/download.py        fetch the three flat-connectome files
src/build_graph.py     annotations + neurotransmitters + weights -> signed graph
src/brain.py           batched LIF over the sparse connectome, (n, batch) layout
src/car_env.py         batched W11 driving environment on rasterised tracks
src/agent.py           lidar -> visual neurons -> descending neurons -> controls
src/train.py           headless ES loop: curriculum, eval, telemetry, checkpoints
src/evaluate.py        long-horizon runs, all starts, stress suite, recording
src/diagnose.py        controlled-stimulus neural diagnostics with verdicts
src/exploits.py        exploit detector: per-car signatures in every rollout
src/fetch_scenery.py   OpenStreetMap buildings, tunnels, coastline (Overpass)
src/build_positions.py soma coordinates -> data/graph_w5/positions.npy
src/viewer.py          live simulation or replay, SSE stream, static server
web/                   drive.js (3D circuit), scenery.js (buildings, tunnel,
                       barriers), brain.js (3D brain), app.js (page)
data/tracks/           monaco.geojson centerline, monaco_scenery.geojson (OSM)
logs/train.jsonl       one JSON record per generation
```

## Tuning

| flag | default | note |
|---|---|---|
| `--popsize` | 64 | ES population, also the parallel body count |
| `--starts-per-gen` | 1 | start points per member per generation (batch = popsize x this) |
| `--episode-steps` | 0 | 0 follows the curriculum (1500 -> 3000) |
| `--eval-every` | 10 | deterministic evaluation of the mean; writes `best.pt` |
| `--seed` | 0 | perturbations and sensory noise |
| `--weight-scale` | 0.15 | below ~0.25 the network stays out of runaway |
| `--adapt-mv` | 0.6 | spike-frequency adaptation; 0 reproduces the paper model |
| `--dt-ms` / `--substeps` | 2.0 / 8 | 16 ms control step; stored in the checkpoint |
| `--layout` | monaco | `loop` for procedural circuits; `--geojson` for any other centerline |

Throughput on an Apple M-series GPU, batch 64: 77 ms per control step
(8 brain steps + environment), 830 body-steps/s, about 4 minutes per
3000-step generation. CUDA and CPU also work (`--device`).

## Roadmap

The environment takes any GeoJSON centerline (`--geojson`), and the scenery
fetcher any circuit's bounding box; real roads (open, not looped) need a
non-periodic progress field and are the next step.

## Viewer assets

Vendored under `web/vendor` and `web/assets`, served locally, never fetched at
page load:

- three.js r170, MIT
- Ferrari 458 Italia model from the three.js examples, by vicent091036,
  CC-BY, scaled to the W11's 5.7 m as a stand-in body
- `asphalt_02`, `aerial_grass_rock` textures and the
  `kloofendal_48d_partly_cloudy_puresky` HDRI from Poly Haven, CC0
- buildings, tunnel, quays: OpenStreetMap contributors, ODbL 1.0
  (`data/tracks/monaco_scenery.geojson`, regenerate with `src/fetch_scenery.py`)
- Monaco centerline: `data/tracks/monaco.geojson` (circuit id `mc-1929`)

## Data and licence

Connectome data: MaleCNS v1.0, HHMI Janelia FlyEM with the University of
Cambridge, MRC LMB and Google Research, CC-BY 4.0. Nothing in `data/raw` or
`data/graph*` is redistributed here; `src/download.py` fetches it from the
official bucket.
