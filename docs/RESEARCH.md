# Research notes: MaleCNS connectome and connectome-driven control

## 1. What the dataset is

The MaleCNS connectome (v1.0) is a synapse-resolution wiring diagram of the
complete central nervous system (brain plus ventral nerve cord) of a single
adult male *Drosophila melanogaster*. It was produced by FlyEM at HHMI Janelia
with the University of Cambridge Department of Zoology, the MRC Laboratory of
Molecular Biology and Google Research, and is released under CC-BY 4.0.

It is the first *whole-CNS* fly connectome: the earlier hemibrain covered part
of one brain hemisphere, FlyWire/FAFB covered a whole female brain but not the
nerve cord, and MANC covered a male nerve cord but not the brain. MaleCNS joins
brain and cord in one animal, so sensory input, central processing, descending
commands and motor neurons exist in a single connected graph. That is exactly
what a closed-loop control experiment needs.

Scale as built in this project (see `src/build_graph.py` output):

| quantity | full graph | pruned graph (>= 5 synapses) |
|---|---|---|
| neurons with annotations | 166,700 | 166,700 |
| directed connections | 25,582,938 | 6,242,118 |
| synapses | 124,177,616 | 89,860,280 |

Functional groups exposed as `roles.json`:

| role | superclass in annotations | count |
|---|---|---|
| photoreceptor | `ol_sensory` | 6,098 |
| visual_projection | `visual_projection` | 9,201 |
| descending | `descending_neuron` | 1,314 |
| motor | `vnc_motor`, `cb_motor` | 815 |
| ascending | `ascending_neuron`, `sensory_ascending` | 2,383 |
| mechanosensory | `vnc_sensory` | 6,370 |

Neurotransmitter composition of the modelled neurons: acetylcholine 62.2%,
glutamate 17.6%, GABA 13.2%, histamine 4.7%, dopamine 0.2%, octopamine 0.1%,
remainder unclear or unpredicted.

## 2. Which files are worth downloading

The bulk download is mostly synapse point clouds that a network model does not
need. Only three files are fetched here (about 1.15 GB total):

- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (13 MB): cell type,
  class, superclass, side, optic-lobe hex coordinates.
- `body-neurotransmitters-male-cns-v1.0.feather` (42 MB): per-neuron consensus
  neurotransmitter prediction.
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1.1 GB): `body_pre`,
  `body_post`, `weight` (synapse count) for every connection.

Skipped: `syn-points` (12.7 GB), `syn-partners` (6.8 GB),
`tbar-neurotransmitters` (2.7 GB), `body-stats` (780 MB), the SWC/precomputed
skeletons and the full neo4j database. They matter for morphology or
per-synapse work, not for a rate/spiking network model.

## 3. Prior art this build follows

- **Shiu, Sterne, Spiller et al. 2024, *Nature* 634:210-219** — the reference
  leaky integrate-and-fire model of the whole fly brain built directly from
  connectivity plus predicted neurotransmitter identity, with no fitting. It
  reproduced feeding and grooming circuits and predicted which neurons drive
  the grooming descending neurons aDN1/aDN2. Implemented in Brian2.
- **Lin et al. 2025 (bioRxiv), CPG circuit for fly walking** — restates the
  exact parameter set and applies the same untuned model to the nerve cord;
  a descending-neuron activation screen recovered DNg100 as the top driver of
  rhythmic leg motor output.
- **Axo-axonic synapses on descending neurons (bioRxiv 2025)** — catalogues
  inputs onto all 1,314 DNs and shows the DN network splits into thirteen
  functional communities; useful context for treating DNs as the motor
  bottleneck.
- **flybrain (github.com/TheMrRaGe/flybrain)** — the closest existing embodied
  build: LIF over the same MaleCNS v1.0 release, with DNa02-based steering.

Parameter set used by Shiu et al. and restated by Lin et al., adopted here:

```
V_rest = V_reset = -52 mV      V_thresh = -45 mV      t_refractory = 2.2 ms
tau_membrane = 20 ms           tau_synaptic = 5 ms    PSP per synapse = 0.275 mV
synaptic delay = 1.8 ms        basal firing = 0       gap junctions excluded
```

Sign convention: acetylcholine, dopamine, serotonin and octopamine excitatory;
GABA and glutamate inhibitory. This build additionally treats **histamine as
inhibitory**, which is the correct fly pharmacology (photoreceptor histamine
opens chloride channels on lamina monopolar cells) and matters here because
photoreceptors are 4.7% of the graph.

## 4. Findings from wiring this brain to a body

Measured in this repository, not taken from the literature.

1. **Integration must be dt-scaled.** Adding `i_syn` straight onto `V` each
   step makes the amount of injected charge depend on `dt`, so a coarse step
   over-drives the network by orders of magnitude. Membrane updates are scaled
   by `dt/tau_m` and the synaptic current is pre-multiplied by `tau_m/tau_s` so
   that one synapse still peaks at 0.275 mV of PSP at any step size.

2. **Constant sensory current has no usable dynamic range.** Driving
   photoreceptors with a steady voltage moved the population from silent at
   0.3 mV/step to refractory-saturated at 1.0 mV/step with nothing between.
   Sensation is therefore delivered as Poisson spike kicks with a rate code,
   which is also what the reference model does.

3. **The full recurrent graph latches without adaptation.** At the literature
   synaptic weight the network settled into 30-100 Hz self-sustained activity
   that barely tracked its input (input 10 Hz vs 400 Hz produced the same
   34 Hz global rate). A mild spike-frequency adaptation current plus global
   synaptic scaling restores input sensitivity. Operating point used:
   `weight_scale = 0.15`, `adapt_mv = 0.6`, giving 0.6-4.6 Hz global and
   2-10 Hz descending-neuron rates that increase monotonically with input.

4. **Visual input enters at the projection neurons, not the retina.** Because
   photoreceptor output is inhibitory, injecting there mostly silences
   downstream circuits unless the whole lamina inversion is also driven. Input
   is delivered to `visual_projection` cells (the LC/LPLC-type looming and
   feature detectors that actually drive descending steering).

5. **Poisson input noise drowned the sensory signal.** With independent input
   draws per population member, two runs of the *same* stimulus differed by
   0.25 in descending-neuron rate space while a left-wall versus right-wall
   stimulus differed by only 0.39: a signal-to-noise ratio of 1.6, so ES was
   ranking noise. Sharing one Poisson draw across the whole population (common
   random numbers) makes identical stimuli produce identical responses and left
   versus right produce clearly opposite steering.

6. **The vehicle had to be slowed to match the brain's control rate.** At a
   16 ms control period, the original 14 m/s car with 0.6 rad of steering lock
   turned at up to 3.7 rad/s, which no readout updating at 60 Hz can stabilise.
   Top speed, steering lock and track width were relaxed accordingly.

7. **MPS has no compressed-sparse support but its COO spmm is fast.**
   `to_sparse_csr()` raises `NotImplementedError` on Apple GPUs; COO spmm on
   MPS ran the 6.24M-edge matrix at 2.2 ms per step versus 38.8 ms for CPU CSR
   at batch 16, a 17x speedup. The model picks the format per device.

## 5. Sources

- [Male CNS Connectome project site](https://male-cns.janelia.org/)
- [MaleCNS download page](https://male-cns.janelia.org/download/)
- [Janelia FlyEM Male CNS Connectome](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- [Shiu et al. 2024, PubMed](https://pubmed.ncbi.nlm.nih.gov/37205514/)
- [Shiu et al. preprint PDF](https://www.biorxiv.org/content/10.1101/2023.05.02.539144.full.pdf)
- [Connectome simulations identify a CPG circuit for fly walking](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v1.full)
- [Axo-axonic synapses on descending neurons](https://pmc.ncbi.nlm.nih.gov/articles/PMC13126034/)
- [flybrain: embodied MaleCNS LIF simulation](https://github.com/TheMrRaGe/flybrain)
- [natverse/malecns R package](https://github.com/natverse/malecns)
- [neuPrint Python client](https://github.com/connectome-neuprint/neuprint-python)
