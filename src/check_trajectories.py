import torch
import numpy as np
from agent import ConnectomeAgent, AgentConfig
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import CarEnv, monaco_config, build_centerline, Track
import defaults

device = pick_device('auto')
connectome = load_connectome('../data/graph_w5')

# Create environments for 4 genomes at the same start point
popsize = 4
batch = popsize
brain = Brain(connectome, batch=batch, config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV), device=device, weight_scale=defaults.WEIGHT_SCALE, precision='fp16')
agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=defaults.SUBSTEPS))

# Load current checkpoint and create perturbed genomes
state = torch.load('../checkpoints/es.pt', map_location=device)
mu = state['mu']
half = popsize // 2
sigma = 0.05
sample_gen = torch.Generator().manual_seed(42)
eps = torch.randn(half, mu.numel(), generator=sample_gen).to(device)
perturb = torch.cat([eps, -eps], dim=0)
params = agent.clamp_params(mu.unsqueeze(0) + sigma * perturb)
theta = agent.unpack(params)

# Create environment with all genomes at the same start point
from dataclasses import replace
cfg = replace(monaco_config(defaults.control_dt_s(defaults.DT_MS, defaults.SUBSTEPS)), geojson_path='../data/tracks/monaco.geojson')
track = Track(build_centerline(cfg), cfg, device)
env = CarEnv(batch, device, cfg, track=track, start_fraction=0.0)

# Run 100 steps and record trajectories
agent.reset()
obs = env.reset()
positions = []
steerings = []
pedals = []
for step in range(100):
    action = agent.act(obs, theta)
    obs, reward, done = env.step(action)
    positions.append(env.pos.clone().cpu())
    steerings.append(action[:, 0].clone().cpu())
    pedals.append(action[:, 1].clone().cpu())
    if done.all():
        print(f"All done at step {step}")
        break

positions = torch.stack(positions)
steerings = torch.stack(steerings)
pedals = torch.stack(pedals)

print("Trajectory diversity check (4 genomes, same start):")
print(f"  Position spread at step 99: x={positions[-1,:,0].std().item():.3f}m, y={positions[-1,:,1].std().item():.3f}m")
print(f"  Steering spread at step 99: {steerings[-1].std().item():.4f}")
print(f"  Pedal spread at step 99: {pedals[-1].std().item():.4f}")
print(f"  Max position distance between genomes: {(positions[-1] - positions[-1][0]).norm(dim=1).max().item():.3f}m")
print()
print("  Genome positions at step 99:")
for i in range(batch):
    print(f"    Genome {i}: ({positions[-1,i,0].item():.1f}, {positions[-1,i,1].item():.1f})")
