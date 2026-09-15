import torch
import numpy as np
from agent import ConnectomeAgent, AgentConfig
from brain import Brain, LIFConfig, load_connectome, pick_device
import defaults

device = pick_device('auto')
connectome = load_connectome('../data/graph_w5')
brain = Brain(connectome, batch=8, config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV), device=device, weight_scale=defaults.WEIGHT_SCALE, precision='fp16')
agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=defaults.SUBSTEPS))

# Load current checkpoint
state = torch.load('../checkpoints/es.pt', map_location=device)
mu = state['mu']
print("mu shape:", mu.shape, "norm:", mu.norm().item())

# Create 8 perturbed genomes like the training loop does
popsize = 8
half = popsize // 2
sigma = 0.05
sample_gen = torch.Generator().manual_seed(42)
eps = torch.randn(half, mu.numel(), generator=sample_gen).to(device)
perturb = torch.cat([eps, -eps], dim=0)
params = agent.clamp_params(mu.unsqueeze(0) + sigma * perturb)

# Check diversity of the perturbed genomes
print("\nGenome diversity check:")
for i in range(popsize):
    p = params[i]
    diff_from_mu = (p - mu).abs().sum().item()
    print(f"  Genome {i}: diff from mu = {diff_from_mu:.6f}")

# Check if antithetic pairs are truly negatives
print("\nAntithetic pair check:")
for i in range(half):
    p1 = params[i]
    p2 = params[i + half]
    diff1 = (p1 - mu).abs().sum().item()
    diff2 = (p2 - mu).abs().sum().item()
    print(f"  Pair {i}: diff1={diff1:.6f}, diff2={diff2:.6f}, ratio={diff1/diff2:.4f}")

# Unpack and check specific parameters
print("\nParameter unpacking check:")
theta = agent.unpack(params)
print("  ray_gain[0]:", theta["ray_gain"][0].tolist())
print("  ray_gain[1]:", theta["ray_gain"][1].tolist())
print("  w_out[0,0,:]:", theta["w_out"][0,0,:].tolist())
print("  w_out[1,0,:]:", theta["w_out"][1,0,:].tolist())
