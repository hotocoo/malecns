import torch
import numpy as np
from agent import ConnectomeAgent, AgentConfig
from brain import Brain, LIFConfig, load_connectome, pick_device
from car_env import CarEnv, monaco_config, build_centerline, Track
from dataclasses import replace
import defaults

device = pick_device('auto')
connectome = load_connectome('../data/graph_w5')

# Create environment for 1 genome
batch = 1
brain = Brain(connectome, batch=batch, config=LIFConfig(dt_ms=defaults.DT_MS, adapt_mv=defaults.ADAPT_MV), device=device, weight_scale=defaults.WEIGHT_SCALE, precision='fp16')
agent = ConnectomeAgent(brain, connectome.neurons, AgentConfig(substeps=defaults.SUBSTEPS))

# Load current checkpoint (use the mean, no perturbation)
state = torch.load('../checkpoints/es.pt', map_location=device)
mu = state['mu']
params = mu.unsqueeze(0)
theta = agent.unpack(params)

# Create environment
cfg = replace(monaco_config(defaults.control_dt_s(defaults.DT_MS, defaults.SUBSTEPS)), geojson_path='../data/tracks/monaco.geojson')
track = Track(build_centerline(cfg), cfg, device)
env = CarEnv(batch, device, cfg, track=track, start_fraction=0.0)

# Run episode and record trajectory
agent.reset()
obs = env.reset()
steps = 0
while steps < 1500:
    action = agent.act(obs, theta)
    obs, reward, done = env.step(action)
    steps += 1
    if steps % 100 == 0:
        print(f"Step {steps}: pos=({env.pos[0,0].item():.1f}, {env.pos[0,1].item():.1f}), speed={env.speed[0].item()*3.6:.0f} km/h, steer={action[0,0].item():.3f}, pedal={action[0,1].item():.3f}, clearance={env.body_clearance(env.pos, env.heading)[0].item():.2f}m")
    if done[0]:
        print(f"Crashed at step {steps}: pos=({env.pos[0,0].item():.1f}, {env.pos[0,1].item():.1f}), clearance={env.body_clearance(env.pos, env.heading)[0].item():.2f}m")
        break
