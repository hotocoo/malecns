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

# Load current checkpoint
state = torch.load('../checkpoints/es.pt', map_location=device)
mu = state['mu']
params = mu.unsqueeze(0)
theta = agent.unpack(params)

# Create environment
cfg = replace(monaco_config(defaults.control_dt_s(defaults.DT_MS, defaults.SUBSTEPS)), geojson_path='../data/tracks/monaco.geojson')
track = Track(build_centerline(cfg), cfg, device)
env = CarEnv(batch, device, cfg, track=track, start_fraction=0.0)

# Run episode and record lidar at crash
agent.reset()
obs = env.reset()
steps = 0
while steps < 400:
    action = agent.act(obs, theta)
    obs, reward, done = env.step(action)
    steps += 1
    if steps >= 300 and steps <= 330:
        lidar = obs[0, :9].tolist()
        speed = obs[0, 9].item()
        print(f"Step {steps}: speed={speed:.2f}, lidar={[f'{d:.2f}' for d in lidar]}, steer={action[0,0].item():.3f}, pedal={action[0,1].item():.3f}")
    if done[0]:
        print(f"Crashed at step {steps}")
        break
