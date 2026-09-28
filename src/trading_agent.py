"""Trading agent interface: market data -> brain input, brain output -> trade decisions."""

from __future__ import annotations

import numpy as np
import torch

from agent import AgentConfig, ConnectomeAgent
from brain import Brain, LIFConfig, load_connectome, pick_device
from trading_env import TradingEnvironment, TradeAction, TradingConfig


OBS_FEATURES = [
    "price_change", "price_change_5", "price_change_20",
    "price_normalized", "position_pct", "equity_normalized",
    "unrealized_pnl", "rsi", "macd", "macd_line",
    "bb_position", "bb_width", "atr", "momentum_5",
    "realized_vol", "spread", "drawdown", "max_drawdown",
    "volatility",
]

REAL_OBS_FEATURES = [
    "SPY_price_change", "QQQ_price_change", "IWM_price_change",
    "SPY_position", "QQQ_position", "IWM_position",
    "SPY_avg_price", "QQQ_avg_price", "IWM_avg_price",
    "SPY_sma_20", "QQQ_sma_20", "IWM_sma_20",
    "SPY_rsi", "QQQ_rsi", "IWM_rsi",
    "cash_pct", "equity", "drawdown",
    "SPY_sma_20",  # duplicate to fill 18 slots (brain expects 19 inputs = 18 + constant)
]

N_OBS = len(OBS_FEATURES)
N_REAL_OBS = len(REAL_OBS_FEATURES)


class TradingAgent:
    """Connects the malecns brain to the trading environment."""

    def __init__(
        self,
        graph_dir: str,
        checkpoint: str | None = None,
        device: str = "auto",
        agent_config: AgentConfig | None = None,
        trading_config: TradingConfig | None = None,
        obs_features: list[str] | None = None,
    ):
        self.device = pick_device(device)
        self.agent_config = agent_config or AgentConfig()
        self.trading_config = trading_config or TradingConfig()
        self.obs_features = obs_features or OBS_FEATURES
        self.n_obs = len(self.obs_features)

        connectome = load_connectome(graph_dir)
        self.brain = Brain(connectome, batch=1, config=LIFConfig(), device=self.device)
        self.agent = ConnectomeAgent(self.brain, connectome.neurons, self.agent_config)
        self.theta = self.agent.unpack(self.agent.initial_params().unsqueeze(0))
        # Move theta to the same device as the brain
        self.theta = {k: v.to(self.device) for k, v in self.theta.items()}

        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.obs_mean = np.zeros(self.n_obs)
        self.obs_std = np.ones(self.n_obs)
        self.n_obs_samples = 0

        self.env = TradingEnvironment(self.trading_config)
        self.obs = self.env.reset()
        self.total_reward = 0.0
        self.episode_rewards = []

        self.agent.reset(batch=1)
        self.agent.seed(0)

    def load_checkpoint(self, path: str):
        """Load trained parameters from checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "mu" in state:
            if isinstance(state["mu"], list):
                for j, name in enumerate(self.theta):
                    if j < len(state["mu"]):
                        self.theta[name] = state["mu"][j].to(self.device)
            else:
                self.theta = self.agent.unpack(state["mu"].reshape(-1).to(self.device).unsqueeze(0))
        if "obs_mean" in state:
            self.obs_mean = np.array(state["obs_mean"])
            self.obs_std = np.array(state["obs_std"])

    def save_checkpoint(self, path: str, extra: dict | None = None):
        """Save trained parameters to checkpoint."""
        params = []
        for name in self.theta:
            params.append(self.theta[name].cpu())
        state = {
            "mu": params,
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        features = np.array([obs[f] for f in self.obs_features], dtype=np.float64)
        self.n_obs_samples += 1
        alpha = 1.0 / self.n_obs_samples
        self.obs_mean = (1 - alpha) * self.obs_mean + alpha * features
        self.obs_std = np.maximum(
            (1 - alpha) * self.obs_std + alpha * (features - self.obs_mean) ** 2,
            1e-6
        )
        self.obs_std = np.sqrt(self.obs_std)
        normalized = (features - self.obs_mean) / self.obs_std
        normalized = np.clip(normalized, 0.0, 10.0)
        obs_array = np.append(normalized, [0.5])
        tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor

    def _output_to_action(self, output: torch.Tensor) -> TradeAction:
        steer = output[0, 0].item()
        throttle = output[0, 1].item()
        size = min(3, max(1, int(abs(throttle) * 3) + 1))
        if steer > 0.1:
            if size == 1: return TradeAction.BUY_SMALL
            elif size == 2: return TradeAction.BUY_MEDIUM
            else: return TradeAction.BUY_LARGE
        elif steer < -0.1:
            if size == 1: return TradeAction.SELL_SMALL
            elif size == 2: return TradeAction.SELL_MEDIUM
            else: return TradeAction.SELL_LARGE
        else:
            return TradeAction.HOLD

    def act(self, obs: dict | None = None) -> TradeAction:
        if obs is None:
            obs = self.obs
        input_tensor = self._obs_to_tensor(obs)
        output = self.agent.act(input_tensor, self.theta)
        return self._output_to_action(output)

    def step(self) -> tuple[dict, float, bool, dict]:
        action = self.act(self.obs)
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        self.total_reward += reward
        if done:
            self.episode_rewards.append(self.total_reward)
        return obs, reward, done, info

    def reset(self) -> dict:
        self.obs = self.env.reset()
        self.total_reward = 0.0
        self.agent.reset(batch=1)
        return self.obs

    def get_performance(self) -> dict:
        return self.env.get_performance_metrics()

    def get_brain_activity(self) -> dict:
        rates = self.brain.v.cpu().numpy().flatten()
        return {
            "rates": rates,
            "spikes": [],
            "output_rates": rates[:7],
        }
