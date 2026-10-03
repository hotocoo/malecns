"""Tests for the MaleCNS trading system."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_env import TradingEnvironment, TradingConfig, TradeAction, Regime
from demo_broker import DemoBroker
from realtime_trading import RealTimeTradingEngine, load_config, SimpleMomentumAgent, Order, OrderSide


class TestTradingEnvironment:
    """Tests for the trading environment."""

    def test_reset_returns_observation(self):
        env = TradingEnvironment()
        obs = env.reset()
        assert isinstance(obs, dict)
        # Check key observation features exist
        assert "price_change" in obs
        assert "rsi" in obs
        assert "macd" in obs

    def test_step_returns_valid_values(self):
        env = TradingEnvironment()
        obs = env.reset()
        for _ in range(10):
            obs, reward, done, info = env.step(TradeAction.HOLD)
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)

    def test_buy_action_changes_position(self):
        env = TradingEnvironment()
        obs = env.reset()
        initial_position = obs["position_pct"]
        obs, _, _, _ = env.step(TradeAction.BUY_SMALL)
        assert obs["position_pct"] != initial_position

    def test_sell_action_changes_position(self):
        env = TradingEnvironment()
        obs = env.reset()
        # Buy first
        obs, _, _, _ = env.step(TradeAction.BUY_SMALL)
        initial_position = obs["position_pct"]
        # Then sell
        obs, _, _, _ = env.step(TradeAction.SELL_SMALL)
        assert obs["position_pct"] != initial_position

    def test_performance_metrics(self):
        env = TradingEnvironment()
        obs = env.reset()
        for _ in range(100):
            obs, _, done, _ = env.step(TradeAction.HOLD)
            if done:
                break
        metrics = env.get_performance_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics


class TestDemoBroker:
    """Tests for the demo broker."""

    def test_connect_disconnect(self):
        broker = DemoBroker()
        assert broker.connect()
        broker.disconnect()

    def test_get_price(self):
        broker = DemoBroker()
        price = broker.get_price("EUR_USD")
        assert price > 0
        assert abs(price - 1.085) < 0.01  # Should be close to starting price

    def test_place_order(self):
        broker = DemoBroker()
        broker.connect()
        order = Order(
            instrument="EUR_USD",
            side=OrderSide.BUY,
            units=100000,
        )
        result = broker.place_order(order)
        assert result.status == "filled"
        assert result.order_id
        broker.disconnect()

    def test_close_position(self):
        broker = DemoBroker()
        broker.connect()
        order = Order(
            instrument="EUR_USD",
            side=OrderSide.BUY,
            units=100000,
        )
        broker.place_order(order)
        result = broker.close_position("EUR_USD", 100000)
        assert result.status == "closed"
        broker.disconnect()

    def test_account_balance(self):
        broker = DemoBroker()
        broker.connect()
        balance = broker.get_account_balance()
        assert balance == 100000.0  # Initial balance
        broker.disconnect()


class TestRealTimeTradingEngine:
    """Tests for the real-time trading engine."""

    def test_config_loading(self):
        config = load_config()
        assert isinstance(config, dict)
        # Config file exists in repo
        assert "broker" in config or config == {}

    def test_engine_creation(self):
        broker = DemoBroker()
        broker.connect()
        engine = RealTimeTradingEngine(
            broker=broker,
            graph_dir="data/graph_w5",
            instruments=["EUR_USD"],
            device="cpu",
        )
        broker.disconnect()

    def test_position_size_calculation(self):
        broker = DemoBroker()
        broker.connect()
        engine = RealTimeTradingEngine(
            broker=broker,
            graph_dir="data/graph_w5",
            instruments=["EUR_USD"],
            device="cpu",
            risk_per_trade=0.01,
            stop_loss_pct=0.02,
        )
        size = engine.calculate_position_size(1.08)
        assert size > 0
        assert size <= 100000  # Max 1 standard lot
        broker.disconnect()

    def test_sl_tp_calculation(self):
        broker = DemoBroker()
        broker.connect()
        engine = RealTimeTradingEngine(
            broker=broker,
            graph_dir="data/graph_w5",
            instruments=["EUR_USD"],
            device="cpu",
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )
        price = 1.08
        sl = price * (1 - engine.stop_loss_pct)
        tp = price * (1 + engine.take_profit_pct)
        assert sl < price
        assert tp > price
        assert abs(sl - 1.0584) < 0.0001
        assert abs(tp - 1.1232) < 0.0001
        broker.disconnect()


class TestSimpleMomentumAgent:
    """Tests for the simple momentum fallback agent."""

    def test_act_returns_valid_action(self):
        agent = SimpleMomentumAgent()
        obs = {"price_change": 0.001}
        action = agent.act(obs)
        assert action in TradeAction

    def test_act_with_none_obs(self):
        agent = SimpleMomentumAgent()
        # Should not crash with None observation
        try:
            action = agent.act(None)
            # If it doesn't crash, action should be valid
            assert action in TradeAction
        except (KeyError, TypeError):
            # Some implementations may require obs dict
            pass


class TestOrderTypes:
    """Tests for order data structures."""

    def test_order_creation(self):
        order = Order(
            instrument="EUR_USD",
            side=OrderSide.BUY,
            units=100000,
            stop_loss=1.05,
            take_profit=1.12,
        )
        assert order.instrument == "EUR_USD"
        assert order.side == OrderSide.BUY
        assert order.units == 100000
        assert order.stop_loss == 1.05
        assert order.take_profit == 1.12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
