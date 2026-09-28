import sys
sys.path.insert(0, "src")
from trading_env_real import RealTradingEnvironment

env = RealTradingEnvironment()
obs = env.reset()
spy_change = obs.get("SPY_price_change", 0)
equity = obs.get("equity", 0)
print("SPY price change:", round(spy_change, 6))
print("Equity:", round(equity, 4))
obs, reward, done, info = env.step(1)
print("After step: reward=", round(reward, 6), "done=", done)
for i in range(10):
    obs, reward, done, info = env.step(2)
    if done:
        break
metrics = env.get_performance_metrics()
print("Return:", round(metrics["total_return"], 4))
print("Sharpe:", round(metrics["sharpe_ratio"], 4))
print("Max DD:", round(metrics["max_drawdown"], 4))
print("Trades:", metrics["total_trades"])
