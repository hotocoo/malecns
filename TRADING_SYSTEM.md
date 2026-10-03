# MaleCNS Trading System

A real-time trading agent powered by the Drosophila (fruit fly) brain connectome with 166,700 neurons and 6.24 million synaptic connections.

## Architecture

```
Market Data → Observations → MaleCNS Brain (166,700 neurons) → Actions → Broker
     ↑                                                        |
     └─────────────── Trading Environment ←───────────────────┘
```

## Key Components

- **Brain**: Spiking neural network using Leaky Integrate-and-Fire (LIF) dynamics, trained with Evolution Strategies (ES)
- **Trading Agent**: Connects brain to trading environment, handles observations and actions
- **Trading Environment**: Simulated or real market data with technical indicators
- **Real-time Engine**: Live trading with OANDA, eToro, or demo broker

## Performance (as of Oct 1, 2026)

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Return | 13.55% | 3.41% | 2.03% |
| Sharpe Ratio | 1.97 | 0.35 | 0.00 |
| Max Drawdown | 2.58% | 4.33% | 3.95% |
| Trades/Episode | 74.2 | 17.0 | 24.4 |

### Stress Tests (Real Market Regimes)

| Regime | Return | Sharpe |
|--------|--------|--------|
| Bull Market 2021 | 2.02% | 1.26 |
| Crash March 2020 | -1.67% | -5.00 |
| Recovery 2020 | 3.28% | 1.57 |
| Sideways 2023 | 2.50% | 1.50 |

## Quick Start

### 1. Demo Trading (No API Key Needed)

```bash
./run_realtime_trading.sh demo --instrument EUR_USD
```

### 2. OANDA Trading

```bash
# Get a free practice account at https://practice-oanda.com
export OANDA_ACCESS_TOKEN="your_token"
export OANDA_ACCOUNT_ID="your_account"
./run_realtime_trading.sh oanda --instrument EUR_USD --paper
```

### 3. eToro Trading

```bash
# Get an API token at https://etoro.com/api
export ETORO_ACCESS_TOKEN="your_token"
./run_realtime_trading.sh etoro --instrument EUR_USD
```

## Training

```bash
# Quick training (100 generations, 16 agents)
python3 src/train_trading_comprehensive.py --generations 100 --population 16

# Extended training (500 generations, 32 agents)
python3 src/train_trading_comprehensive.py --generations 500 --population 32 --episodes 5
```

## Evaluation

```bash
python3 src/evaluate_trading.py --checkpoint checkpoints/trading_comprehensive_best.pt
```

## Data

- **ETFs**: SPY, QQQ, IWM (1,687 days, Jan 2020 - Sep 2026)
- **Forex**: EUR_USD, GBP_USD, USD_JPY (1,300 days)
- Downloaded via yfinance

## Checkpoints

- `checkpoints/trading_comprehensive_best.pt` - Best performing agent
- `checkpoints/trading_comprehensive.pt` - Latest checkpoint

## Virtual Panelists

The trading system includes a virtual expert panel that analyzes market conditions:

- **Technical Analyst**: RSI, MACD, Bollinger Bands, ATR
- **Fundamental Analyst**: Price trends, volume analysis
- **Risk Manager**: Position sizing, stop-loss, take-profit
- **Market Psychologist**: Sentiment analysis, fear/greed indicators

## Configuration

Edit `.trading_config.json`:

```json
{
    "broker": "demo",
    "instrument": "EUR_USD",
    "instruments": ["EUR_USD", "GBP_USD", "USD_JPY"],
    "paper": true,
    "risk_per_trade": 0.01,
    "max_positions": 3,
    "stop_loss": 0.02,
    "take_profit": 0.04
}
```

## Recent Improvements (Oct 4, 2026)

- **Config file loading**: System now reads `.trading_config.json` for default settings
- **Thread safety**: Added locking to prevent duplicate position openings
- **Rate limiting**: Forex data fetching now includes retry logic with exponential backoff
- **Error handling**: Graceful fallback to simple momentum trading if brain fails to load
- **Cleaner logs**: Suppressed PyTorch sparse tensor beta warnings

## Requirements

- Python 3.10+
- PyTorch 2.0+
- yfinance
- requests
