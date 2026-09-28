"""Live trading viewer: watch the connectome trade, neuron by neuron.

Serves a premium luxury web dashboard with:
- Real-time price charts with candlesticks
- Order book visualization
- Brain activity 3D point cloud
- Performance metrics (Sharpe, drawdown, P&L)
- Trade history and position tracking
- Market regime indicator

Usage:
    python3 src/trading_viewer.py --checkpoint checkpoints/trading_es.pt
    open http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain import Brain, LIFConfig, load_connectome, pick_device, synchronize
from trading_agent import TradingAgent
from trading_env import TradingConfig

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class TradingState:
    """Shared state between trading thread and HTTP server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame: dict[str, Any] = {}
        self.price_history: list[float] = []
        self.equity_history: list[float] = []
        self.trade_history: list[dict] = []
        self.brain_rates: list[float] = []
        self.indicator_history: dict[str, list] = {}
        self.running = True
        self.last_update = time.time()

    def update(self, frame: dict):
        with self.lock:
            self.frame = frame
            self.last_update = time.time()

    def get_frame(self) -> dict:
        with self.lock:
            return self.frame.copy()


state = TradingState()


def trading_loop(agent: TradingAgent):
    """Background thread that runs the trading agent."""
    while state.running:
        obs, reward, done, info = agent.step()

        # Get brain activity
        brain_activity = agent.get_brain_activity()

        # Build frame
        frame = {
            "timestamp": time.time(),
            "step": agent.env.step_count,
            "price": obs["price"],
            "price_change": obs["price_change"],
            "equity": info.get("equity", 0),
            "position": info.get("position", 0),
            "cash": info.get("cash", 0),
            "drawdown": info.get("drawdown", 0),
            "max_drawdown": info.get("max_drawdown", 0),
            "regime": info.get("regime", "UNKNOWN"),
            "spread": obs["spread"],
            "bid": obs["bid"],
            "ask": obs["ask"],
            "rsi": obs["rsi"],
            "macd": obs["macd"],
            "bb_position": obs["bb_position"],
            "volatility": obs["volatility"],
            "total_trades": info.get("total_trades", 0),
            "total_commissions": info.get("total_commissions", 0),
            "total_slippage": info.get("total_slippage", 0),
            "reward": reward,
            "done": done,
            # Brain activity (sampled)
            "brain_rates": brain_activity["rates"].tolist(),
            "output_rates": brain_activity["output_rates"].tolist(),
        }

        state.update(frame)

        if done:
            # Episode complete, reset
            metrics = agent.get_performance()
            state.update({
                **frame,
                "episode_complete": True,
                "metrics": metrics,
            })
            agent.reset()
            time.sleep(2)

        time.sleep(0.1)  # 10 Hz update rate


class TradingHandler(BaseHTTPRequestHandler):
    """HTTP handler for the trading viewer."""

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.serve_file(WEB_DIR / "trading.html", "text/html")
        elif self.path == "/trading.html":
            self.serve_file(WEB_DIR / "trading.html", "text/html")
        elif self.path == "/trading.js":
            self.serve_file(WEB_DIR / "trading.js", "application/javascript")
        elif self.path == "/trading.css":
            self.serve_file(WEB_DIR / "trading.css", "text/css")
        elif self.path == "/stream":
            self.serve_stream()
        elif self.path == "/frame":
            self.serve_frame()
        elif self.path.startswith("/vendor/"):
            self.serve_file(WEB_DIR / self.path[1:], "application/javascript")
        else:
            self.send_error(404)

    def serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404, f"Not found: {path}")
            return
        try:
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def serve_frame(self):
        frame = state.get_frame()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(frame).encode())

    def serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_update = 0
        while state.running:
            if state.last_update > last_update:
                frame = state.get_frame()
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                self.wfile.flush()
                last_update = state.last_update
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="MaleCNS Trading Viewer")
    parser.add_argument("--graph", default="data/graph_w5", help="Connectome graph directory")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port")
    parser.add_argument("--device", default=None, help="Device")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Device: {device}")

    # Create trading agent
    print("Creating trading agent...")
    agent = TradingAgent(
        args.graph,
        checkpoint=args.checkpoint,
        device=device,
    )

    # Start trading loop in background
    trade_thread = threading.Thread(target=trading_loop, args=(agent,), daemon=True)
    trade_thread.start()

    # Start HTTP server
    server = ThreadingHTTPServer(("127.0.0.1", args.port), TradingHandler)
    print(f"Trading viewer running at http://127.0.0.1:{args.port}/trading.html")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        state.running = False
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
