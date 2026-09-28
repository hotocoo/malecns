"""Supervise training: start, monitor, restart on failure.

Runs in an infinite loop:
1. Check if training is running
2. If not, start it
3. Monitor for crashes and stalls
4. Restart if needed
5. Log everything
"""

from __future__ import annotations

import json
import time
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

SUPERVISOR_LOG = LOG_DIR / "training_supervisor.log"
TRAINING_LOG = LOG_DIR / "train_trading_parallel.jsonl"
CHECKPOINT = "checkpoints/trading_es_v2.pt"
GRAPH = "data/graph_w5"

# Training parameters
TRAINING_ARGS = [
    "python3", "src/train_trading_parallel.py",
    "--graph", GRAPH,
    "--out", CHECKPOINT, "--workers", "4",
    "--population", "48",
    "--generations", "100000",
    "--episodes", "5",
    "--episode-length", "252",
    "--mutation", "0.05",
    "--log-interval", "5",
    "--save-interval", "25",
    "--stagnation-window", "50",
]


def log(message: str):
    """Log a message to the supervisor log."""
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(SUPERVISOR_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def is_training_running() -> bool:
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_trading_parallel.py"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def start_training(resume: bool = True) -> subprocess.Popen | None:
    """Start the training process."""
    args = TRAINING_ARGS.copy()
    if resume:
        if Path(CHECKPOINT).exists():
            args.extend(["--resume", CHECKPOINT])
    
    log(f"Starting training: {' '.join(args)}")
    
    try:
        # Run in background, redirect output to log
        training_log = open(LOG_DIR / "training_output.log", "a")
        process = subprocess.Popen(
            args,
            stdout=training_log,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
            start_new_session=True,
        )
        log(f"Training started with PID {process.pid}")
        return process
    except Exception as e:
        log(f"Failed to start training: {e}")
        return None


def check_training_health() -> str:
    """Check training health. Returns: 'healthy', 'stalled', 'not_running'"""
    if not is_training_running():
        return "not_running"
    
    # Check if log is being updated
    if not TRAINING_LOG.exists():
        return "stalled"
    
    try:
        mtime = TRAINING_LOG.stat().st_mtime
        age = time.time() - mtime
        if age > 1800:  # 30 minutes without update
            return "stalled"
    except Exception:
        return "stalled"
    
    return "healthy"


def main():
    log("=== MaleCNS Trading Training Supervisor ===")
    log(f"Started at {datetime.now().isoformat()}")
    
    restart_count = 0
    max_restarts = 5
    restart_cooldown = 60  # seconds between restart attempts
    last_restart_time = 0
    
    while True:
        try:
            health = check_training_health()
            
            if health == "not_running":
                log("Training not running. Checking if restart needed...")
                
                now = time.time()
                if now - last_restart_time < restart_cooldown:
                    log(f"In restart cooldown. Waiting...")
                    time.sleep(10)
                    continue
                
                if restart_count >= max_restarts:
                    log(f"Max restarts ({max_restarts}) reached. Giving up.")
                    time.sleep(60)
                    continue
                
                restart_count += 1
                last_restart_time = now
                log(f"Restarting training (attempt {restart_count}/{max_restarts})...")
                start_training(resume=True)
            elif health == "stalled":
                log("Training appears stalled (no log updates for 10+ minutes).")
                
                now = time.time()
                if now - last_restart_time < restart_cooldown:
                    log(f"In restart cooldown. Waiting...")
                    time.sleep(10)
                    continue
                
                restart_count += 1
                last_restart_time = now
                log(f"Restarting stalled training (attempt {restart_count}/{max_restarts})...")
                # Kill the stalled process
                try:
                    subprocess.run(["pkill", "-f", "train_trading_parallel.py"], timeout=5)
                    time.sleep(5)
                except Exception:
                    pass
                start_training(resume=True)
            else:
                # Healthy
                if restart_count > 0:
                    restart_count = 0
                    log("Training is healthy. Resetting restart counter.")
            
            # Check every 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            log("Supervisor stopped by user.")
            break
        except Exception as e:
            log(f"Supervisor error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
