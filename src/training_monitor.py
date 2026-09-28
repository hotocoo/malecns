"""Monitor training progress and detect issues.

Checks:
- Training is still running
- Fitness is improving (not stuck)
- No crashes or errors
- Logs are being written

Can be run as a cron job or in the background.
"""

from __future__ import annotations

import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
TRAINING_LOG = LOG_DIR / "train_trading_v2.jsonl"
STATUS_FILE = LOG_DIR / "training_status.json"
ALERT_FILE = LOG_DIR / "training_alerts.jsonl"


def is_training_running() -> bool:
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_trading_v2.py"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def get_latest_metrics() -> dict | None:
    """Read the latest training log entry."""
    if not TRAINING_LOG.exists():
        return None
    
    try:
        with open(TRAINING_LOG, "r") as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
    except Exception as e:
        print(f"Error reading training log: {e}")
    return None


def check_improvement(window: int = 20, threshold: float = 0.0001) -> dict:
    """Check if training is improving over a window of generations."""
    if not TRAINING_LOG.exists():
        return {"improving": False, "reason": "No training log"}
    
    try:
        with open(TRAINING_LOG, "r") as f:
            lines = f.readlines()
            
        if len(lines) < window:
            return {"improving": True, "reason": "Not enough data yet"}
        
        recent = [json.loads(line) for line in lines[-window:]]
        first_fitness = recent[0]["best_fitness"]
        last_fitness = recent[-1]["best_fitness"]
        improvement = last_fitness - first_fitness
        
        return {
            "improving": improvement > threshold,
            "improvement": improvement,
            "first_fitness": first_fitness,
            "last_fitness": last_fitness,
            "generations_checked": window,
        }
    except Exception as e:
        return {"improving": False, "reason": f"Error: {e}"}


def check_log_age(max_age_seconds: int = 300) -> dict:
    """Check if the training log is being updated."""
    if not TRAINING_LOG.exists():
        return {"fresh": False, "reason": "No training log"}
    
    try:
        mtime = TRAINING_LOG.stat().st_mtime
        age = time.time() - mtime
        return {
            "fresh": age < max_age_seconds,
            "age_seconds": age,
            "max_age_seconds": max_age_seconds,
        }
    except Exception as e:
        return {"fresh": False, "reason": f"Error: {e}"}


def write_alert(alert: dict):
    """Write an alert to the alerts log."""
    alert["timestamp"] = time.time()
    alert["datetime"] = datetime.now().isoformat()
    try:
        with open(ALERT_FILE, "a") as f:
            f.write(json.dumps(alert) + "\n")
    except Exception as e:
        print(f"Error writing alert: {e}")


def write_status(status: dict):
    """Write current status to status file."""
    status["timestamp"] = time.time()
    status["datetime"] = datetime.now().isoformat()
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"Error writing status: {e}")


def main():
    print("=== MaleCNS Trading Training Monitor ===")
    print(f"Time: {datetime.now().isoformat()}")
    print()
    
    # Check if training is running
    running = is_training_running()
    print(f"Training running: {running}")
    
    # Get latest metrics
    metrics = get_latest_metrics()
    if metrics:
        print(f"Latest generation: {metrics['generation']}")
        print(f"Best fitness: {metrics['best_fitness']:.4f}")
        print(f"Best return: {metrics['best_return']*100:.2f}%")
        print(f"Mutation strength: {metrics['mutation_strength']:.4f}")
        print(f"Stagnation counter: {metrics['stagnation_counter']}")
    else:
        print("No training metrics found")
    
    # Check improvement
    improvement = check_improvement()
    print(f"Improving: {improvement['improving']}")
    if "improvement" in improvement:
        print(f"  Improvement over window: {improvement['improvement']:.6f}")
    
    # Check log freshness
    log_fresh = check_log_age()
    print(f"Log fresh: {log_fresh['fresh']}")
    if "age_seconds" in log_fresh:
        print(f"  Log age: {log_fresh['age_seconds']:.0f}s")
    
    # Determine status
    status = "healthy"
    alerts = []
    
    if not running:
        status = "not_running"
        alerts.append({"type": "not_running", "severity": "error", "message": "Training process not running"})
    elif not log_fresh["fresh"]:
        status = "stalled"
        alerts.append({"type": "stalled", "severity": "warning", "message": f"Training log not updated for {log_fresh['age_seconds']:.0f}s"})
    elif not improvement["improving"]:
        status = "stagnant"
        alerts.append({"type": "stagnant", "severity": "warning", "message": "Training not improving over recent window"})
    
    print(f"\nStatus: {status}")
    
    # Write alerts
    for alert in alerts:
        print(f"  ALERT: {alert['message']}")
        write_alert(alert)
    
    # Write status
    status_data = {
        "status": status,
        "running": running,
        "log_fresh": log_fresh["fresh"],
        "improving": improvement["improving"],
        "latest_metrics": metrics,
        "alerts": alerts,
    }
    write_status(status_data)
    
    # Exit code based on status
    if status == "healthy":
        return 0
    elif status == "stalled" or status == "stagnant":
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
