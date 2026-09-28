"""Generate a training progress report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
TRAINING_LOG = LOG_DIR / "train_trading_v2.jsonl"
STATUS_FILE = LOG_DIR / "training_status.json"


def load_training_data():
    """Load all training log entries."""
    if not TRAINING_LOG.exists():
        return []
    
    entries = []
    try:
        with open(TRAINING_LOG, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error reading training log: {e}")
    return entries


def generate_report():
    """Generate a comprehensive training report."""
    entries = load_training_data()
    
    if not entries:
        print("No training data found.")
        return
    
    print("=" * 80)
    print("MALECNS TRADING TRAINING REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 80)
    print()
    
    # Latest status
    latest = entries[-1]
    print(f"Current Generation: {latest['generation']}")
    print(f"Best Fitness: {latest['best_fitness']:.4f}")
    print(f"Best Return: {latest['best_return']*100:.2f}%")
    print(f"Mutation Strength: {latest['mutation_strength']:.4f}")
    print(f"Stagnation Counter: {latest['stagnation_counter']}")
    print(f"Elapsed Time: {latest['elapsed']:.0f}s ({latest['elapsed']/3600:.1f}h)")
    print(f"Avg Gen Time: {latest['gen_time']:.1f}s")
    print()
    
    # Recent progress
    print("Recent Progress (last 10 logged generations):")
    print(f"{'Gen':>6} | {'Best Fit':>10} | {'Best Ret':>10} | {'Sharpe':>8} | {'DD':>8} | {'Mut':>8}")
    print("-" * 70)
    for entry in entries[-10:]:
        print(f"{entry['generation']:6d} | {entry['best_fitness']:10.4f} | "
              f"{entry['best_return']*100:9.2f}% | {entry['gen_best_sharpe']:8.2f} | "
              f"{entry['gen_best_drawdown']*100:7.2f}% | {entry['mutation_strength']:8.4f}")
    print()
    
    # Improvement analysis
    if len(entries) >= 10:
        first = entries[0]
        last = entries[-1]
        improvement = last['best_fitness'] - first['best_fitness']
        print(f"Total Improvement: {improvement:.4f} ({first['best_fitness']:.4f} -> {last['best_fitness']:.4f})")
        print(f"Generations Trained: {last['generation'] - first['generation']}")
        
        # Check for stagnation
        recent_entries = entries[-20:]
        if len(recent_entries) >= 2:
            recent_improvement = recent_entries[-1]['best_fitness'] - recent_entries[0]['best_fitness']
            if recent_improvement < 0.0001:
                print("WARNING: Training appears to be stagnating!")
            else:
                print(f"Recent improvement (last 20 logged): {recent_improvement:.4f}")
    
    # Status file
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                status = json.load(f)
            print(f"\nMonitor Status: {status.get('status', 'unknown')}")
            if status.get('alerts'):
                print("Alerts:")
                for alert in status['alerts']:
                    print(f"  - {alert['message']}")
        except Exception:
            pass
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    generate_report()
