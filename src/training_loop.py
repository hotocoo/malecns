"""Run MaleCNS trading training in a loop, restarting if it fails."""

import subprocess
import time
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRAINING_CMD = [
    "python3", "src/train_trading_parallel.py",
    "--graph", "data/graph_w5",
    "--out", "checkpoints/trading_es_v2.pt",
    "--workers", "2",
    "--population", "16",
    "--generations", "100000",
    "--episodes", "3",
    "--episode-length", "100",
    "--mutation", "0.05",
    "--log-interval", "5",
    "--save-interval", "25",
    "--stagnation-window", "50",
    "--resume", "checkpoints/trading_es_v2.pt",
]

def main():
    log_file = open(LOG_DIR / "training_loop.log", "a")
    
    while True:
        try:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Starting training...\n")
            log_file.flush()
            
            process = subprocess.Popen(
                TRAINING_CMD,
                stdout=open(LOG_DIR / "training_output.log", "a"),
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            
            # Wait for process to finish
            exit_code = process.wait()
            
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Training exited with code {exit_code}\n")
            log_file.flush()
            
            if exit_code == 0:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Training completed successfully. Exiting loop.\n")
                log_file.flush()
                break
            else:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Training failed. Restarting in 30 seconds...\n")
                log_file.flush()
                time.sleep(30)
                
        except Exception as e:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Error: {e}\n")
            log_file.flush()
            time.sleep(30)
    
    log_file.close()

if __name__ == "__main__":
    main()
