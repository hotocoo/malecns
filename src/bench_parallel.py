import train

raise SystemExit(train.main([
    "--generations", "1",
    "--islands", "14",
    "--popsize", "128",
    "--starts-per-gen", "6",
    "--episode-steps", "125",
    "--checkpoint", "/tmp/malecns_parallel.pt",
    "--log", "/tmp/malecns_parallel.jsonl",
    "--eval-every", "0",
]))
