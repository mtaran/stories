#!/usr/bin/env python3
"""
Proof of Concept entrypoint: Process 100 stories from SimpleStories dataset.

Usage:
    python run_poc.py submit [train|test]   # Submit batch job (default: train)
    python run_poc.py check [train|test]    # Check job status
    python run_poc.py fetch [train|test]    # Fetch results and create parquet
"""

import sys
from simplify_stories import run

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python run_poc.py {submit|check|fetch} [train|test]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode not in ["submit", "check", "fetch"]:
        print(f"Invalid mode: {mode}")
        print("Valid modes: submit, check, fetch")
        sys.exit(1)

    # Get split (default to "train")
    split = sys.argv[2].lower() if len(sys.argv) == 3 else "train"
    if split not in ["train", "test"]:
        print(f"Invalid split: {split}")
        print("Valid splits: train, test")
        sys.exit(1)

    print(f"=== PoC Mode: Processing 100 stories from {split} split ===\n")
    run(mode=mode, n_stories=100, job_suffix="poc", split=split)
