#!/usr/bin/env python3
"""
Proof of Concept entrypoint: Process 100 stories from SimpleStories dataset.

Usage:
    python run_poc.py submit   # Submit batch job
    python run_poc.py check    # Check job status
    python run_poc.py fetch    # Fetch results and create parquet
"""

import sys
from simplify_stories import run

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_poc.py {submit|check|fetch}")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode not in ["submit", "check", "fetch"]:
        print(f"Invalid mode: {mode}")
        print("Valid modes: submit, check, fetch")
        sys.exit(1)

    print(f"=== PoC Mode: Processing 100 stories ===\n")
    run(mode=mode, n_stories=100, job_suffix="poc")
