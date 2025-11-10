#!/usr/bin/env python3
"""
Full dataset entrypoint: Process all ~2M stories from SimpleStories dataset.

Usage:
    python run_full.py submit   # Submit batch job
    python run_full.py check    # Check job status
    python run_full.py fetch    # Fetch results and create parquet
"""

import sys
from simplify_stories import run

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_full.py {submit|check|fetch}")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode not in ["submit", "check", "fetch"]:
        print(f"Invalid mode: {mode}")
        print("Valid modes: submit, check, fetch")
        sys.exit(1)

    print(f"=== Full Dataset Mode: Processing ALL stories ===\n")
    run(mode=mode, n_stories=None, job_suffix="full")
