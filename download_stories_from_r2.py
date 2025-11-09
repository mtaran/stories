#!/usr/bin/env python3
"""
Download stories from R2 bucket by slice range.
Usage: python download_stories_from_r2.py "0:10"
Downloads stories/0000000.json through stories/0000009.json
"""

import sys
import os
import json
import boto3
from botocore.client import Config
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional, will use environment variables directly

# R2 configuration from environment
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
BUCKET_NAME = "mechinterp"

# 2M items = 2,000,000 -> max index 1,999,999 = 7 digits
PADDING_WIDTH = 7


def parse_slice(slice_str):
    """Parse slice string like '0:10' or '100:200' into start, end."""
    parts = slice_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice format: {slice_str}. Expected format: 'start:end'")

    start = int(parts[0])
    end = int(parts[1])

    if start < 0 or end < 0:
        raise ValueError("Slice indices must be non-negative")
    if start >= end:
        raise ValueError("Start index must be less than end index")

    return start, end


def download_stories(start, end):
    """Download stories from R2 in the specified range."""

    # Configure R2 client (S3-compatible)
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    # Create output directory
    output_dir = Path("downloaded_stories")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading stories {start} to {end-1} from R2...")

    for idx in range(start, end):
        # Pad index to 7 digits
        filename = f"{idx:0{PADDING_WIDTH}d}.json"
        s3_key = f"stories/{filename}"
        local_path = output_dir / filename

        try:
            # Download from R2
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            content = response["Body"].read()

            # Save to local file
            with open(local_path, "wb") as f:
                f.write(content)

            if (idx - start + 1) % 10 == 0:
                print(f"  Downloaded {idx - start + 1}/{end - start} files...")

        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")

    print(f"✓ Downloaded {end - start} files to {output_dir}/")


if __name__ == "__main__":
    # Validate environment variables
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("Error: Missing R2 credentials in environment variables:")
        print("  R2_ACCOUNT_ID")
        print("  R2_ACCESS_KEY_ID")
        print("  R2_SECRET_ACCESS_KEY")
        exit(1)

    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python download_stories_from_r2.py 'start:end'")
        print("Example: python download_stories_from_r2.py '0:10'")
        exit(1)

    try:
        start, end = parse_slice(sys.argv[1])
        download_stories(start, end)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
