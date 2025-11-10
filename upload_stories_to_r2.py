#!/usr/bin/env python3
"""
Upload ALL items from SimpleStories dataset to R2 bucket.
Each item saved as stories/{N}.json where N is 0-padded for 2M items (7 digits).
Processes in batches to manage memory usage.
"""

import json
import os
import boto3
from datasets import load_dataset
from botocore.client import Config
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# R2 configuration from environment
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
BUCKET_NAME = "mechinterp"

# 2M items = 2,000,000 -> max index 1,999,999 = 7 digits
PADDING_WIDTH = 7
MAX_WORKERS = 20  # Number of parallel upload threads
BATCH_SIZE = 10000  # Process this many items at a time to manage memory

# Thread-safe counter for progress reporting
upload_counter = 0
counter_lock = Lock()


def upload_single_item(idx, item):
    """Upload a single item to R2."""
    global upload_counter

    # Create a new S3 client for this thread
    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    # Pad index to 7 digits (for 2M items: 0000000 to 1999999)
    filename = f"{idx:0{PADDING_WIDTH}d}.json"
    s3_key = f"stories/{filename}"

    # Convert item to JSON
    json_data = json.dumps(item, ensure_ascii=False, indent=2)

    # Upload to R2
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=json_data.encode("utf-8"),
        ContentType="application/json"
    )

    # Update progress counter (thread-safe)
    with counter_lock:
        upload_counter += 1
        if upload_counter % 100 == 0:
            print(f"  Uploaded {upload_counter} items...")

    return idx


def upload_batch(batch_items):
    """Upload a batch of items in parallel."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all upload tasks for this batch
        futures = [executor.submit(upload_single_item, idx, item) for idx, item in batch_items]

        # Wait for all uploads to complete
        for future in as_completed(futures):
            try:
                future.result()  # Raises exception if upload failed
            except Exception as e:
                print(f"Error uploading item: {e}")


def upload_to_r2():
    """Stream ALL items from SimpleStories and upload to R2 in batches."""
    global upload_counter
    upload_counter = 0

    # Load dataset in streaming mode to avoid downloading everything
    print("Loading SimpleStories dataset in streaming mode...")
    dataset = load_dataset(
        "SimpleStories/SimpleStories",
        split="train",
        streaming=True
    )

    print(f"Starting upload with {MAX_WORKERS} workers, processing in batches of {BATCH_SIZE}...")

    # Process dataset in batches
    batch = []
    current_idx = 0

    for item in dataset:
        batch.append((current_idx, item))
        current_idx += 1

        # When batch is full, upload it
        if len(batch) >= BATCH_SIZE:
            print(f"Uploading batch {current_idx - BATCH_SIZE} to {current_idx - 1}...")
            upload_batch(batch)
            batch = []

    # Upload any remaining items in the last batch
    if batch:
        print(f"Uploading final batch ({len(batch)} items)...")
        upload_batch(batch)

    max_idx = current_idx - 1
    print(f"✓ Successfully uploaded {current_idx} items to s3://{BUCKET_NAME}/stories/")
    print(f"  Files: {0:0{PADDING_WIDTH}d}.json through {max_idx:0{PADDING_WIDTH}d}.json")


if __name__ == "__main__":
    # Validate environment variables
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        print("Error: Missing R2 credentials in environment variables:")
        print("  R2_ACCOUNT_ID")
        print("  R2_ACCESS_KEY_ID")
        print("  R2_SECRET_ACCESS_KEY")
        exit(1)

    upload_to_r2()
