#!/usr/bin/env python3
"""
Upload SimplerStories dataset to HuggingFace Hub.

This script:
1. Loads the original SimpleStories/SimpleStories dataset
2. Loads the simplified texts from the parquet file created by simplify_stories.py
3. Adds the 'simplified' column at the BEGINNING of the dataset
4. Uploads to HuggingFace Hub under mtaran/SimplerStories

Usage:
    python upload_to_hub.py <path_to_simplified_parquet>

Example:
    python upload_to_hub.py batch_data/output_full/stories_simplified.parquet
"""

import sys
from pathlib import Path

import duckdb
from datasets import load_dataset


def upload_simpler_stories(simplified_parquet_path: str):
    """
    Upload SimplerStories dataset to HuggingFace Hub.

    Args:
        simplified_parquet_path: Path to the parquet file with simplified column
    """
    parquet_path = Path(simplified_parquet_path)

    if not parquet_path.exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        print(f"Make sure you've run 'python run_full.py fetch' or 'python run_poc.py fetch' first")
        sys.exit(1)

    # Load the original SimpleStories dataset
    print(f"📂 Loading original SimpleStories/SimpleStories dataset...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="train")
    print(f"✓ Loaded {len(dataset):,} examples from SimpleStories")

    # Load simplified texts from parquet
    print(f"\n📂 Loading simplified texts from: {parquet_path}")
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE simplified_data AS SELECT * FROM read_parquet('{parquet_path}')")

    # Get the simplified column as a list
    simplified_texts = conn.execute("SELECT simplified FROM simplified_data ORDER BY ROWID").fetchall()
    simplified_texts = [row[0] for row in simplified_texts]

    conn.close()

    print(f"✓ Loaded {len(simplified_texts):,} simplified texts")

    # Verify counts match
    if len(dataset) != len(simplified_texts):
        print(f"❌ Mismatch! Dataset has {len(dataset):,} examples but parquet has {len(simplified_texts):,}")
        print(f"Make sure the parquet file corresponds to the same portion of the dataset")
        sys.exit(1)

    # Add the simplified column to the dataset
    print(f"\n📊 Adding 'simplified' column to dataset...")

    def add_simplified(example, idx):
        # Add simplified as the first field by returning a new dict with simplified first
        return {"simplified": simplified_texts[idx], **example}

    dataset = dataset.map(add_simplified, with_indices=True, desc="Adding simplified column")

    print(f"✓ Dataset updated with {len(dataset):,} examples")
    print(f"✓ Columns: {', '.join(dataset.column_names)}")

    # Show sample
    print(f"\n📖 Sample (first example):")
    sample = dataset[0]
    print(f"  Simplified: {sample['simplified'][:150]}...")
    if 'story' in sample:
        print(f"  Original:   {sample['story'][:150]}...")

    # Upload to HuggingFace Hub
    print(f"\n🚀 Uploading to HuggingFace Hub: mtaran/SimplerStories")
    print(f"   (This may take a while for large datasets...)")

    try:
        dataset.push_to_hub(
            "mtaran/SimplerStories",
            private=False,
            commit_message="Add simplified stories column to SimpleStories dataset"
        )
        print(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/mtaran/SimplerStories")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\nMake sure you're logged in to HuggingFace Hub:")
        print(f"  huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_to_hub.py <path_to_simplified_parquet>")
        print("\nExample:")
        print("  python upload_to_hub.py batch_data/output_full/stories_simplified.parquet")
        print("  python upload_to_hub.py batch_data/output_poc/stories_simplified.parquet")
        sys.exit(1)

    simplified_parquet_path = sys.argv[1]
    upload_simpler_stories(simplified_parquet_path)
