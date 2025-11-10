#!/usr/bin/env python3
"""
Upload SimplerStories dataset to HuggingFace Hub.

This script:
1. Loads the original SimpleStories/SimpleStories dataset (both train and test splits)
2. Loads the simplified texts from both train and test parquet files
3. Adds the 'simplified' column at the BEGINNING of each split
4. Uploads to HuggingFace Hub under mtaran/SimplerStories

Usage:
    python upload_to_hub.py <path_to_train_parquet> <path_to_test_parquet>

Example:
    python upload_to_hub.py batch_data/output_full_train/stories_simplified.parquet batch_data/output_full_test/stories_simplified.parquet
"""

import sys
from pathlib import Path

import duckdb
from datasets import load_dataset, DatasetDict


def load_simplified_texts(parquet_path: Path, split_name: str):
    """
    Load simplified texts from a parquet file.

    Args:
        parquet_path: Path to the parquet file
        split_name: Name of the split (for logging)

    Returns:
        List of simplified texts
    """
    if not parquet_path.exists():
        print(f"❌ Parquet file not found: {parquet_path}")
        print(f"Make sure you've run 'python run_full.py fetch {split_name}' first")
        sys.exit(1)

    print(f"📂 Loading simplified texts from {split_name} parquet: {parquet_path}")
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE simplified_data AS SELECT * FROM read_parquet('{parquet_path}')")

    # Get the simplified column as a list
    simplified_texts = conn.execute("SELECT simplified FROM simplified_data ORDER BY ROWID").fetchall()
    simplified_texts = [row[0] for row in simplified_texts]

    conn.close()

    print(f"✓ Loaded {len(simplified_texts):,} simplified texts from {split_name}")
    return simplified_texts


def upload_simpler_stories(train_parquet_path: str, test_parquet_path: str):
    """
    Upload SimplerStories dataset to HuggingFace Hub.

    Args:
        train_parquet_path: Path to the train parquet file with simplified column
        test_parquet_path: Path to the test parquet file with simplified column
    """
    train_path = Path(train_parquet_path)
    test_path = Path(test_parquet_path)

    # Load the original SimpleStories dataset
    print(f"📂 Loading original SimpleStories/SimpleStories dataset...\n")
    original_dataset = load_dataset("SimpleStories/SimpleStories")

    train_dataset = original_dataset["train"]
    test_dataset = original_dataset["test"]

    print(f"✓ Loaded train split: {len(train_dataset):,} examples")
    print(f"✓ Loaded test split: {len(test_dataset):,} examples")

    # Load simplified texts from both parquet files
    print()
    train_simplified = load_simplified_texts(train_path, "train")
    test_simplified = load_simplified_texts(test_path, "test")

    # Verify counts match
    if len(train_dataset) != len(train_simplified):
        print(f"❌ Train mismatch! Dataset has {len(train_dataset):,} examples but parquet has {len(train_simplified):,}")
        sys.exit(1)

    if len(test_dataset) != len(test_simplified):
        print(f"❌ Test mismatch! Dataset has {len(test_dataset):,} examples but parquet has {len(test_simplified):,}")
        sys.exit(1)

    # Add the simplified column to the train dataset
    print(f"\n📊 Adding 'simplified' column to train split...")

    def add_simplified_train(example, idx):
        return {"simplified": train_simplified[idx], **example}

    train_dataset = train_dataset.map(add_simplified_train, with_indices=True, desc="Adding simplified column to train")

    # Add the simplified column to the test dataset
    print(f"📊 Adding 'simplified' column to test split...")

    def add_simplified_test(example, idx):
        return {"simplified": test_simplified[idx], **example}

    test_dataset = test_dataset.map(add_simplified_test, with_indices=True, desc="Adding simplified column to test")

    # Create a DatasetDict with both splits
    final_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    print(f"\n✓ Dataset updated:")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Test: {len(test_dataset):,} examples")
    print(f"✓ Columns: {', '.join(train_dataset.column_names)}")

    # Show samples from both splits
    print(f"\n📖 Sample from train split (first example):")
    train_sample = train_dataset[0]
    print(f"  Simplified: {train_sample['simplified'][:150]}...")
    if 'story' in train_sample:
        print(f"  Original:   {train_sample['story'][:150]}...")

    print(f"\n📖 Sample from test split (first example):")
    test_sample = test_dataset[0]
    print(f"  Simplified: {test_sample['simplified'][:150]}...")
    if 'story' in test_sample:
        print(f"  Original:   {test_sample['story'][:150]}...")

    # Upload to HuggingFace Hub
    print(f"\n🚀 Uploading to HuggingFace Hub: mtaran/SimplerStories")
    print(f"   (This may take a while for large datasets...)")

    try:
        final_dataset.push_to_hub(
            "mtaran/SimplerStories",
            private=False,
            commit_message="Add simplified stories column to both train and test splits"
        )
        print(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/mtaran/SimplerStories")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\nMake sure you're logged in to HuggingFace Hub:")
        print(f"  huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_to_hub.py <path_to_train_parquet> <path_to_test_parquet>")
        print("\nExample:")
        print("  python upload_to_hub.py batch_data/output_full_train/stories_simplified.parquet batch_data/output_full_test/stories_simplified.parquet")
        sys.exit(1)

    train_parquet_path = sys.argv[1]
    test_parquet_path = sys.argv[2]
    upload_simpler_stories(train_parquet_path, test_parquet_path)
