"""
Main module for simplifying SimpleStories dataset using Gemini Batch API.
"""

import json
import os
import time
from pathlib import Path
from typing import Literal

import duckdb
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Constants
PROMPT_TEMPLATE = """Simplify the vocabulary, flowery/metaphorical expressions and grammar such that a 4-5 year old would understand:
\"\"\"
{story}
\"\"\"
Output the simplified version of the story and nothing else!"""

MODEL_NAME = "gemini-2.0-flash"
BATCH_DIR = Path("batch_data")
BATCH_DIR.mkdir(exist_ok=True)
MAX_STORIES_PER_BATCH = 10000


class StorySimplifier:
    """Handles the batch simplification of stories using Gemini API."""

    def __init__(self, n_stories: int = None, job_name_suffix: str = ""):
        """
        Initialize the simplifier.

        Args:
            n_stories: Number of stories to process. None for all.
            job_name_suffix: Suffix for job name (e.g., "poc" or "full")
        """
        self.n_stories = n_stories
        self.job_name_suffix = job_name_suffix
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # File paths
        self.job_info_path = BATCH_DIR / f"job_info_{job_name_suffix}.json"
        self.output_parquet_dir = BATCH_DIR / f"output_{job_name_suffix}"

    def submit(self):
        """Submit batch jobs to Gemini API (split into chunks of 10k stories)."""
        print(f"Loading SimpleStories dataset...")
        dataset = load_dataset("SimpleStories/SimpleStories", split="train")

        if self.n_stories:
            dataset = dataset.select(range(self.n_stories))
            print(f"Selected first {self.n_stories} stories")
        else:
            print(f"Processing all {len(dataset)} stories")

        total_stories = len(dataset)
        num_batches = (total_stories + MAX_STORIES_PER_BATCH - 1) // MAX_STORIES_PER_BATCH

        print(f"\n📊 Splitting into {num_batches} batch(es) of max {MAX_STORIES_PER_BATCH} stories each")

        all_batch_info = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * MAX_STORIES_PER_BATCH
            end_idx = min(start_idx + MAX_STORIES_PER_BATCH, total_stories)
            batch_dataset = dataset.select(range(start_idx, end_idx))

            print(f"\n--- Batch {batch_idx + 1}/{num_batches} (stories {start_idx}-{end_idx-1}) ---")

            # Create JSONL file for this batch
            input_jsonl_path = BATCH_DIR / f"batch_input_{self.job_name_suffix}_{batch_idx}.jsonl"

            print(f"Preparing batch requests...")
            with open(input_jsonl_path, "w") as f:
                for idx, example in enumerate(batch_dataset):
                    story = example["story"]
                    # Use global index for key
                    global_idx = start_idx + idx
                    request = {
                        "key": f"story-{global_idx}",
                        "request": {
                            "contents": [{
                                "parts": [{
                                    "text": PROMPT_TEMPLATE.format(story=story)
                                }]
                            }],
                            "generation_config": {
                                "temperature": 0.2
                            }
                        }
                    }
                    f.write(json.dumps(request) + "\n")

            print(f"Wrote {len(batch_dataset)} requests to {input_jsonl_path}")

            # Upload file to Gemini File API
            print(f"Uploading batch file to Gemini File API...")
            uploaded_file = self.client.files.upload(
                file=str(input_jsonl_path),
                config=types.UploadFileConfig(
                    display_name=f'simplify-stories-{self.job_name_suffix}-{batch_idx}',
                    mime_type='text/plain'
                )
            )
            print(f"Uploaded file: {uploaded_file.name}")

            # Create batch job
            print(f"Creating batch job...")
            batch_job = self.client.batches.create(
                model=MODEL_NAME,
                src=uploaded_file.name,
                config={
                    'display_name': f"simplify-stories-{self.job_name_suffix}-{batch_idx}",
                },
            )

            # Save batch info
            batch_info = {
                "batch_idx": batch_idx,
                "job_name": batch_job.name,
                "uploaded_file_name": uploaded_file.name,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "n_stories": len(batch_dataset),
                "created_at": time.time()
            }
            all_batch_info.append(batch_info)

            print(f"✓ Batch job created: {batch_job.name}")

            # Clean up input JSONL file
            input_jsonl_path.unlink()
            print(f"✓ Cleaned up input file: {input_jsonl_path}")

        # Save all batch info
        job_info = {
            "total_stories": total_stories,
            "num_batches": num_batches,
            "batches": all_batch_info,
            "created_at": time.time()
        }
        with open(self.job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"\n✓ All {num_batches} batch job(s) submitted!")
        print(f"✓ Job info saved to: {self.job_info_path}")
        print(f"\nNext step: run with 'check' to monitor status")

    def check(self):
        """Check the status of all batch jobs."""
        if not self.job_info_path.exists():
            print(f"❌ No job info found at {self.job_info_path}")
            print("Run with 'submit' first")
            return None

        with open(self.job_info_path) as f:
            job_info = json.load(f)

        batches = job_info["batches"]
        num_batches = job_info["num_batches"]

        print(f"📊 Checking status of {num_batches} batch job(s)...\n")

        status_counts = {
            "JOB_STATE_SUCCEEDED": 0,
            "JOB_STATE_RUNNING": 0,
            "JOB_STATE_PENDING": 0,
            "JOB_STATE_FAILED": 0,
            "JOB_STATE_CANCELLED": 0,
            "JOB_STATE_EXPIRED": 0
        }

        for batch in batches:
            batch_idx = batch["batch_idx"]
            job_name = batch["job_name"]

            try:
                batch_job = self.client.batches.get(name=job_name)
                state = batch_job.state.name
                status_counts[state] = status_counts.get(state, 0) + 1

                status_emoji = {
                    "JOB_STATE_SUCCEEDED": "✓",
                    "JOB_STATE_RUNNING": "⏳",
                    "JOB_STATE_PENDING": "⏳",
                    "JOB_STATE_FAILED": "❌",
                    "JOB_STATE_CANCELLED": "❌",
                    "JOB_STATE_EXPIRED": "❌"
                }.get(state, "?")

                print(f"{status_emoji} Batch {batch_idx + 1}/{num_batches}: {state} (stories {batch['start_idx']}-{batch['end_idx']-1})")

                if state == "JOB_STATE_FAILED" and hasattr(batch_job, 'error'):
                    print(f"   Error: {batch_job.error}")

            except Exception as e:
                print(f"❌ Batch {batch_idx + 1}/{num_batches}: Error checking status: {e}")
                status_counts["JOB_STATE_FAILED"] += 1

        print(f"\n📈 Summary:")
        print(f"   ✓ Succeeded: {status_counts['JOB_STATE_SUCCEEDED']}/{num_batches}")
        print(f"   ⏳ Running: {status_counts['JOB_STATE_RUNNING']}/{num_batches}")
        print(f"   ⏳ Pending: {status_counts['JOB_STATE_PENDING']}/{num_batches}")
        print(f"   ❌ Failed: {status_counts['JOB_STATE_FAILED']}/{num_batches}")
        print(f"   ❌ Cancelled: {status_counts['JOB_STATE_CANCELLED']}/{num_batches}")
        print(f"   ❌ Expired: {status_counts['JOB_STATE_EXPIRED']}/{num_batches}")

        if status_counts['JOB_STATE_SUCCEEDED'] == num_batches:
            print(f"\n✓ All batches completed successfully!")
            print(f"Next step: run with 'fetch' to download results")
        elif status_counts['JOB_STATE_RUNNING'] > 0 or status_counts['JOB_STATE_PENDING'] > 0:
            print(f"\n⏳ Some batches still in progress. Check again in a few minutes.")
        else:
            print(f"\n❌ Some batches have failed or been cancelled.")

        return status_counts

    def fetch(self):
        """Fetch results from all batches and create new parquet files with simplified column."""
        if not self.job_info_path.exists():
            print(f"❌ No job info found at {self.job_info_path}")
            print("Run with 'submit' first")
            return

        with open(self.job_info_path) as f:
            job_info = json.load(f)

        batches = job_info["batches"]
        num_batches = job_info["num_batches"]
        total_stories = job_info["total_stories"]

        print(f"📊 Checking all {num_batches} batch(es) before fetching...\n")

        # First, check that all batches are succeeded
        all_succeeded = True
        for batch in batches:
            batch_idx = batch["batch_idx"]
            job_name = batch["job_name"]

            batch_job = self.client.batches.get(name=job_name)

            if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
                print(f"❌ Batch {batch_idx + 1}/{num_batches} is not in succeeded state: {batch_job.state.name}")
                all_succeeded = False

        if not all_succeeded:
            print(f"\n❌ Not all batches have succeeded. Run 'check' to see current status")
            return

        print(f"✓ All batches succeeded! Proceeding with fetch...\n")

        # Download and parse all results
        all_results = {}

        for batch in batches:
            batch_idx = batch["batch_idx"]
            job_name = batch["job_name"]

            print(f"--- Batch {batch_idx + 1}/{num_batches} ---")
            print(f"Fetching results for: {job_name}")

            batch_job = self.client.batches.get(name=job_name)

            # Download results file
            if batch_job.dest and batch_job.dest.file_name:
                result_file_name = batch_job.dest.file_name
                print(f"Downloading results from: {result_file_name}")

                file_content = self.client.files.download(file=result_file_name)

                # Save results temporarily
                results_path = BATCH_DIR / f"results_{self.job_name_suffix}_{batch_idx}.jsonl"
                with open(results_path, "wb") as f:
                    f.write(file_content)

                print(f"✓ Results saved temporarily to: {results_path}")

                # Parse results
                print("Parsing results...")
                with open(results_path, "r") as f:
                    for line in f:
                        result = json.loads(line)
                        key = result.get("key")

                        if "response" in result:
                            # Extract the simplified text
                            try:
                                response = result["response"]
                                candidates = response.get("candidates", [])
                                if candidates:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    if parts:
                                        simplified_text = parts[0].get("text", "")
                                        all_results[key] = simplified_text
                            except (KeyError, IndexError) as e:
                                print(f"Warning: Could not parse response for {key}: {e}")
                                all_results[key] = ""
                        elif "error" in result:
                            print(f"Warning: Error for {key}: {result['error']}")
                            all_results[key] = ""

                print(f"✓ Parsed {len(all_results)} total results so far")

                # Clean up results JSONL file
                results_path.unlink()
                print(f"✓ Cleaned up results file: {results_path}")

            else:
                print(f"❌ No result file found for batch {batch_idx + 1}")

        print(f"\n✓ Fetched and parsed all {len(all_results)} results")

        # Load original dataset
        print("\nLoading original dataset...")
        dataset = load_dataset("SimpleStories/SimpleStories", split="train")

        if self.n_stories:
            dataset = dataset.select(range(self.n_stories))

        # Create temporary parquet with original data
        temp_original_path = BATCH_DIR / f"temp_original_{self.job_name_suffix}.parquet"
        dataset.to_parquet(str(temp_original_path))

        # Use DuckDB to add simplified column
        print("Creating new parquet files with simplified column...")
        self.output_parquet_dir.mkdir(exist_ok=True)

        conn = duckdb.connect()

        # Load original data with explicit row index
        conn.execute(f"""
            CREATE TABLE original AS
            SELECT
                ROW_NUMBER() OVER () - 1 AS idx,
                *
            FROM read_parquet('{temp_original_path}')
        """)

        # Create a table with simplified results
        # Extract index from key (story-0 -> 0) and convert to int
        simplified_data = [(int(key.split('-')[1]), text) for key, text in all_results.items()]
        conn.execute("CREATE TABLE simplified (idx INTEGER, simplified_text VARCHAR)")
        conn.executemany("INSERT INTO simplified VALUES (?, ?)", simplified_data)

        # Join and create new table using the explicit index
        conn.execute("""
            CREATE TABLE result AS
            SELECT
                original.* EXCLUDE (idx),
                COALESCE(simplified.simplified_text, '') as simplified
            FROM original
            LEFT JOIN simplified ON original.idx = simplified.idx
        """)

        # Export to parquet
        output_parquet_path = self.output_parquet_dir / "stories_simplified.parquet"
        conn.execute(f"COPY result TO '{output_parquet_path}' (FORMAT PARQUET)")

        print(f"✓ Created parquet file: {output_parquet_path}")

        # Clean up temp file
        temp_original_path.unlink()
        print(f"✓ Cleaned up temp file: {temp_original_path}")

        # Show sample
        print("\nSample of results:")
        sample = conn.execute("SELECT story, simplified FROM result LIMIT 3").fetchall()
        for idx, (original, simplified) in enumerate(sample, 1):
            print(f"\n--- Story {idx} ---")
            print(f"Original: {original[:200]}...")
            print(f"Simplified: {simplified[:200]}...")

        conn.close()

        print(f"\n✓ Done! Output saved to: {self.output_parquet_dir}")


def run(mode: Literal["submit", "check", "fetch"], n_stories: int = None, job_suffix: str = ""):
    """
    Run the story simplification pipeline.

    Args:
        mode: Operation mode - "submit", "check", or "fetch"
        n_stories: Number of stories to process (None for all)
        job_suffix: Suffix for job identification
    """
    simplifier = StorySimplifier(n_stories=n_stories, job_name_suffix=job_suffix)

    if mode == "submit":
        simplifier.submit()
    elif mode == "check":
        simplifier.check()
    elif mode == "fetch":
        simplifier.fetch()
    else:
        print(f"❌ Invalid mode: {mode}")
        print("Valid modes: submit, check, fetch")
