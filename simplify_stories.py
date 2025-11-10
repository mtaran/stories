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
        self.input_jsonl_path = BATCH_DIR / f"batch_input_{job_name_suffix}.jsonl"
        self.job_info_path = BATCH_DIR / f"job_info_{job_name_suffix}.json"
        self.results_path = BATCH_DIR / f"results_{job_name_suffix}.jsonl"
        self.output_parquet_dir = BATCH_DIR / f"output_{job_name_suffix}"

    def submit(self):
        """Submit batch job to Gemini API."""
        print(f"Loading SimpleStories dataset...")
        dataset = load_dataset("SimpleStories/SimpleStories", split="train")

        if self.n_stories:
            dataset = dataset.select(range(self.n_stories))
            print(f"Selected first {self.n_stories} stories")
        else:
            print(f"Processing all {len(dataset)} stories")

        # Prepare batch requests
        print(f"Preparing batch requests...")
        with open(self.input_jsonl_path, "w") as f:
            for idx, example in enumerate(dataset):
                story = example["story"]
                request = {
                    "key": f"story-{idx}",
                    "request": {
                        "contents": [{
                            "parts": [{
                                "text": PROMPT_TEMPLATE.format(story=story)
                            }]
                        }]
                    }
                }
                f.write(json.dumps(request) + "\n")

        print(f"Wrote {len(dataset)} requests to {self.input_jsonl_path}")

        # Upload file to Gemini File API
        print(f"Uploading batch file to Gemini File API...")
        uploaded_file = self.client.files.upload(
            file=str(self.input_jsonl_path),
            config=types.UploadFileConfig(
                display_name=f'simplify-stories-{self.job_name_suffix}',
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
                'display_name': f"simplify-stories-{self.job_name_suffix}",
            },
        )

        # Save job info for later
        job_info = {
            "job_name": batch_job.name,
            "uploaded_file_name": uploaded_file.name,
            "n_stories": len(dataset),
            "created_at": time.time()
        }
        with open(self.job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"✓ Batch job created: {batch_job.name}")
        print(f"✓ Job info saved to: {self.job_info_path}")
        print(f"\nNext step: run with 'check' to monitor status")

        return batch_job.name

    def check(self):
        """Check the status of the batch job."""
        if not self.job_info_path.exists():
            print(f"❌ No job info found at {self.job_info_path}")
            print("Run with 'submit' first")
            return None

        with open(self.job_info_path) as f:
            job_info = json.load(f)

        job_name = job_info["job_name"]
        print(f"Checking status of job: {job_name}")

        batch_job = self.client.batches.get(name=job_name)

        print(f"\nJob Status: {batch_job.state.name}")

        if hasattr(batch_job, 'request_count'):
            print(f"Total requests: {batch_job.request_count}")

        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            print("✓ Job completed successfully!")
            print(f"\nNext step: run with 'fetch' to download results")
        elif batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"❌ Job failed!")
            if hasattr(batch_job, 'error'):
                print(f"Error: {batch_job.error}")
        elif batch_job.state.name == 'JOB_STATE_RUNNING':
            print("⏳ Job is still running...")
            print("Check again in a few minutes")
        elif batch_job.state.name == 'JOB_STATE_PENDING':
            print("⏳ Job is pending...")
            print("Check again in a few minutes")
        elif batch_job.state.name == 'JOB_STATE_CANCELLED':
            print("❌ Job was cancelled")
        elif batch_job.state.name == 'JOB_STATE_EXPIRED':
            print("❌ Job expired (took longer than 48 hours)")

        return batch_job.state.name

    def fetch(self):
        """Fetch results and create new parquet files with simplified column."""
        if not self.job_info_path.exists():
            print(f"❌ No job info found at {self.job_info_path}")
            print("Run with 'submit' first")
            return

        with open(self.job_info_path) as f:
            job_info = json.load(f)

        job_name = job_info["job_name"]
        print(f"Fetching results for job: {job_name}")

        batch_job = self.client.batches.get(name=job_name)

        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            print(f"❌ Job is not in succeeded state: {batch_job.state.name}")
            print("Run 'check' to see current status")
            return

        # Download results file
        if batch_job.dest and batch_job.dest.file_name:
            result_file_name = batch_job.dest.file_name
            print(f"Downloading results from: {result_file_name}")

            file_content = self.client.files.download(file=result_file_name)

            # Save results locally
            with open(self.results_path, "wb") as f:
                f.write(file_content)

            print(f"✓ Results saved to: {self.results_path}")
        else:
            print("❌ No result file found in batch job")
            return

        # Parse results
        print("Parsing results...")
        results_dict = {}

        with open(self.results_path, "r") as f:
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
                                results_dict[key] = simplified_text
                    except (KeyError, IndexError) as e:
                        print(f"Warning: Could not parse response for {key}: {e}")
                        results_dict[key] = ""
                elif "error" in result:
                    print(f"Warning: Error for {key}: {result['error']}")
                    results_dict[key] = ""

        print(f"✓ Parsed {len(results_dict)} results")

        # Load original dataset
        print("Loading original dataset...")
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
        simplified_data = [(int(key.split('-')[1]), text) for key, text in results_dict.items()]
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
