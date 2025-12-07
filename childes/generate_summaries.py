#!/usr/bin/env python3
"""
Generate narrative summaries from CHILDES CHAT transcripts.

Downloads corpus data from TalkBank, extracts utterances using pylangacq,
and prepares data for summarization.

Usage:
    python generate_summaries.py [OPTIONS]

Examples:
    # Extract first 5 sessions from Hall corpus (default)
    python generate_summaries.py

    # Extract specific number of sessions
    python generate_summaries.py --sessions 10

    # Extract from a different corpus
    python generate_summaries.py --corpus Brown --sessions 3

    # Specify language group
    python generate_summaries.py --language Eng-NA --corpus MacWhinney

Requirements:
    pip install pylangacq requests
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pylangacq
import requests


# Speaker role mapping for readable names
SPEAKER_NAMES = {
    'CHI': 'the child',
    'MOT': 'the mother',
    'FAT': 'the father',
    'EXP': 'the investigator',
    'TEA': 'the teacher',
    'SIS': 'the sister',
    'BRO': 'the brother',
    'GRM': 'the grandmother',
    'GRF': 'the grandfather',
    'FCH': 'another child',
    'MCH': 'another child',
    'UNK': 'someone',
    'GRO': 'a group member',
    'TEL': 'the television',
    'ADU': 'an adult',
}


def get_speaker_name(code: str) -> str:
    """Get readable speaker name from code."""
    return SPEAKER_NAMES.get(code, code)


def login_talkbank(email: str, password: str) -> requests.Session:
    """Authenticate with TalkBank and return a session."""
    session = requests.Session()

    login_url = "https://sla2.talkbank.org/logInUser"
    login_data = {"email": email, "pswd": password}

    resp = session.post(login_url, json=login_data, timeout=30)
    result = resp.json()

    if not result.get('success'):
        raise RuntimeError(f"TalkBank login failed: {result.get('respMsg', 'Unknown error')}")

    print(f"Logged in to TalkBank as {email}")
    return session


def download_corpus(session: requests.Session, language: str, corpus: str, output_dir: str) -> str:
    """Download a corpus from TalkBank and extract it."""
    url = f"https://git.talkbank.org/childes/data/{language}/{corpus}.zip"
    print(f"Downloading {corpus} corpus from {url}...")

    resp = session.get(url, timeout=300)

    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download corpus: HTTP {resp.status_code}")

    if len(resp.content) < 1000 or resp.content[:2] != b'PK':
        raise RuntimeError(f"Downloaded file is not a valid zip archive")

    # Save and extract
    zip_path = os.path.join(output_dir, f"{corpus}.zip")
    with open(zip_path, 'wb') as f:
        f.write(resp.content)

    extract_dir = os.path.join(output_dir, corpus)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def clean_chat_content(content: str) -> str:
    """Remove %mor and %gra tiers from CHAT content to avoid alignment issues."""
    content = re.sub(r'^%mor:\t.+\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^%gra:\t.+\n', '', content, flags=re.MULTILINE)
    return content


def extract_utterances(filepath: str, temp_dir: str) -> dict:
    """
    Extract utterances from a CHAT file using pylangacq.
    Returns dict with situation, participants, and utterances.
    """
    # Read and clean the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    cleaned_content = clean_chat_content(content)

    # Write cleaned file to temp location for pylangacq
    basename = os.path.basename(filepath)
    clean_path = os.path.join(temp_dir, basename)
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    # Parse with pylangacq
    reader = pylangacq.read_chat(clean_path)

    # Get headers for metadata
    headers = reader.headers()[0] if reader.headers() else {}
    situation = headers.get('Situation', 'everyday conversation')

    # Get participants (returns a set of speaker codes)
    participants = reader.participants()
    participant_info = sorted(participants) if participants else []

    # Extract utterances
    utterances = []
    for utt in reader.utterances():
        participant = utt.participant
        words = [t.word for t in utt.tokens if t.word]
        text = ' '.join(words)

        if text.strip():
            utterances.append({
                'speaker': participant,
                'text': text
            })

    return {
        'situation': situation,
        'participants': ', '.join(participant_info) if participant_info else 'Unknown',
        'utterances': utterances
    }


def format_transcript_for_summary(utterances: list, situation: str) -> str:
    """Format all utterances for the summarization prompt."""
    lines = []
    for utt in utterances:
        speaker_name = get_speaker_name(utt['speaker'])
        lines.append(f"{speaker_name}: {utt['text']}")

    transcript = '\n'.join(lines)

    # Calculate target sentence count (10-30 utterances per sentence)
    n_utterances = len(utterances)
    min_sentences = max(n_utterances // 30, 10)
    max_sentences = max(n_utterances // 10, 20)

    return f"""This is a transcript from a child language study.
Context: {situation}

The transcript has {n_utterances} utterances. Write a narrative summary of {min_sentences}-{max_sentences} sentences.

IMPORTANT RULES:
- Each sentence must describe a SPECIFIC event, topic, or exchange from the transcript
- NEVER use generic filler like "The conversation continued" or "They discussed various topics"
- Include concrete details: names of objects, games, foods, places, people mentioned
- Capture the flow: what happened first, then what, any conflicts or resolutions
- Note emotional moments: excitement, frustration, silliness, affection

Write in past tense, third person.

Transcript:
{transcript}

Now write the narrative summary ({min_sentences}-{max_sentences} sentences). Remember: every sentence must contain SPECIFIC details from the transcript, no generic filler:"""


def process_session(filepath: str, temp_dir: str, output_dir: str) -> dict:
    """Process a single CHAT session file and extract data for summarization."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nProcessing: {basename}")

    # Extract utterances using pylangacq
    data = extract_utterances(filepath, temp_dir)
    utterances = data['utterances']
    situation = data['situation']
    participants = data['participants']

    print(f"  Situation: {situation[:80]}{'...' if len(situation) > 80 else ''}")
    print(f"  Total utterances: {len(utterances)}")

    if len(utterances) == 0:
        print("  No utterances found, skipping.")
        return None

    # Generate the prompt for summarization
    prompt = format_transcript_for_summary(utterances, situation)

    # Save extracted data as JSON for Claude Code to summarize
    output = {
        'session_id': basename,
        'situation': situation,
        'participants': participants,
        'total_utterances': len(utterances),
        'utterances': utterances,
        'summarization_prompt': prompt
    }

    output_path = os.path.join(output_dir, f"{basename}_extracted.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved extracted data to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Extract CHILDES CHAT transcripts for summarization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Default: 5 sessions from Eng-NA/Hall
  %(prog)s --sessions 10                      # First 10 sessions from Hall
  %(prog)s --corpus Brown --sessions 3        # First 3 sessions from Brown
  %(prog)s --language Eng-UK --corpus Belfast # Belfast corpus from Eng-UK
        """
    )
    parser.add_argument('--language', '-l', default='Eng-NA',
                        help='Language group (default: Eng-NA)')
    parser.add_argument('--corpus', '-c', default='Hall',
                        help='Corpus name (default: Hall)')
    parser.add_argument('--sessions', '-n', type=int, default=5,
                        help='Number of sessions to process (default: 5)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: ./childes/extracted)')
    parser.add_argument('--email', default='maksym.taran@gmail.com',
                        help='TalkBank email (default: maksym.taran@gmail.com)')
    parser.add_argument('--password', default='D2z8jQ6@9mrR9Yu',
                        help='TalkBank password')

    args = parser.parse_args()

    # Set up directories
    script_dir = Path(__file__).parent.absolute()
    output_dir = args.output or str(script_dir / 'extracted')
    os.makedirs(output_dir, exist_ok=True)

    # Create temp directory for downloads and cleaned files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Login and download corpus
        session = login_talkbank(args.email, args.password)
        corpus_dir = download_corpus(session, args.language, args.corpus, temp_dir)

        # Find CHAT files
        cha_files = sorted(glob.glob(os.path.join(corpus_dir, '**', '*.cha'), recursive=True))

        if not cha_files:
            print(f"No .cha files found in {corpus_dir}", file=sys.stderr)
            sys.exit(1)

        # Limit to requested number of sessions
        cha_files = cha_files[:args.sessions]

        print(f"\nFound {len(cha_files)} CHAT files to process:")
        for f in cha_files:
            print(f"  - {os.path.basename(f)}")

        # Process each session
        results = []
        for filepath in cha_files:
            try:
                result = process_session(filepath, temp_dir, output_dir)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  Error processing {filepath}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"Extracted {len(results)} sessions to {output_dir}")
        print(f"{'='*60}")
        print(f"\nRun summarization with Claude Code to generate narrative summaries.")


if __name__ == "__main__":
    main()
