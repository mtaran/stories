#!/usr/bin/env python3
"""
Generate narrative summaries from CHILDES CHAT transcripts.
Uses pylangacq for proper CHAT file parsing and Claude for summarization.
Compresses 10-30 utterances into single summary sentences.
"""

import glob
import os
import re
import anthropic
import pylangacq


def clean_chat_file(filepath: str) -> str:
    """Remove %mor and %gra tiers from a CHAT file to avoid alignment issues."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove %mor and %gra lines (morphological tiers that cause alignment issues)
    content = re.sub(r'^%mor:\t.+\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'^%gra:\t.+\n', '', content, flags=re.MULTILINE)

    return content


def extract_utterances_with_pylangacq(filepath: str, clean_dir: str) -> dict:
    """
    Extract utterances from a CHAT file using pylangacq.
    Returns dict with situation, participants, and utterances.
    """
    basename = os.path.basename(filepath)

    # Clean the file to avoid morphological tier alignment issues
    cleaned_content = clean_chat_file(filepath)
    clean_path = os.path.join(clean_dir, basename)
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    # Parse with pylangacq
    reader = pylangacq.read_chat(clean_path)

    # Get headers for metadata
    headers = reader.headers()[0] if reader.headers() else {}
    situation = headers.get('Situation', 'everyday conversation')

    # Get participants
    participants = reader.participants()
    participant_info = []
    if participants:
        for file_participants in participants:
            for code, info in file_participants.items():
                role = info.get('role', code)
                participant_info.append(f"{code} ({role})")

    # Extract utterances
    utterances = []
    for utt in reader.utterances():
        participant = utt.participant
        # Get words from tokens
        words = [t.word for t in utt.tokens if t.word]
        text = ' '.join(words)

        if text.strip():
            utterances.append({
                'speaker': participant,
                'text': text
            })

    return {
        'situation': situation,
        'participants': ', '.join(participant_info),
        'utterances': utterances
    }


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
}


def get_speaker_name(code: str) -> str:
    """Get readable speaker name from code."""
    return SPEAKER_NAMES.get(code, f'{code}')


def format_chunk_for_summary(chunk: list, situation: str) -> str:
    """Format a chunk of utterances for the summarization prompt."""
    lines = []
    for utt in chunk:
        speaker_name = get_speaker_name(utt['speaker'])
        lines.append(f"{speaker_name}: {utt['text']}")

    dialogue = '\n'.join(lines)

    return f"""Context: {situation}

Conversation segment ({len(chunk)} utterances):
{dialogue}

Write ONE concise sentence (15-30 words) summarizing what happens in this conversation segment.
Write in past tense, third person. Focus on the key actions, topics, or exchanges.
Just output the single summary sentence, nothing else."""


def summarize_with_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """Use Claude Haiku to summarize a chunk of utterances."""
    response = client.messages.create(
        model="claude-haiku-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def process_session(client: anthropic.Anthropic, filepath: str, clean_dir: str, output_dir: str) -> dict:
    """Process a single CHAT session file and generate summary."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nProcessing: {basename}")

    # Extract utterances using pylangacq
    data = extract_utterances_with_pylangacq(filepath, clean_dir)
    utterances = data['utterances']
    situation = data['situation']
    participants = data['participants']

    print(f"  Situation: {situation[:80]}...")
    print(f"  Total utterances: {len(utterances)}")

    if len(utterances) == 0:
        print("  No utterances found, skipping.")
        return None

    # Chunk utterances (aim for ~20 utterances per chunk)
    chunk_size = 20
    chunks = [utterances[i:i+chunk_size] for i in range(0, len(utterances), chunk_size)]

    # Combine small final chunk with previous
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    print(f"  Chunks to summarize: {len(chunks)}")

    # Generate summaries for each chunk using Claude
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"    Summarizing chunk {i+1}/{len(chunks)}...", end=' ', flush=True)
        prompt = format_chunk_for_summary(chunk, situation)
        summary = summarize_with_claude(client, prompt)
        summaries.append(summary)
        print("done")

    # Save output
    output = {
        'session_id': basename,
        'situation': situation,
        'participants': participants,
        'total_utterances': len(utterances),
        'summary_sentences': len(summaries),
        'compression_ratio': f"{len(utterances)/len(summaries):.1f}:1",
        'narrative': summaries
    }

    output_path = os.path.join(output_dir, f"{basename}_summary.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Session: {basename}\n")
        f.write(f"Situation: {situation}\n")
        f.write(f"Participants: {participants}\n")
        f.write(f"Total utterances: {len(utterances)}\n")
        f.write(f"Summary sentences: {len(summaries)}\n")
        f.write(f"Compression ratio: {len(utterances)/len(summaries):.1f} utterances per sentence\n")
        f.write(f"\n{'='*60}\nNARRATIVE SUMMARY\n{'='*60}\n\n")
        for sentence in summaries:
            f.write(f"{sentence}\n\n")

    print(f"  Saved to: {output_path}")
    return output


def main():
    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Get first 5 CHAT files from Hall corpus
    cha_files = sorted(glob.glob("/home/user/stories/childes/Hall/**/*.cha", recursive=True))[:5]

    print(f"Found {len(cha_files)} CHAT files to process:")
    for f in cha_files:
        print(f"  - {os.path.basename(f)}")

    # Create directories
    clean_dir = "/home/user/stories/childes/Hall_cleaned"
    output_dir = "/home/user/stories/childes/summaries"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Process each session
    results = []
    for filepath in cha_files:
        try:
            result = process_session(client, filepath, clean_dir, output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Completed! Generated {len(results)} summary files in {output_dir}")
    print(f"{'='*60}")

    # Show sample output
    if results:
        print(f"\nSample from first session ({results[0]['session_id']}):")
        print(f"  First 5 summary sentences:")
        for i, sent in enumerate(results[0]['narrative'][:5], 1):
            print(f"    {i}. {sent}")


if __name__ == "__main__":
    main()
