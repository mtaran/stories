#!/usr/bin/env python3
"""
Generate narrative summaries from CHILDES CHAT transcripts.
Compresses 10-30 utterances into single summary sentences using content-based narrativization.
"""

import re
import glob
import os
import random


def extract_utterances_simple(filepath):
    """Extract utterances from a CHAT file without parsing morphology."""
    utterances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get the situation context
    situation = ""
    sit_match = re.search(r'@Situation:\s*(.+)', content)
    if sit_match:
        situation = sit_match.group(1).strip()

    # Get participants
    participants = ""
    part_match = re.search(r'@Participants:\s*(.+)', content)
    if part_match:
        participants = part_match.group(1).strip()

    # Extract lines starting with *SPEAKER:
    pattern = r'^\*([A-Z]{3}):\s*(.+?)(?=\n[%@*]|\Z)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for speaker, text in matches:
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
        text = re.sub(r'<.*?>', '', text)    # Remove angle brackets
        text = re.sub(r'&~?\w+', '', text)   # Remove phonological fragments
        text = re.sub(r'\+\.\.\.', '...', text)
        text = re.sub(r'\+["/!]', '', text)
        text = re.sub(r'xxx', '', text)      # Remove unintelligible markers
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'0\w+', '', text)
        text = re.sub(r'\(\.\)', '', text)   # Remove pause markers
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[.,!?]+$', '', text)  # Remove punctuation-only

        if text and len(text) > 1:
            utterances.append({'speaker': speaker, 'text': text})

    return {
        'situation': situation,
        'participants': participants,
        'utterances': utterances
    }


# Speaker role mapping
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


def get_speaker_name(code):
    """Get readable speaker name from code."""
    return SPEAKER_NAMES.get(code, f'the {code.lower()}')


def extract_key_words(text):
    """Extract potentially meaningful words from text."""
    # Remove common words and punctuation
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                  'by', 'from', 'up', 'about', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'between', 'under',
                  'again', 'further', 'then', 'once', 'here', 'there', 'when',
                  'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                  'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
                  'if', 'or', 'because', 'as', 'until', 'while', 'i', 'me',
                  'my', 'myself', 'we', 'our', 'ours', 'you', 'your', 'yours',
                  'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they',
                  'them', 'their', 'what', 'which', 'who', 'whom', 'this',
                  'that', 'these', 'those', 'am', 'oh', 'yeah', 'yes', 'no',
                  'okay', 'well', 'now', 'dont', "don't", 'gonna', 'wanna',
                  'gotta', 'got', 'get', 'go', 'come', 'know', 'think', 'see',
                  'look', 'want', 'let', 'say', 'said', 'tell', 'told', 'make',
                  'put', 'give', 'take', 'like', 'huh', 'uh', 'um', 'hmm',
                  'right', 'okay', 'alright', 'really', 'thing', 'things'}

    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    meaningful = [w for w in words if w not in stop_words]
    return meaningful


def extract_topics_from_chunk(chunk):
    """Extract actual topics/nouns from utterances."""
    all_words = []
    for u in chunk:
        all_words.extend(extract_key_words(u['text']))

    # Count word frequency
    word_counts = {}
    for w in all_words:
        word_counts[w] = word_counts.get(w, 0) + 1

    # Get top words
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    return [w for w, c in sorted_words[:5] if c >= 2]


def extract_sample_quotes(chunk, max_quotes=2):
    """Extract short sample quotes from the chunk."""
    quotes = []
    for u in chunk:
        text = u['text'].strip()
        # Look for interesting short phrases
        if 5 < len(text) < 50 and not text.startswith('...'):
            # Clean up
            text = re.sub(r'\s+', ' ', text)
            if text and text[-1] in '.!?':
                quotes.append((u['speaker'], text))

    # Sample diverse quotes
    if quotes:
        random.shuffle(quotes)
        return quotes[:max_quotes]
    return []


def summarize_chunk(chunk, chunk_idx, total_chunks):
    """Generate a narrative summary sentence for a chunk of utterances."""

    speakers = list(set([u['speaker'] for u in chunk]))
    speaker_counts = {}
    for u in chunk:
        speaker_counts[u['speaker']] = speaker_counts.get(u['speaker'], 0) + 1

    # Get dominant speaker
    dominant = max(speaker_counts, key=speaker_counts.get)

    # Get topics
    topics = extract_topics_from_chunk(chunk)

    # Get sample quotes
    quotes = extract_sample_quotes(chunk)

    # Count features
    questions = sum(1 for u in chunk if '?' in u['text'])
    exclamations = sum(1 for u in chunk if '!' in u['text'])

    # Build narrative sentence
    speaker_names = [get_speaker_name(s) for s in speakers[:3]]

    # Vary the sentence structure based on content
    templates = []

    # With topics
    if topics:
        topic_str = topics[0] if len(topics) == 1 else f"{topics[0]} and {topics[1]}" if len(topics) >= 2 else "conversation"

        if 'CHI' in speakers:
            if len(speakers) == 2:
                other = get_speaker_name([s for s in speakers if s != 'CHI'][0])
                if questions > len(chunk) * 0.3:
                    templates.append(f"The child asked {other} about {topic_str}.")
                    templates.append(f"{other.capitalize()} and the child discussed {topic_str}.")
                else:
                    templates.append(f"The child and {other} talked about {topic_str}.")
                    templates.append(f"{other.capitalize()} interacted with the child regarding {topic_str}.")
            else:
                others = [get_speaker_name(s) for s in speakers if s != 'CHI'][:2]
                templates.append(f"The child engaged with {' and '.join(others)} about {topic_str}.")
        else:
            templates.append(f"{speaker_names[0].capitalize()} and {speaker_names[1] if len(speaker_names) > 1 else 'others'} discussed {topic_str}.")

    # With quotes
    if quotes and random.random() > 0.5:
        speaker, quote = quotes[0]
        speaker_name = get_speaker_name(speaker)
        if len(quote) < 40:
            templates.append(f"{speaker_name.capitalize()} said \"{quote}\"")

    # Activity-based
    all_text = ' '.join([u['text'].lower() for u in chunk])

    if re.search(r'\b(school|class|teacher)\b', all_text):
        templates.append("The conversation turned to school-related matters.")
    if re.search(r'\b(eat|breakfast|food|hungry)\b', all_text):
        templates.append("They discussed food and eating.")
    if re.search(r'\b(play|game|toy)\b', all_text):
        templates.append("Play and games came up in the conversation.")
    if re.search(r'\b(ready|hurry|time|late)\b', all_text):
        templates.append("There was discussion about getting ready or being on time.")
    if re.search(r'\b(name|call|bobby|tony|chris)\b', all_text):
        templates.append("Names and identity were mentioned.")

    # Emotional/interactive patterns
    if exclamations > len(chunk) * 0.2:
        templates.append("The exchange was animated and expressive.")
    if questions > len(chunk) * 0.4:
        templates.append("The conversation was filled with questions and answers.")

    # Position-based variations
    if chunk_idx == 0:
        templates.append("The recording began with greetings and initial exchanges.")
    elif chunk_idx >= total_chunks - 2:
        templates.append("The conversation continued as the session neared its end.")

    # Generic fallbacks with variation
    if 'CHI' in speakers:
        child_dominant = speaker_counts.get('CHI', 0) > len(chunk) * 0.4
        if child_dominant:
            templates.append("The child was talkative during this exchange.")
            templates.append("The child led much of the conversation.")
        else:
            templates.append("Adults guided the conversation with the child.")
            templates.append("The child participated in the ongoing dialogue.")

    # Select a template
    if templates:
        return random.choice(templates)

    # Ultimate fallback
    return f"The participants ({', '.join(speaker_names)}) continued their conversation."


def process_session(filepath, output_dir):
    """Process a single CHAT session file and generate summary."""

    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nProcessing: {basename}")

    data = extract_utterances_simple(filepath)
    utterances = data['utterances']
    situation = data['situation'] or "everyday conversation"
    participants = data['participants']

    print(f"  Situation: {situation[:80]}...")
    print(f"  Total utterances: {len(utterances)}")

    if len(utterances) == 0:
        print("  No utterances found, skipping.")
        return None

    # Chunk utterances (aim for ~20 utterances per chunk)
    chunk_size = 20
    chunks = [utterances[i:i+chunk_size] for i in range(0, len(utterances), chunk_size)]

    # Filter out very small final chunks
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]

    print(f"  Chunks to summarize: {len(chunks)}")

    # Set seed for reproducibility but with file-specific variation
    random.seed(hash(basename) % 2**32)

    # Generate summaries for each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, i, len(chunks))
        summaries.append(summary)

    # Output
    output = {
        'session_id': basename,
        'situation': situation,
        'participants': participants,
        'total_utterances': len(utterances),
        'summary_sentences': len(summaries),
        'compression_ratio': f"{len(utterances)/len(summaries):.1f}:1",
        'narrative': summaries
    }

    # Save to file
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
    # Get first 5 CHAT files from Hall corpus
    cha_files = sorted(glob.glob("/home/user/stories/childes/Hall/**/*.cha", recursive=True))[:5]

    print(f"Found {len(cha_files)} CHAT files to process:")
    for f in cha_files:
        print(f"  - {os.path.basename(f)}")

    # Create output directory
    output_dir = "/home/user/stories/childes/summaries"
    os.makedirs(output_dir, exist_ok=True)

    # Process each session
    results = []
    for filepath in cha_files:
        try:
            result = process_session(filepath, output_dir)
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
        print(f"  First 10 summary sentences:")
        for i, sent in enumerate(results[0]['narrative'][:10], 1):
            print(f"    {i}. {sent}")


if __name__ == "__main__":
    main()
