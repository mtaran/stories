# CHILDES Session Summaries

Generate narrative summaries from CHILDES CHAT transcripts.

## Workflow (for Claude Code)

### Step 1: Extract transcripts

```bash
python childes/generate_summaries.py --sessions 6
```

This downloads the corpus from TalkBank, extracts utterances using pylangacq, and saves plain text transcripts to `childes/transcripts/`.

Options:
- `--language` / `-l`: Language group (default: Eng-NA)
- `--corpus` / `-c`: Corpus name (default: Hall)
- `--sessions` / `-n`: Number of sessions (default: 5)

### Step 2: Summarize with Task agents

For each transcript file, use a Task agent with this prompt:

```
Read /home/user/stories/childes/transcripts/{session}_transcript.txt

This is a child language study transcript. Write a narrative summary following these rules:
- Each sentence must describe a SPECIFIC event, topic, or exchange
- NEVER use generic filler like "The conversation continued" or "They discussed various topics"
- Include concrete details: names of objects, games, foods, places, people mentioned
- Capture the flow: what happened first, then what, any conflicts or resolutions
- Note emotional moments: excitement, frustration, silliness, affection
- Write in past tense, third person
- Target: [use the range from the transcript header] sentences

Save to /home/user/stories/childes/summaries/{session}_summary.txt:

Session: {session}
Situation: [from header]
Participants: [from header]
Total utterances: [from header]
Summary sentences: [your count]
Compression ratio: [utterances/sentences, rounded]:1

============================================================
NARRATIVE SUMMARY
============================================================

[Each sentence on its own line with blank line between]

Return: sentence count and first 3 sentences.
```

Use `model: haiku` and `subagent_type: general-purpose`.

## Requirements

```bash
pip install pylangacq requests
```

TalkBank credentials are embedded in the script.
