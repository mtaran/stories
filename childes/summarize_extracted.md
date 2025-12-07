# Summarizing Extracted CHILDES Sessions

After running `generate_summaries.py`, use Task agents with Haiku to summarize each extracted JSON file.

## Task Agent Prompt Template

For each `{session}_extracted.json` file in `childes/extracted/`, use this prompt:

```
Read /home/user/stories/childes/extracted/{session}_extracted.json

Extract the "summarization_prompt" field and follow its instructions to generate a narrative summary.

Save the result to /home/user/stories/childes/summaries/{session}_summary.txt in this format:

Session: {session}
Situation: [from JSON situation field]
Participants: [from JSON participants field]
Total utterances: [from JSON total_utterances field]
Summary sentences: [count your sentences]
Compression ratio: [total_utterances / summary_sentences]:1

============================================================
NARRATIVE SUMMARY
============================================================

[Each sentence on its own line, with blank line between sentences]

Return: number of sentences generated and first 3 sentences.
```

## Example

For session "jaf":

```
Read /home/user/stories/childes/extracted/jaf_extracted.json

Extract the "summarization_prompt" field and follow its instructions to generate a narrative summary.

Save the result to /home/user/stories/childes/summaries/jaf_summary.txt in this format:

Session: jaf
Situation: [from JSON situation field]
Participants: [from JSON participants field]
Total utterances: [from JSON total_utterances field]
Summary sentences: [count your sentences]
Compression ratio: [total_utterances / summary_sentences]:1

============================================================
NARRATIVE SUMMARY
============================================================

[Each sentence on its own line, with blank line between sentences]

Return: number of sentences generated and first 3 sentences.
```

Use `model: haiku` and `subagent_type: general-purpose` for the Task agent.
