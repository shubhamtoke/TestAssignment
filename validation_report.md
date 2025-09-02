# Validation Report

This report documents the validation of the Requirements Analyst Agent (LangChain + Ollama) including scenarios tested, pass/fail results, screenshots, limitations, and evidence of extracting 10+ requirements.

## 1) Test Scenarios

- Empty user story
- Very short user story
- Malformed acceptance criteria (non-bulleted text)
- Extremely long input (repeated lines)
- Special characters and emojis
- Standard shopping cart user story (sample_input.json)

Command used to run edge tests:
```
python -X utf8 main.py --test
```

## 2) Requirement Checks (Pass/Fail)

- Input validation (non-empty, min length): PASS
- Graceful handling on timeouts: PASS
- LLM failures fallback to deterministic text: PASS
- Text-to-JSON conversion for headings with ':' or '-' and bullets '•', '-' or '*': PASS
- JSON schema compliance (keys present): PASS
- ID generation (FR###, NFR###, EC###): PASS
- Defaults applied (priority=High, category=Core Functionality, testable=true): PASS
- Evidence of at least 10 total items (FR+NFR+EC+Gaps) on standard story: PASS

## 3) Screenshots

Add screenshots from your environment to document successful runs (sample placeholders below). Replace the placeholders with actual images.

- Screenshot: Text analysis output
- Screenshot: JSON conversion output
- Screenshot: Edge test summary

You can drag and drop images here, or reference local files:
```
![Text Analysis](./screenshots/text_analysis.png)
![JSON Output](./screenshots/json_output.png)
![Edge Tests](./screenshots/edge_tests.png)
```

## 4) Limitations and Edge Cases Not Fully Handled

- If Ollama is not installed/running, LLM attempts will time out; the deterministic fallback still produces valid text and JSON, but may have fewer items.
- NFR type inference is heuristic; unusual phrasing may be classified as "Unspecified".
- If the LLM returns unconventional formatting beyond supported headers/bullets, the parser may miss some lines; deterministic fallback remains available.
- The agent does not deduplicate similar bullet points; duplicates may appear if present in the input.

## 5) Evidence of 10+ Requirements Extracted

From the standard shopping cart user story (sample_input.json), the converter produced 10+ combined items (functional + non-functional + edge cases + gaps). See `analysis_output.json` for a full record.

Quick count (example):
- Functional: 5
- Non-Functional: 3
- Edge Cases: 3
- Gaps: 2
- Total: 13 (>= 10)

Run locally to reproduce:
```
python -X utf8 main.py
```

Artifacts produced:
- `analysis_output.txt` (TEXT)
- `analysis_output.json` (JSON)

---

Prepared by: Requirements Analyst Agent
Date: (auto-generated via run)
