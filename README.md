# FactChecker CLI

> **Python stdlib only** — no third-party dependencies.  
> Fact-check claims against a local text corpus (`data/corpus/*.txt`).  
> Optional **localhost-only** Ollama LLM for verification and explanations.

---

## Startup banner

When you start the program in **interactive mode**, you see:

```
  _____         _      _            _
 |  ___|_ _ ___| | __ | |_ ___  ___| |__
 | |_ / _` / __| |/ / | __/ _ \/ __| '_ \
 |  _| (_| \__ \   <  | ||  __/ (__| | | |
 |_|  \__,_|___/_|\_\  \__\___|\___|_| |_|

Interactive Offline Factchecker
Offline fact-checker over a local text corpus. Type /help for commands.
Type /help for commands.

factcheck>
```

Then the `factcheck>` prompt is ready for your claims or commands.

---

## Quick start

### Interactive (from project root)

```bash
python factcheck.py
```

or

```bash
python factchecker/factcheck.py
```

### One-shot

```bash
python factcheck.py --claim "Tokyo is the capital of Japan." --mode pretty
python factcheck.py --claim "Tokyo is the capital of Japan." --mode json
python factcheck.py --claim "Your claim here." --corpus data/corpus --k 10
```

### With Ollama (localhost)

```bash
python factcheck.py --claim "Tokyo is the capital of Japan." --ollama --model deepseek-r1:8b --mode pretty
python factcheck.py --claim "Your claim." --ollama --lang ja --timeout 90
```

---

## CLI options (one-shot)

| Option | Default | Description |
|--------|---------|-------------|
| `--claim` | — | One-shot: fact-check this claim and exit |
| `--mode` | `pretty` | Output: `pretty` or `json` |
| `--corpus` | `data/corpus` | Corpus directory (.txt files) |
| `--k` | 10 | Top-k documents for retrieval |
| `--max_sentences_per_doc` | 2 | Max sentences per doc (capped at 2) |
| `--ollama` | off | Enable LLM (Ollama) for verification |
| `--ollama-host` | `http://127.0.0.1:11434` | Ollama URL (localhost only) |
| `--model` | `deepseek-r1:8b` | Ollama model name |
| `--llm-role` | `both` | `verifier` \| `explainer` \| `both` \| `off` |
| `--timeout` | 60 | Ollama request timeout (seconds) |
| `--lang` | `en` | Explanation language: `en` or `ja` |
| `--debug` | off | Log Step1/Step2 raw and sanitized explanation to stderr |

---

## Interactive commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/exit`, `/quit`, `/q` | Exit the REPL |
| `/clear` | Clear screen |
| `/status` | Corpus path, k, mode, session id, doc count |
| `/k <n>` | Set top-k documents |
| `/corpus <path>` | Set corpus directory |
| `/mode pretty\|json` | Output mode |
| `/json on\|off` | Alias for mode |
| `/sources used\|all` | SOURCES: used only (or top 2) vs all |
| `/ollama on\|off` | Enable/disable Ollama LLM |
| `/ollama_host <url>` | Set Ollama host (localhost only) |
| `/model <name>` | Set Ollama model |
| `/lang en\|ja` | Explanation language |
| `/load <file>` | Evaluate claims from file (line-by-line; skip empty and `#`) |
| `/save [file]` | Save session to JSONL (default: `sessions/<timestamp>.jsonl`) |
| `/resume <file>` | Restore session from JSONL |
| `/compact` | Keep last 20 history items |

**Multi-line claim:** type `"""`, then your text, then `"""` again.

---

## Verdicts and evidence

| Concept | Description |
|--------|-------------|
| **Verdicts** | `SUPPORTED` / `REFUTED` / `NOT_ENOUGH_INFO` |
| **Evidence** | Each item: `doc_id`, `score`, `snippet`, `span`. Indices 1..N. |
| **used_evidence** | Indices actually cited for the verdict (no auto-fill). |
| **Pretty SOURCES** | Default `sources_mode=used`: show only used evidence (or top 2 if none). Use `/sources all` for all. |

---

## How it works

**Rule-based path**

- BM25-like retrieval → sentence overlap filter (≥1 shared term) → cap at 4 evidence items.
- Verifier uses key-term overlap, capital-of mismatch, direct negation, meta-refutation, and **directional contradiction** (north/south, east/west with border context).
- NOT_ENOUGH_INFO confidence in [0.20, 0.60] by best overlap.

**Ollama path (optional)**

- **Step 1:** Strict JSON (verdict, confidence, used_evidence, alignment, brief_reason).
- **Step 2:** Long explanation; citations sanitized (out-of-range indices removed).
- On failure (timeout, invalid JSON, or validation), **fallback to rule-based** with a note:
  - `[LLM unavailable (timeout/connection); used rule-based fallback.]`
  - `[LLM output invalid JSON; used rule-based fallback.]`
  - `[LLM output invalid (validation); used rule-based fallback.]`
- Fallback results have confidence capped at 0.75.

---

## Example session

```
factcheck> /status
factcheck> /k 10
factcheck> /mode pretty
factcheck> /sources used
factcheck> Tokyo is the capital of Japan.
factcheck> /ollama on
factcheck> /model deepseek-r1:8b
factcheck> US borders Canada to the south.
factcheck> /save
factcheck> /resume sessions/1234567890.jsonl
factcheck> /compact
factcheck> /exit
```

---

## Project layout

```
factchecker/
  factcheck.py       # Entry + REPL, one-shot
  core/
    schema.py        # Verdict, EvidenceItem, FactcheckResult, SessionState
    tokenize.py      # tokenize, key_terms, split_sentences
    retrieve.py      # Retriever (BM25-like), overlap filter, cap 4
    verify.py        # Rule-based verifier + verify_with_ollama (two-step LLM)
    ollama_client.py # Ollama API (localhost), Step1/Step2 prompts, validation
    io_utils.py      # REPL, session save/resume, pretty/JSON output
data/corpus/         # .txt documents (filename = doc_id)
sessions/            # Saved session JSONL files
```
