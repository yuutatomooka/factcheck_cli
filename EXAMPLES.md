# Example outputs (natural-language-first format)

## 1. Supported claim

**Claim:** `Tokyo is the capital of Japan.`

**Pretty mode:**
```
---
CLAIM: Tokyo is the capital of Japan.
VERDICT: SUPPORTED
CONFIDENCE: 1.00
EVIDENCE USED: [1], [2], [3], [4]
SOURCES:
  [1] japan.txt — "Tokyo is the capital city of Japan."
  [2] japan.txt — "It is one of the most populous cities in the world."
  [3] asia.txt — "The capital of Japan is Tokyo."
  [4] asia.txt — "Asia is the largest continent by land area."
---
EXPLANATION:
Evidence directly states the claim. Best match in japan.txt.
```

**JSON mode:** includes `claim`, `verdict`, `confidence`, `evidence` (with `idx`), `rationale`, `explanation`, `used_evidence`.

---

## 2. Refuted claim

**Claim:** `Tokyo is the capital of South Korea.`

With a corpus that contains e.g. "The capital of Japan is Tokyo" (asia.txt / japan.txt), the verifier detects a **capital-of mismatch** and returns:

```
---
CLAIM: Tokyo is the capital of South Korea.
VERDICT: REFUTED
CONFIDENCE: 0.85
EVIDENCE USED: [1], [2], [3], [4]
SOURCES:
  [1] japan.txt — "Tokyo is the capital city of Japan."
  ...
---
EXPLANATION:
Evidence states a conflicting fact: japan.txt states the capital is of a different country.
```

---

## 3. NOT_ENOUGH_INFO claim

**Claim:** `The moon is made of cheese.`

No evidence supports or refutes this; the corpus has no relevant content.

```
---
CLAIM: The moon is made of cheese.
VERDICT: NOT_ENOUGH_INFO
CONFIDENCE: 0.40
EVIDENCE USED: []
SOURCES:
  [1] usa.txt — "is the capital city of the United States of America."
  ...
---
EXPLANATION:
Evidence does not directly support or contradict the claim.
```

With **Ollama enabled**, the LLM can add what kind of evidence would be needed (e.g. scientific text about the moon’s composition).

---

## Running examples

```bash
# Rule-based (no Ollama)
python factcheck.py --claim "Tokyo is the capital of Japan." --mode pretty

# With Ollama (LLM primary; fallback to rules on failure)
python factcheck.py --claim "Tokyo is the capital of Japan." --ollama --model deepseek-r1:8b --mode pretty

# JSON
python factcheck.py --claim "Tokyo is the capital of Japan." --mode json

# Japanese explanation (when using Ollama)
python factcheck.py --claim "Tokyo is the capital of Japan." --ollama --lang ja --mode pretty
```

Interactive: `/ollama on`, `/model deepseek-r1:8b`, `/lang ja`, then enter claims.
