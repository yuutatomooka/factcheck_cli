"""Ollama API client using Python stdlib only (urllib). Localhost only."""
import json
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_HOST = "http://127.0.0.1:11434"
VALID_VERDICTS = frozenset({"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"})

# Step 1 aligner output schema (for validation)
ALIGNER_VERDICT_KEY = "verdict"
ALIGNER_CONFIDENCE_KEY = "confidence"
ALIGNER_USED_EVIDENCE_KEY = "used_evidence"
ALIGNER_ALIGNMENT_KEY = "alignment"
ALIGNER_BRIEF_REASON_KEY = "brief_reason"
ALIGNMENT_ROLES = frozenset({"support", "refute"})


def _extract_json_from_response(text: str) -> dict | None:
    """Extract JSON object from LLM response. Strips markdown code blocks if present. Returns None on failure."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    # Remove optional ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _is_localhost(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in ("127.0.0.1", "localhost", "::1", "")
    except Exception:
        return False


def generate(
    host: str,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout: float = 60.0,
) -> str:
    """
    POST to /api/generate with stream=false, temperature=0. Fallback when /api/chat fails.
    Uses Ollama's 'system' field in the JSON payload when provided (no string concatenation).
    """
    if not _is_localhost(host):
        raise ValueError("Only localhost is allowed")
    url = host.rstrip("/") + "/api/generate"
    body_dict: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    if system is not None:
        body_dict["system"] = system
    body = json.dumps(body_dict).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return (out.get("response") or "").strip()


def ping(host: str = DEFAULT_HOST, timeout: float = 5.0) -> dict[str, Any]:
    """
    Check Ollama availability. Returns a status dict with 'ok' (bool) and optional 'error', 'message'.
    Only allows localhost.
    """
    if not _is_localhost(host):
        return {"ok": False, "error": "Only localhost is allowed", "message": None}
    url = host.rstrip("/") + "/api/tags"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "error": None, "message": data.get("models", [])}
    except URLError as e:
        return {"ok": False, "error": str(e.reason) if getattr(e, "reason", None) else str(e), "message": None}
    except (HTTPError, OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": str(e), "message": None}


def chat(
    host: str,
    model: str,
    system_content: str,
    user_content: str,
    timeout: float = 60.0,
) -> str:
    """
    POST to /api/chat with stream=false, temperature=0. Returns assistant message content.
    Raises on non-localhost or request/parse errors.
    """
    if not _is_localhost(host):
        raise ValueError("Only localhost is allowed")
    url = host.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
    }).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    msg = out.get("message") or {}
    return (msg.get("content") or "").strip()


def chat_or_generate(
    host: str,
    model: str,
    system_content: str,
    user_content: str,
    timeout: float = 60.0,
) -> str:
    """Try /api/chat; on failure fall back to /api/generate."""
    try:
        return chat(host, model, system_content, user_content, timeout=timeout)
    except Exception:
        try:
            return generate(host, model, user_content, system=system_content, timeout=timeout)
        except Exception as e:
            print(f"Ollama generate fallback failed: {e}", file=sys.stderr)
            raise


def validate_verifier_fields(
    verdict: str | None,
    confidence: float,
    used_evidence: list[int],
    explanation: str,
    num_evidence: int,
) -> tuple[bool, str | None]:
    """
    Validate parsed LLM verifier output. Returns (True, None) if valid, (False, reason) if invalid.
    - verdict must be SUPPORTED, REFUTED, or NOT_ENOUGH_INFO
    - confidence must be float in [0, 1]
    - if verdict is SUPPORTED or REFUTED: USED_EVIDENCE must not be empty; EXPLANATION must contain
      citations like [1] referencing valid evidence indices; at least one citation must be in USED_EVIDENCE.
    """
    if verdict is None or verdict not in VALID_VERDICTS:
        return (False, "invalid or missing verdict")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        return (False, "confidence not in [0, 1]")
    valid_indices = set(range(1, num_evidence + 1))
    if verdict in ("SUPPORTED", "REFUTED"):
        if not used_evidence:
            return (False, "SUPPORTED/REFUTED requires non-empty USED_EVIDENCE")
        if not set(used_evidence) <= valid_indices:
            return (False, "USED_EVIDENCE contains invalid indices")
        # EXPLANATION must contain at least one citation like [1], [2], etc.
        citation_pattern = re.compile(r"\[(\d+)\]")
        cited = set()
        for m in citation_pattern.finditer(explanation or ""):
            n = int(m.group(1))
            if 1 <= n <= num_evidence:
                cited.add(n)
        if not cited:
            return (False, "EXPLANATION must contain citations like [1] or [2]")
        if not (cited & set(used_evidence)):
            return (False, "citations in EXPLANATION must overlap with USED_EVIDENCE")
    return (True, None)


# Manual test (stdlib-only, localhost Ollama): Claim "The capital city of the United States is Washington D.C."
# with evidence including e.g. "Washington D.C. is the capital city of the United States." — expected: SUPPORTED.
# Evidence containing "New York is NOT the capital..." must NOT cause REFUTED for that claim (support-precedence
# or validation should yield SUPPORTED).


def parse_verifier_response(response: str, num_evidence: int) -> tuple[str | None, float, list[int], str]:
    """
    Parse LLM verifier output: VERDICT:, CONFIDENCE:, USED_EVIDENCE:, then EXPLANATION.
    Returns (verdict, confidence, used_evidence, explanation). verdict None if invalid.
    used_evidence must be subset of 1..num_evidence; confidence clamped 0..1.
    """
    verdict = None
    confidence = 0.0
    used_evidence: list[int] = []
    explanation = ""
    lines = response.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("VERDICT:"):
            v = line[7:].strip().upper()
            if v in VALID_VERDICTS:
                verdict = v
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                c = float(line[11:].strip())
                confidence = max(0.0, min(1.0, c))
            except ValueError:
                pass
        elif line.upper().startswith("USED_EVIDENCE:"):
            rest = line[13:].strip()
            for part in re.split(r"[,;\s]+", rest):
                part = part.strip().strip("[]")
                if part.isdigit():
                    idx = int(part)
                    if 1 <= idx <= num_evidence and idx not in used_evidence:
                        used_evidence.append(idx)
        elif line.upper().startswith("EXPLANATION:"):
            explanation = line[12:].strip()
            i += 1
            while i < len(lines):
                explanation += "\n" + lines[i]
                i += 1
            explanation = explanation.strip()
            break
        i += 1
    if not explanation and "explanation" in response.lower():
        idx = response.lower().find("explanation")
        explanation = response[idx:].split(":", 1)[-1].strip()
    used_evidence.sort()
    return (verdict, confidence, used_evidence, explanation or response.strip())


def verifier_aligner_system_prompt() -> str:
    """System prompt for Step 1: strict JSON-only verifier+aligner."""
    return (
        "You are a strict evidence-grounded fact-checking engine. "
        "Use ONLY the provided evidence snippets. Do NOT use external knowledge. "
        "Return ONLY valid JSON matching the schema. No extra text. "
        "Quotes must be copied verbatim from snippets. "
        "If evidence is insufficient or ambiguous, return NOT_ENOUGH_INFO."
    )


def verifier_aligner_user_prompt(claim: str, evidence_lines: list[str]) -> str:
    """User prompt for Step 1: claim + evidence candidates + schema."""
    evidence_block = "\n".join(evidence_lines)
    return (
        f"Claim: {claim}\n\n"
        "Evidence candidates:\n"
        f"{evidence_block}\n\n"
        "Return ONLY a single JSON object with these exact keys:\n"
        '"verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",\n'
        '"confidence": <float 0 to 1>,\n'
        '"used_evidence": [1, 2] (list of indices you use; empty [] only for NOT_ENOUGH_INFO),\n'
        '"alignment": [ {"idx": 1, "doc_id": "<from snippet>", "quote": "<exact substring from that snippet>", "role": "support"|"refute"} ],\n'
        '"brief_reason": "1-2 sentences with citations like [1][2]."\n\n'
        "Rules: quote MUST be copied verbatim from the snippet. "
        "For REFUTED, at least one alignment item must have role \"refute\". "
        "brief_reason must include citations [n] that overlap with used_evidence. "
        "Directional rule: north vs south, east vs west are opposites. "
        "If evidence states a direction (e.g. borders X to the north) and the claim states the opposite (e.g. to the south), output REFUTED with used_evidence containing that snippet."
    )


def parse_aligner_response(response: str) -> dict | None:
    """Parse Step 1 response as JSON only. Returns dict or None."""
    return _extract_json_from_response(response)


def validate_aligner_output(parsed: dict, evidence: list, num_evidence: int) -> tuple[bool, str | None]:
    """
    Validate Step 1 aligner output. evidence is list of EvidenceItem; indices 1..num_evidence.
    Returns (True, None) if valid, (False, reason) otherwise.
    """
    if not isinstance(parsed, dict):
        return (False, "output is not a dict")
    verdict = parsed.get(ALIGNER_VERDICT_KEY)
    if verdict is None or str(verdict).upper() not in VALID_VERDICTS:
        return (False, "invalid or missing verdict")
    verdict = str(verdict).upper()
    try:
        confidence = float(parsed.get(ALIGNER_CONFIDENCE_KEY, 0))
    except (TypeError, ValueError):
        return (False, "confidence not a number")
    if confidence < 0 or confidence > 1:
        return (False, "confidence not in [0, 1]")
    used_raw = parsed.get(ALIGNER_USED_EVIDENCE_KEY)
    if not isinstance(used_raw, list):
        return (False, "used_evidence must be a list")
    used_evidence = []
    for x in used_raw:
        try:
            idx = int(x)
            if 1 <= idx <= num_evidence:
                used_evidence.append(idx)
        except (TypeError, ValueError):
            pass
    valid_indices = set(range(1, num_evidence + 1))
    if verdict in ("SUPPORTED", "REFUTED"):
        if not used_evidence:
            return (False, "SUPPORTED/REFUTED requires non-empty used_evidence")
        if not set(used_evidence) <= valid_indices:
            return (False, "used_evidence contains invalid indices")
    alignment_raw = parsed.get(ALIGNER_ALIGNMENT_KEY)
    if alignment_raw is not None:
        if not isinstance(alignment_raw, list):
            return (False, "alignment must be a list")
        idx_to_snippet: dict[int, str] = {}
        for i, ev in enumerate(evidence, 1):
            idx_to_snippet[i] = ev.snippet if hasattr(ev, "snippet") else str(ev.get("snippet", ""))
        for item in alignment_raw:
            if not isinstance(item, dict):
                return (False, "alignment item must be an object")
            idx = item.get("idx")
            try:
                idx = int(idx) if idx is not None else None
            except (TypeError, ValueError):
                idx = None
            if idx is None or idx not in valid_indices:
                return (False, "alignment idx must be in 1..N")
            if verdict in ("SUPPORTED", "REFUTED") and idx not in used_evidence:
                return (False, "alignment idx must be in used_evidence")
            quote = item.get("quote")
            if quote is None:
                quote = ""
            if not isinstance(quote, str):
                return (False, "alignment quote must be a string")
            snippet = idx_to_snippet.get(idx, "")
            if quote.strip() and quote not in snippet:
                return (False, "alignment quote must be exact substring of snippet")
            role = item.get("role")
            if role is not None and str(role).lower() not in ALIGNMENT_ROLES:
                return (False, "alignment role must be support or refute")
        # REFUTED: alignment with role="refute" is optional; we only require used_evidence non-empty and brief_reason citations (below)
    brief_reason = parsed.get(ALIGNER_BRIEF_REASON_KEY) or ""
    if not isinstance(brief_reason, str):
        brief_reason = ""
    if verdict in ("SUPPORTED", "REFUTED"):
        citation_pattern = re.compile(r"\[(\d+)\]")
        cited = set()
        for m in citation_pattern.finditer(brief_reason):
            n = int(m.group(1))
            if 1 <= n <= num_evidence:
                cited.add(n)
        if not cited:
            return (False, "brief_reason must contain citations like [1] or [2]")
        if not (cited & set(used_evidence)):
            return (False, "citations in brief_reason must overlap used_evidence")
    return (True, None)


def explainer_system_prompt(lang: str = "en") -> str:
    """System prompt for Step 2: evidence-grounded explainer."""
    lang_instruction = " Respond in Japanese (日本語)." if lang == "ja" else " Respond in English."
    return (
        "You are an evidence-grounded explainer. "
        "Use the given verdict exactly; do not contradict it. "
        "Do not restate the verdict label; explain the evidence in line with the given verdict. "
        "Use ONLY the provided evidence snippets and alignment quotes. "
        "Write 6–10 sentences, cite evidence indices like [1][2]. "
        "Do not invent facts. If insufficient, explain what is missing."
        + lang_instruction
    )


def explainer_system_prompt_strict_no_verdict(lang: str = "en") -> str:
    """Stricter system prompt for Step 2 regeneration: no verdict labels, neutral explanation only."""
    lang_instruction = " Respond in Japanese (日本語)." if lang == "ja" else " Respond in English."
    return (
        "You are an evidence-grounded explainer. "
        "Do not include any verdict label words (SUPPORTED, REFUTED, NOT_ENOUGH_INFO). "
        "Explain neutrally using evidence citations only. "
        "Use ONLY the provided evidence snippets and alignment. Cite indices like [1][2]."
        + lang_instruction
    )


def explainer_user_prompt(
    claim: str,
    verdict: str,
    confidence: float,
    alignment_text: str,
    evidence_block: str,
    num_evidence: int,
) -> str:
    """User prompt for Step 2: claim, verdict, alignment, evidence. num_evidence = N (indices 1..N)."""
    range_note = (
        f"There are exactly {num_evidence} evidence items indexed [1]..[{num_evidence}]. "
        "Never cite indices outside this range."
    )
    if num_evidence < 3:
        range_note += " Do not mention [3] or [4]."
    return (
        f"Claim: {claim}\n\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence}\n\n"
        "Alignment (quotes from evidence):\n"
        f"{alignment_text}\n\n"
        "Evidence list:\n"
        f"{evidence_block}\n\n"
        f"{range_note}\n\n"
        "Write a detailed explanation (6–10 sentences): "
        "If SUPPORTED, explain how the aligned quotes entail the claim. "
        "If REFUTED, explain what the evidence says instead of the claim. "
        "If NOT_ENOUGH_INFO, explain what missing evidence would decide. "
        "Cite indices like [1][2] where relevant (only within [1]..[" + str(num_evidence) + "])."
    )


def parse_verdict_and_rationale(response: str) -> tuple[str | None, str]:
    """
    Parse LLM response for VERDICT: and RATIONALE: lines.
    Returns (verdict, rationale). verdict is None if not found or invalid.
    """
    verdict = None
    rationale = ""
    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            v = line[7:].strip().upper()
            if v in VALID_VERDICTS:
                verdict = v
        elif line.upper().startswith("RATIONALE:"):
            rationale = line[10:].strip()
    if not rationale and "rationale" in response.lower():
        # Fallback: take text after last VERDICT line or whole response
        parts = re.split(r"\bVERDICT\s*:\s*\w+", response, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            rationale = parts[-1].replace("RATIONALE:", "").strip()
        else:
            rationale = response.strip()
    return (verdict, rationale or "No rationale provided.")
