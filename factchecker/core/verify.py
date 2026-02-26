"""Rule-based verifier. Verdict: SUPPORTED / REFUTED / NOT_ENOUGH_INFO. Optional Ollama LLM."""
import re
import string
import sys
from typing import Literal, Optional

from .schema import FactcheckResult, EvidenceItem, Verdict
from .tokenize import tokenize, key_terms


def sanitize_citations(explanation: str, n: int) -> tuple[str, list[int]]:
    """
    Remove bracket citations [k] where k is outside 1..n. Only remove the bracket tokens.
    Returns (sanitized_text, list of removed invalid indices).

    Manual test: With only 2 evidence snippets, Step2 output mentioning [3] should be
    sanitized to remove [3]. NEI confidence should be at least 0.20 even when overlap is zero.
    """
    if n < 1:
        return (explanation, [])
    valid = set(range(1, n + 1))
    pattern = re.compile(r"\[\d+\]")
    invalid_cited: list[int] = []
    def repl(m):
        try:
            k = int(m.group(0)[1:-1])
            if k not in valid:
                invalid_cited.append(k)
                return ""
        except ValueError:
            pass
        return m.group(0)
    out = pattern.sub(repl, explanation)
    out = re.sub(r"  +", " ", out)
    out = re.sub(r"\n\n+", "\n\n", out)
    out = out.strip()
    if invalid_cited:
        removed = sorted(set(invalid_cited))
        out = out + ' (Note: removed invalid citations: ' + ", ".join(f"[{i}]" for i in removed) + ".)"
    return (out, invalid_cited)


def _explanation_contradicts_verdict(explanation: str, verdict: str) -> bool:
    """True if explanation contains an explicit verdict label that contradicts the given verdict."""
    if not explanation or not verdict:
        return False
    up = explanation.upper()
    if verdict == "SUPPORTED" and re.search(r"\bREFUTED\b", up):
        return True
    if verdict == "REFUTED" and re.search(r"\bSUPPORTED\b", up):
        return True
    if verdict == "NOT_ENOUGH_INFO" and (re.search(r"\bSUPPORTED\b", up) or re.search(r"\bREFUTED\b", up)):
        return True
    return False


def _strip_verdict_words_from_explanation(explanation: str) -> str:
    """Replace verdict labels with [verdict] to avoid contradicting the actual verdict."""
    return re.sub(r"\b(SUPPORTED|REFUTED|NOT_ENOUGH_INFO)\b", "[verdict]", explanation, flags=re.IGNORECASE)


def _overlap_ratio(claim_key_terms: set[str], snippet: str) -> float:
    """(# claim key terms in snippet) / (# claim key terms). 0 if no key terms."""
    if not claim_key_terms:
        return 0.0
    snippet_tokens = set(tokenize(snippet))
    matched = len(claim_key_terms & snippet_tokens)
    return matched / len(claim_key_terms)


def _best_snippet(claim_key_terms: set[str], evidence: list[EvidenceItem]) -> tuple[EvidenceItem | None, float, int]:
    """Return (evidence item with highest overlap_ratio, best_overlap_ratio, 1-based index)."""
    if not evidence or not claim_key_terms:
        return (None, 0.0, 0)
    best_ev = evidence[0]
    best_r = _overlap_ratio(claim_key_terms, best_ev.snippet)
    best_idx = 1
    for i, ev in enumerate(evidence[1:], 2):
        r = _overlap_ratio(claim_key_terms, ev.snippet)
        if r > best_r:
            best_r = r
            best_ev = ev
            best_idx = i
    return (best_ev, best_r, best_idx)


def _claim_subject_tokens(claim: str) -> set[str]:
    """Tokens that likely refer to the claim's subject: key terms plus capitalized words (e.g. Washington, D.C.)."""
    key_set = set(key_terms(list(tokenize(claim))))
    words = claim.strip().split()
    capitalized = set()
    for w in words:
        w_clean = w.strip(string.punctuation)
        if w_clean and w_clean[0].isupper():
            capitalized.add(w_clean.lower())
    return key_set | capitalized


def _snippet_negation_subject(snippet: str) -> set[str]:
    """Subject of the negation in snippet: tokens before ' is not ' or ' does not '. Empty if no negation."""
    low = snippet.lower()
    if " is not " in low:
        before = low.split(" is not ", 1)[0].strip()
        return set(tokenize(before))
    m = re.search(r"^(.+?)\s+does\s+not\s+\w+", low)
    if m:
        return set(tokenize(m.group(1).strip()))
    return set()


def _negation_same_subject_as_claim(claim: str, snippet: str) -> bool:
    """True only if the snippet's negation refers to the same subject/entity as the claim (by token overlap)."""
    claim_subj = _claim_subject_tokens(claim)
    snippet_subj = _snippet_negation_subject(snippet)
    if not snippet_subj:
        return False
    return bool(claim_subj & snippet_subj)


def _direct_negation_in_snippet(snippet: str, claim_key_terms: set[str], claim: str = "") -> bool:
    """
    True if snippet contains a direct negation of the core proposition.
    Conservative: "X is not Y", "does not <verb>" only when key-term overlap is present
    and the negation refers to the SAME subject as the claim (e.g. "New York is NOT..." does not refute "Washington D.C. is...").
    False when negation is inside "the statement that ... is false/incorrect" (that supports the claim).
    """
    low = snippet.lower()
    tokens = set(tokenize(snippet))
    if not (tokens & claim_key_terms):
        return False
    # Meta-support: "the statement that [negative clause] is false/incorrect" → supports claim, not negation
    if "the statement that" in low and ("false" in low or "incorrect" in low):
        return False
    # "<subject> is not <object>" or "does not <verb>" — only refute if same subject as claim
    if re.search(r"\bis\s+not\s+", low) or re.search(r"\bdoes\s+not\s+\w+", low):
        # Require same subject so "New York is NOT the capital" does not refute "Washington D.C. is the capital"
        return _negation_same_subject_as_claim(claim, snippet) if claim else True
    return False


def _extract_capital_of_object(text: str) -> str | None:
    """Extract object after 'capital of' (e.g. 'South Korea', 'Japan'). Stops at ' is ', [.,;:], or end."""
    m = re.search(r"\bcapital\s+of\s+([^.,;:]+?)(?:\s+is\s+|\s*[.,;:]|\s*$)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).strip().lower()


def _capital_of_mismatch(claim: str, claim_key_terms: set[str], evidence: list[EvidenceItem]) -> tuple[bool, float, str, int]:
    """
    If claim has 'capital' and 'of', and some snippet has different object after 'capital of'
    with strong subject overlap, return (True, confidence, rationale, 1-based index). Else (False, 0, "", 0).
    """
    claim_low = claim.lower()
    if "capital" not in claim_low or " of " not in claim_low:
        return (False, 0.0, "", 0)
    claim_obj = _extract_capital_of_object(claim)
    if not claim_obj:
        return (False, 0.0, "", 0)
    for i, ev in enumerate(evidence, 1):
        r = _overlap_ratio(claim_key_terms, ev.snippet)
        if r < 0.5:
            continue
        ev_obj = _extract_capital_of_object(ev.snippet)
        if not ev_obj:
            continue
        if claim_obj != ev_obj:
            conf = min(1.0, 0.7 + 0.3 * r)
            return (True, round(conf, 2), f"Evidence states a conflicting fact: {ev.doc_id} states the capital is of a different country.", i)
    return (False, 0.0, "", 0)


def _direct_negation_refuted(claim: str, claim_key_terms: set[str], evidence: list[EvidenceItem]) -> tuple[bool, float, str, int]:
    """REFUTED only if some snippet has direct negation, overlap >= 0.5, same subject. Returns (ok, conf, rationale, 1-based index)."""
    for i, ev in enumerate(evidence, 1):
        r = _overlap_ratio(claim_key_terms, ev.snippet)
        if r < 0.5:
            continue
        if _direct_negation_in_snippet(ev.snippet, claim_key_terms, claim):
            conf = min(1.0, 0.7 + 0.3 * r)
            return (True, round(conf, 2), f"Evidence directly contradicts the claim ({ev.doc_id}).", i)
    return (False, 0.0, "", 0)


def _meta_refutation(claim: str, claim_key_terms: set[str], evidence: list[EvidenceItem]) -> tuple[bool, float, str, int]:
    """
    REFUTED if snippet contains "the statement that" + "false"/"incorrect" and quoted part is positive form of claim.
    Returns (ok, conf, rationale, 1-based index).
    """
    for i, ev in enumerate(evidence, 1):
        r = _overlap_ratio(claim_key_terms, ev.snippet)
        if r < 0.5:
            continue
        low = ev.snippet.lower()
        if "the statement that" not in low or ("false" not in low and "incorrect" not in low):
            continue
        m = re.search(r"the\s+statement\s+that\s+(.+?)\s+is\s+(?:false|incorrect)", low, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        quoted = m.group(1).strip()
        quoted_tokens = set(tokenize(quoted))
        if not quoted_tokens:
            continue
        if "not" in quoted_tokens or "never" in quoted_tokens or "no " in quoted:
            continue
        overlap = len(claim_key_terms & quoted_tokens) / len(claim_key_terms) if claim_key_terms else 0
        if overlap >= 0.5:
            conf = min(1.0, 0.7 + 0.3 * r)
            return (True, round(conf, 2), f"Evidence states that the claim is false or incorrect ({ev.doc_id}).", i)
    return (False, 0.0, "", 0)


# Opposite directional phrases (claim direction -> evidence direction that refutes)
_DIRECTION_OPPOSITES = [
    ("north", "south"),
    ("south", "north"),
    ("east", "west"),
    ("west", "east"),
]


def _directional_contradiction(claim: str, claim_key_terms: set[str], evidence: list[EvidenceItem]) -> tuple[bool, float, str, int]:
    """
    If claim asserts a direction (e.g. to the south) and high-overlap evidence asserts the opposite (e.g. to the north)
    in a border context, return (True, confidence, rationale, 1-based index). Else (False, 0, "", 0).
    Conservative: snippet must contain 'border' or 'borders' and entity from claim if present.
    """
    claim_low = claim.lower()
    # Require at least one directional phrase in claim
    claim_has = None
    for a, b in _DIRECTION_OPPOSITES:
        if f"to the {a}" in claim_low or f" {a} " in claim_low:
            claim_has = a
            evidence_must_have = b
            break
    if claim_has is None:
        return (False, 0.0, "", 0)
    # Entity: capitalized words in claim (likely place names)
    claim_words = claim.strip().split()
    entity_tokens = set()
    for w in claim_words:
        w_clean = w.strip(string.punctuation)
        if w_clean and len(w_clean) > 1 and w_clean[0].isupper():
            entity_tokens.add(w_clean.lower())
    for i, ev in enumerate(evidence, 1):
        r = _overlap_ratio(claim_key_terms, ev.snippet)
        if r < 0.5:
            continue
        snip_low = ev.snippet.lower()
        if "border" not in snip_low and "borders" not in snip_low:
            continue
        if f"to the {evidence_must_have}" not in snip_low and f" {evidence_must_have} " not in snip_low:
            continue
        if entity_tokens:
            if not any(e in snip_low for e in entity_tokens):
                continue
        conf = min(0.95, max(0.80, 0.7 + 0.25 * r))
        rationale = f"Evidence states the opposite direction ({ev.doc_id}): borders to the {evidence_must_have}, not {claim_has}."
        return (True, round(conf, 2), rationale, i)
    return (False, 0.0, "", 0)


# Refutation cues that must appear in brief_reason (A) to accept REFUTED without B/C
_REFUTATION_CUES = ("not", "false", "incorrect", "wrong", "opposite", "contradict", "refute")


def _refutation_support_ok(
    claim: str,
    claim_key_terms: set[str],
    evidence: list[EvidenceItem],
    used_evidence: list[int],
    brief_reason: str,
) -> bool:
    """
    Accept REFUTED only if at least one of:
    A) brief_reason contains a refutation cue (not, no, false, incorrect, wrong, opposite, contradict, refute)
    B) some used_evidence snippet shows directional contradiction (claim vs snippet)
    C) some used_evidence snippet contains direct negation (is not / does not) with same subject.
    Used after Step1 validation; prevents accepting low-quality REFUTED that only cites supportive evidence.

    Manual test: REFUTED claim where evidence is supportive only (no cue/directional/negation) -> Step1 REFUTED
    should be rejected and fallback to rule-based.
    """
    brief_low = (brief_reason or "").lower()
    # A) Refutation cue in brief_reason (citations already required by validation)
    if any(cue in brief_low for cue in _REFUTATION_CUES):
        return True
    # B) Directional contradiction in any used snippet
    for idx in used_evidence:
        if 1 <= idx <= len(evidence):
            ev = evidence[idx - 1]
            dir_ok, _, _, _ = _directional_contradiction(claim, claim_key_terms, [ev])
            if dir_ok:
                return True
    # C) Direct negation in any used snippet (same-subject check inside _direct_negation_in_snippet)
    for idx in used_evidence:
        if 1 <= idx <= len(evidence):
            if _direct_negation_in_snippet(evidence[idx - 1].snippet, claim_key_terms, claim):
                return True
    return False


FALLBACK_REASON_UNAVAILABLE = "unavailable"
FALLBACK_REASON_INVALID_JSON = "invalid_json"
FALLBACK_REASON_INVALID_OUTPUT = "invalid_output"

_FALLBACK_NOTES = {
    FALLBACK_REASON_UNAVAILABLE: " [LLM unavailable (timeout/connection); used rule-based fallback.]",
    FALLBACK_REASON_INVALID_JSON: " [LLM output invalid JSON; used rule-based fallback.]",
    FALLBACK_REASON_INVALID_OUTPUT: " [LLM output invalid (validation); used rule-based fallback.]",
}


def fallback_cap_result(
    result: FactcheckResult,
    max_confidence: float = 0.75,
    reason: str = FALLBACK_REASON_UNAVAILABLE,
) -> FactcheckResult:
    """Cap confidence and append fallback note when using rule-based verify() after Ollama failure.
    reason: 'unavailable' | 'invalid_json' | 'invalid_output'. When LLM output fails validation
    (not timeout), use invalid_output so the note says 'invalid (validation)' not 'unavailable'.

    Manual test: When LLM output fails validation but is not a timeout, the note must say
    "LLM output invalid (validation)" not "unavailable". Use reason=invalid_output in that case.
    """
    cap = min(result.confidence, max_confidence)
    note = _FALLBACK_NOTES.get(reason, _FALLBACK_NOTES[FALLBACK_REASON_UNAVAILABLE])
    return FactcheckResult(
        claim=result.claim,
        verdict=result.verdict,
        confidence=round(cap, 2),
        evidence=result.evidence,
        rationale=result.rationale + note,
        explanation=(result.explanation or result.rationale) + note,
        used_evidence=result.used_evidence,
    )


def verify(claim: str, evidence: list[EvidenceItem]) -> FactcheckResult:
    """
    Three-step decision:
    A) SUPPORTED if best_overlap_ratio >= 0.75 and best_snippet has no direct negation.
    B) REFUTED if capital-of mismatch, direct negation, or meta-refutation (with overlap >= 0.5).
    C) Otherwise NOT_ENOUGH_INFO.
    """
    claim_tokens = set(tokenize(claim))
    claim_key = set(key_terms(list(claim_tokens)))
    if not evidence:
        return FactcheckResult(
            claim=claim,
            verdict="NOT_ENOUGH_INFO",
            confidence=0.20,
            evidence=[],
            rationale="No evidence found in the corpus.",
            explanation="No evidence found in the corpus.",
            used_evidence=[],
        )

    best_ev, best_overlap_ratio, best_idx = _best_snippet(claim_key, evidence)

    # Directional contradiction (before SUPPORTED): north/south, east/west opposite with border context
    dir_ok, dir_conf, dir_rationale, dir_idx = _directional_contradiction(claim, claim_key, evidence)
    if dir_ok:
        return FactcheckResult(
            claim=claim,
            verdict="REFUTED",
            confidence=dir_conf,
            evidence=evidence,
            rationale=dir_rationale,
            explanation=dir_rationale,
            used_evidence=[dir_idx] if dir_idx else [],
        )

    # Step A: SUPPORTED first (prioritize support over refutation). used_evidence = single best snippet index.
    if best_overlap_ratio >= 0.75 and best_ev is not None and not _direct_negation_in_snippet(best_ev.snippet, claim_key, claim):
        confidence = max(0.0, min(1.0, 0.6 + 0.4 * best_overlap_ratio))
        rationale = f"Evidence directly states the claim. Best match in {best_ev.doc_id}."
        return FactcheckResult(
            claim=claim,
            verdict="SUPPORTED",
            confidence=round(confidence, 2),
            evidence=evidence,
            rationale=rationale,
            explanation=rationale,
            used_evidence=[best_idx] if best_idx else [],
        )

    # Step B: REFUTED. used_evidence = single index of snippet that triggered refutation.
    cap_ok, cap_conf, cap_rationale, cap_idx = _capital_of_mismatch(claim, claim_key, evidence)
    if cap_ok:
        return FactcheckResult(
            claim=claim,
            verdict="REFUTED",
            confidence=cap_conf,
            evidence=evidence,
            rationale=cap_rationale,
            explanation=cap_rationale,
            used_evidence=[cap_idx] if cap_idx else [],
        )

    neg_ok, neg_conf, neg_rationale, neg_idx = _direct_negation_refuted(claim, claim_key, evidence)
    if neg_ok:
        # Support precedence: strong direct supporting evidence (no negation) → prefer SUPPORTED
        if best_overlap_ratio >= 0.80 and best_ev is not None and not _direct_negation_in_snippet(best_ev.snippet, claim_key, claim):
            confidence = min(0.75, 0.6 + 0.4 * best_overlap_ratio)
            rationale = f"Evidence directly states the claim. Best match in {best_ev.doc_id}."
            return FactcheckResult(
                claim=claim,
                verdict="SUPPORTED",
                confidence=round(confidence, 2),
                evidence=evidence,
                rationale=rationale,
                explanation=rationale,
                used_evidence=[best_idx] if best_idx else [],
            )
        return FactcheckResult(
            claim=claim,
            verdict="REFUTED",
            confidence=neg_conf,
            evidence=evidence,
            rationale="Evidence directly contradicts the claim.",
            explanation=neg_rationale,
            used_evidence=[neg_idx] if neg_idx else [],
        )

    meta_ok, meta_conf, meta_rationale, meta_idx = _meta_refutation(claim, claim_key, evidence)
    if meta_ok:
        return FactcheckResult(
            claim=claim,
            verdict="REFUTED",
            confidence=meta_conf,
            evidence=evidence,
            rationale=meta_rationale,
            explanation=meta_rationale,
            used_evidence=[meta_idx] if meta_idx else [],
        )

    # Step C: NOT_ENOUGH_INFO — confidence in [0.20, 0.60] by best key-term overlap
    confidence = max(0.20, min(0.60, 0.20 + 0.40 * best_overlap_ratio))
    rationale = "Evidence does not directly support or contradict the claim."
    return FactcheckResult(
        claim=claim,
        verdict="NOT_ENOUGH_INFO",
        confidence=round(confidence, 2),
        evidence=evidence,
        rationale=rationale,
        explanation=rationale,
        used_evidence=[],
    )


def verify_with_ollama(
    claim: str,
    evidence: list[EvidenceItem],
    *,
    host: str,
    model: str,
    timeout: float = 60.0,
    llm_role: Literal["verifier", "explainer", "both"] = "both",
    lang: str = "en",
    debug: bool = False,
) -> tuple[Optional[FactcheckResult], Optional[str]]:
    """
    Two-step LLM: (1) JSON-only verifier+aligner, (2) optional long explainer.
    Returns (result, None) on success; (None, fallback_reason) on failure.
    When evidence is empty, returns rule-based NEI directly with no fallback reason (Ollama not called).

    Manual test: Claim with empty corpus -> NEI, no LLM fallback note.
    """
    from . import ollama_client
    num_evidence = len(evidence)
    if num_evidence == 0:
        return (verify(claim, []), None)

    # explainer-only: rule-based verdict + Step 2 for long explanation
    if llm_role == "explainer":
        rule_result = verify(claim, evidence)
        alignment_parts = []
        for i in rule_result.used_evidence:
            if 1 <= i <= len(evidence):
                ev = evidence[i - 1]
                alignment_parts.append(f"  [{i}] (support): \"{ev.snippet}\"")
        alignment_text = "\n".join(alignment_parts) if alignment_parts else "  (none)"
        evidence_block_s2 = "\n".join(f"[{i}] {e.doc_id} — {e.snippet!r}" for i, e in enumerate(evidence, 1))
        system_s2 = ollama_client.explainer_system_prompt(lang)
        user_s2 = ollama_client.explainer_user_prompt(
            claim, rule_result.verdict, rule_result.confidence, alignment_text, evidence_block_s2, len(evidence)
        )
        explanation = rule_result.explanation
        response_s2_raw = None
        try:
            response_s2 = ollama_client.chat_or_generate(host, model, system_s2, user_s2, timeout=timeout)
            if response_s2 and response_s2.strip():
                response_s2_raw = response_s2.strip()
                explanation, _ = sanitize_citations(response_s2_raw, len(evidence))
        except Exception as e:
            print(f"Ollama explainer failed: {e}; using rule rationale.", file=sys.stderr)
        explanation, _ = sanitize_citations(explanation, len(evidence))
        if debug:
            if response_s2_raw is not None:
                print("[FACTCHECK_DEBUG] Step2 raw:", response_s2_raw[:500] + ("..." if len(response_s2_raw) > 500 else ""), file=sys.stderr)
            print("[FACTCHECK_DEBUG] Sanitized explanation:", explanation[:300] + ("..." if len(explanation) > 300 else ""), file=sys.stderr)
        return (FactcheckResult(
            claim=claim,
            verdict=rule_result.verdict,
            confidence=rule_result.confidence,
            evidence=evidence,
            rationale=rule_result.rationale,
            explanation=explanation,
            used_evidence=rule_result.used_evidence,
        ), None)

    evidence_lines = [f"[{i}] doc_id={e.doc_id} snippet={e.snippet!r}" for i, e in enumerate(evidence, 1)]
    system_s1 = ollama_client.verifier_aligner_system_prompt()
    user_s1 = ollama_client.verifier_aligner_user_prompt(claim, evidence_lines)
    try:
        response_s1 = ollama_client.chat_or_generate(host, model, system_s1, user_s1, timeout=timeout)
    except Exception as e:
        print(f"Ollama request failed: {e}", file=sys.stderr)
        return (None, FALLBACK_REASON_UNAVAILABLE)
    if debug:
        print("[FACTCHECK_DEBUG] Step1 raw:", response_s1[:800] + ("..." if len(response_s1) > 800 else ""), file=sys.stderr)
    parsed = ollama_client.parse_aligner_response(response_s1)
    if parsed is None:
        print("Ollama Step 1 response was not valid JSON; falling back to rule-based verifier.", file=sys.stderr)
        return (None, FALLBACK_REASON_INVALID_JSON)
    valid, reason = ollama_client.validate_aligner_output(parsed, evidence, num_evidence)
    if not valid:
        print(f"Ollama verifier output invalid ({reason}); falling back to rule-based verifier.", file=sys.stderr)
        return (None, FALLBACK_REASON_INVALID_OUTPUT)
    verdict = str(parsed.get("verdict", "")).upper()
    brief_reason = (parsed.get("brief_reason") or "").strip()
    used_raw = parsed.get("used_evidence") or []
    used_evidence_pre = [int(x) for x in used_raw if isinstance(x, (int, float)) and 1 <= int(x) <= num_evidence]
    used_evidence_pre = list(dict.fromkeys(used_evidence_pre))
    claim_key = set(key_terms(list(tokenize(claim))))
    if verdict == "REFUTED" and not _refutation_support_ok(claim, claim_key, evidence, used_evidence_pre, brief_reason):
        print("Ollama REFUTED lacks refutation support (cue/directional/negation); falling back to rule-based verifier.", file=sys.stderr)
        return (None, FALLBACK_REASON_INVALID_OUTPUT)
    if debug:
        print("[FACTCHECK_DEBUG] Step1 parsed verdict:", parsed.get("verdict"), "confidence:", parsed.get("confidence"), "used_evidence:", parsed.get("used_evidence"), file=sys.stderr)
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    # NEI confidence policy: [0.20, 0.60] by best key-term overlap
    claim_key = set(key_terms(list(tokenize(claim))))
    best_ev, best_overlap_ratio, best_idx = _best_snippet(claim_key, evidence)
    if verdict == "NOT_ENOUGH_INFO":
        confidence = max(0.20, min(0.60, 0.20 + 0.40 * best_overlap_ratio))
    used_raw = parsed.get("used_evidence") or []
    used_evidence = [int(x) for x in used_raw if isinstance(x, (int, float)) and 1 <= int(x) <= num_evidence]
    used_evidence = list(dict.fromkeys(used_evidence))
    brief_reason = (parsed.get("brief_reason") or "").strip() or f"Verdict: {verdict}. Confidence: {confidence}."

    # Guardrail F: REFUTED but near-exact supporting snippet (overlap >= 0.80, no negation) -> override to SUPPORTED, confidence capped 0.75
    if (
        verdict == "REFUTED"
        and best_ev is not None
        and best_overlap_ratio >= 0.80
        and not _direct_negation_in_snippet(best_ev.snippet, claim_key, claim)
    ):
        verdict = "SUPPORTED"
        confidence = min(confidence, 0.75)
        used_evidence = [best_idx] if best_idx else used_evidence
        brief_reason = brief_reason + " [Overridden to SUPPORTED due to direct supporting evidence in the corpus.]"

    explanation = brief_reason
    response_s2_raw = None
    if llm_role in ("explainer", "both"):
        alignment_list = parsed.get("alignment") or []
        alignment_parts = []
        for item in alignment_list:
            if not isinstance(item, dict):
                continue
            idx = item.get("idx")
            quote = item.get("quote") or ""
            role = item.get("role", "support")
            alignment_parts.append(f"  [{idx}] ({role}): \"{quote}\"")
        alignment_text = "\n".join(alignment_parts) if alignment_parts else "  (none)"
        evidence_block_s2 = "\n".join(f"[{i}] {e.doc_id} — {e.snippet!r}" for i, e in enumerate(evidence, 1))
        system_s2 = ollama_client.explainer_system_prompt(lang)
        user_s2 = ollama_client.explainer_user_prompt(claim, verdict, confidence, alignment_text, evidence_block_s2, num_evidence)
        try:
            response_s2 = ollama_client.chat_or_generate(host, model, system_s2, user_s2, timeout=timeout)
            if response_s2 and response_s2.strip():
                response_s2_raw = response_s2.strip()
                explanation = response_s2_raw
                # Conflict resolution: if Step2 contradicts verdict, try one regeneration with strict prompt
                if _explanation_contradicts_verdict(explanation, verdict):
                    try:
                        system_s2_strict = ollama_client.explainer_system_prompt_strict_no_verdict(lang)
                        response_s2_retry = ollama_client.chat_or_generate(host, model, system_s2_strict, user_s2, timeout=timeout)
                        if response_s2_retry and response_s2_retry.strip():
                            retry_text = response_s2_retry.strip()
                            if not _explanation_contradicts_verdict(retry_text, verdict):
                                explanation = retry_text
                            else:
                                explanation = _strip_verdict_words_from_explanation(explanation) + " (Note: explanation contradicted verdict; explanation sanitized.)"
                        else:
                            explanation = _strip_verdict_words_from_explanation(explanation) + " (Note: explanation contradicted verdict; explanation sanitized.)"
                    except Exception:
                        explanation = _strip_verdict_words_from_explanation(explanation) + " (Note: explanation contradicted verdict; explanation sanitized.)"
        except Exception as e:
            print(f"Ollama explainer failed: {e}; using brief_reason.", file=sys.stderr)
    # Always sanitize citations before returning (fixes invalid indices and contradictions)
    explanation, _ = sanitize_citations(explanation, num_evidence)
    if debug:
        if response_s2_raw is not None:
            print("[FACTCHECK_DEBUG] Step2 raw:", response_s2_raw[:500] + ("..." if len(response_s2_raw) > 500 else ""), file=sys.stderr)
        print("[FACTCHECK_DEBUG] Sanitized explanation:", explanation[:300] + ("..." if len(explanation) > 300 else ""), file=sys.stderr)

    return (FactcheckResult(
        claim=claim,
        verdict=verdict,
        confidence=round(confidence, 2),
        evidence=evidence,
        rationale=brief_reason,
        explanation=explanation,
        used_evidence=used_evidence,
    ), None)
