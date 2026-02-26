"""Data schemas for fact-checker (dataclasses / typed dicts, stdlib only)."""
from dataclasses import dataclass, field
from typing import List, Optional, Literal

Verdict = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


@dataclass
class EvidenceItem:
    """Single evidence snippet from a document."""
    doc_id: str
    score: float
    snippet: str
    span: tuple[int, int]  # (start, end) character indices


@dataclass
class FactcheckResult:
    """Result of fact-checking a claim."""
    claim: str
    verdict: Verdict
    confidence: float
    evidence: List[EvidenceItem]
    rationale: str
    explanation: str = ""  # Natural-language body with citations [1][2]
    used_evidence: List[int] = field(default_factory=list)  # Evidence indices 1..N cited


LlmRole = Literal["verifier", "explainer", "both", "off"]


@dataclass
class SessionState:
    """Session state for save/resume and Ollama config."""
    session_id: str
    corpus_path: str
    k: int
    mode: Literal["pretty", "json"]
    history: List[dict] = field(default_factory=list)
    ollama_enabled: bool = False
    ollama_host: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:8b"
    llm_role: LlmRole = "both"
    timeout: float = 60.0
    lang: str = "en"  # Explanation language: en, ja
    sources_mode: str = "used"  # "used" | "all" for SOURCES in pretty output
    debug: bool = False  # Log Step1/Step2 raw and sanitized to stderr
