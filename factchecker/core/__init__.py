"""Core fact-checker components."""
from .schema import Verdict, EvidenceItem, FactcheckResult, SessionState
from .tokenize import tokenize, key_terms, split_sentences, STOPWORDS
from .retrieve import Retriever
from .verify import verify, verify_with_ollama, fallback_cap_result

__all__ = [
    "Verdict", "EvidenceItem", "FactcheckResult", "SessionState",
    "tokenize", "key_terms", "split_sentences", "STOPWORDS",
    "Retriever", "verify", "verify_with_ollama", "fallback_cap_result",
]
