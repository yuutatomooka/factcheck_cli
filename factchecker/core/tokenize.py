"""Tokenization using Python stdlib only."""
import re
import string

# Small built-in stopword list for key-term extraction
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "as", "at", "from", "it", "its", "that", "this", "be", "was",
    "were", "been", "have", "has", "had", "do", "does", "did", "will", "would",
})


def tokenize(text: str) -> list[str]:
    """Tokenize: lowercase, remove punctuation, split on whitespace."""
    if not text or not text.strip():
        return []
    text = text.lower().strip()
    # Remove punctuation (replace with space then split)
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()


def key_terms(tokens: list[str], min_length: int = 4) -> list[str]:
    """Return tokens that are key terms: length >= min_length, not stopwords."""
    return [t for t in tokens if len(t) >= min_length and t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences; never drop leading text.
    Split on . ? ! when followed by space or end, but not when period is part of abbreviation (e.g. D.C.).
    Use negative lookbehind (?<![A-Z]) so we don't split after a single capital letter (D.C., U.S.).
    If no splits, return [text.strip()].
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    # Split on . or ? or ! followed by space or end; do not split after [A-Z] (abbreviation like D.C.)
    parts = re.split(r"(?<![A-Z])[.?!]\s+", text)
    sentences = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        # Restore trailing terminator except for last part (which may not have one)
        if i < len(parts) - 1:
            if not p.endswith(".") and not p.endswith("?") and not p.endswith("!"):
                p = p + "."
        else:
            if p and p[-1] not in ".?!":
                p = p + "."
        sentences.append(p)
    if not sentences:
        return [text]
    return sentences
