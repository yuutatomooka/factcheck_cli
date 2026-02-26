"""BM25-like retrieval using Python stdlib only."""
import math
import os
from collections import defaultdict

from .tokenize import tokenize, split_sentences
from .schema import EvidenceItem


def _tf(term: str, tokens: list[str]) -> float:
    """Term frequency (raw count) in token list."""
    return float(tokens.count(term))


class Retriever:
    """Loads corpus, computes DF/IDF, scores and ranks documents."""

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.doc_ids: list[str] = []
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_text: dict[str, str] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._idf: dict[str, float] = {}
        self._n_docs = 0
        self._built = False

    def build(self) -> int:
        """Load documents from corpus dir (filename = doc_id). Build DF/IDF. Returns doc count."""
        if not os.path.isdir(self.corpus_path):
            return 0
        self._doc_tokens.clear()
        self._doc_text.clear()
        self._df.clear()
        self.doc_ids = []
        for name in sorted(os.listdir(self.corpus_path)):
            path = os.path.join(self.corpus_path, name)
            if not os.path.isfile(path) or not name.endswith(".txt"):
                continue
            doc_id = name
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            self._doc_text[doc_id] = text
            tokens = tokenize(text)
            self._doc_tokens[doc_id] = tokens
            self.doc_ids.append(doc_id)
            for t in set(tokens):
                self._df[t] += 1
        self._n_docs = len(self.doc_ids)
        # IDF: idf = log((N+1)/(df+1)) + 1
        for term, df in self._df.items():
            self._idf[term] = math.log((self._n_docs + 1) / (df + 1)) + 1
        self._built = True
        return self._n_docs

    def search(
        self,
        query: str,
        k: int = 10,
        max_sentences_per_doc: int = 3,
    ) -> list[EvidenceItem]:
        """
        Rank docs by sum over query terms of tf(term, doc) * idf(term).
        Use only top k_docs_for_evidence = min(k, 2) docs; per doc keep top
        min(max_sentences_per_doc, 2) sentences. Total evidence <= 4 by default.
        """
        if not self._built or self._n_docs == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scores: list[tuple[str, float]] = []
        for doc_id in self.doc_ids:
            tokens = self._doc_tokens[doc_id]
            score = 0.0
            for term in q_set:
                tf_val = _tf(term, tokens)
                idf_val = self._idf.get(term, 1.0)
                score += tf_val * idf_val
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: -x[1])
        k_docs_for_evidence = min(max(1, k), 2)
        sentences_per_doc = min(max(1, max_sentences_per_doc), 2)
        top = scores[:k_docs_for_evidence]

        evidence: list[EvidenceItem] = []
        for doc_id, score in top:
            text = self._doc_text.get(doc_id, "")
            sentences = split_sentences(text)
            sent_scores: list[tuple[str, int, int, float]] = []
            pos = 0
            for sent in sentences:
                sent_tokens = set(tokenize(sent))
                overlap = len(q_set & sent_tokens)
                if overlap < 1:
                    continue
                sent_scores.append((sent, pos, pos + len(sent), float(overlap)))
                pos += len(sent) + 2
            sent_scores.sort(key=lambda x: -x[3])
            taken = 0
            for sent, start, end, _ in sent_scores:
                if taken >= sentences_per_doc:
                    break
                evidence.append(
                    EvidenceItem(doc_id=doc_id, score=round(score, 2), snippet=sent, span=(start, end))
                )
                taken += 1
        # Cap at 4 by sentence overlap (re-score by overlap for ordering)
        if len(evidence) > 4:
            with_overlap = [(e, len(q_set & set(tokenize(e.snippet)))) for e in evidence]
            with_overlap.sort(key=lambda x: -x[1])
            evidence = [e for e, _ in with_overlap[:4]]
        return evidence
