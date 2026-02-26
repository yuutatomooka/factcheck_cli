#!/usr/bin/env python3
"""
Interactive Offline Factchecker CLI.
Python stdlib only; no network. Corpus: data/corpus/*.txt
"""
import argparse
import os
import sys
import time

# Project root (parent of factchecker/) on sys.path for "from factchecker.core import ..."
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR) if os.path.basename(_SCRIPT_DIR) == "factchecker" else _SCRIPT_DIR
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from factchecker.core import (
    SessionState,
    Retriever,
    verify,
    verify_with_ollama,
    fallback_cap_result,
)
from factchecker.core.io_utils import (
    run_repl,
    print_pretty,
    print_json,
    result_to_dict,
    load_session,
)


BANNER = r"""
  _____         _      _            _
 |  ___|_ _ ___| | __ | |_ ___  ___| |__
 | |_ / _` / __| |/ / | __/ _ \/ __| '_ \
 |  _| (_| \__ \   <  | ||  __/ (__| | | |
 |_|  \__,_|___/_|\_\  \__\___|\___|_| |_|
"""
DESCRIPTION = "Offline fact-checker over a local text corpus. Type /help for commands."


def default_corpus_path() -> str:
    # factcheck.py lives in factchecker/; data/corpus is sibling (project root/data/corpus)
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base) if os.path.basename(base) == "factchecker" else base
    return os.path.join(root, "data", "corpus")


def build_retriever(corpus_path: str) -> Retriever:
    r = Retriever(corpus_path)
    r.build()
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline fact-checker CLI")
    parser.add_argument("--claim", type=str, help="One-shot: fact-check this claim and exit")
    parser.add_argument("--mode", choices=["pretty", "json"], default="pretty", help="Output mode (one-shot)")
    parser.add_argument("--corpus", type=str, default=None, help="Corpus directory (default: data/corpus)")
    parser.add_argument("--k", type=int, default=10, help="Top-k documents")
    parser.add_argument("--max_sentences_per_doc", type=int, default=2, help="Max sentences per doc for evidence (capped at 2)")
    parser.add_argument("--ollama", action="store_true", help="Enable LLM (Ollama) for verification")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434", help="Ollama host URL (localhost only)")
    parser.add_argument("--model", type=str, default="deepseek-r1:8b", help="Ollama model name")
    parser.add_argument("--llm-role", choices=["verifier", "explainer", "both", "off"], default="both", help="Which components use LLM")
    parser.add_argument("--timeout", type=float, default=60, help="Ollama request timeout (seconds)")
    parser.add_argument("--lang", choices=["en", "ja"], default="en", help="Explanation language")
    parser.add_argument("--debug", action="store_true", help="Log Step1/Step2 raw output and sanitized explanation to stderr")
    args = parser.parse_args()

    corpus_path = args.corpus or default_corpus_path()
    k = max(1, args.k)
    max_sentences_per_doc = max(1, min(args.max_sentences_per_doc, 2))
    mode = args.mode
    ollama_enabled = args.ollama
    ollama_host = args.ollama_host or "http://127.0.0.1:11434"
    model = args.model or "deepseek-r1:8b"
    llm_role = args.llm_role or "both"
    timeout = max(1.0, args.timeout)

    if args.claim is not None:
        # One-shot mode
        retriever = build_retriever(corpus_path)
        evidence = retriever.search(args.claim.strip(), k=k, max_sentences_per_doc=max_sentences_per_doc)
        claim = args.claim.strip()
        result = None
        fallback_reason = None
        if ollama_enabled and llm_role != "off":
            result, fallback_reason = verify_with_ollama(
                claim, evidence,
                host=ollama_host, model=model, timeout=timeout,
                llm_role="both" if llm_role == "both" else llm_role,
                lang=args.lang,
                debug=args.debug,
            )
        if result is None:
            result = verify(claim, evidence)
            if fallback_reason is not None:
                result = fallback_cap_result(result, reason=fallback_reason)
        if mode == "json":
            print_json(result)
        else:
            print_pretty(result)
        return 0

    # Interactive mode
    print(BANNER)
    print("Interactive Offline Factchecker")
    print(DESCRIPTION)
    print("Type /help for commands.")
    print()

    session_id = str(int(time.time()))
    state = SessionState(
        session_id=session_id,
        corpus_path=corpus_path,
        k=k,
        mode="pretty",
        history=[],
        ollama_enabled=ollama_enabled,
        ollama_host=ollama_host,
        model=model,
        llm_role=llm_role,
        timeout=timeout,
        lang="en",
        debug=args.debug,
    )
    retriever = build_retriever(state.corpus_path)

    def doc_count_getter() -> int:
        return retriever._n_docs

    def on_state_change(new_state: SessionState) -> None:
        nonlocal retriever
        if new_state.corpus_path != retriever.corpus_path:
            retriever = build_retriever(new_state.corpus_path)

    def run_factcheck(claim: str, current_state: SessionState):
        evidence = retriever.search(claim, k=current_state.k, max_sentences_per_doc=max_sentences_per_doc)
        fallback_reason = None
        if getattr(current_state, "ollama_enabled", False) and getattr(current_state, "llm_role", "off") != "off":
            result, fallback_reason = verify_with_ollama(
                claim, evidence,
                host=getattr(current_state, "ollama_host", "http://127.0.0.1:11434"),
                model=getattr(current_state, "model", "deepseek-r1:8b"),
                timeout=getattr(current_state, "timeout", 60.0),
                llm_role="both" if current_state.llm_role == "both" else current_state.llm_role,
                lang=getattr(current_state, "lang", "en"),
                debug=getattr(current_state, "debug", False),
            )
            if result is not None:
                return result
        result = verify(claim, evidence)
        if fallback_reason is not None:
            result = fallback_cap_result(result, reason=fallback_reason)
        return result

    run_repl(
        initial_state=state,
        run_factcheck=run_factcheck,
        doc_count_getter=doc_count_getter,
        on_state_change=on_state_change,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
