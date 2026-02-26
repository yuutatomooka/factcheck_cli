"""REPL, session handling, pretty/JSON output."""
import json
import os
import sys
from typing import Callable, Optional

from .schema import (
    SessionState,
    FactcheckResult,
    EvidenceItem,
)


def _ollama_kw(state: SessionState) -> dict:
    """Ollama-related kwargs for SessionState from current state."""
    return {
        "ollama_enabled": getattr(state, "ollama_enabled", False),
        "ollama_host": getattr(state, "ollama_host", "http://127.0.0.1:11434"),
        "model": getattr(state, "model", "deepseek-r1:8b"),
        "llm_role": getattr(state, "llm_role", "both"),
        "timeout": getattr(state, "timeout", 60.0),
        "lang": getattr(state, "lang", "en"),
        "sources_mode": getattr(state, "sources_mode", "used"),
        "debug": getattr(state, "debug", False),
    }


def result_to_dict(r: FactcheckResult) -> dict:
    """Serialize FactcheckResult to JSON-serializable dict. Preserve used_evidence as-is; if missing/empty keep []."""
    used = getattr(r, "used_evidence", None)
    if used is None:
        used = []
    expl = getattr(r, "explanation", None) or r.rationale
    d = {
        "claim": r.claim,
        "verdict": r.verdict,
        "confidence": r.confidence,
        "evidence": [
            {"doc_id": e.doc_id, "score": e.score, "snippet": e.snippet, "span": list(e.span), "idx": i}
            for i, e in enumerate(r.evidence, 1)
        ],
        "rationale": r.rationale,
        "explanation": expl,
        "used_evidence": list(used),
    }
    return d


def dict_to_evidence(d: dict) -> EvidenceItem:
    """Deserialize evidence item from dict."""
    return EvidenceItem(
        doc_id=d["doc_id"],
        score=float(d["score"]),
        snippet=d["snippet"],
        span=tuple(d["span"]),
    )


def print_pretty(result: FactcheckResult, sources_mode: str = "used") -> None:
    """
    Natural-language-first report: structured header + SOURCES + EXPLANATION.
    sources_mode "used": show only used_evidence indices (or top 2 if empty).
    sources_mode "all": show all evidence snippets.
    """
    used = getattr(result, "used_evidence", None)
    if used is None:
        used = []
    expl = getattr(result, "explanation", None) or result.rationale
    print("---")
    print("CLAIM:", result.claim)
    print("VERDICT:", result.verdict)
    print("CONFIDENCE: {:.2f}".format(result.confidence))
    print("EVIDENCE USED:", ", ".join(f"[{i}]" for i in used) if used else "[]")
    print("SOURCES:")
    idx_to_ev = {i: ev for i, ev in enumerate(result.evidence, 1)}
    if sources_mode == "all":
        for i, ev in enumerate(result.evidence, 1):
            print("  [{}] {} — \"{}\"".format(i, ev.doc_id, ev.snippet))
    elif used:
        for i in used:
            if i in idx_to_ev:
                ev = idx_to_ev[i]
                print("  [{}] {} — \"{}\"".format(i, ev.doc_id, ev.snippet))
    else:
        for i, ev in enumerate(result.evidence[:2], 1):
            print("  [{}] {} — \"{}\"".format(i, ev.doc_id, ev.snippet))
    print("---")
    print("EXPLANATION:")
    print(expl)


def print_json(result: FactcheckResult) -> None:
    """Print result as single JSON object to stdout (JSON mode: stdout only)."""
    d = result_to_dict(result)
    print(json.dumps(d, ensure_ascii=False))


def run_repl(
    *,
    initial_state: SessionState,
    run_factcheck: Callable[[str, SessionState], FactcheckResult],
    doc_count_getter: Callable[[], int],
    on_state_change: Optional[Callable[[SessionState], None]] = None,
) -> None:
    """
    Interactive REPL. Uses stdin/stdout; debug to stderr.
    on_state_change(state) is called when settings or history change (e.g. for auto-save).
    """
    state = SessionState(
        session_id=initial_state.session_id,
        corpus_path=initial_state.corpus_path,
        k=initial_state.k,
        mode=initial_state.mode,
        history=list(initial_state.history),
        **_ollama_kw(initial_state),
    )
    multi_line_buffer: list[str] = []
    last_interrupt_time: Optional[float] = None
    INTERRUPT_DOUBLE_THRESHOLD = 1.0  # seconds

    def apply_state(s: SessionState) -> None:
        nonlocal state
        state = s
        if on_state_change:
            on_state_change(state)

    def show_help() -> None:
        lines = [
            "/help          - Show this command list",
            "/exit, /quit, /q - Exit (session may be auto-saved)",
            "/clear         - Clear the screen",
            "/status        - Show corpus path, k, mode, session id, doc count",
            "/k <n>         - Set top-k documents",
            "/corpus <path> - Set corpus directory",
            "/mode pretty|json - Output mode",
            "/json on|off   - Alias for mode",
            "/ollama on|off - Enable/disable LLM (Ollama)",
            "/ollama_host <url> - Set Ollama host (localhost only)",
            "/model <name>  - Set Ollama model",
            "/llm_role verifier|explainer|both|off - Which parts use LLM",
            "/lang en|ja    - Explanation language (en default)",
            "/sources used|all - SOURCES: used only (or top 2) vs all",
            "/ping          - Check Ollama availability",
            "/load <file>   - Load claims from file and evaluate",
            "/save [file]   - Save session to JSONL",
            "/resume <file> - Load session from JSONL",
            "/compact       - Reduce history to last N items",
        ]
        for line in lines:
            print(line)

    def do_clear() -> None:
        if sys.platform == "win32":
            os.system("cls")
        else:
            os.system("clear")

    def do_status(doc_count: int) -> None:
        print("corpus path:", state.corpus_path)
        print("k:", state.k)
        print("mode:", state.mode)
        print("session id:", state.session_id)
        print("documents loaded:", doc_count)

    def do_save(path: Optional[str] = None) -> None:
        import time
        if path is None:
            os.makedirs("sessions", exist_ok=True)
            path = "sessions/{}.jsonl".format(int(time.time()))
        try:
            with open(path, "w", encoding="utf-8") as f:
                meta = {
                    "session_id": state.session_id,
                    "corpus_path": state.corpus_path,
                    "k": state.k,
                    "mode": state.mode,
                    "ollama_enabled": getattr(state, "ollama_enabled", False),
                    "ollama_host": getattr(state, "ollama_host", "http://127.0.0.1:11434"),
                    "model": getattr(state, "model", "deepseek-r1:8b"),
                    "llm_role": getattr(state, "llm_role", "both"),
                    "timeout": getattr(state, "timeout", 60.0),
                    "lang": getattr(state, "lang", "en"),
                    "sources_mode": getattr(state, "sources_mode", "used"),
                    "debug": getattr(state, "debug", False),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for entry in state.history:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print("Session saved to", path)
        except OSError as e:
            print("Error saving session:", e, file=sys.stderr)

    def do_resume(path: str, doc_count_getter: Callable[[], int]) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return False
            first = json.loads(lines[0])
            if "session_id" in first and "corpus_path" in first:
                apply_state(SessionState(
                    session_id=first.get("session_id", state.session_id),
                    corpus_path=first.get("corpus_path", state.corpus_path),
                    k=first.get("k", state.k),
                    mode=first.get("mode", state.mode),
                    history=[json.loads(ln) for ln in lines[1:]],
                    ollama_enabled=first.get("ollama_enabled", False),
                    ollama_host=first.get("ollama_host", "http://127.0.0.1:11434"),
                    model=first.get("model", "deepseek-r1:8b"),
                    llm_role=first.get("llm_role", "both"),
                    timeout=float(first.get("timeout", 60)),
                    lang=first.get("lang", "en"),
                    sources_mode=first.get("sources_mode", "used"),
                    debug=first.get("debug", False),
                ))
                print("Session restored from", path)
                return True
        except (OSError, json.JSONDecodeError) as e:
            print("Error resuming session:", e, file=sys.stderr)
        return False

    def do_compact(n: int = 20) -> None:
        apply_state(SessionState(
            session_id=state.session_id,
            corpus_path=state.corpus_path,
            k=state.k,
            mode=state.mode,
            history=state.history[-n:],
            **_ollama_kw(state),
        ))
        print("History compacted to last", n, "items.")

    def do_ping() -> None:
        from . import ollama_client
        host = getattr(state, "ollama_host", "http://127.0.0.1:11434")
        timeout = getattr(state, "timeout", 60.0)
        status = ollama_client.ping(host, timeout=min(5.0, timeout))
        if state.mode == "json":
            print(json.dumps(status, ensure_ascii=False))
        else:
            if status.get("ok"):
                print("Ollama: OK", status.get("message", ""))
            else:
                print("Ollama: FAILED", status.get("error", "unknown"), file=sys.stderr)

    def process_claim(claim: str, doc_count_getter: Callable[[], int]) -> None:
        claim = claim.strip()
        if not claim:
            return
        result = run_factcheck(claim, state)
        entry = {"claim": claim, "result": result_to_dict(result)}
        apply_state(SessionState(
            session_id=state.session_id,
            corpus_path=state.corpus_path,
            k=state.k,
            mode=state.mode,
            history=state.history + [entry],
            **_ollama_kw(state),
        ))
        if state.mode == "json":
            print_json(result)
        else:
            print_pretty(result, getattr(state, "sources_mode", "used"))

    def load_file(filepath: str, doc_count_getter: Callable[[], int]) -> None:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    process_claim(line, doc_count_getter)
        except OSError as e:
            print("Error loading file:", e, file=sys.stderr)

    def handle_line(line: str) -> Optional[str]:
        """Returns 'exit' to break loop, None to continue."""
        nonlocal multi_line_buffer, last_interrupt_time
        if multi_line_buffer:
            if line.strip() == '"""':
                claim = "\n".join(multi_line_buffer).strip()
                multi_line_buffer = []
                if claim:
                    process_claim(claim, doc_count_getter)
            else:
                multi_line_buffer.append(line)
            return None

        if not line.strip():
            return None
        if line.strip() == '"""':
            multi_line_buffer = []
            return None

        parts = line.strip().split()
        if not parts[0].startswith("/"):
            process_claim(line.strip(), doc_count_getter)
            return None

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "/quit", "/q"):
            return "exit"
        if cmd == "/help":
            show_help()
            return None
        if cmd == "/clear":
            do_clear()
            return None
        if cmd == "/status":
            do_status(doc_count_getter())
            return None
        if cmd == "/k":
            if args and args[0].isdigit():
                n = int(args[0])
                if n > 0:
                    apply_state(SessionState(
                        state.session_id, state.corpus_path, n, state.mode, state.history, **_ollama_kw(state)
                    ))
                    print("k set to", n)
            else:
                print("Usage: /k <positive integer>", file=sys.stderr)
            return None
        if cmd == "/corpus":
            if args:
                path = args[0]
                if os.path.isdir(path):
                    apply_state(SessionState(
                        state.session_id, path, state.k, state.mode, state.history, **_ollama_kw(state)
                    ))
                    print("Corpus set to", path)
                else:
                    print("Error: not a valid directory:", path, file=sys.stderr)
            else:
                print("Usage: /corpus <path>", file=sys.stderr)
            return None
        if cmd == "/mode":
            if args and args[0].lower() in ("pretty", "json"):
                m = "pretty" if args[0].lower() == "pretty" else "json"
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, m, state.history, **_ollama_kw(state)
                ))
                print("Mode set to", m)
            else:
                print("Usage: /mode pretty|json", file=sys.stderr)
            return None
        if cmd == "/json":
            if args and args[0].lower() in ("on", "off"):
                m = "json" if args[0].lower() == "on" else "pretty"
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, m, state.history, **_ollama_kw(state)
                ))
                print("JSON", args[0].lower(), "- mode", m)
            else:
                print("Usage: /json on|off", file=sys.stderr)
            return None
        if cmd == "/ollama":
            if args and args[0].lower() in ("on", "off"):
                en = args[0].lower() == "on"
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    ollama_enabled=en, ollama_host=state.ollama_host, model=state.model,
                    llm_role=state.llm_role, timeout=state.timeout, lang=getattr(state, "lang", "en")
                ))
                print("Ollama", "on" if en else "off")
            else:
                print("Usage: /ollama on|off", file=sys.stderr)
            return None
        if cmd == "/ollama_host":
            if args:
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    ollama_enabled=state.ollama_enabled, ollama_host=args[0], model=state.model,
                    llm_role=state.llm_role, timeout=state.timeout, lang=getattr(state, "lang", "en")
                ))
                print("Ollama host set to", args[0])
            else:
                print("Usage: /ollama_host <url>", file=sys.stderr)
            return None
        if cmd == "/model":
            if args:
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    ollama_enabled=state.ollama_enabled, ollama_host=state.ollama_host, model=args[0],
                    llm_role=state.llm_role, timeout=state.timeout, lang=getattr(state, "lang", "en")
                ))
                print("Model set to", args[0])
            else:
                print("Usage: /model <name>", file=sys.stderr)
            return None
        if cmd == "/llm_role":
            if args and args[0].lower() in ("verifier", "explainer", "both", "off"):
                role = args[0].lower()
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    ollama_enabled=state.ollama_enabled, ollama_host=state.ollama_host, model=state.model,
                    llm_role=role, timeout=state.timeout, lang=getattr(state, "lang", "en")
                ))
                print("llm_role set to", role)
            else:
                print("Usage: /llm_role verifier|explainer|both|off", file=sys.stderr)
            return None
        if cmd == "/ping":
            do_ping()
            return None
        if cmd == "/lang":
            if args and args[0].lower() in ("en", "ja"):
                lang_val = args[0].lower()
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    ollama_enabled=state.ollama_enabled, ollama_host=state.ollama_host, model=state.model,
                    llm_role=state.llm_role, timeout=state.timeout, lang=lang_val,
                    sources_mode=getattr(state, "sources_mode", "used"),
                ))
                print("Language set to", lang_val)
            else:
                print("Usage: /lang en|ja", file=sys.stderr)
            return None
        if cmd == "/sources":
            if args and args[0].lower() in ("used", "all"):
                sm = args[0].lower()
                apply_state(SessionState(
                    state.session_id, state.corpus_path, state.k, state.mode, state.history,
                    **_ollama_kw(state),
                    sources_mode=sm,
                ))
                print("sources_mode set to", sm)
            else:
                print("Usage: /sources used|all", file=sys.stderr)
            return None
        if cmd == "/load":
            if args:
                load_file(args[0], doc_count_getter)
            else:
                print("Usage: /load <file>", file=sys.stderr)
            return None
        if cmd == "/save":
            do_save(args[0] if args else None)
            return None
        if cmd == "/resume":
            if args:
                do_resume(args[0], doc_count_getter)
            else:
                print("Usage: /resume <file>", file=sys.stderr)
            return None
        if cmd == "/compact":
            do_compact(20)
            return None
        print("Unknown command. Type /help for list.", file=sys.stderr)
        return None

    # Main loop with Ctrl+C handling
    import time
    while True:
        try:
            prompt = "... " if multi_line_buffer else "factcheck> "
            try:
                line = input(prompt)
            except EOFError:
                break
            if handle_line(line) == "exit":
                break
        except KeyboardInterrupt:
            now = time.time()
            if last_interrupt_time is not None and (now - last_interrupt_time) < INTERRUPT_DOUBLE_THRESHOLD:
                print("\nExiting.", file=sys.stderr)
                break
            last_interrupt_time = now
            multi_line_buffer = []
            print("\nInterrupted. Type /exit to quit.", file=sys.stderr)


def load_session(path: str) -> Optional[SessionState]:
    """Load session from JSONL file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        first = json.loads(lines[0])
        return SessionState(
            session_id=first.get("session_id", ""),
            corpus_path=first.get("corpus_path", ""),
            k=int(first.get("k", 10)),
            mode=first.get("mode", "pretty"),
            history=[json.loads(ln) for ln in lines[1:]],
            ollama_enabled=first.get("ollama_enabled", False),
            ollama_host=first.get("ollama_host", "http://127.0.0.1:11434"),
            model=first.get("model", "deepseek-r1:8b"),
            llm_role=first.get("llm_role", "both"),
            timeout=float(first.get("timeout", 60)),
            lang=first.get("lang", "en"),
            sources_mode=first.get("sources_mode", "used"),
            debug=first.get("debug", False),
        )
    except (OSError, json.JSONDecodeError):
        return None


def save_session(state: SessionState, path: str) -> bool:
    """Save session to JSONL. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            meta = {
                "session_id": state.session_id,
                "corpus_path": state.corpus_path,
                "k": state.k,
                "mode": state.mode,
                "ollama_enabled": getattr(state, "ollama_enabled", False),
                "ollama_host": getattr(state, "ollama_host", "http://127.0.0.1:11434"),
                "model": getattr(state, "model", "deepseek-r1:8b"),
                "llm_role": getattr(state, "llm_role", "both"),
                "timeout": getattr(state, "timeout", 60.0),
                "lang": getattr(state, "lang", "en"),
                "sources_mode": getattr(state, "sources_mode", "used"),
                "debug": getattr(state, "debug", False),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for entry in state.history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False
