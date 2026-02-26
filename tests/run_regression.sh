#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
FACTCHECK="${FACTCHECK:-$ROOT/factcheck.py}"
CORPUS="${CORPUS:-$ROOT/data/corpus}"
OUTDIR="${OUTDIR:-$ROOT/tests/out}"
MODEL="${MODEL:-deepseek-r1:8b}"
TIMEOUT="${TIMEOUT:-60}"

mkdir -p "$OUTDIR"

echo "== Regression run =="
echo "PY=$PY"
echo "FACTCHECK=$FACTCHECK"
echo "CORPUS=$CORPUS"
echo "MODEL=$MODEL"
echo "OUTDIR=$OUTDIR"
echo

# Pretty (no ollama) baseline
echo "== 1) Rule-based (pretty) =="
"$PY" "$FACTCHECK" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "The United States borders Canada to the south" > "$OUTDIR/rule_1.txt"
"$PY" "$FACTCHECK" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "The United States borders Canada to the north" > "$OUTDIR/rule_2.txt"
"$PY" "$FACTCHECK" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "Washington D.C. is the capital of USA." > "$OUTDIR/rule_3.txt"
"$PY" "$FACTCHECK" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "New York is the capital of United States." > "$OUTDIR/rule_4.txt"
"$PY" "$FACTCHECK" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "The population of Mars is 2 billion." > "$OUTDIR/rule_5.txt"

# JSON mode check (stdout must be valid JSON)
echo "== 2) JSON mode (rule-based) =="
"$PY" "$FACTCHECK" --mode json --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "The United States borders Canada to the south" > "$OUTDIR/rule_1.json"
"$PY" - <<PY
import json, sys
json.load(open("$OUTDIR/rule_1.json", "r", encoding="utf-8"))
print("JSON OK:", "$OUTDIR/rule_1.json")
PY

# Ollama path (if available)
echo "== 3) Ollama (pretty, debug to stderr) =="
set +e
"$PY" "$FACTCHECK" --ollama --model "$MODEL" --timeout "$TIMEOUT" --mode pretty --corpus "$CORPUS" --k 10 --max_sentences_per_doc 2 --claim "The United States borders Canada to the south" > "$OUTDIR/llm_1.txt" 2> "$OUTDIR/llm_1.err"
LLM_RC=$?
set -e
echo "Ollama exit code: $LLM_RC (errors logged to $OUTDIR/llm_1.err)"
echo

echo "== Done =="
echo "Outputs in: $OUTDIR"