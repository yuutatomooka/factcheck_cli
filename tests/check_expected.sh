#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-tests/out}"

expect() {
  local file="$1"
  local verdict="$2"
  if ! grep -q "VERDICT: $verdict" "$file"; then
    echo "FAIL: $file (expected VERDICT: $verdict)" >&2
    exit 1
  fi
  echo "OK: $file -> $verdict"
}

expect "$OUTDIR/rule_1.txt" "REFUTED"
expect "$OUTDIR/rule_2.txt" "SUPPORTED"
expect "$OUTDIR/rule_3.txt" "SUPPORTED"
expect "$OUTDIR/rule_4.txt" "REFUTED"
expect "$OUTDIR/rule_5.txt" "NOT_ENOUGH_INFO"

echo "All expected verdicts matched."