#!/usr/bin/env bash
# Loop-OFF arm of the single-shot vs. iterative-loop ablation.
#
# One paper per invocation, run SEQUENTIALLY (never two at once: the Runner sets
# no --cpus, so two containers each claim every core and roughly double wall clock).
#
# Usage: ./run-ablation.sh "<paper stem>" <claim-id>
#
# Writes to SEPARATE directories so the loop-ON evidence in orchestrator/output,
# coder/output and critic/output is never touched.
set -uo pipefail

PAPER="$1"
CLAIM="$2"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

LOGDIR="docs/notes/ablation/data/logs"
mkdir -p "$LOGDIR"

echo "=== LOOP-OFF: $PAPER (claim $CLAIM) ==="
START=$(date +%s)

uv run --extra orchestrator python -m orchestrator.pipeline \
    --input "reader/output/${PAPER}.json" \
    --max-stage full \
    --force \
    --no-build \
    --claim-id "$CLAIM" \
    --retry-budget 0 \
    --no-critic \
    --output orchestrator/output-ablation \
    --coder-output coder/output-ablation \
    > "$LOGDIR/${PAPER}.orchestrator.log" 2>&1
RC=$?

END=$(date +%s)
echo "orchestrator exit=$RC wall=$((END-START))s"

# Judge the loop-off arm with the same arithmetic Critic used for loop-ON.
uv run --extra orchestrator python -m critic.pipeline \
    --state "orchestrator/output-ablation/${PAPER}" \
    --coder-output coder/output-ablation \
    --output critic/output-ablation \
    > "$LOGDIR/${PAPER}.critic.log" 2>&1
echo "critic exit=$?"

echo "${PAPER}|${CLAIM}|${RC}|$((END-START))" >> docs/notes/ablation/data/timings.psv
