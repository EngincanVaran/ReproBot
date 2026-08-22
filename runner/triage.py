"""Classify a failed sandbox run: bug in the generated script, or broken environment?

One Claude call, forced `tool_choice`, dataclass parse - the same shape as
`reader/validator.py`, for the same reason: a free-text answer would have to be
string-matched downstream, and the whole point of this call is to produce a
routable label.

Two deliberate scope limits:

1. **Called only on a genuine, non-timeout failure.** Success and timeout are
   decided mechanically in `docker_runner.py` with no API call at all - a
   zero-exit run needs no explanation, and a timeout's cause is "it did not
   finish in N seconds", which no model can improve on. That keeps the common
   path (everything passed) free.
2. **A plain function, not a class.** `reader/` and `coder/` use an ABC because
   they have a growing family of interchangeable stages; triage is one fixed
   step with one call site, so an ABC here would be abstraction nobody uses.

The `recoverable_error` / `environment_error` split is the routing decision the
future Orchestrator needs: `recoverable_error` is what feeds back into
`CodeWriter.write(feedback=...)` for another Coder attempt, while
`environment_error` means regenerating the script would change nothing and the
image or the mounts are what has to be fixed.

Usage:
    from runner.triage import triage_failure
    result = triage_failure(
        exit_code=1, failed_stage="probe", log_excerpt=excerpt, client=Anthropic()
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger

MODEL = "claude-haiku-4-5"

VALID_CATEGORIES = ("recoverable_error", "environment_error")

TRIAGE_PROMPT = """You are triaging a failed run of an automatically generated \
machine-learning training script. The script was produced from a research paper \
by an LLM and executed inside a Docker sandbox (Python 3.11 on Linux, CPU-only \
torch/torchvision/transformers already installed in the image). It exited \
non-zero.

Your ONLY job is to decide which of two categories the failure belongs to:

- `recoverable_error` - the fault is in the GENERATED SCRIPT itself, and \
rewriting the script could fix it. Examples: a Python traceback from the \
script's own code, a tensor shape or dtype mismatch, an undefined name or \
attribute, a wrong argument passed to a library function, a deprecated or \
non-existent API being called, bad indexing, a division by zero, a dataloader \
that yields zero batches.
- `environment_error` - the fault is in the CONTAINER or its inputs, and \
rewriting the script would change nothing. Examples: a missing Python package \
that the image never installed, a network/download failure fetching the \
dataset, a permissions error on a mounted directory, the process being killed \
by the OS out-of-memory killer (often exit code 137), a CUDA/GPU device being \
requested when none exists, disk full, a corrupted cache.

Judge from the evidence given, not from what you would expect. Note that a \
missing import can be either: if the script imports a package the image does \
not carry, that is an `environment_error`; if it imports a name that does not \
exist inside an installed package, that is a `recoverable_error`.

The log excerpt is TRUNCATED from the middle - a marker shows where. Do not \
treat the truncation as evidence of anything.

Then, for a `recoverable_error` only, write a specific `suggested_fix`: what \
the regenerated script should do differently. It will be handed verbatim to the \
model that rewrites the script, so make it actionable and concrete, naming the \
function or line involved. For an `environment_error`, leave `suggested_fix` \
empty - there is nothing the script author can do about it.

Call the `record_triage` tool with your verdict."""

TRIAGE_TOOL: dict[str, Any] = {
    "name": "record_triage",
    "description": (
        "Record the classification of a failed sandbox run - whether the fault lies in "
        "the generated script (recoverable) or in the container environment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(VALID_CATEGORIES),
                "description": (
                    "'recoverable_error' if a rewrite of the generated script could fix "
                    "this; 'environment_error' if the container or its inputs are at fault "
                    "and a rewrite would change nothing."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Short, concrete justification pointing at the specific evidence in "
                    "the log excerpt or exit code that decided the category."
                ),
            },
            "suggested_fix": {
                "type": "string",
                "description": (
                    "For 'recoverable_error': a specific, actionable instruction for the "
                    "model that will rewrite the script, naming the function or line "
                    "involved. Empty string for 'environment_error'."
                ),
            },
        },
        "required": ["category", "reasoning", "suggested_fix"],
    },
}


@dataclass
class TriageResult:
    category: str
    reasoning: str
    suggested_fix: str | None
    model: str = MODEL


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the record_triage tool")


def parse_triage(payload: dict[str, object]) -> TriageResult:
    """Parse the tool payload into a `TriageResult`, rejecting an unknown category.

    Split out from the API call so the parsing rules are exercisable against a
    synthetic payload without spending a call. The category is re-checked here
    even though the tool schema declares an `enum`: the schema is a strong hint
    to the model, not a guarantee the SDK enforces, and a silently mislabelled
    category would misroute the whole retry decision downstream.
    """
    category = str(payload.get("category", ""))
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"triage returned an unknown category {category!r}; "
            f"expected one of {', '.join(VALID_CATEGORIES)}"
        )
    suggested_fix = str(payload.get("suggested_fix", "")).strip()
    return TriageResult(
        category=category,
        reasoning=str(payload.get("reasoning", "")),
        suggested_fix=suggested_fix or None,
    )


def triage_failure(
    *,
    exit_code: int | None,
    failed_stage: str,
    log_excerpt: str,
    client: Anthropic,
) -> TriageResult:
    """Classify one genuine (non-timeout) sandbox failure.

    Inputs are deliberately only the three things that carry signal: the exit
    code, which escalation stage failed, and the truncated log excerpt. The
    generated script's source is NOT sent - the traceback already names the
    failing line, and sending several hundred lines of code to a triage call
    would cost far more than the classification is worth.
    """
    logger.info(
        f"  [triage] classifying failure of stage '{failed_stage}' "
        f"(exit {exit_code}) with {MODEL}..."
    )
    user_content = (
        f"{TRIAGE_PROMPT}\n\n"
        f"--- FAILED STAGE ---\n\n{failed_stage}\n\n"
        f"--- EXIT CODE ---\n\n{exit_code}\n\n"
        f"--- LOG EXCERPT (truncated) ---\n\n{log_excerpt}"
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[cast(ToolParam, TRIAGE_TOOL)],
        tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": "record_triage"}),
        messages=[{"role": "user", "content": user_content}],
    )
    logger.info(
        f"  [triage] usage: {message.usage.input_tokens} in / "
        f"{message.usage.output_tokens} out, stop_reason={message.stop_reason}"
    )
    result = parse_triage(_tool_input(message))

    logger.info(f"  [triage] category: {result.category}")
    logger.info(f"  [triage] reasoning: {result.reasoning}")
    if result.suggested_fix:
        logger.info(f"  [triage] suggested fix: {result.suggested_fix}")
    else:
        logger.info("  [triage] no suggested fix (nothing a script rewrite could change)")

    return result
