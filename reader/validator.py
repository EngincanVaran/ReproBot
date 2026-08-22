"""Cross-check the combined extraction output against the paper text.

Pulled out of `reader/pipeline.py` so the pipeline's retry loop can call
`ExtractionValidator.validate()` as many times as needed without pipeline.py
owning the validation prompt/tool/parsing itself. `results` is a dict keyed
by extractor name (e.g. `{"claims": <ClaimsExtraction>, "hyperparameters":
<HyperparametersExtraction>}`) rather than named parameters, so this stays
generic as more extraction stages are added later - this module's *logic*
never needs to change to support them, only the prompt's per-stage example
parentheticals, so the cross-checks it can already perform are actually asked
for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from anthropic import Anthropic
from loguru import logger

from reader.tooluse import as_list, request_tool_use

MODEL = "claude-sonnet-5"

VALIDATION_PROMPT = """You are cross-checking structured extractions made from an ML \
paper's Markdown transcription against the paper's actual text, to catch \
mistakes before they propagate downstream.

Below you are given:
1. The full paper Markdown.
2. A combined JSON object with the extractions already produced from it, \
one entry per extraction stage (e.g. `method_summary`: a prose summary of the \
paper's own problem, core idea and novelty, deliberately carrying no result \
numbers, hyperparameters, dataset detail or layer dimensions; \
`architecture_notes`: the paper's own model architecture - components, key \
equations, depth/width scaling - together with `unstated_details`, an explicit \
list of architectural details the paper does NOT state, which is a required \
finding rather than a gap in the extraction; `claims`: the paper's own \
reported results; `hyperparameters`: the paper's own training configuration; \
`data_pipeline`: the paper's own per-dataset preprocessing/augmentation/ \
split and any reference URLs).

Your job is NOT to fix anything - only to FLAG problems. Look for:
- Inconsistencies between extractions (e.g. a `model_variant` named in one \
extraction that has no matching entry in another, or vice versa).
- Anything in the paper text that looks like it belongs in one of these \
extractions but is missing from it (e.g. a results table or an \
"Implementation Details" detail that was not captured).
- Anything extracted that looks wrong when checked against the paper text \
(e.g. a value, source citation, or dataset name that doesn't match what the \
paper actually says at the cited location).
- Anything extracted that looks like it may actually be a baseline/prior-work \
value rather than the paper's own, when checked against the paper text.

Only flag genuine, specific issues you can point to - do not invent \
speculative concerns. If you find nothing wrong, return an empty list.

For each issue, call out which field/extraction it relates to using the \
extraction's key name from the JSON object followed by a colon and the \
specific detail (e.g. "claims: c3", "hyperparameters: learning rate", \
"data_pipeline: CIFAR-10", "architecture_notes: mlpconv layer", \
"method_summary: novelty"), or, if the issue is a genuine inconsistency \
BETWEEN two or more extractions rather than something owned by exactly one \
of them, use "cross-check: " followed by which extractions are involved \
(e.g. "cross-check: claims vs. hyperparameters"). Give a short, concrete \
description of the problem.

Call the `record_validation` tool with the results."""

VALIDATION_TOOL: dict[str, Any] = {
    "name": "record_validation",
    "description": (
        "Record any flagged inconsistencies or gaps found while cross-checking the "
        "combined extraction output against the paper text. Flags only - no fixes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "description": "One entry per flagged issue. Empty if nothing was found.",
                "items": {
                    "type": "object",
                    "properties": {
                        "relates_to": {
                            "type": "string",
                            "description": (
                                "Which field/extraction this concerns, e.g. 'claims: c3', "
                                "'hyperparameters: learning rate', 'data_pipeline: "
                                "CIFAR-10', 'architecture_notes: mlpconv layer', "
                                "'method_summary: novelty', or 'cross-check: claims vs. "
                                "hyperparameters'."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "Short, concrete description of the issue.",
                        },
                    },
                    "required": ["relates_to", "description"],
                },
            },
        },
        "required": ["flags"],
    },
}


@dataclass
class ValidationFlag:
    relates_to: str
    description: str


@dataclass
class ValidationResult:
    flags: list[ValidationFlag]


def _parse_flag(raw: object) -> ValidationFlag:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a validation flag object, got {raw!r}")
    return ValidationFlag(
        relates_to=str(raw["relates_to"]),
        description=str(raw["description"]),
    )


class ExtractionValidator:
    """Cross-checks a combined extraction output against the paper text and
    flags (does not fix) inconsistencies or gaps."""

    def validate(
        self, markdown_text: str, results: dict[str, Any], client: Anthropic
    ) -> ValidationResult:
        """`results` is keyed by extractor name (`stage.name`), values are that
        stage's extraction-result dataclass instances. No file I/O here - the
        pipeline owns reading input and writing output."""
        combined = {
            key: asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
            for key, value in results.items()
        }
        user_content = (
            f"{VALIDATION_PROMPT}\n\n"
            f"--- PAPER MARKDOWN ---\n\n{markdown_text}\n\n"
            f"--- COMBINED EXTRACTION ---\n\n{json.dumps(combined, indent=2)}"
        )
        payload = request_tool_use(
            client,
            log_prefix="validate",
            model=MODEL,
            max_tokens=8192,
            tool=VALIDATION_TOOL,
            user_content=user_content,
            # Nothing goes in `required_keys`: an empty `flags` list is this
            # stage's *success* case ("no inconsistencies or gaps found"), so
            # treating emptiness as a malformed response would make every clean
            # validation re-request itself and double its cost. `flags` still has
            # to arrive, though - a payload with no `flags` key at all silently
            # reads as "clean" and is exactly the failure this helper exists for.
            may_be_empty_keys=("flags",),
        )
        flags = [_parse_flag(raw) for raw in as_list(payload.get("flags"))]

        logger.info(f"  [validate] flags raised: {len(flags)}")
        for flag in flags:
            logger.info(f"    [{flag.relates_to}] {flag.description}")
        if not flags:
            logger.info("  [validate] no inconsistencies or gaps found")

        return ValidationResult(flags=flags)
