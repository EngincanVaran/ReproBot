"""Extract structured claims about a paper's own proposed method from its VLM Markdown.

Input is the *already-extracted* Markdown produced by `ocr.vlm_extract`
(`ocr/output/vlm/<paper>.md`), not a PDF — the Reader agent's claims
extraction reads that clean, VLM-transcribed text, not raw pages. This
sends the full Markdown to Claude and uses tool-use (a JSON-schema tool
definition, `tool_choice` forced to it) to get back a validated list of
claims describing the paper's own method — baseline/prior-work rows from
the same results tables are deliberately excluded.

This module is an importable extraction step, not a standalone script —
`reader/pipeline.py` is the entry point that loads a paper's Markdown, runs
`ClaimsExtractor.extract()`, and combines the result with the other
extraction steps into one `reader_output.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from anthropic import Anthropic
from loguru import logger

from reader.base import Extractor
from reader.tooluse import as_int, as_list, request_tool_use

MODEL = "claude-sonnet-5"

PROMPT = """You are reading the Claude-VLM Markdown transcription of one ML paper. \
Your job is to extract this paper's OWN reported experimental results as \
structured claims - NOT the baseline/prior-work numbers it compares itself \
against.

Steps:
1. Scan the whole document for results tables (Markdown tables reporting a \
metric such as error rate or accuracy, usually captioned "Table N: ...").
2. For each results table you look at, note its caption and the transcribed \
page number it appears under (the nearest preceding "## Page N" heading); \
record one entry per table in `tables_examined`, e.g. \
"Table 1: Test set error rates for CIFAR-10 (transcribed page 5)".
3. Within each results table, classify every data row as either:
   - the paper's OWN proposed method: the row's method name is the paper's \
own model/architecture (matches how the paper refers to itself in running \
text, e.g. "our method", "we obtain a test error of X%"), and it is \
typically NOT tagged with a bracketed citation like "[11]" pointing to a \
different paper, and is often bolded in the Markdown table.
   - a BASELINE / prior-work row: it cites another paper (e.g. "[14]", \
"[8]") and describes a method this paper compares itself against, not \
proposes.
   Count every row you examine, of both kinds and across every table, \
toward `candidates_considered`.
4. Emit one claim per DISTINCT own-method row, not just the single best \
number - if the paper reports multiple headline configurations for the \
same metric+dataset pair (e.g. "NIN + Dropout" vs. \
"NIN + Dropout + Data Augmentation"), each is a separate claim; put the \
configuration name in `model_variant`.
5. Do NOT emit a claim for any baseline/prior-work row, even from a table \
that also contains own-method rows.
6. `source` must cite exactly which table and transcribed page the claim \
came from, e.g. "Table 1, transcribed page 5".
7. `reported_value` must be the bare number (e.g. 10.41 for "10.41%"), with \
the unit given separately in `unit` (e.g. "%").
8. Do not normalize, translate, or rename the metric - use the paper's own \
wording (e.g. "test error", "top-1 accuracy").

Call the `record_claims` tool with the results."""

CLAIM_TOOL: dict[str, Any] = {
    "name": "record_claims",
    "description": (
        "Record the paper's own-method claims extracted from its results "
        "tables, plus bookkeeping about what was examined."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tables_examined": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per results table looked at, e.g. 'Table 1: Test set "
                    "error rates for CIFAR-10 (transcribed page 5)'."
                ),
            },
            "candidates_considered": {
                "type": "integer",
                "description": (
                    "Total number of data rows examined across all results tables, "
                    "including both own-method rows and baseline/prior-work rows, "
                    "before filtering out baselines."
                ),
            },
            "claims": {
                "type": "array",
                "description": (
                    "Only rows describing the paper's own proposed method, after "
                    "excluding baseline/prior-work rows."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "description": "Short unique id within this paper, e.g. 'c1', 'c2'.",
                        },
                        "metric": {
                            "type": "string",
                            "description": "The paper's own wording, e.g. 'test error'.",
                        },
                        "dataset": {"type": "string", "description": "e.g. 'CIFAR-10'."},
                        "reported_value": {
                            "type": "number",
                            "description": "Bare numeric value, e.g. 10.41 for '10.41%'.",
                        },
                        "unit": {"type": "string", "description": "e.g. '%'."},
                        "source": {
                            "type": "string",
                            "description": "e.g. 'Table 1, transcribed page 5'.",
                        },
                        "model_variant": {
                            "type": "string",
                            "description": (
                                "Optional configuration label distinguishing this claim "
                                "from other own-method claims for the same metric+dataset, "
                                "e.g. 'NIN + Dropout' vs. 'NIN + Dropout + Data "
                                "Augmentation'. Omit if the paper reports only one "
                                "configuration for this metric+dataset."
                            ),
                        },
                    },
                    "required": [
                        "claim_id",
                        "metric",
                        "dataset",
                        "reported_value",
                        "unit",
                        "source",
                    ],
                },
            },
        },
        "required": ["tables_examined", "candidates_considered", "claims"],
    },
}


@dataclass
class Claim:
    claim_id: str
    metric: str
    dataset: str
    reported_value: float
    unit: str
    source: str
    model_variant: str | None = None


@dataclass
class ClaimsExtraction:
    claims: list[Claim]
    tables_examined: list[str]
    candidates_considered: int


def _parse_claim(raw: object) -> Claim:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a claim object, got {raw!r}")
    model_variant = raw.get("model_variant")
    return Claim(
        claim_id=str(raw["claim_id"]),
        metric=str(raw["metric"]),
        dataset=str(raw["dataset"]),
        reported_value=float(raw["reported_value"]),
        unit=str(raw["unit"]),
        source=str(raw["source"]),
        model_variant=str(model_variant) if model_variant is not None else None,
    )


class ClaimsExtractor(Extractor[ClaimsExtraction]):
    """Extracts a paper's own reported results (claims) from its VLM Markdown."""

    name: ClassVar[str] = "claims"

    def extract(
        self, markdown_text: str, client: Anthropic, feedback: str | None = None
    ) -> ClaimsExtraction:
        """Send one paper's already-loaded VLM Markdown to Claude and extract its
        own-method claims. No file I/O here - the pipeline owns reading input and
        writing output."""
        prompt = PROMPT
        if feedback:
            prompt = (
                f"{PROMPT}\n\n"
                f"A prior validation pass flagged this specific issue with your "
                f"previous attempt: {feedback}\nAddress it specifically in this "
                f"attempt."
            )

        payload = request_tool_use(
            client,
            log_prefix="claims",
            model=MODEL,
            max_tokens=8192,
            tool=CLAIM_TOOL,
            user_content=f"{prompt}\n\n---\n\n{markdown_text}",
            required_keys=("tables_examined", "candidates_considered", "claims"),
        )

        tables_examined = [str(table) for table in as_list(payload.get("tables_examined"))]
        candidates_considered = as_int(
            payload.get("candidates_considered"), "claims", "candidates_considered"
        )
        claims = [_parse_claim(raw) for raw in as_list(payload.get("claims"))]

        logger.info(f"  [claims] tables examined ({len(tables_examined)}):")
        for table in tables_examined:
            logger.info(f"    - {table}")
        logger.info(
            f"  [claims] candidates considered (own-method + baseline rows): "
            f"{candidates_considered}"
        )
        logger.info(f"  [claims] kept as own-method claims: {len(claims)}")
        for claim in claims:
            variant = f" [{claim.model_variant}]" if claim.model_variant else ""
            logger.info(
                f"    {claim.claim_id}: {claim.metric} = {claim.reported_value}{claim.unit} "
                f"on {claim.dataset}{variant} ({claim.source})"
            )

        return ClaimsExtraction(
            claims=claims,
            tables_examined=tables_examined,
            candidates_considered=candidates_considered,
        )
