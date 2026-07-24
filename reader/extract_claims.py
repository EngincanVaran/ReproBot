"""Extract structured claims about a paper's own proposed method from its VLM Markdown.

Input is the *already-extracted* Markdown produced by `ocr.vlm_extract`
(`ocr/output/vlm/<paper>.md`), not a PDF — the Reader agent's claims
extraction reads that clean, VLM-transcribed text, not raw pages. This
sends the full Markdown to Claude and uses tool-use (a JSON-schema tool
definition, `tool_choice` forced to it) to get back a validated list of
claims describing the paper's own method — baseline/prior-work rows from
the same results tables are deliberately excluded.

Requires the `reader` extra: `uv sync --extra reader`, plus an
ANTHROPIC_API_KEY in `.env` at the repo root (copy `.env.example`).

Usage:
    uv run python -m reader.extract_claims --input "ocr/output/vlm/2013-12 - Network In Network.md"
    uv run python -m reader.extract_claims --input ocr/output/vlm --output reader/output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from dotenv import load_dotenv
from loguru import logger

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
class ClaimsResult:
    markdown_path: Path
    json_path: Path
    claims: list[Claim]
    tables_examined: list[str]
    candidates_considered: int


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the record_claims tool")


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


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


def run_claims_extraction(markdown_path: Path, output_dir: Path, client: Anthropic) -> ClaimsResult:
    """Send one paper's VLM Markdown to Claude and extract its own-method claims."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{markdown_path.stem}.json"

    markdown_text = markdown_path.read_text(encoding="utf-8")
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        tools=[cast(ToolParam, CLAIM_TOOL)],
        tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": "record_claims"}),
        messages=[{"role": "user", "content": f"{PROMPT}\n\n---\n\n{markdown_text}"}],
    )
    payload = _tool_input(message)

    tables_examined = [str(table) for table in _as_list(payload.get("tables_examined"))]
    candidates_considered = int(payload.get("candidates_considered") or 0)  # type: ignore[call-overload]
    claims = [_parse_claim(raw) for raw in _as_list(payload.get("claims"))]

    logger.info(f"  tables examined ({len(tables_examined)}):")
    for table in tables_examined:
        logger.info(f"    - {table}")
    logger.info(f"  candidates considered (own-method + baseline rows): {candidates_considered}")
    logger.info(f"  kept as own-method claims: {len(claims)}")
    for claim in claims:
        variant = f" [{claim.model_variant}]" if claim.model_variant else ""
        logger.info(
            f"    {claim.claim_id}: {claim.metric} = {claim.reported_value}{claim.unit} "
            f"on {claim.dataset}{variant} ({claim.source})"
        )

    json_path.write_text(
        json.dumps(
            {
                "paper": markdown_path.stem,
                "source_markdown": str(markdown_path),
                "tables_examined": tables_examined,
                "candidates_considered": candidates_considered,
                "claims": [asdict(claim) for claim in claims],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ClaimsResult(
        markdown_path=markdown_path,
        json_path=json_path,
        claims=claims,
        tables_examined=tables_examined,
        candidates_considered=candidates_considered,
    )


def extract_dataset(input_dir: Path, output_dir: Path) -> None:
    """Run claims extraction over every Markdown file in input_dir, skipping papers already done."""
    markdown_paths = sorted(input_dir.glob("*.md"))
    if not markdown_paths:
        logger.warning(f"No Markdown files found in {input_dir}")
        return

    client = Anthropic()
    for markdown_path in markdown_paths:
        json_path = output_dir / f"{markdown_path.stem}.json"
        if json_path.exists():
            logger.info(f"[skip]    {markdown_path.name} (already extracted)")
            continue

        logger.info(f"[extract] {markdown_path.name}")
        try:
            result = run_claims_extraction(markdown_path, output_dir, client)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {markdown_path.name}: {exc}")
            continue
        logger.info(f"          -> {result.json_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ocr/output/vlm"),
        help="VLM-extracted Markdown file, or a directory of them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reader/output"),
        help="Directory for extracted claims JSON",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        extract_dataset(args.input, args.output)
    else:
        client = Anthropic()
        logger.info(f"[extract] {args.input.name}")
        result = run_claims_extraction(args.input, args.output, client)
        logger.info(f"          -> {result.json_path}")


if __name__ == "__main__":
    main()
