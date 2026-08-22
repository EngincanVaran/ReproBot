"""Reader pipeline: run every extraction stage over one paper's VLM Markdown,
then validate and retry.

This is the entry point for `reader/` - `method_summary.py`,
`architecture_notes.py`, `claims.py`, `hyperparameters.py`, and
`data_pipeline.py` provide `Extractor` subclasses (prompt + tool schema +
parsing + `extract()`), not standalone scripts. `ReaderPipeline` loads a
paper's Markdown once, runs each stage against it, runs `ExtractionValidator`
to cross-check the combined output against the source paper, and - if
validation flags anything that maps to a specific stage - re-runs that stage
with the flag folded into its prompt as feedback, up to `max_retries` total
validation passes. Flags that don't map to exactly one stage (e.g.
"cross-check: claims vs. hyperparameters") are left in the final output as
report-only. One `<paper>.json` is written per paper. Further extraction
stages plug in by adding one more `Extractor` to the `stages` list passed to
`ReaderPipeline` - no other code here needs to change (which is exactly how
`method_summary` and `architecture_notes` were added: two new classes, two new
list entries, zero changes to the loop).

Requires the `reader` extra: `uv sync --extra reader`, plus an
ANTHROPIC_API_KEY in `.env` at the repo root (copy `.env.example`).

Usage:
    uv run python -m reader.pipeline --input "ocr/output/vlm/2013-12 - Network In Network.md"
    uv run python -m reader.pipeline --input ocr/output/vlm --output reader/output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from reader.architecture_notes import ArchitectureNotesExtractor
from reader.base import Extractor
from reader.claims import ClaimsExtractor
from reader.data_pipeline import DataPipelineExtractor
from reader.hyperparameters import HyperparametersExtractor
from reader.method_summary import MethodSummaryExtractor
from reader.validator import ExtractionValidator, ValidationFlag, ValidationResult


@dataclass
class ReaderOutput:
    paper: str
    source_markdown: str
    results: dict[str, Any]
    validation: ValidationResult
    validation_passes: int
    retried_stages: list[str]


def _route_flags(
    flags: list[ValidationFlag], known_stage_names: set[str]
) -> tuple[dict[str, list[ValidationFlag]], list[ValidationFlag]]:
    """Split flags into those that map to exactly one known stage (by the
    `relates_to` prefix before the first colon, e.g. "claims: c3" -> "claims")
    and those that don't (e.g. "cross-check: claims vs. hyperparameters")."""
    routed: dict[str, list[ValidationFlag]] = {}
    unroutable: list[ValidationFlag] = []
    for flag in flags:
        prefix = flag.relates_to.split(":", 1)[0].strip()
        if prefix in known_stage_names:
            routed.setdefault(prefix, []).append(flag)
        else:
            unroutable.append(flag)
    return routed, unroutable


class ReaderPipeline:
    """Runs every extraction stage over one paper's VLM Markdown, validates
    the combined result, and retries the specific stage(s) a validation flag
    points at - up to `max_retries` total validation passes."""

    def __init__(
        self,
        stages: list[Extractor[Any]],
        validator: ExtractionValidator,
        max_retries: int = 3,
    ) -> None:
        self.stages = stages
        self.validator = validator
        self.max_retries = max_retries

    def run(self, markdown_path: Path, client: Anthropic) -> ReaderOutput:
        """Run every stage once, validate, and retry flagged stages until
        validation is clean or `max_retries` validation passes have run."""
        markdown_text = markdown_path.read_text(encoding="utf-8")

        known_stage_names = {stage.name for stage in self.stages}
        stages_by_name = {stage.name: stage for stage in self.stages}

        results: dict[str, Any] = {}
        for stage in self.stages:
            logger.info(f"[{stage.name}] extracting...")
            results[stage.name] = stage.extract(markdown_text, client)

        validation_pass = 1
        logger.info(
            f"[validate] cross-checking combined extraction against paper text "
            f"(pass {validation_pass})..."
        )
        validation = self.validator.validate(markdown_text, results, client)

        retried_stages: list[str] = []

        while validation.flags and validation_pass < self.max_retries:
            routed, unroutable = _route_flags(validation.flags, known_stage_names)

            for flag in unroutable:
                logger.info(
                    f"  [validate] leaving flag as report-only - does not map to a "
                    f"single known stage: [{flag.relates_to}] {flag.description}"
                )

            if not routed:
                logger.info(
                    "  [validate] no remaining flags route to a single known stage; "
                    "stopping the retry loop here"
                )
                break

            validation_pass += 1
            logger.info(f"[retry] validation pass {validation_pass} - re-running flagged stages")
            for stage_name, flags in routed.items():
                feedback = "; ".join(flag.description for flag in flags)
                logger.info(
                    f"  [retry] re-running '{stage_name}' - {len(flags)} flag(s) "
                    f"triggered this retry: {feedback}"
                )
                stage = stages_by_name[stage_name]
                results[stage_name] = stage.extract(markdown_text, client, feedback=feedback)
                if stage_name not in retried_stages:
                    retried_stages.append(stage_name)

            logger.info(
                f"[validate] cross-checking combined extraction against paper text "
                f"(pass {validation_pass})..."
            )
            validation = self.validator.validate(markdown_text, results, client)

        if validation.flags:
            logger.warning(
                f"[validate] finished with {len(validation.flags)} unresolved flag(s) "
                f"remaining after {validation_pass} validation pass(es)"
            )
        else:
            logger.info(f"[validate] clean - no flags after {validation_pass} validation pass(es)")

        return ReaderOutput(
            paper=markdown_path.stem,
            source_markdown=str(markdown_path),
            results=results,
            validation=validation,
            validation_passes=validation_pass,
            retried_stages=retried_stages,
        )


def run_pipeline(
    markdown_path: Path,
    output_dir: Path,
    client: Anthropic,
    pipeline: ReaderPipeline | None = None,
) -> ReaderOutput:
    """Run the reader pipeline over one paper's VLM Markdown and write the
    combined `<paper>.json`."""
    if pipeline is None:
        # Stage order is cosmetic, not functional: every stage reads the same
        # Markdown and none consumes another's output, so this list could be in
        # any order and produce the same result. It is ordered the way a person
        # reads a paper - and the way the Coder needs it - broad to specific:
        # what the paper is (method_summary), what to build (architecture_notes),
        # what it must score (claims), how to train it (hyperparameters), what to
        # feed it (data_pipeline). That order is also the order the console log
        # scrolls past in, which is the only thing it really affects.
        pipeline = ReaderPipeline(
            stages=[
                MethodSummaryExtractor(),
                ArchitectureNotesExtractor(),
                ClaimsExtractor(),
                HyperparametersExtractor(),
                DataPipelineExtractor(),
            ],
            validator=ExtractionValidator(),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{markdown_path.stem}.json"

    output = pipeline.run(markdown_path, client)

    payload: dict[str, Any] = {
        "paper": output.paper,
        "source_markdown": output.source_markdown,
        **{name: asdict(result) for name, result in output.results.items()},
        "validation": {
            "flags": [asdict(flag) for flag in output.validation.flags],
            "attempts": output.validation_passes,
            "retried_stages": output.retried_stages,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"  -> {json_path}")
    return output


def run_dataset(input_dir: Path, output_dir: Path) -> None:
    """Run the full pipeline over every Markdown file in input_dir, skipping
    papers already done."""
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
            run_pipeline(markdown_path, output_dir, client)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {markdown_path.name}: {exc}")
            continue


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
        help="Directory for <paper>.json files",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        run_dataset(args.input, args.output)
    else:
        client = Anthropic()
        logger.info(f"[extract] {args.input.name}")
        run_pipeline(args.input, args.output, client)


if __name__ == "__main__":
    main()
