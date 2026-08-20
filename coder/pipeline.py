"""Coder pipeline: turn one paper's Reader output into a runnable training script.

This is the entry point for `coder/` - `script_writer.py` provides a
`CodeWriter` subclass (prompt + tool schema + parsing + `write()`), not a
standalone script. `CoderPipeline` resolves a paper's TWO inputs (its
authoritative `reader/output/<paper>.json` and its `ocr/output/vlm/<paper>.md`,
which is where architecture description still lives because `reader/` has no
`architecture_notes` stage yet), makes one Claude tool-use call, runs two
deterministic zero-cost safety gates over the result, and writes
`coder/output/<paper>/train.py` plus a `coder_output.json` of bookkeeping.

The two gates are pure Python, cost nothing, and catch the failures that would
otherwise only surface inside the Runner's Docker sandbox:

1. `ast.parse` - the script must be syntactically valid Python before it is
   written as a `.py`. A truncated or malformed script is written to
   `train.py.invalid` instead, alongside `coder_output.failed.json`, and the
   run is reported as failed rather than silently producing broken code.
2. Required-flag check - every flag in `script_writer.REQUIRED_CLI_FLAGS` must
   literally appear in the script text. Those flags are the interface contract
   with the future `runner/` stage, which uses them to force fast capped runs.
   Missing ones are warned about by name and recorded in `missing_cli_flags`;
   the model's own `cli_flags_included` self-report is cross-checked against
   the script text too, since a model can report a flag it did not write.

This stage never executes the generated script and never imports torch.

Requires the `coder` extra: `uv sync --extra coder`, plus an ANTHROPIC_API_KEY
in `.env` at the repo root (copy `.env.example`).

Usage:
    uv run python -m coder.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json"
    uv run python -m coder.pipeline --input reader/output --output coder/output
    uv run python -m coder.pipeline --input reader/output/paper.json --claim-id c34
"""

from __future__ import annotations

import argparse
import ast
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from coder.base import CodeWriter
from coder.script_writer import REQUIRED_CLI_FLAGS, TrainingScript, TrainingScriptWriter


class MissingPaperMarkdownError(FileNotFoundError):
    """The paper's VLM Markdown - a required second input - was not found."""


class ScriptSyntaxError(RuntimeError):
    """The generated script failed the `ast.parse` gate and was not written as .py."""


@dataclass
class CoderOutput:
    paper: str
    source_reader_output: str
    source_markdown: str
    script_path: str
    result: TrainingScript
    missing_cli_flags: list[str]


def check_syntax(script_content: str) -> SyntaxError | None:
    """Deterministic, zero-LLM-cost gate: is this valid Python at all?

    Returns the `SyntaxError` rather than raising, so the caller can still
    persist the offending source for debugging before failing the run.
    """
    try:
        ast.parse(script_content)
    except SyntaxError as exc:
        return exc
    return None


def check_required_flags(script_content: str, self_reported: list[str]) -> list[str]:
    """Deterministic, zero-LLM-cost gate: does the script really define every
    flag the future Runner will pass it?

    Checks the script text itself (the only thing that actually matters) and
    separately cross-checks the model's `cli_flags_included` self-report, which
    can claim a flag the script never defines. Returns the flags missing from
    the script text.
    """
    missing_from_script = [flag for flag in REQUIRED_CLI_FLAGS if flag not in script_content]
    missing_from_report = [flag for flag in REQUIRED_CLI_FLAGS if flag not in self_reported]

    if missing_from_script:
        logger.warning(
            f"  [gate:flags] script text is MISSING {len(missing_from_script)} required "
            f"CLI flag(s): {', '.join(missing_from_script)}"
        )
    else:
        logger.info(f"  [gate:flags] all {len(REQUIRED_CLI_FLAGS)} required CLI flags present")

    overreported = [flag for flag in self_reported if flag not in script_content]
    if overreported:
        logger.warning(
            f"  [gate:flags] model self-reported flag(s) that do not appear in the "
            f"script text: {', '.join(overreported)}"
        )
    underreported = [
        flag for flag in missing_from_report if flag in script_content
    ]  # in the script but not self-reported
    if underreported:
        logger.warning(
            f"  [gate:flags] script defines flag(s) the model did not self-report: "
            f"{', '.join(underreported)}"
        )

    return missing_from_script


def resolve_paper_markdown(reader_json_path: Path, markdown_dir: Path) -> Path:
    """Match a `reader/output/<paper>.json` to its `<markdown_dir>/<paper>.md`.

    The Markdown is a REQUIRED second input, not an optional enrichment - it is
    the only place the paper's architecture description exists. A missing one is
    an error, never a quiet fallback to reader-output-only generation.
    """
    markdown_path = markdown_dir / f"{reader_json_path.stem}.md"
    if not markdown_path.exists():
        raise MissingPaperMarkdownError(
            f"no paper Markdown at {markdown_path} - the Coder needs the paper text "
            f"for architecture detail, which reader/output/*.json does not carry. "
            f"Run `uv run python -m ocr.vlm_extract` for this paper first, or point "
            f"--paper-markdown-dir at the right directory."
        )
    return markdown_path


class CoderPipeline:
    """Runs the code-generation stage over one paper's Reader output + Markdown,
    then applies the deterministic syntax and CLI-flag gates to what came back."""

    def __init__(self, writer: CodeWriter[TrainingScript]) -> None:
        self.writer = writer

    def run(
        self,
        reader_json_path: Path,
        markdown_path: Path,
        output_dir: Path,
        client: Anthropic,
    ) -> CoderOutput:
        """Generate one training script, gate it, and write it under
        `output_dir/<paper>/`. Raises `ScriptSyntaxError` if the syntax gate
        fails (the offending source is still written, as `train.py.invalid`)."""
        paper = reader_json_path.stem
        logger.info(f"[{self.writer.name}] reading reader output: {reader_json_path}")
        reader_output: dict[str, Any] = json.loads(reader_json_path.read_text(encoding="utf-8"))
        logger.info(f"[{self.writer.name}] reading paper Markdown: {markdown_path}")
        paper_markdown = markdown_path.read_text(encoding="utf-8")

        claims = reader_output.get("claims", {}).get("claims", [])
        logger.info(
            f"[{self.writer.name}] reader output carries {len(claims)} claim(s), "
            f"{len(reader_output.get('hyperparameters', {}).get('hyperparameters', []))} "
            f"hyperparameter(s), "
            f"{len(reader_output.get('data_pipeline', {}).get('datasets', []))} dataset(s)"
        )

        logger.info(f"[{self.writer.name}] generating training script...")
        result = self.writer.write(reader_output, paper_markdown, client)

        paper_dir = output_dir / paper
        paper_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[gate:syntax] ast.parse over {len(result.script_content)} chars...")
        syntax_error = check_syntax(result.script_content)
        missing_cli_flags = check_required_flags(result.script_content, result.cli_flags_included)

        if syntax_error is not None:
            invalid_path = paper_dir / "train.py.invalid"
            invalid_path.write_text(result.script_content, encoding="utf-8")
            logger.error(
                f"  [gate:syntax] FAILED - {syntax_error.msg} at line "
                f"{syntax_error.lineno}, offset {syntax_error.offset}"
            )
            if syntax_error.text:
                logger.error(f"  [gate:syntax] offending line: {syntax_error.text.rstrip()}")
            logger.error(f"  [gate:syntax] wrote the invalid source to {invalid_path}")
            _write_bookkeeping(
                paper_dir / "coder_output.failed.json",
                paper=paper,
                reader_json_path=reader_json_path,
                markdown_path=markdown_path,
                script_path=invalid_path,
                result=result,
                syntax_ok=False,
                syntax_error=(
                    f"{syntax_error.msg} (line {syntax_error.lineno}, offset {syntax_error.offset})"
                ),
                missing_cli_flags=missing_cli_flags,
            )
            raise ScriptSyntaxError(
                f"generated script for '{paper}' is not valid Python: "
                f"{syntax_error.msg} at line {syntax_error.lineno}"
            )

        logger.info("  [gate:syntax] passed - script is valid Python")
        script_path = paper_dir / "train.py"
        script_path.write_text(result.script_content, encoding="utf-8")
        logger.info(f"  -> {script_path}")
        _write_bookkeeping(
            paper_dir / "coder_output.json",
            paper=paper,
            reader_json_path=reader_json_path,
            markdown_path=markdown_path,
            script_path=script_path,
            result=result,
            syntax_ok=True,
            syntax_error=None,
            missing_cli_flags=missing_cli_flags,
        )
        logger.info(f"  -> {paper_dir / 'coder_output.json'}")
        reproduce_path = paper_dir / "reproduce.sh"
        _write_reproduce_script(reproduce_path, paper=paper, script_path=script_path, result=result)
        logger.info(f"  -> {reproduce_path} (run './reproduce.sh smoke' for a fast check)")

        return CoderOutput(
            paper=paper,
            source_reader_output=str(reader_json_path),
            source_markdown=str(markdown_path),
            script_path=str(script_path),
            result=result,
            missing_cli_flags=missing_cli_flags,
        )


def _wrap_comment(text: str, *, bullet: str = "", width: int = 76) -> str:
    """Wrap one value into indented `#` comment lines.

    The model's bookkeeping fields are prose, and `architecture_used` in
    particular runs to well over a thousand characters - emitted raw, it would
    make a single unreadable comment line wider than most terminals.
    """
    body = textwrap.wrap(" ".join(text.split()), width=width) or [""]
    indent = " " * (3 + len(bullet))
    head = f"#   {bullet}{body[0]}"
    return "\n".join([head, *(f"#{indent}{line}" for line in body[1:])])


def _write_reproduce_script(
    sh_path: Path,
    *,
    paper: str,
    script_path: Path,
    result: TrainingScript,
) -> None:
    """Write a runnable `reproduce.sh` next to the generated training script.

    Built deterministically from data already in hand - claim, hyperparameters,
    script path - rather than asked of the model. A shell wrapper is pure
    templating, so generating it in Python costs nothing, cannot hallucinate a
    flag the script does not define, and cannot be swallowed by the tool-field
    leak `script_writer._recover_leaked_fields` exists to undo.

    Two invocations are emitted, and the difference between them is the whole
    point: the full run carries no flags at all, because every default in the
    generated script is already the paper's own value. The smoke run is what
    `runner/` will drive, capped small enough to finish on a CPU.
    """
    hyperparameters = "\n".join(
        _wrap_comment(f"{hyperparameter.name}: {hyperparameter.value_used}")
        for hyperparameter in result.hyperparameters_used
    )
    assumptions = "\n".join(
        _wrap_comment(assumption, bullet="- ") for assumption in result.assumptions
    ) or ("#   (none recorded)")
    script_name = script_path.name
    sh_path.write_text(
        f"""#!/usr/bin/env bash
# Reproduce: {paper}
# Target claim: {result.claim_targeted}
#
# GENERATED by coder/pipeline.py - do not edit by hand; regenerate instead.
#
# Architecture and dataset (abridged - full text in coder_output.json):
{_wrap_comment(result.architecture_used, bullet="arch:    ")}
{_wrap_comment(result.dataset_used, bullet="dataset: ")}
#
# Hyperparameters taken from the paper (via reader/output/{paper}.json):
{hyperparameters}
#
# Assumptions the Coder had to make (not stated in the paper):
{assumptions}
#
# This script needs torch/torchvision/transformers, which are deliberately NOT
# in this repo's uv lock (see CLAUDE.md's platform-trap note). Run it inside
# runner/'s Docker image, or in a throwaway venv on a machine with wheels:
#   python3.11 -m venv .venv && . .venv/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install transformers accelerate
set -euo pipefail
cd "$(dirname "$0")"

MODE="${{1:-full}}"

case "$MODE" in
  full)
    # No flags: every default in {script_name} is already the paper's own value.
    # This is the real reproduction attempt.
    python {script_name}
    ;;
  smoke)
    # Fast structural check - proves the script runs end to end and writes
    # metrics.json. The resulting accuracy is NOT comparable to the paper.
    python {script_name} \\
      --epochs 1 \\
      --max-train-samples 256 \\
      --max-eval-samples 256 \\
      --metrics-output metrics.smoke.json
    ;;
  *)
    echo "usage: $0 [full|smoke]" >&2
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    sh_path.chmod(0o755)


def _write_bookkeeping(
    json_path: Path,
    *,
    paper: str,
    reader_json_path: Path,
    markdown_path: Path,
    script_path: Path,
    result: TrainingScript,
    syntax_ok: bool,
    syntax_error: str | None,
    missing_cli_flags: list[str],
) -> None:
    """Write the run's bookkeeping JSON next to the script.

    `script_content` is deliberately NOT duplicated in here - the script lives
    on disk at `script_path` and this file stays readable. `script_version` and
    `diff_from_previous` are present but inert: today every run is version 1
    with no previous version, but the planned Coder<->Runner retry loop will
    rewrite a script in place and needs somewhere to record which revision this
    is and what changed. Emitting them now keeps this schema stable when that
    lands, instead of forcing consumers to handle two shapes.
    """
    payload: dict[str, Any] = {
        "paper": paper,
        "source_reader_output": str(reader_json_path),
        "source_markdown": str(markdown_path),
        "script_path": str(script_path),
        "script_version": 1,
        "diff_from_previous": None,
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "missing_cli_flags": missing_cli_flags,
        "claim_targeted": result.claim_targeted,
        "claim_selection_reasoning": result.claim_selection_reasoning,
        "architecture_used": result.architecture_used,
        "dataset_used": result.dataset_used,
        "hyperparameters_used": [
            {"name": hyperparameter.name, "value_used": hyperparameter.value_used}
            for hyperparameter in result.hyperparameters_used
        ],
        "assumptions": result.assumptions,
        "cli_flags_included": result.cli_flags_included,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(
    reader_json_path: Path,
    markdown_dir: Path,
    output_dir: Path,
    client: Anthropic,
    pipeline: CoderPipeline | None = None,
    target_claim_id: str | None = None,
) -> CoderOutput:
    """Resolve one paper's two inputs, generate its training script, and write
    `<output_dir>/<paper>/train.py` + `coder_output.json`."""
    if pipeline is None:
        pipeline = CoderPipeline(writer=TrainingScriptWriter(target_claim_id=target_claim_id))
    markdown_path = resolve_paper_markdown(reader_json_path, markdown_dir)
    return pipeline.run(reader_json_path, markdown_path, output_dir, client)


def run_dataset(
    reader_json_paths: list[Path],
    markdown_dir: Path,
    output_dir: Path,
    target_claim_id: str | None = None,
) -> None:
    """Generate a script for every reader output given, skipping papers already
    done and continuing past any single paper's failure."""
    if not reader_json_paths:
        logger.warning("No reader output JSON files to process")
        return

    client = Anthropic()
    generated = 0
    failed = 0
    for reader_json_path in reader_json_paths:
        paper_dir = output_dir / reader_json_path.stem
        if (paper_dir / "coder_output.json").exists():
            logger.info(f"[skip]     {reader_json_path.name} (script already generated)")
            continue

        logger.info(f"[generate] {reader_json_path.name}")
        try:
            run_pipeline(
                reader_json_path,
                markdown_dir,
                output_dir,
                client,
                target_claim_id=target_claim_id,
            )
            generated += 1
        except MissingPaperMarkdownError as exc:
            logger.error(f"[error]    {reader_json_path.name}: {exc}")
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]    {reader_json_path.name}: {exc}")
            failed += 1
            continue

    logger.info(f"[done] {generated} script(s) generated, {failed} failed")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reader/output"),
        help="A reader/output/<paper>.json file, or a directory of them",
    )
    parser.add_argument(
        "--paper-markdown-dir",
        type=Path,
        default=Path("ocr/output/vlm"),
        help="Directory holding each paper's VLM Markdown (<paper>.md), a required second input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coder/output"),
        help="Directory for <paper>/train.py + <paper>/coder_output.json",
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        default=None,
        help="Target this exact claim_id instead of letting Claude pick the headline claim",
    )
    args = parser.parse_args()

    reader_json_paths = sorted(args.input.glob("*.json")) if args.input.is_dir() else [args.input]

    run_dataset(reader_json_paths, args.paper_markdown_dir, args.output, args.claim_id)


if __name__ == "__main__":
    main()
