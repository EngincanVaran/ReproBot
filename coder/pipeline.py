"""Coder pipeline: turn one paper's reader/ output into a runnable script.

This is the entry point for `coder/` - `method_translator.py`, `code_
synthesizer.py`, and `dependency_resolver.py` are importable steps, not
standalone scripts. Unlike `reader/pipeline.py`'s `ReaderPipeline`, there is
no class holding a configurable list of stages here: coder/'s three steps
are a fixed sequential chain with differently-shaped inputs/outputs (reader
JSON -> CodePlan -> script text -> Dependencies), not a homogeneous list of
interchangeable stages to loop over - see coder/README.md for the full
rationale. `run_pipeline()` just calls the three in order, same shape as
`ocr/vlm_extract.py`'s function-based `run_vlm()`/`extract_dataset()`
rather than reader/'s class-based pipeline.

No retry loop yet, unlike reader/'s validation-driven one - that needs
`runner/` and `critic/` to exist first (a Critic verdict is what would
supply the `feedback` param `MethodTranslator.translate()` and
`CodeSynthesizer.synthesize()` already accept but nothing calls yet).

Requires the `coder` extra: `uv sync --extra coder`, plus an
ANTHROPIC_API_KEY in `.env` at the repo root (copy `.env.example`).

Usage:
    uv run python -m coder.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json"
    uv run python -m coder.pipeline --input reader/output --output coder/output
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from coder.code_synthesizer import CodeSynthesizer
from coder.dependency_resolver import resolve_dependencies
from coder.method_translator import MethodTranslator
from coder.models import CodePlan, Dependencies


@dataclass
class CoderOutput:
    paper: str
    code_plan: CodePlan
    dependencies: Dependencies
    script_path: str
    requirements_path: str
    synthesis_notes: str | None


def run_pipeline(reader_output_path: Path, output_dir: Path, client: Anthropic) -> CoderOutput:
    """Run the coder pipeline over one paper's reader/ output and write the
    generated script, its requirements.txt, and a combined <paper>.json."""
    reader_output = json.loads(reader_output_path.read_text(encoding="utf-8"))
    paper = str(reader_output.get("paper", reader_output_path.stem))

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{reader_output_path.stem}.py"
    requirements_path = output_dir / f"{reader_output_path.stem}_requirements.txt"
    json_path = output_dir / f"{reader_output_path.stem}.json"

    logger.info(f"[method_translator] translating reader output for '{paper}'...")
    code_plan = MethodTranslator().translate(reader_output, client)

    logger.info(f"[code_synthesizer] synthesizing script for '{paper}'...")
    synthesized = CodeSynthesizer().synthesize(code_plan, client)

    try:
        ast.parse(synthesized.script)
    except SyntaxError as exc:
        raise ValueError(f"Generated script for '{paper}' is not valid Python: {exc}") from exc
    logger.info("  [pipeline] generated script parses as valid Python")

    logger.info(f"[dependency_resolver] resolving dependencies for '{paper}'...")
    dependencies = resolve_dependencies(synthesized.script)

    script_path.write_text(synthesized.script, encoding="utf-8")
    requirements_text = (
        "\n".join(f"{dep.package}=={dep.version}" for dep in dependencies.requirements) + "\n"
    )
    requirements_path.write_text(requirements_text, encoding="utf-8")

    output = CoderOutput(
        paper=paper,
        code_plan=code_plan,
        dependencies=dependencies,
        script_path=str(script_path),
        requirements_path=str(requirements_path),
        synthesis_notes=synthesized.notes,
    )
    payload: dict[str, Any] = {
        "paper": output.paper,
        "code_plan": asdict(output.code_plan),
        "dependencies": asdict(output.dependencies),
        "script_path": output.script_path,
        "requirements_path": output.requirements_path,
        "synthesis_notes": output.synthesis_notes,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"  -> {script_path}")
    logger.info(f"  -> {requirements_path}")
    logger.info(f"  -> {json_path}")
    return output


def run_dataset(input_dir: Path, output_dir: Path) -> None:
    """Run the full pipeline over every reader/ output JSON in input_dir,
    skipping papers already generated."""
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        logger.warning(f"No reader output JSON files found in {input_dir}")
        return

    client = Anthropic()
    for reader_output_path in json_paths:
        script_path = output_dir / f"{reader_output_path.stem}.py"
        if script_path.exists():
            logger.info(f"[skip]    {reader_output_path.name} (already generated)")
            continue

        logger.info(f"[generate] {reader_output_path.name}")
        try:
            run_pipeline(reader_output_path, output_dir, client)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {reader_output_path.name}: {exc}")
            continue


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reader/output"),
        help="reader/ output JSON file, or a directory of them",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coder/output"),
        help="Directory for <paper>.py / <paper>_requirements.txt / <paper>.json files",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        run_dataset(args.input, args.output)
    else:
        client = Anthropic()
        logger.info(f"[generate] {args.input.name}")
        run_pipeline(args.input, args.output, client)


if __name__ == "__main__":
    main()
