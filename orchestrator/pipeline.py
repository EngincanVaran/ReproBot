"""Orchestrator pipeline: drive the Coder<->Runner retry loop over one or more papers.

The only entry point for `orchestrator/` - `state.py` holds the shared-memory
object and `loop.py` the loop and its transition rules, exactly as
`runner/pipeline.py` is the entry point over `docker_runner.py` + `triage.py`.
Same CLI shape as every earlier stage: `--input` / `--output`, skip anything
already done, and never let one paper's failure stop the batch.

What a run does, per paper:

1. Load `reader/output/<paper>.json` into a fresh `ReproState` (project plan
   §1.2's shared-memory object) and resolve the paper's VLM Markdown.
2. Generate a training script with `coder/`, archive it as
   `attempts/v1_train.py`, and execute it with `runner/` in the Docker sandbox.
3. On a failure, route on the triage category: an `environment_error` or a
   timeout stops immediately, a `recoverable_error` regenerates the script with
   the triage's `suggested_fix` as feedback and runs again.
4. Stop on success, on an exhausted retry budget, or early if a regenerated
   script comes back near-identical to the one that just failed.
5. Write `orchestrator/output/<paper>/state.json`, with every attempt's script
   kept beside it under `attempts/`.

This stage makes no LLM calls of its own. Every call it costs comes from the
stages it drives: one Sonnet generation per attempt (`coder/`) and one Haiku
triage per genuine failure (`runner/`). It also builds no Critic and compares no
numbers - see `orchestrator/README.md`.

Requires the `orchestrator` extra (`uv sync --extra orchestrator`), an
ANTHROPIC_API_KEY in `.env`, and a reachable Docker daemon.

Usage:
    uv run python -m orchestrator.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json"
    uv run python -m orchestrator.pipeline --input reader/output --max-stage probe --retry-budget 2
    uv run python -m orchestrator.pipeline --input reader/output/paper.json --use-existing-script
"""  # noqa: E501

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from coder.pipeline import CoderPipeline, MissingPaperMarkdownError, resolve_paper_markdown
from coder.script_writer import TrainingScriptWriter
from orchestrator.loop import DEFAULT_PLATEAU_THRESHOLD, DEFAULT_RETRY_BUDGET, Orchestrator
from runner.docker_runner import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE,
    DEFAULT_STAGE_TIMEOUTS,
    STAGE_ORDER,
    DockerRunner,
    DockerUnavailableError,
    ImageMissingError,
    parse_timeout_overrides,
    stages_up_to,
)

# Matches `runner/`'s own default depth, so "the Orchestrator ran it" and "the
# Runner ran it" mean the same amount of execution. `--max-stage probe` is the
# cheap loop: it proves a script starts, which is the failure class the retry
# loop exists to repair.
DEFAULT_MAX_STAGE = "capped"
OUTPUT_FILENAME = "state.json"


def discover_reader_outputs(input_path: Path) -> list[Path]:
    """Resolve `--input` to a list of `reader/output/<paper>.json` files."""
    if not input_path.exists():
        raise FileNotFoundError(f"--input path does not exist: {input_path}")
    return sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]


def resolve_source_pdf(paper_id: str, dataset_dir: Path) -> Path | None:
    """Find the paper's original PDF for §1.2's `source_pdf`, if it is there.

    Purely provenance: nothing in the loop reads the PDF (the pipeline consumes
    `ocr/`'s Markdown), so a missing one is recorded as `null` rather than
    treated as an error.
    """
    candidate = dataset_dir / f"{paper_id}.pdf"
    return candidate if candidate.exists() else None


def run_dataset(
    reader_json_paths: list[Path],
    markdown_dir: Path,
    coder_output_dir: Path,
    output_root: Path,
    dataset_dir: Path,
    orchestrator: Orchestrator,
    client: Anthropic,
    use_existing_script: bool = False,
    force: bool = False,
) -> None:
    """Loop over every paper given, skipping ones already done and continuing past failures.

    A per-paper exception here means the ORCHESTRATOR could not run the loop at
    all (no Markdown, the daemon died mid-batch). A generated script failing is
    not an exception - it is a normal recorded outcome with its own verdict.
    """
    if not reader_json_paths:
        logger.warning("No reader output JSON files to orchestrate")
        return

    verdicts: Counter[str] = Counter()
    errored = 0
    for reader_json_path in reader_json_paths:
        paper_id = reader_json_path.stem
        output_dir = output_root / paper_id
        if (output_dir / OUTPUT_FILENAME).exists() and not force:
            logger.info(f"[skip]        {paper_id} (already orchestrated; --force to re-run)")
            continue

        logger.info(f"[orchestrate] {paper_id}")
        try:
            markdown_path = resolve_paper_markdown(reader_json_path, markdown_dir)
            state = orchestrator.run(
                reader_json_path=reader_json_path,
                markdown_path=markdown_path,
                coder_output_dir=coder_output_dir,
                output_dir=output_dir,
                client=client,
                source_pdf=resolve_source_pdf(paper_id, dataset_dir),
                use_existing_script=use_existing_script,
            )
        except MissingPaperMarkdownError as exc:
            logger.error(f"[error]       {paper_id}: {exc}")
            errored += 1
            continue
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]       {paper_id}: {exc}")
            errored += 1
            continue

        path = state.save(output_dir / OUTPUT_FILENAME)
        logger.info(f"  -> {path}")
        verdicts[str(state.verdict)] += 1

    summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(verdicts.items()))
    logger.info(
        f"[done] {sum(verdicts.values())} paper(s) orchestrated"
        + (f" ({summary})" if summary else "")
        + f", {errored} could not be run at all"
    )


def build_parser() -> argparse.ArgumentParser:
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
        help="Directory holding each paper's VLM Markdown (<paper>.md), a required Coder input",
    )
    parser.add_argument(
        "--coder-output",
        type=Path,
        default=Path("coder/output"),
        help="Where the Coder writes <paper>/train.py + reproduce.sh, and where the Runner runs it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("orchestrator/output"),
        help="Directory for <paper>/state.json, <paper>/attempts/ and <paper>/logs/",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Where the original PDFs live, recorded as the state's source_pdf provenance",
    )
    parser.add_argument(
        "--claim-id",
        type=str,
        default=None,
        help="Target this exact claim_id instead of letting Claude pick the headline claim",
    )
    parser.add_argument(
        "--retry-budget",
        type=int,
        default=DEFAULT_RETRY_BUDGET,
        help=(
            f"How many REGENERATIONS are allowed after the first attempt "
            f"(default: {DEFAULT_RETRY_BUDGET}). 0 makes the loop single-shot"
        ),
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=DEFAULT_PLATEAU_THRESHOLD,
        help=(
            f"Stop early when a regenerated script is at least this line-similar to the "
            f"one that just failed (default: {DEFAULT_PLATEAU_THRESHOLD}). Every "
            f"comparison logs its actual ratio, so this can be tuned from real runs"
        ),
    )
    parser.add_argument(
        "--use-existing-script",
        action="store_true",
        help=(
            "Seed attempt 1 from the train.py already in the paper directory instead of "
            "generating one. Retries still regenerate normally"
        ),
    )

    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--max-stage",
        choices=STAGE_ORDER,
        default=None,
        help=(
            f"Escalate probe -> ... -> this stage on every attempt (default: "
            f"{DEFAULT_MAX_STAGE}). 'full' needs a GPU and hours"
        ),
    )
    stage_group.add_argument(
        "--mode",
        choices=STAGE_ORDER,
        default=None,
        help="Run ONLY this one mode per attempt, skipping the escalation ladder",
    )

    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Sandbox image tag")
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the sandbox image before running (--no-build to reuse the existing one)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Host directory for the shared CIFAR-10 / HuggingFace caches",
    )
    parser.add_argument(
        "--timeout",
        action="append",
        metavar="MODE=SECONDS",
        help=(
            "Override one stage's wall-clock budget; repeatable. Defaults: "
            + ", ".join(f"{mode}={seconds}s" for mode, seconds in DEFAULT_STAGE_TIMEOUTS.items())
        ),
    )
    parser.add_argument("--memory", default=None, help="Container memory limit, e.g. 8g")
    parser.add_argument("--cpus", default=None, help="Container CPU limit, e.g. 4")
    parser.add_argument(
        "--network",
        default="bridge",
        help="Container network mode ('none' once the dataset cache is warm)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run papers that already have a state.json",
    )
    return parser


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    # Usage mistakes get argparse's own error handling (usage line, exit 2)
    # rather than a traceback out of a helper three frames down.
    try:
        stage_timeouts = parse_timeout_overrides(args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    if args.retry_budget < 0:
        parser.error(f"--retry-budget must be >= 0, got {args.retry_budget}")
    if not 0.0 < args.plateau_threshold <= 1.0:
        parser.error(f"--plateau-threshold must be in (0, 1], got {args.plateau_threshold}")

    modes = (args.mode,) if args.mode else stages_up_to(args.max_stage or DEFAULT_MAX_STAGE)
    if "full" in modes:
        logger.warning(
            "[setup] 'full' is in the stage list - that is the paper's real setup: it "
            "needs a GPU and can run for hours, and the retry loop would repeat it"
        )

    # There is deliberately no --no-triage flag here, unlike runner/'s CLI: the
    # triage category IS this stage's routing decision. Without it every failure
    # would end as `untriaged_error` on the first attempt and the loop would never
    # retry anything.
    runner = DockerRunner(
        image=args.image,
        cache_dir=args.cache_dir,
        stage_timeouts=stage_timeouts,
        memory=args.memory,
        cpus=args.cpus,
        network=args.network,
        run_triage=True,
    )
    try:
        runner.check_daemon()
        runner.ensure_image(build=args.build)
    except (DockerUnavailableError, ImageMissingError) as exc:
        logger.error(f"[setup] {exc}")
        raise SystemExit(1) from exc

    orchestrator = Orchestrator(
        coder=CoderPipeline(writer=TrainingScriptWriter(target_claim_id=args.claim_id)),
        runner=runner,
        modes=modes,
        retry_budget=args.retry_budget,
        plateau_threshold=args.plateau_threshold,
    )

    reader_json_paths = discover_reader_outputs(args.input)
    logger.info(
        f"[setup] {len(reader_json_paths)} paper(s) to orchestrate: "
        f"{', '.join(path.stem for path in reader_json_paths)}"
    )
    run_dataset(
        reader_json_paths,
        args.paper_markdown_dir,
        args.coder_output,
        args.output,
        args.dataset_dir,
        orchestrator,
        Anthropic(),
        use_existing_script=args.use_existing_script,
        force=args.force,
    )


if __name__ == "__main__":
    main()
