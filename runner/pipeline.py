"""Runner pipeline: execute coder/'s generated training scripts in a Docker sandbox.

This is the entry point for `runner/` - `docker_runner.py` owns the daemon
mechanics and `triage.py` the single failure-classification call, the same way
`coder/pipeline.py` is the entry point over `script_writer.py`. Same CLI shape as
the earlier stages: `--input` / `--output`, skip anything already done, and never
let one paper's failure stop the batch.

What a run does, per paper:

1. Resolve `coder/output/<paper>/` - the directory holding `train.py`,
   `reproduce.sh` and (after a run) `metrics.<mode>.json`.
2. Run `bash reproduce.sh <mode>` inside the sandbox image for each escalation
   stage in turn, stopping at the first non-zero exit.
3. Read the metrics back out of the bind mount, capture the full logs to disk,
   and build a truncated excerpt for downstream prompts.
4. On a genuine (non-timeout) failure only, spend one Haiku call classifying it
   as a script bug or an environment problem.
5. Write `runner/output/<paper>/runner_output.json`.

The interface to a generated script is `bash reproduce.sh <mode>` and nothing
else - no `python` command is ever constructed and no `--flag` is ever passed,
which is what keeps this stage paper-agnostic.

ANTHROPIC_API_KEY is needed only when something fails: the client is constructed
lazily, so a batch where every paper passes runs with no key present at all.

Requires the `runner` extra (`uv sync --extra runner`) plus a reachable Docker
daemon. Nothing here imports torch - that lives only inside the image.

Usage:
    uv run python -m runner.pipeline --input "coder/output/2016-05 - Wide Residual Networks"
    uv run python -m runner.pipeline --input coder/output --max-stage smoke
    uv run python -m runner.pipeline --mode probe --no-build
    uv run python -m runner.pipeline --timeout probe=300 --timeout smoke=1800
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from runner.docker_runner import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IMAGE,
    DEFAULT_STAGE_TIMEOUTS,
    STAGE_ORDER,
    DockerRunner,
    DockerUnavailableError,
    ImageMissingError,
    RunnerOutput,
    parse_timeout_overrides,
    stages_up_to,
)

DEFAULT_MAX_STAGE = "capped"
OUTPUT_FILENAME = "runner_output.json"


def discover_paper_dirs(input_path: Path) -> list[Path]:
    """Resolve `--input` to a list of paper directories.

    Accepts either one `coder/output/<paper>/` directory or the `coder/output`
    parent holding many. The discriminator is the presence of `reproduce.sh`,
    which is the only file this stage actually invokes - so a directory without
    one is not a runnable paper regardless of what else it contains.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"--input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(
            f"--input must be a coder/output/<paper> directory or the coder/output "
            f"parent, got a file: {input_path}"
        )
    if (input_path / "reproduce.sh").exists():
        return [input_path]
    return sorted(
        child
        for child in input_path.iterdir()
        if child.is_dir() and (child / "reproduce.sh").exists()
    )


def write_runner_output(output_dir: Path, result: RunnerOutput) -> Path:
    """Persist one paper's `RunnerOutput` as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUTPUT_FILENAME
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path


def run_dataset(
    paper_dirs: list[Path],
    output_root: Path,
    runner: DockerRunner,
    modes: tuple[str, ...],
    force: bool = False,
) -> None:
    """Run every paper given, skipping ones already done and continuing past failures.

    A per-paper failure here means the RUNNER broke (daemon died, mount refused),
    not that the generated script failed - a script failure is a normal, recorded
    outcome with `status: "error"`, not an exception.
    """
    if not paper_dirs:
        logger.warning("No paper directories with a reproduce.sh to run")
        return

    client: Anthropic | None = None
    if runner.run_triage:
        try:
            client = Anthropic()
        except Exception as exc:  # noqa: BLE001 - triage is optional, running is not
            logger.warning(
                f"[setup] could not construct an Anthropic client ({exc}); failures will "
                f"be recorded untriaged. Set ANTHROPIC_API_KEY in .env to enable triage."
            )

    passed = 0
    failed = 0
    errored = 0
    for paper_dir in paper_dirs:
        output_dir = output_root / paper_dir.name
        if (output_dir / OUTPUT_FILENAME).exists() and not force:
            logger.info(f"[skip]    {paper_dir.name} (already run; --force to re-run)")
            continue

        logger.info(f"[execute] {paper_dir.name}")
        try:
            result = runner.run_paper(
                paper_dir,
                logs_dir=output_dir / "logs",
                modes=modes,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            logger.error(f"[error]   {paper_dir.name}: {exc}")
            errored += 1
            continue

        path = write_runner_output(output_dir, result)
        logger.info(f"  -> {path}")
        if result.status == "success":
            passed += 1
        else:
            failed += 1

    logger.info(
        f"[done] {passed} paper(s) passed, {failed} failed their stages, "
        f"{errored} could not be run at all"
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("coder/output"),
        help="A coder/output/<paper> directory, or the coder/output parent holding many",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runner/output"),
        help="Directory for <paper>/runner_output.json and <paper>/logs/",
    )

    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--max-stage",
        choices=STAGE_ORDER,
        default=None,
        help=(
            f"Escalate probe -> ... -> this stage, stopping at the first failure "
            f"(default: {DEFAULT_MAX_STAGE}). 'full' needs a GPU and hours - it is "
            f"never reached by default and must be asked for by name."
        ),
    )
    stage_group.add_argument(
        "--mode",
        choices=STAGE_ORDER,
        default=None,
        help="Run ONLY this one mode, skipping the escalation ladder entirely",
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
        help=(
            "Host directory for the shared CIFAR-10 / HuggingFace caches, bind-mounted "
            "into every run so the ~170 MB dataset is downloaded once, not per paper"
        ),
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
    parser.add_argument(
        "--memory",
        default=None,
        help="Container memory limit passed to `docker run --memory` (e.g. 8g). Unset by default",
    )
    parser.add_argument(
        "--cpus",
        default=None,
        help="Container CPU limit passed to `docker run --cpus` (e.g. 4). Unset by default",
    )
    parser.add_argument(
        "--network",
        default="bridge",
        help=(
            "Container network mode. Stays 'bridge' by default because torchvision "
            "downloads CIFAR-10 on the first run; once the cache is warm, 'none' "
            "makes the sandbox fully offline"
        ),
    )
    parser.add_argument(
        "--no-triage",
        action="store_true",
        help="Never make the Haiku failure-classification call (runs fully offline)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run papers that already have a runner_output.json",
    )
    args = parser.parse_args()

    # A malformed --timeout is a usage mistake, so it gets argparse's own error
    # handling (usage line, exit 2) rather than a traceback out of a helper.
    try:
        stage_timeouts = parse_timeout_overrides(args.timeout)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode:
        modes: tuple[str, ...] = (args.mode,)
    else:
        modes = stages_up_to(args.max_stage or DEFAULT_MAX_STAGE)
    if "full" in modes:
        logger.warning(
            "[setup] 'full' is in the stage list - this is the paper's real setup: "
            "it needs a GPU and can run for hours"
        )

    runner = DockerRunner(
        image=args.image,
        cache_dir=args.cache_dir,
        stage_timeouts=stage_timeouts,
        memory=args.memory,
        cpus=args.cpus,
        network=args.network,
        run_triage=not args.no_triage,
    )
    # A missing daemon or image is a setup problem, not a bug worth a traceback -
    # and it is the single most likely reason a first run does not start, so it
    # gets a one-line actionable message instead of a stack trace.
    try:
        runner.check_daemon()
        runner.ensure_image(build=args.build)
    except (DockerUnavailableError, ImageMissingError) as exc:
        logger.error(f"[setup] {exc}")
        raise SystemExit(1) from exc

    paper_dirs = discover_paper_dirs(args.input)
    logger.info(
        f"[setup] {len(paper_dirs)} paper(s) to run: {', '.join(p.name for p in paper_dirs)}"
    )
    run_dataset(paper_dirs, args.output, runner, modes, force=args.force)


if __name__ == "__main__":
    main()
