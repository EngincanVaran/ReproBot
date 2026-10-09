"""Report generator entry point: shared-memory state in, Markdown replication report out.

The last stage of the project plan's pipeline (§2.6). Same CLI shape as every other
stage - `--input`/`--output`, skip what is already done, never let one paper's failure
stop the batch - and, like `critic/`'s verdict, it makes **no API call**: a report is a
rearrangement of facts the loop already established, so it needs no model, no Docker and
no network.

What it reads per paper:

* `orchestrator/output/<paper>/state.json` - the claims, the verdict, the review, every
  attempt and the whole history;
* `coder/output/<paper>/coder_output.json` - the Coder's own bookkeeping (what it used,
  what it had to guess), which the state deliberately does not duplicate;
* `coder/output/<paper>/metrics.<mode>.history.jsonl` - the learning curve of the judged
  run, drawn into `curve.svg` beside the report.

Usage:
    uv run python -m report.pipeline --input orchestrator/output
    uv run python -m report.pipeline --input "orchestrator/output/<paper>" --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from report.curves import read_curve, render_curve
from report.render import ReportInputs, render, render_index
from report.summary import render_summary

DEFAULT_INPUT = Path("orchestrator/output")
DEFAULT_CODER_OUTPUT = Path("coder/output")
DEFAULT_OUTPUT = Path("report/output")
REPORT_FILENAME = "report.md"
SUMMARY_FILENAME = "SUMMARY.md"
CURVE_FILENAME = "curve.svg"


def state_files(target: Path) -> list[Path]:
    """Resolve `--input` to a list of state.json files, the way `critic/` does."""
    if target.is_file():
        return [target]
    if (target / "state.json").exists():
        return [target / "state.json"]
    # `<dir>/<paper>/state.json` is one run per paper; `<dir>/<run>/<paper>/state.json`
    # is several runs of the same papers side by side, which is what a repeated
    # evaluation produces. Both are accepted, so a summary can span either layout.
    return sorted(target.glob("*/state.json")) + sorted(target.glob("*/*/state.json"))


def judged_mode(state: dict[str, Any]) -> str:
    """Which stage produced the number the Critic judged; `full` unless told otherwise."""
    critic = state.get("critic_output") or {}
    mode = critic.get("metrics_mode")
    if isinstance(mode, str):
        return mode
    attempts = state.get("attempts") or []
    if attempts and isinstance(attempts[-1].get("stage_reached"), str):
        return str(attempts[-1]["stage_reached"])
    return "full"


def load_bookkeeping(paper_dir: Path) -> dict[str, Any]:
    """The Coder's `coder_output.json`, or an empty dict when the directory is gone.

    Generated output is gitignored and a paper directory can be cleaned away, so a
    missing file costs the report its "what was run" detail, never the report itself.
    """
    path = paper_dir / "coder_output.json"
    if not path.exists():
        logger.warning(f"  [report] no coder_output.json at {path} - setup section will be thin")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"  [report] {path} is not readable JSON ({exc}) - skipping it")
        return {}
    return payload if isinstance(payload, dict) else {}


def build_report(state_path: Path, coder_output: Path, output_dir: Path) -> tuple[str, str]:
    """Write one paper's report (and its curve) and return its id and relative link."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    paper = str(state.get("paper_id") or state_path.parent.name)
    paper_dir = coder_output / paper
    mode = judged_mode(state)

    bookkeeping = load_bookkeeping(paper_dir)
    coder_state = dict(state.get("coder_output") or {})
    coder_state["bookkeeping"] = bookkeeping
    state["coder_output"] = coder_state

    run_label = state_path.parent.parent.name
    report_dir = (
        output_dir / paper if run_label in ("output", "") else output_dir / run_label / paper
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    curve = read_curve(paper_dir / f"metrics.{mode}.history.jsonl")
    href = None
    if curve is not None:
        (report_dir / CURVE_FILENAME).write_text(
            render_curve(curve, f"{paper} - {mode} run"), encoding="utf-8"
        )
        href = CURVE_FILENAME
        logger.info(f"  [report] curve: {len(curve.steps)} records from the {mode} run")
    else:
        logger.info(f"  [report] no usable metrics.{mode}.history.jsonl - report without a curve")

    markdown = render(ReportInputs(state=state, curve=curve, curve_href=href))
    (report_dir / REPORT_FILENAME).write_text(markdown, encoding="utf-8")

    critic = state.get("critic_output") or {}
    logger.info(
        f"  [report] {paper}: fidelity {critic.get('verdict') or 'not_evaluated'}, "
        f"execution {state.get('verdict')}, {len(markdown)} chars"
    )
    href = f"{paper}/{REPORT_FILENAME}"
    if report_dir.parent != output_dir:
        href = f"{report_dir.parent.name}/{href}"
    return paper, href


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="A state.json, a paper's orchestrator output dir, or a directory of them",
    )
    parser.add_argument(
        "--coder-output",
        type=Path,
        default=DEFAULT_CODER_OUTPUT,
        help="Where <paper>/coder_output.json and the history files live",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Rewrite reports that already exist")
    args = parser.parse_args()

    paths = state_files(args.input)
    if not paths:
        parser.error(f"no state.json found under {args.input}")
    logger.info(f"[setup] {len(paths)} state file(s) to report on")

    written: list[tuple[str, dict[str, Any], str]] = []
    failed = 0
    for path in paths:
        paper = path.parent.name
        run_label = path.parent.parent.name
        prefix = args.output if run_label in ("output", "") else args.output / run_label
        existing = prefix / paper / REPORT_FILENAME
        if existing.exists() and not args.force:
            logger.info(f"[skip]   {paper} (already reported; --force to rewrite)")
            href = f"{paper}/{REPORT_FILENAME}"
            if prefix != args.output:
                href = f"{run_label}/{href}"
            written.append((paper, json.loads(path.read_text(encoding="utf-8")), href))
            continue
        logger.info(f"[report] {paper}")
        try:
            name, href = build_report(path, args.coder_output, args.output)
        except Exception as exc:  # noqa: BLE001 - report the rest of the batch
            logger.error(f"[error]  {paper}: {exc}")
            failed += 1
            continue
        written.append((name, json.loads(path.read_text(encoding="utf-8")), href))

    if written:
        index = args.output / "README.md"
        index.write_text(render_index(written), encoding="utf-8")
        logger.info(f"  -> {index}")
        summary = args.output / SUMMARY_FILENAME
        summary.write_text(render_summary(written), encoding="utf-8")
        logger.info(f"  -> {summary} (per paper, across runs)")
    logger.info(f"[done] {len(written)} report(s) written, {failed} failed")


if __name__ == "__main__":
    main()
