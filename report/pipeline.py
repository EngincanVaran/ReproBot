"""Report generator entry point: write Markdown replication reports for finished runs.

Reads what the earlier stages left on disk (an orchestrator `state.json`, or the separate
Reader / Runner / Critic files) and writes `report/output/<paper>.md`, an SVG learning
curve beside it when the run has a per-epoch curve, and `index.md` across papers.
It makes no model call and needs no API key, so regenerating a report is always safe.

Usage:
    uv run python -m report.pipeline --paper "2016-05 - Wide Residual Networks"
    uv run python -m report.pipeline --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from report.build import build_index, build_report, slug
from report.sources import Roots, Run, list_papers, load_run


def write_reports(runs: list[Run], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for run in runs:
        report = build_report(run)
        path = output / f"{slug(run.paper)}.md"
        path.write_text(report.markdown, encoding="utf-8")
        written.append(path)
        for name, svg in report.assets.items():
            (output / name).write_text(svg, encoding="utf-8")
        logger.info(
            f"[report] {run.paper}: {len(report.markdown.splitlines())} lines, "
            f"{len(report.assets)} chart(s) -> {path}"
        )
    index = output / "index.md"
    index.write_text(build_index(runs), encoding="utf-8")
    logger.info(f"[report] index of {len(runs)} paper(s) -> {index}")
    return [*written, index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--paper", action="append", help="paper name; repeatable")
    which.add_argument("--all", action="store_true", help="every paper with a finished run")
    parser.add_argument("--output", type=Path, default=Path("report/output"))
    parser.add_argument("--orchestrator-output", type=Path, default=Path("orchestrator/output"))
    args = parser.parse_args()

    roots = Roots(orchestrator=args.orchestrator_output)
    papers = list_papers(roots) if args.all else args.paper
    runs = []
    for paper in papers:
        try:
            runs.append(load_run(paper, roots))
        except FileNotFoundError as exc:
            if not args.all:
                parser.error(str(exc))
            logger.warning(f"[report] skipping {paper}: {exc}")
    if not runs:
        parser.error("nothing to report: no finished run found")
    write_reports(runs, args.output)


if __name__ == "__main__":
    main()
