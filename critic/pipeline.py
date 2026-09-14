"""Critic entry point: judge a finished run against the paper's claim.

Two ways in, same verdict logic (`critic/judge.py`):

* `--state` - an Orchestrator run: `orchestrator/output/<paper>/state.json` (or its
  directory). Claims come from its `reader_output`, the reproduced value from its
  `runner_output`, and the stage from its last attempt. `--write` stores the verdict
  in the state's reserved `critic_output`.
* `--reader-json` + `--metrics-json` - a run made without the Orchestrator (for
  example `runner.pipeline --mode full`), judged from the two files directly.

Every verdict is also written to `critic/output/<paper>.json`. The verdict needs no API
key, no Docker and no network: it is arithmetic. `--review` adds the Critic v2 Sonnet
review of the script against the paper (one API call, `critic/review.py`).

Usage:
    uv run python -m critic.pipeline --state "orchestrator/output/<paper>"
    uv run python -m critic.pipeline --state orchestrator/output --write
    uv run python -m critic.pipeline --reader-json "reader/output/<paper>.json" \\
        --metrics-json "coder/output/<paper>/metrics.full.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from critic.claims import group_claims
from critic.judge import Judgement, judge

if TYPE_CHECKING:
    from critic.review import Review

DEFAULT_OUTPUT = Path("critic/output")


type StateInputs = tuple[list[dict[str, Any]], dict[str, Any] | None, str, str | None, list[float]]


def state_inputs(state: dict[str, Any]) -> StateInputs:
    """Claims, metrics, runner status, metrics stage and extra seed values from a state."""
    reader = state.get("reader_output") or {}
    claims = list((reader.get("claims") or {}).get("claims") or [])
    runner = state.get("runner_output") or {}
    status = str(runner.get("status") or "missing")
    attempts = state.get("attempts") or []
    stage = attempts[-1].get("stage_reached") if attempts else None
    critic = state.get("critic_output") or {}
    seeds = [float(v) for v in critic.get("seed_run_values", []) if isinstance(v, int | float)]
    return claims, runner.get("reproduced_metrics"), status, stage, seeds


def judge_state_file(path: Path) -> tuple[dict[str, Any], Judgement]:
    state = json.loads(path.read_text(encoding="utf-8"))
    claims, metrics, status, stage, seeds = state_inputs(state)
    return state, judge(
        claims, metrics, runner_status=status, metrics_mode=stage, extra_run_values=seeds
    )


def log_judgement(paper: str, judgement: Judgement) -> None:
    logger.info(f"[critic] {paper}")
    level = "INFO" if judgement.verdict == "pass" else "WARNING"
    logger.log(level, f"  [critic] VERDICT: {judgement.summary()}")
    if judgement.claimed is not None:
        logger.info(f"  [critic] reason: {judgement.reason}")
    if judgement.tolerance is not None:
        logger.info(
            f"  [critic] tolerance {judgement.tolerance:.4g} from {judgement.evidence} "
            f"(precision {judgement.reporting_precision}, test noise {judgement.test_noise}, "
            f"run spread {judgement.run_spread})"
        )
    if len(judgement.merged_claim_ids) > 1:
        logger.info(f"  [critic] merged duplicate claims: {', '.join(judgement.merged_claim_ids)}")
    if judgement.recommendation != "none":
        logger.info(f"  [critic] recommendation: {judgement.recommendation}")


def run_review(
    judgement: Judgement,
    metrics: dict[str, Any] | None,
    reader_output: dict[str, Any],
    paper_dir: Path,
) -> Review:
    """The Sonnet review of one judged run, from the files the run left behind."""
    from anthropic import Anthropic
    from dotenv import load_dotenv

    from critic.review import review_run

    load_dotenv()
    coder_output = json.loads((paper_dir / "coder_output.json").read_text(encoding="utf-8"))
    markdown = Path(
        str(coder_output.get("source_markdown") or reader_output.get("source_markdown"))
    )
    return review_run(
        Anthropic(),
        judgement=judgement,
        metrics=metrics,
        reader_output=reader_output,
        coder_output=coder_output,
        script=(paper_dir / "train.py").read_text(encoding="utf-8"),
        paper_markdown=markdown.read_text(encoding="utf-8") if markdown.is_file() else "",
        history_path=paper_dir / "metrics.full.history.jsonl",
    )


def write_output(output_dir: Path, paper: str, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{paper}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def state_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if (target / "state.json").exists():
        return [target / "state.json"]
    return sorted(target.glob("*/state.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--state", type=Path, help="state.json, a paper's output dir, or a dir of them"
    )
    parser.add_argument("--reader-json", type=Path, help="reader/output/<paper>.json")
    parser.add_argument("--metrics-json", type=Path, help="a metrics.full.json to judge")
    parser.add_argument(
        "--extra-metrics",
        type=Path,
        nargs="*",
        default=[],
        help="metrics files of extra seeds of the same run (e.g. metrics.seed2.json)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="also run the Sonnet review of the script against the paper (needs an API key)",
    )
    parser.add_argument(
        "--coder-output",
        type=Path,
        default=Path("coder/output"),
        help="where <paper>/train.py, coder_output.json and the history files live",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write", action="store_true", help="also store the verdict in the state's critic_output"
    )
    args = parser.parse_args()

    if args.state:
        paths = state_files(args.state)
        if not paths:
            parser.error(f"no state.json found under {args.state}")
        for path in paths:
            state, judgement = judge_state_file(path)
            paper = str(state.get("paper_id") or path.parent.name)
            log_judgement(paper, judgement)
            payload = judgement.to_dict()
            if args.review:
                reader = state.get("reader_output") or {}
                metrics = (state.get("runner_output") or {}).get("reproduced_metrics")
                payload["review"] = run_review(
                    judgement, metrics, reader, args.coder_output / paper
                ).to_dict()
            logger.info(f"  -> {write_output(args.output, paper, payload)}")
            if args.write:
                state["critic_output"] = {**(state.get("critic_output") or {}), **payload}
                path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                logger.info(f"  -> critic_output stored in {path}")
        return

    if not (args.reader_json and args.metrics_json):
        parser.error("give --state, or both --reader-json and --metrics-json")
    reader = json.loads(args.reader_json.read_text(encoding="utf-8"))
    claims = list(reader.get("claims", {}).get("claims", []))
    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    extra = []
    for path in args.extra_metrics:
        value = json.loads(path.read_text(encoding="utf-8")).get("value")
        if isinstance(value, int | float):
            extra.append(float(value))
    name = args.metrics_json.name
    mode = name.split(".")[1] if name.startswith("metrics.") and name.count(".") >= 2 else None
    judgement = judge(
        claims, metrics, runner_status="success", metrics_mode=mode, extra_run_values=extra
    )
    paper = args.reader_json.stem
    log_judgement(paper, judgement)
    groups = [group.to_dict() for group in group_claims(claims)]
    payload = {**judgement.to_dict(), "claim_groups": groups}
    if args.review:
        payload["review"] = run_review(
            judgement, metrics, reader, args.metrics_json.parent
        ).to_dict()
    logger.info(f"  -> {write_output(args.output, paper, payload)}")


if __name__ == "__main__":
    main()
