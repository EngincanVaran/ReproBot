"""Rebuild a run's learning-curve history from its HF `Trainer` stdout log.

`critic/review.py` reads `metrics.<mode>.history.jsonl` so the model review can judge the
learning curve, not just the final number. `main`'s Runner check-ups write that file live
while a script trains; a run made without them (for example on this branch) has only the
Trainer's printed dict lines in `runner/output/<paper>/logs/<mode>.stdout.log`. This turns
those lines into the same records, so the review is not blind to the curve.

Only Trainer-format, accuracy-based runs are understood: per-epoch `eval_accuracy` /
`eval_loss` lines and `loss` logging lines. The error metric is derived as
`100 - eval_accuracy` when the claim's metric is lower-is-better. Nothing is invented: an
epoch with no logged training loss simply has no `train_loss`.

Usage:
    uv run python -m critic.history --paper "<paper>" --chance 90
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

from critic.judge import infer_higher_is_better


def parse_trainer_log(text: str) -> tuple[dict[int, dict[str, Any]], dict[int, list[float]]]:
    """Per-epoch evaluation dicts, and the training losses logged inside each epoch."""
    evals: dict[int, dict[str, Any]] = {}
    losses: dict[int, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(row, dict) or "epoch" not in row or "train_runtime" in row:
            continue
        epoch = float(row["epoch"])
        if "eval_accuracy" in row:
            evals[round(epoch)] = row
        elif "loss" in row:
            losses.setdefault(math.ceil(epoch), []).append(float(row["loss"]))
    return evals, losses


def build_records(
    evals: dict[int, dict[str, Any]],
    losses: dict[int, list[float]],
    metrics: dict[str, Any],
    *,
    target: float | None,
    chance: float | None,
) -> list[dict[str, Any]]:
    metric = str(metrics.get("metric") or "")
    higher = infer_higher_is_better(metric)
    steps_total = max(evals) if evals else None
    records = []
    for epoch in sorted(evals):
        row = evals[epoch]
        accuracy = float(row["eval_accuracy"])
        record: dict[str, Any] = {
            "step": epoch,
            "steps_total": steps_total,
            "kind": "epoch",
            "metric": metric,
            "train_loss": statistics.fmean(losses[epoch]) if epoch in losses else None,
            "eval_loss": row.get("eval_loss"),
            "eval_metric": accuracy if higher else 100 - accuracy,
            "higher_is_better": higher,
            "chance_metric": chance,
            "target_value": target,
            "num_train_samples": metrics.get("num_train_samples"),
            "num_eval_samples": metrics.get("num_eval_samples"),
        }
        records.append({k: v for k, v in record.items() if v is not None})
    return records


def claim_target(reader: dict[str, Any], claim_id: object) -> float | None:
    claims = reader.get("claims", [])
    if isinstance(claims, dict):
        claims = claims.get("claims", [])
    for claim in claims:
        if claim.get("claim_id") == claim_id and isinstance(
            claim.get("reported_value"), int | float
        ):
            return float(claim["reported_value"])
    return None


def write_history(
    paper: str,
    *,
    mode: str = "full",
    chance: float | None = None,
    coder_output: Path = Path("coder/output"),
    runner_output: Path = Path("runner/output"),
    reader_output: Path = Path("reader/output"),
) -> Path:
    """Write `metrics.<mode>.history.jsonl` for a paper's run and return its path.

    Raises `FileNotFoundError` when the metrics or the stage log is missing, and
    `ValueError` when the log holds no per-epoch evaluation lines.
    """
    paper_dir = coder_output / paper
    metrics_path = paper_dir / f"metrics.{mode}.json"
    if not metrics_path.exists():
        metrics_path = paper_dir / "metrics.json"
    log_path = runner_output / paper / "logs" / f"{mode}.stdout.log"
    for path in (metrics_path, log_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reader_path = reader_output / f"{paper}.json"
    target = None
    if reader_path.exists():
        target = claim_target(
            json.loads(reader_path.read_text(encoding="utf-8")), metrics.get("claim_id")
        )
    evals, losses = parse_trainer_log(log_path.read_text(encoding="utf-8", errors="replace"))
    if not evals:
        raise ValueError(f"no per-epoch evaluation lines found in {log_path}")

    records = build_records(evals, losses, metrics, target=target, chance=chance)
    out = paper_dir / f"metrics.{mode}.history.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    logger.info(f"[history] {len(records)} epochs from {log_path.name} -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True, help="paper name, as in coder/output/<paper>")
    parser.add_argument("--mode", default="full", help="stage whose log to read (default: full)")
    parser.add_argument(
        "--chance", type=float, default=None, help="the metric's chance level, e.g. 90 for CIFAR-10"
    )
    parser.add_argument("--coder-output", type=Path, default=Path("coder/output"))
    parser.add_argument("--runner-output", type=Path, default=Path("runner/output"))
    parser.add_argument("--reader-output", type=Path, default=Path("reader/output"))
    args = parser.parse_args()
    try:
        write_history(
            args.paper,
            mode=args.mode,
            chance=args.chance,
            coder_output=args.coder_output,
            runner_output=args.runner_output,
            reader_output=args.reader_output,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
