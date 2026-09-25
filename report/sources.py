"""Load one paper's finished run, from whichever pipeline produced it.

`orchestrator.pipeline` keeps a whole run in `state.json`. `runner.pipeline` and
`critic.pipeline` write their own files instead. Both shapes are normalised into `Run`
so the report builder never cares which one it got.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from critic.history import build_records, claim_target, parse_trainer_log

CURVE_MIN_POINTS = 3


@dataclass
class Roots:
    """Where each stage keeps its output. All relative to the repo root by default."""

    reader: Path = Path("reader/output")
    coder: Path = Path("coder/output")
    runner: Path = Path("runner/output")
    critic: Path = Path("critic/output")
    orchestrator: Path = Path("orchestrator/output")


@dataclass
class Run:
    paper: str
    origin: str
    reader: dict[str, Any]
    coder: dict[str, Any]
    runner: dict[str, Any] | None
    critic: dict[str, Any] | None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    loop_verdict: str | None = None
    retry_count: int | None = None
    retry_budget: int | None = None
    script: str | None = None
    script_path: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _read_history(
    paper_dir: Path, log_dirs: list[Path], metrics: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The learning curve: the script's own history file, else rebuilt from its Trainer log."""
    path = paper_dir / "metrics.full.history.jsonl"
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return [r for r in rows if isinstance(r, dict)]
    for log_dir in log_dirs:
        log = log_dir / "full.stdout.log"
        if log.is_file() and metrics:
            evals, losses = parse_trainer_log(log.read_text(encoding="utf-8", errors="replace"))
            if evals:
                return build_records(evals, losses, metrics, target=None, chance=None)
    return []


def _script(coder: dict[str, Any], paper_dir: Path) -> tuple[str | None, str | None]:
    for candidate in (coder.get("script_path"), str(paper_dir / "train.py")):
        if candidate and Path(candidate).is_file():
            return Path(candidate).read_text(encoding="utf-8"), str(candidate)
    return None, None


def list_papers(roots: Roots) -> list[str]:
    """Every paper with something to report: an orchestrated run or a Runner result."""
    names = {p.parent.name for p in roots.orchestrator.glob("*/state.json")}
    names |= {p.parent.name for p in roots.runner.glob("*/runner_output.json")}
    return sorted(names)


def load_run(paper: str, roots: Roots | None = None) -> Run:
    """Load `paper`. Raises `FileNotFoundError` when it has neither a state nor a Runner file."""
    roots = roots or Roots()
    paper_dir = roots.coder / paper
    coder_file = _read_json(paper_dir / "coder_output.json") or {}
    state = _read_json(roots.orchestrator / paper / "state.json")

    if state is not None:
        runner = state.get("runner_output")
        attempts = list(state.get("attempts") or [])
        last = attempts[-1] if attempts else {}
        if runner:
            runner = {**runner, "metrics_mode": last.get("stage_reached")}
        coder = {**coder_file, **(state.get("coder_output") or {})}
        logs = runner.get("logs_path") if runner else None
        run = Run(
            paper=paper,
            origin="orchestrator",
            reader=state.get("reader_output") or {},
            coder=coder,
            runner=runner,
            critic=state.get("critic_output"),
            attempts=attempts,
            events=list(state.get("history") or []),
            loop_verdict=state.get("verdict"),
            retry_count=state.get("retry_count"),
            retry_budget=state.get("retry_budget"),
        )
        log_dirs = [Path(logs)] if logs else []
    else:
        runner = _read_json(roots.runner / paper / "runner_output.json")
        if runner is None:
            raise FileNotFoundError(f"no orchestrator state or Runner output for {paper!r}")
        run = Run(
            paper=paper,
            origin="files",
            reader=_read_json(roots.reader / f"{paper}.json") or {},
            coder=coder_file,
            runner=runner,
            critic=_read_json(roots.critic / f"{paper}.json"),
        )
        log_dirs = [roots.runner / paper / "logs"]

    metrics = (run.runner or {}).get("reproduced_metrics")
    run.script, run.script_path = _script(run.coder, paper_dir)
    run.history = _read_history(paper_dir, log_dirs, metrics)
    target = claim_target(run.reader, (metrics or {}).get("claim_id"))
    for record in run.history:
        if target is not None:
            record.setdefault("target_value", target)
    logger.info(f"[report] loaded {paper} from {run.origin}: {len(run.history)} curve point(s)")
    return run
