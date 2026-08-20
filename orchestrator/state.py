"""`ReproState` - the shared-memory object every stage reads and writes.

This is the project plan's §1.2 schema made concrete: one structured object per
paper, threaded through the whole pipeline, holding `reader_output`,
`coder_output`, `runner_output`, `critic_output`, the retry counters and an
append-only `history`. §1.2 is explicit about *why* it is one object rather than
scattered state: it is what lets the Orchestrator answer "have we already tried
this" and lets a future Report Generator produce a claim-by-claim table without
re-deriving anything.

Three deliberate decisions about how faithfully this mirrors the plan:

1. **`critic_output` exists and is always `None`.** The Critic is not built. The
   field is here so the schema does not change shape when it lands - a consumer
   written today against this JSON keeps working. Nothing in `orchestrator/`
   writes it, and nothing compares a reproduced number to a claimed one.
2. **`history` entries are structured, not free strings.** §1.2 sketches
   `history` as a list of prose strings. Splitting each entry into
   `timestamp` / `stage` / `event` / `detail` is the same append-only audit log
   with its fields separated, which is what makes "have we already tried this"
   answerable by a query instead of a substring search.
3. **`attempts` and `verdict` are additions, not in §1.2.** §1.2's `history` is
   narrative; `attempts` is the per-attempt *record* the loop actually routes on
   (which version ran, what feedback produced it, what the Runner said, how the
   triage classified it, how similar it was to the previous attempt). `verdict`
   is the loop's own terminal answer, distinct from the Critic's future
   `critic_output.verdict`, which judges numbers rather than execution.

Serialisation is plain `dataclasses.asdict` out and an explicit `from_dict` back,
so `state.json` round-trips exactly. Nothing here imports `anthropic`, `docker`,
or any pipeline stage - this module is pure data and is exercisable on its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

# The loop's terminal answers. Deliberately NOT the Critic's pass/retry/fail
# vocabulary from §2.5 - these describe how far execution got, not whether a
# reproduced number matched a claimed one. Keeping the two vocabularies separate
# is what stops a `success` here from ever being mistaken for a reproduction.
type Verdict = Literal[
    "success",
    "environment_error",
    "retry_budget_exhausted",
    "plateaued",
    "timeout",
    "untriaged_error",
    "coder_failed",
]

STAGE_ORCHESTRATOR = "orchestrator"
STAGE_CODER = "coder"
STAGE_RUNNER = "runner"


@dataclass
class CoderOutputState:
    """§1.2's `coder_output`. All three fields are populated for real by the loop.

    `script_version` increments once per generated attempt and
    `diff_from_previous` is a truncated unified diff against the previous
    attempt's script - both were emitted inert (`1` / `null`) by
    `coder/pipeline.py` because a single-shot stage has no previous version to
    diff against.
    """

    script_path: str
    script_version: int
    diff_from_previous: str | None


@dataclass
class RunnerOutputState:
    """§1.2's `runner_output` - the four fields the plan names, no more.

    `runner/`'s own `RunnerOutput` carries far more (per-stage timings, container
    names, an 8000-char log excerpt, the full triage result). Copying all of it
    in here would make `state.json` unreadable and duplicate a file that already
    exists per attempt. The extra detail that the loop actually routes on is
    lifted into `AttemptRecord` instead, and `logs_path` points at the rest.
    """

    status: str
    reproduced_metrics: dict[str, Any] | None
    logs_path: str
    error_trace: str | None


@dataclass
class AttemptRecord:
    """One pass through generate -> run, and everything the loop decided from it.

    `script_path` is the ARCHIVED copy under `attempts/`, not the live
    `coder/output/<paper>/train.py`, which the next attempt overwrites. Keeping
    every version is what makes a plateau claim checkable after the fact rather
    than something you have to take the ratio's word for.
    """

    version: int
    feedback_given: str | None
    script_path: str
    script_chars: int
    plateau_ratio: float | None
    runner_status: str | None
    stage_reached: str | None
    failed_stage: str | None
    exit_code: int | None
    wall_clock_seconds: float | None
    triage_category: str | None
    triage_reasoning: str | None
    suggested_fix: str | None
    syntax_error: str | None
    logs_path: str | None
    action: str
    verdict: str | None
    reason: str


@dataclass
class HistoryEntry:
    """One stage transition, appended and never mutated."""

    timestamp: str
    stage: str
    event: str
    detail: str | None = None


@dataclass
class ReproState:
    """The whole shared-memory object for one paper.

    Field order and names follow §1.2. `retry_budget` counts *regenerations*, not
    total attempts: a budget of 2 means at most attempt 1 plus two retries.
    """

    paper_id: str
    source_pdf: str | None = None
    reader_output: dict[str, Any] | None = None
    coder_output: CoderOutputState | None = None
    runner_output: RunnerOutputState | None = None
    # Typed as a dict rather than `None` on purpose: when the Critic lands it
    # fills this in and the annotation does not have to change. Nothing writes
    # it today - see this module's docstring.
    critic_output: dict[str, Any] | None = None
    retry_count: int = 0
    retry_budget: int = 3
    history: list[HistoryEntry] = field(default_factory=list)
    verdict: Verdict | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)

    # -- history ------------------------------------------------------------ #

    def record(self, stage: str, event: str, detail: str | None = None) -> HistoryEntry:
        """Append one transition to the audit log and return it.

        The only way `history` is ever written. Timestamps are UTC and
        second-resolution: this log is read by a human reconstructing a run, not
        used for profiling (per-attempt wall clock lives in `AttemptRecord`).
        """
        entry = HistoryEntry(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            stage=stage,
            event=event,
            detail=detail,
        )
        self.history.append(entry)
        return entry

    # -- (de)serialisation -------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        """Rebuild a state from `to_dict()` output, reconstructing nested dataclasses.

        Written out explicitly rather than reached for via a serialisation
        library: the nesting is three shallow dataclasses deep, and an explicit
        constructor is the thing that actually fails loudly when a future field
        is added to one side and not the other.
        """
        coder_output = payload.get("coder_output")
        runner_output = payload.get("runner_output")
        return cls(
            paper_id=str(payload["paper_id"]),
            source_pdf=_optional_str(payload.get("source_pdf")),
            reader_output=_optional_dict(payload.get("reader_output")),
            coder_output=CoderOutputState(**coder_output) if coder_output else None,
            runner_output=RunnerOutputState(**runner_output) if runner_output else None,
            critic_output=_optional_dict(payload.get("critic_output")),
            retry_count=int(payload.get("retry_count", 0)),
            retry_budget=int(payload.get("retry_budget", 3)),
            history=[HistoryEntry(**entry) for entry in payload.get("history", [])],
            verdict=payload.get("verdict"),
            attempts=[AttemptRecord(**attempt) for attempt in payload.get("attempts", [])],
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"expected an object or null, got {type(value).__name__}")
    return value
