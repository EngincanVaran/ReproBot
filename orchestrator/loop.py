"""The Coder<->Runner retry loop, and the pure functions that decide its transitions.

This is the piece the whole pipeline was built to make possible: `coder/` writes
a script but never runs it, `runner/` runs a script and classifies its failure
but never rewrites it, and until now nothing joined the two. The loop is

    generate -> run in Docker -> triage the failure -> regenerate WITH that
    feedback -> run again

bounded by a retry budget and cut short early by a plateau guard.

WHAT STOPS THE LOOP, AND WHY EACH ONE IS SEPARATE
-------------------------------------------------
* `success`                - a stage passed. Done.
* `environment_error`      - triage says the container or its inputs are at
                             fault. Regenerating the script cannot fix a broken
                             image, so retrying would burn the budget learning
                             nothing. THIS DISTINCTION IS THE POINT of spending a
                             triage call at all.
* `timeout`                - the run exceeded its wall-clock budget. `runner/`
                             deliberately produces NO triage for a timeout (its
                             cause is mechanical), so there is no `suggested_fix`
                             to feed back and nothing to tell the Coder to change.
                             Treating it as a code bug would send the Coder off to
                             "fix" a script that may be perfectly correct and
                             merely slow.
* `retry_budget_exhausted` - the budget ran out with the script still failing.
* `plateaued`              - the regenerated script is near-identical to the one
                             that just failed, so running it would produce the
                             same failure at Docker's price. Project plan §2.1
                             requires this guard explicitly, citing PaperBench's
                             finding that agents plateau rather than making
                             steady long-horizon progress.
* `untriaged_error`        - the run failed but no triage came back (triage
                             disabled, or no API key). Without a category there
                             is no routing decision to make, so the loop stops
                             rather than guessing which of the two it was.
* `coder_failed`           - the Coder itself raised. Not a script bug; a stage
                             failure.

A `ScriptSyntaxError` from `coder/`'s own `ast.parse` gate is NOT one of these.
That is a code bug the Coder can fix, it is detected before any container starts,
and the fix costs no Docker run - so the syntax message is fed straight back as
feedback and the loop continues. It still consumes one unit of retry budget,
because a model that emits invalid Python three times running is not converging.

PURE TRANSITIONS
----------------
Every decision above is made by one of three small pure functions
(`decide_after_run`, `decide_after_syntax_error`, `decide_after_regeneration`)
that take primitives and return a `LoopDecision`. They touch no state object, no
filesystem and no API, so the whole routing table is verifiable without Docker or
an API key - and a later migration to a LangGraph conditional edge is a matter of
calling the same function from a node (see `orchestrator/README.md` for why that
migration is deliberately not done yet).
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from loguru import logger

from coder.pipeline import CoderPipeline, ScriptSyntaxError
from orchestrator.state import (
    STAGE_CODER,
    STAGE_ORCHESTRATOR,
    STAGE_RUNNER,
    AttemptRecord,
    CoderOutputState,
    ReproState,
    RunnerOutputState,
    Verdict,
)
from runner.docker_runner import DockerRunner, RunnerOutput, metrics_path_for
from runner.triage import TriageResult

DEFAULT_RETRY_BUDGET = 3

# Two consecutive scripts at or above this line-level similarity count as a
# plateau. 0.98 is a STARTING POINT, not a measurement: the honest calibration
# data does not exist yet, which is why every comparison logs its actual ratio
# and why the threshold is a constructor parameter rather than a constant.
#
# What makes 0.98 defensible for THIS Coder: `TrainingScriptWriter` regenerates
# the whole file from scratch on every attempt, so two independent Sonnet
# generations of a ~430-line script differ in far more than 1% of their lines
# even when converging on the same design. Scoring above 0.98 therefore means the
# model reproduced its previous output almost verbatim - the plateau §2.1
# describes. If the Coder ever becomes patch-based, this threshold would be far
# too aggressive (a correct one-line fix scores ~0.998) and should move toward
# 0.999.
DEFAULT_PLATEAU_THRESHOLD = 0.98

# `diff_from_previous` is bookkeeping inside `state.json`, not the artifact of
# record - every version's full source is archived under `attempts/`. An
# unbounded diff of a 430-line rewrite would dominate the file it lives in.
MAX_DIFF_CHARS = 8000

type LoopAction = Literal["done", "stop", "retry", "run"]


@dataclass(frozen=True)
class LoopDecision:
    """What the loop does next, and the verdict/feedback that goes with it."""

    action: LoopAction
    verdict: Verdict | None
    reason: str
    feedback: str | None = None


# --------------------------------------------------------------------------- #
# Pure helpers. Nothing below this line touches Docker, the network, or the
# filesystem, so the entire routing table is verifiable without either.
# --------------------------------------------------------------------------- #


def normalise_lines(script: str) -> list[str]:
    """Reduce a script to the lines that carry meaning, for similarity scoring.

    Trailing whitespace and blank lines are removed because they are the two
    things that differ between two otherwise identical generations without
    changing what the script does - counting them would let a purely cosmetic
    reflow hide a genuine plateau.
    """
    return [stripped for line in script.splitlines() if (stripped := line.rstrip())]


def script_similarity(previous: str, current: str) -> float:
    """Similarity of two scripts in [0, 1], compared LINE by line.

    Two decisions worth stating, because both are traps:

    1. **Lines, not characters.** A line is the unit a code change happens in,
       and it is the same unit `unified_diff` reports, so the ratio and the diff
       in `state.json` describe the same thing.
    2. **`autojunk=False`.** `SequenceMatcher`'s default heuristic treats any
       element appearing in more than 1% of a >=200-element sequence as junk. On
       a long script that silently swallows every common line (`return`, a bare
       `)`, blank scaffolding) and returns a ratio that means nothing. Off, the
       comparison is exact.
    """
    matcher = difflib.SequenceMatcher(
        None, normalise_lines(previous), normalise_lines(current), autojunk=False
    )
    return matcher.ratio()


def make_diff(
    previous: str | None,
    current: str,
    *,
    from_version: int,
    to_version: int,
    max_chars: int = MAX_DIFF_CHARS,
) -> str | None:
    """Unified diff between two attempts' scripts, or None for the first attempt.

    Diffed over the RAW lines, not the normalised ones: this is an audit artifact
    describing what actually changed on disk, so a whitespace-only edit should be
    visible in it even though the similarity score deliberately ignores one.
    """
    if previous is None:
        return None
    diff = "\n".join(
        difflib.unified_diff(
            previous.splitlines(),
            current.splitlines(),
            fromfile=f"v{from_version}_train.py",
            tofile=f"v{to_version}_train.py",
            lineterm="",
        )
    )
    if not diff:
        return "(no textual difference)"
    if len(diff) <= max_chars:
        return diff
    dropped = len(diff) - max_chars
    return f"{diff[:max_chars]}\n...[{dropped} chars of diff truncated]..."


def feedback_from_triage(triage: TriageResult) -> str:
    """The text handed to `CodeWriter.write(feedback=...)` on a recoverable failure.

    `suggested_fix` is written by `runner/triage.py` specifically to be passed
    verbatim, so it is used as-is when present. It can come back empty even for a
    `recoverable_error`, though - the schema requires the field, not that the
    model fills it - and losing an entire retry to an empty string would be a
    silly way to fail, so the reasoning is used as a fallback.
    """
    if triage.suggested_fix:
        return triage.suggested_fix
    logger.warning(
        "  [decide] triage returned a recoverable_error with an EMPTY suggested_fix; "
        "falling back to its reasoning as the feedback text"
    )
    return (
        f"The previous script failed when executed. Triage reasoning: {triage.reasoning}"
        if triage.reasoning
        else "The previous script failed when executed, with no further detail available."
    )


def decide_after_run(
    *,
    status: str,
    triage: TriageResult | None,
    retry_count: int,
    retry_budget: int,
) -> LoopDecision:
    """Route one finished Docker run. The loop's central decision.

    Checked in this order on a failure, and the order is load-bearing:
    environment first (a retry cannot help), then budget, then the retry itself.
    The plateau guard is deliberately NOT here - it compares the *regenerated*
    script against its predecessor, which does not exist yet at this point.
    """
    if status == "success":
        return LoopDecision("done", "success", "a stage passed; nothing to retry")
    if status == "timeout":
        return LoopDecision(
            "stop",
            "timeout",
            "the run exceeded its wall-clock budget; runner/ produces no triage for a "
            "timeout by design, so there is no specific fix to feed back - a slow but "
            "correct script must not be 'fixed' into a different one",
        )
    if status != "error":
        raise ValueError(f"unknown runner status {status!r}; expected success/error/timeout")

    if triage is None:
        return LoopDecision(
            "stop",
            "untriaged_error",
            "the run failed but no triage was produced (triage disabled, or no API key), "
            "so there is no category to route on",
        )
    if triage.category == "environment_error":
        return LoopDecision(
            "stop",
            "environment_error",
            f"triage says the container or its inputs are at fault, not the script, so "
            f"regenerating it would change nothing: {triage.reasoning}",
        )
    if retry_count >= retry_budget:
        return LoopDecision(
            "stop",
            "retry_budget_exhausted",
            f"the failure is recoverable but the retry budget ({retry_budget}) is spent",
        )
    return LoopDecision(
        "retry",
        None,
        f"triage says the fault is in the generated script: {triage.reasoning}",
        feedback=feedback_from_triage(triage),
    )


def decide_after_syntax_error(
    *,
    message: str,
    retry_count: int,
    retry_budget: int,
) -> LoopDecision:
    """Route a `ScriptSyntaxError` raised by coder/'s own `ast.parse` gate.

    Always a code bug the Coder can fix, and one detected before any container
    starts - so this retry costs nothing but an API call. It still consumes
    budget: a model emitting invalid Python repeatedly is not converging either.
    """
    if retry_count >= retry_budget:
        return LoopDecision(
            "stop",
            "retry_budget_exhausted",
            f"the generated script does not parse and the retry budget ({retry_budget}) is spent",
        )
    return LoopDecision(
        "retry",
        None,
        "the generated script is not valid Python - caught by coder/'s ast.parse gate "
        "before any container started, so this retry costs no Docker run",
        feedback=(
            f"The script you generated was not valid Python and was rejected before it "
            f"could run: {message}. Emit a COMPLETE, syntactically valid script this time "
            f"- no truncation, no placeholders, no unclosed brackets or strings."
        ),
    )


def decide_after_regeneration(*, similarity: float, threshold: float) -> LoopDecision:
    """Plateau guard: is the regenerated script actually different from the last one?

    Project plan §2.1's explicit requirement, and the reason it runs BEFORE the
    Docker run rather than after: the whole saving is not spending a container on
    a "fix" that changed nothing.
    """
    if similarity >= threshold:
        return LoopDecision(
            "stop",
            "plateaued",
            f"the regenerated script is {similarity:.4f} similar to the one that just "
            f"failed (threshold {threshold}); running it would reproduce the same "
            f"failure at Docker's price",
        )
    return LoopDecision(
        "run",
        None,
        f"the regenerated script differs enough to be worth running "
        f"(similarity {similarity:.4f} < threshold {threshold})",
    )


def build_attempt(
    *,
    version: int,
    feedback: str | None,
    script_path: Path,
    script_text: str,
    decision: LoopDecision,
    plateau_ratio: float | None = None,
    runner_output: RunnerOutput | None = None,
    syntax_error: str | None = None,
) -> AttemptRecord:
    """Flatten one attempt's inputs, outcome and decision into a single record."""
    triage = runner_output.triage if runner_output else None
    return AttemptRecord(
        version=version,
        feedback_given=feedback,
        script_path=str(script_path),
        script_chars=len(script_text),
        plateau_ratio=plateau_ratio,
        runner_status=runner_output.status if runner_output else None,
        stage_reached=runner_output.stage_reached if runner_output else None,
        failed_stage=runner_output.failed_stage if runner_output else None,
        exit_code=runner_output.exit_code if runner_output else None,
        wall_clock_seconds=runner_output.wall_clock_seconds if runner_output else None,
        triage_category=triage.category if triage else None,
        triage_reasoning=triage.reasoning if triage else None,
        suggested_fix=triage.suggested_fix if triage else None,
        syntax_error=syntax_error,
        logs_path=runner_output.logs_path if runner_output else None,
        action=decision.action,
        verdict=decision.verdict,
        reason=decision.reason,
    )


# --------------------------------------------------------------------------- #
# The loop itself.
# --------------------------------------------------------------------------- #


class Orchestrator:
    """Drives generate -> run -> triage -> regenerate over one paper.

    Owns no mechanics of its own: `CoderPipeline` writes the script and runs its
    gates, `DockerRunner` executes it and classifies the failure, and this class
    only decides what happens next and records it in the shared-memory object.
    """

    def __init__(
        self,
        coder: CoderPipeline,
        runner: DockerRunner,
        modes: tuple[str, ...] = ("probe",),
        retry_budget: int = DEFAULT_RETRY_BUDGET,
        plateau_threshold: float = DEFAULT_PLATEAU_THRESHOLD,
    ) -> None:
        self.coder = coder
        self.runner = runner
        self.modes = modes
        self.retry_budget = retry_budget
        self.plateau_threshold = plateau_threshold

    # -- workspace hygiene -------------------------------------------------- #

    def clear_stale_metrics(self, paper_dir: Path) -> None:
        """Delete `metrics.<mode>.json` before a run, for every mode about to run.

        Single-shot `runner/` never had to think about this; a loop does. The
        metrics file lives in the bind-mounted paper directory and survives the
        container, so attempt 2 crashing before it writes one would let
        `read_metrics_file` hand back ATTEMPT 1's numbers as if they were this
        run's. Removing them first makes "no metrics" mean no metrics.
        """
        for mode in self.modes:
            stale = metrics_path_for(paper_dir, mode)
            if stale.exists():
                stale.unlink()
                logger.info(f"  [workspace] removed stale {stale.name} from a previous attempt")

    # -- the loop ----------------------------------------------------------- #

    def run(
        self,
        *,
        reader_json_path: Path,
        markdown_path: Path,
        coder_output_dir: Path,
        output_dir: Path,
        client: Anthropic,
        source_pdf: Path | None = None,
        use_existing_script: bool = False,
    ) -> ReproState:
        """Run the loop over one paper and return its shared-memory object.

        `use_existing_script` seeds attempt 1 from whatever `train.py` is already
        in the paper directory instead of generating one. That is how you re-drive
        the loop over a script you already have (or a deliberately broken one)
        without paying for a regeneration you do not need; every RETRY still
        regenerates normally.
        """
        paper_id = reader_json_path.stem
        paper_dir = coder_output_dir / paper_id
        attempts_dir = output_dir / "attempts"
        attempts_dir.mkdir(parents=True, exist_ok=True)

        state = ReproState(
            paper_id=paper_id,
            source_pdf=str(source_pdf) if source_pdf else None,
            reader_output=json.loads(reader_json_path.read_text(encoding="utf-8")),
            retry_budget=self.retry_budget,
        )
        logger.info(f"[loop] {paper_id}")
        logger.info(
            f"[loop] retry budget {self.retry_budget}, plateau threshold "
            f"{self.plateau_threshold}, runner stages {' -> '.join(self.modes)}"
        )
        state.record(
            STAGE_ORCHESTRATOR,
            "loop started",
            f"retry_budget={self.retry_budget}, plateau_threshold={self.plateau_threshold}, "
            f"modes={'/'.join(self.modes)}",
        )

        version = 1
        feedback: str | None = None
        previous_script: str | None = None

        while True:
            logger.info(
                f"[attempt {version}] retry_count={state.retry_count}/{state.retry_budget}"
                + (f", feedback: {feedback}" if feedback else ", no feedback (first attempt)")
            )

            # ---- CODER ---------------------------------------------------- #
            syntax_error: str | None = None
            if version == 1 and use_existing_script:
                script_path = paper_dir / "train.py"
                if not script_path.exists():
                    raise FileNotFoundError(
                        f"--use-existing-script was given but there is no {script_path}. "
                        f"Generate one first with `uv run python -m coder.pipeline`, or "
                        f"drop the flag to have the loop generate attempt 1 itself."
                    )
                script_text = script_path.read_text(encoding="utf-8")
                logger.info(
                    f"  [coder] reusing the existing script at {script_path} "
                    f"({len(script_text.splitlines())} lines) - generation skipped for "
                    f"attempt 1 only"
                )
                state.record(STAGE_CODER, f"reused existing script as v{version}", str(script_path))
            else:
                try:
                    coder_output = self.coder.run(
                        reader_json_path,
                        markdown_path,
                        coder_output_dir,
                        client,
                        feedback=feedback,
                    )
                    script_path = Path(coder_output.script_path)
                    script_text = script_path.read_text(encoding="utf-8")
                    state.record(
                        STAGE_CODER,
                        f"generated script v{version}",
                        f"claim={coder_output.result.claim_targeted}, "
                        f"{len(script_text.splitlines())} lines"
                        + (f", feedback applied: {feedback}" if feedback else ""),
                    )
                except ScriptSyntaxError as exc:
                    syntax_error = str(exc)
                    script_path = paper_dir / "train.py.invalid"
                    script_text = (
                        script_path.read_text(encoding="utf-8") if script_path.exists() else ""
                    )
                    logger.error(f"  [coder] syntax gate rejected v{version}: {syntax_error}")
                    state.record(STAGE_CODER, f"script v{version} failed the syntax gate", str(exc))
                except Exception as exc:  # noqa: BLE001 - a stage failure, not a script bug
                    logger.error(f"  [coder] the Coder stage itself failed: {exc}")
                    decision = LoopDecision(
                        "stop", "coder_failed", f"the Coder stage raised: {exc}"
                    )
                    state.attempts.append(
                        build_attempt(
                            version=version,
                            feedback=feedback,
                            script_path=paper_dir / "train.py",
                            script_text="",
                            decision=decision,
                        )
                    )
                    self._finish(state, decision)
                    break

            # ---- archive + diff ------------------------------------------- #
            suffix = ".invalid" if syntax_error else ""
            archived = attempts_dir / f"v{version}_train.py{suffix}"
            archived.write_text(script_text, encoding="utf-8")
            logger.info(f"  [archive] v{version} -> {archived}")

            diff = make_diff(
                previous_script, script_text, from_version=version - 1, to_version=version
            )
            state.coder_output = CoderOutputState(
                script_path=str(script_path),
                script_version=version,
                diff_from_previous=diff,
            )
            if diff is not None:
                logger.info(
                    f"  [diff] v{version - 1} -> v{version}: {len(diff.splitlines())} diff "
                    f"line(s), {len(diff)} chars recorded in state.coder_output"
                )

            # ---- plateau guard (before spending a container) --------------- #
            plateau_ratio: float | None = None
            if previous_script is not None:
                plateau_ratio = script_similarity(previous_script, script_text)
                logger.info(
                    f"  [plateau] v{version - 1} vs v{version} line similarity "
                    f"{plateau_ratio:.4f} (stop at >= {self.plateau_threshold})"
                )
                decision = decide_after_regeneration(
                    similarity=plateau_ratio, threshold=self.plateau_threshold
                )
                if decision.action == "stop":
                    state.attempts.append(
                        build_attempt(
                            version=version,
                            feedback=feedback,
                            script_path=archived,
                            script_text=script_text,
                            decision=decision,
                            plateau_ratio=plateau_ratio,
                            syntax_error=syntax_error,
                        )
                    )
                    self._finish(state, decision)
                    break
                logger.info(f"  [plateau] {decision.reason}")

            # ---- syntax failure: retry without touching Docker ------------- #
            if syntax_error is not None:
                decision = decide_after_syntax_error(
                    message=syntax_error,
                    retry_count=state.retry_count,
                    retry_budget=state.retry_budget,
                )
                state.attempts.append(
                    build_attempt(
                        version=version,
                        feedback=feedback,
                        script_path=archived,
                        script_text=script_text,
                        decision=decision,
                        plateau_ratio=plateau_ratio,
                        syntax_error=syntax_error,
                    )
                )
                if decision.action == "stop":
                    self._finish(state, decision)
                    break
                feedback, previous_script, version = self._next_attempt(
                    state, decision, script_text, version
                )
                continue

            # ---- RUNNER --------------------------------------------------- #
            logs_dir = output_dir / "logs" / f"attempt-{version}"
            self.clear_stale_metrics(paper_dir)
            runner_output = self.runner.run_paper(paper_dir, logs_dir, self.modes, client)
            state.runner_output = RunnerOutputState(
                status=runner_output.status,
                reproduced_metrics=runner_output.reproduced_metrics,
                logs_path=runner_output.logs_path,
                error_trace=runner_output.error_trace,
            )
            state.record(
                STAGE_RUNNER,
                f"ran v{version}: {runner_output.status}",
                f"stage_reached={runner_output.stage_reached}, "
                f"failed_stage={runner_output.failed_stage}, "
                f"exit_code={runner_output.exit_code}, "
                f"{runner_output.wall_clock_seconds}s"
                + (
                    f", triage={runner_output.triage.category}"
                    if runner_output.triage
                    else ", no triage"
                ),
            )

            decision = decide_after_run(
                status=runner_output.status,
                triage=runner_output.triage,
                retry_count=state.retry_count,
                retry_budget=state.retry_budget,
            )
            state.attempts.append(
                build_attempt(
                    version=version,
                    feedback=feedback,
                    script_path=archived,
                    script_text=script_text,
                    decision=decision,
                    plateau_ratio=plateau_ratio,
                    runner_output=runner_output,
                )
            )

            if decision.action in ("done", "stop"):
                self._finish(state, decision)
                break
            feedback, previous_script, version = self._next_attempt(
                state, decision, script_text, version
            )

        return state

    # -- transition bookkeeping --------------------------------------------- #

    def _next_attempt(
        self, state: ReproState, decision: LoopDecision, script_text: str, version: int
    ) -> tuple[str | None, str, int]:
        """Charge one unit of retry budget and return the next attempt's inputs."""
        state.retry_count += 1
        next_version = version + 1
        logger.warning(f"  [decide] RETRY - {decision.reason}")
        logger.warning(f"  [decide] feedback to the Coder: {decision.feedback}")
        state.record(
            STAGE_ORCHESTRATOR,
            f"retry {state.retry_count}/{state.retry_budget} -> v{next_version}",
            decision.feedback,
        )
        return decision.feedback, script_text, next_version

    def _finish(self, state: ReproState, decision: LoopDecision) -> None:
        """Stamp the terminal verdict onto the state and log why the loop stopped."""
        state.verdict = decision.verdict
        if decision.verdict == "success":
            logger.info(f"[loop] VERDICT: {decision.verdict}")
            logger.info(f"[loop] reason: {decision.reason}")
        else:
            logger.warning(f"[loop] VERDICT: {decision.verdict}")
            logger.warning(f"[loop] reason: {decision.reason}")
        logger.info(
            f"[loop] finished after {len(state.attempts)} attempt(s), "
            f"{state.retry_count}/{state.retry_budget} retries used"
        )
        state.record(
            STAGE_ORCHESTRATOR,
            f"loop finished: {decision.verdict}",
            decision.reason,
        )
