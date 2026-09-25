"""Routing of a run halted by the Runner's live check-up."""

from orchestrator.loop import decide_after_run
from orchestrator.state import AttemptRecord, ReproState


def test_halted_run_retries_with_the_checkup_evidence_as_feedback() -> None:
    decision = decide_after_run(
        status="halted",
        triage=None,
        retry_count=0,
        retry_budget=2,
        halt_feedback="stopped at epoch 1: loss below its lower bound",
    )
    assert decision.action == "retry"
    assert decision.verdict is None
    assert decision.feedback == "stopped at epoch 1: loss below its lower bound"


def test_halted_run_without_feedback_still_gets_generic_guidance() -> None:
    decision = decide_after_run(status="halted", triage=None, retry_count=0, retry_budget=1)
    assert decision.action == "retry"
    assert decision.feedback


def test_halted_run_stops_when_the_budget_is_spent() -> None:
    decision = decide_after_run(status="halted", triage=None, retry_count=2, retry_budget=2)
    assert (decision.action, decision.verdict) == ("stop", "retry_budget_exhausted")


def test_state_written_before_checkups_still_loads() -> None:
    legacy_attempt = {
        "version": 1,
        "feedback_given": None,
        "script_path": "attempts/v1_train.py",
        "script_chars": 10,
        "plateau_ratio": None,
        "runner_status": "success",
        "stage_reached": "full",
        "failed_stage": None,
        "exit_code": 0,
        "wall_clock_seconds": 1.0,
        "triage_category": None,
        "triage_reasoning": None,
        "suggested_fix": None,
        "syntax_error": None,
        "logs_path": None,
        "action": "done",
        "verdict": "success",
        "reason": "ok",
    }
    state = ReproState.from_dict({"paper_id": "p", "attempts": [legacy_attempt]})
    assert isinstance(state.attempts[0], AttemptRecord)
    assert state.attempts[0].checkup_rule is None
