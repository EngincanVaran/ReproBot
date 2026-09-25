"""Routing of Critic verdicts: when the loop accepts, runs extra seeds, or retries."""

from typing import Any

import pytest

from orchestrator.loop import decide_after_critic


def decide(verdict: str, recommendation: str = "none", **overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "seeds_already_run": False,
        "can_run_seeds": True,
        "fidelity_retries_used": 0,
        "fidelity_retry_budget": 1,
        "retry_count": 0,
        "retry_budget": 3,
        "has_unstated_choices": True,
    }
    kwargs.update(overrides)
    return decide_after_critic(verdict=verdict, recommendation=recommendation, **kwargs).action


@pytest.mark.parametrize("verdict", ["pass", "not_evaluated"])
def test_pass_and_not_evaluated_are_final(verdict: str) -> None:
    assert decide(verdict) == "accept"


def test_inconclusive_runs_seeds_once() -> None:
    assert decide("inconclusive", "run_more_seeds") == "run_seeds"
    assert decide("inconclusive", "run_more_seeds", seeds_already_run=True) == "accept"


def test_inconclusive_accepts_when_seeds_are_unaffordable() -> None:
    assert decide("inconclusive", "run_more_seeds", can_run_seeds=False) == "accept"


def test_fail_earns_one_guided_retry() -> None:
    assert decide("fail", "guided_retry") == "retry"


@pytest.mark.parametrize(
    "overrides",
    [
        {"fidelity_retries_used": 1},  # the one fidelity retry is spent
        {"retry_count": 3},  # the general retry budget is spent
        {"has_unstated_choices": False},  # nothing unstated to revisit
    ],
)
def test_fail_is_final_without_room_to_retry(overrides: dict[str, Any]) -> None:
    assert decide("fail", "guided_retry", **overrides) == "accept"
