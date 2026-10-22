"""A numeric pass the review shows is not the paper's method must not read as reproduced.

Replayed from real reviews of 2026-10-22: Pang & Lee's regenerated script invented 1,400
keyword-planted "reviews" and passed at 92.8% against 82.9%; GCN and Kim CNN passed
honestly and must keep their passes.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from critic.blocking import pass_blocking_reasons
from orchestrator.loop import decide_after_critic
from report.render import ReportInputs, render

FIXTURES = Path(__file__).parent / "fixtures"


def review(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}_review.json").read_text())
    return data


def test_an_invented_dataset_blocks_the_pass() -> None:
    reasons = pass_blocking_reasons(review("pang_lee"))
    assert any("major deviations" in reason for reason in reasons)
    assert any("(data source)" in reason for reason in reasons)


@pytest.mark.parametrize("name", ["gcn", "kim_cnn"])
def test_honest_passes_are_not_blocked(name: str) -> None:
    # GCN carries a verified MEDIUM finding and an unverified one; neither may block.
    assert pass_blocking_reasons(review(name)) == []


def test_no_review_blocks_nothing() -> None:
    assert pass_blocking_reasons(None) == []


def decide(**overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "verdict": "pass",
        "recommendation": "none",
        "seeds_already_run": False,
        "can_run_seeds": True,
        "fidelity_retries_used": 1,
        "fidelity_retry_budget": 1,
        "retry_count": 1,
        "retry_budget": 3,
        "has_unstated_choices": True,
        "has_stated_problems": True,
        "pass_blocked": True,
    }
    kwargs.update(overrides)
    return decide_after_critic(**kwargs).action


def test_a_blocked_pass_is_fixed_on_the_normal_budget() -> None:
    assert decide() == "fix"  # the spent fidelity retry does not limit a correctness fix
    assert decide(retry_count=3) == "accept"
    assert decide(has_stated_problems=False) == "accept"
    assert decide(pass_blocked=False) == "accept"


def pass_state(review_name: str) -> dict[str, Any]:
    critic = {
        "verdict": "pass",
        "claim_id": "c5",
        "metric": "accuracy",
        "unit": "%",
        "claimed": 82.9,
        "reproduced": 92.7864,
        "tolerance": 1.401,
        "evidence": "measured spread of 3 runs",
        "run_values": [92.31, 93.78, 92.27],
        "reason": "the reproduced value is within tolerance of the claim",
        "review": review(review_name),
    }
    return {"paper_name": "paper", "verdict": "success", "attempts": [], "critic_output": critic}


def test_the_report_withholds_reproduced_on_a_blocked_pass() -> None:
    text = render(ReportInputs(state=pass_state("pang_lee")))
    assert "**not established**" in text and "**reproduced**" not in text
    assert "`pass` (blocked by the review)" in text
    assert "(data source)" in text


def test_the_report_keeps_reproduced_on_an_honest_pass() -> None:
    text = render(ReportInputs(state=pass_state("kim_cnn")))
    assert "**reproduced**" in text and "blocked" not in text
