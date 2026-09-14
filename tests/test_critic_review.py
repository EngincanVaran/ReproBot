"""Critic v2's deterministic guards and routing, replayed on the first real review.

`fixtures/wijaya_review_payload.json` is the payload Sonnet returned on 2026-09-14 for
the Wijaya run, as delivered: its `findings` leaked inside `curve_assessment`.
"""

import json
from pathlib import Path
from typing import Any

from critic.judge import judge
from critic.review import (
    REVIEW_TOOL,
    Finding,
    Hypothesis,
    Review,
    _normalise,
    deviation_feedback,
    parse_review,
    unstated_feedback,
    unsupported_numbers,
    verify,
)
from orchestrator.loop import decide_after_critic
from reader.tooluse import recover_leaked_fields

FIXTURE = Path(__file__).parent / "fixtures" / "wijaya_review_payload.json"

SCRIPT = """\
import torch
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.001)
"""
PAPER = _normalise(
    "The author utilizes the Adam (Adaptive Moments) optimization method with a learning "
    "rate equal to 0.001. The model was trained for 1000 epochs. We performed standard "
    "normalization or z-score normalization."
)


def finding(**overrides: Any) -> Finding:
    base: dict[str, Any] = {
        "kind": "matches_paper",
        "aspect": "optimizer",
        "severity": "low",
        "paper_quote": "Adam (Adaptive Moments) optimization method with a learning rate",
        "script_lines": [4],
        "script_snippet": "optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)",
        "explanation": "Adam with lr 0.001 as stated.",
    }  # fmt: skip
    base.update(overrides)
    return Finding(**base)


def review_of(*findings: Finding, hypotheses: list[Hypothesis] | None = None) -> Review:
    return verify(
        Review("faithful", "summary", "curve", list(findings), hypotheses or []),
        script=SCRIPT,
        paper_sources=PAPER,
        facts_text='{"claimed": 3.02, "reproduced": 3.3292}',
    )


def test_real_leaked_findings_are_recovered() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "findings" not in payload
    recovered = recover_leaked_fields(payload, REVIEW_TOOL["input_schema"]["required"], "t")
    review = parse_review(recovered)
    assert len(review.findings) == 10
    assert "</curve_assessment>" not in review.curve_assessment
    assert {f.kind for f in review.findings} == {"matches_paper", "unstated_choice"}


def test_a_cited_finding_is_verified() -> None:
    assert review_of(finding()).findings[0].verified


def test_an_invented_number_is_caught() -> None:
    bad = finding(explanation="Adam with lr 0.003 as stated.")
    checked = review_of(bad).findings[0]
    assert not checked.verified
    assert any("0.003" in problem for problem in checked.problems)


def test_a_misquoted_paper_is_caught() -> None:
    bad = finding(paper_quote="we use SGD with momentum 0.9")
    assert not review_of(bad).findings[0].verified


def test_missing_script_lines_and_snippets_are_caught() -> None:
    assert not review_of(finding(script_lines=[99])).findings[0].verified
    assert not review_of(finding(script_snippet="optimizer = SGD(model)")).findings[0].verified


def test_elided_snippets_and_quotes_are_allowed() -> None:
    elided = finding(
        paper_quote="standard normalization ... z-score normalization",
        script_snippet="scaler = StandardScaler()\n...\nX_train = scaler.fit_transform(X_train)",
    )
    assert review_of(elided).findings[0].verified


def test_a_deviation_needs_a_quote_of_what_is_stated() -> None:
    bad = finding(kind="deviates_from_paper", paper_quote="not stated")
    assert not review_of(bad).findings[0].verified


def test_number_guard_readings() -> None:
    known = [3.02, 1000.0, 0.001, 237.0, 239.0]
    assert unsupported_numbers("claimed 3.0 after 1,000 epochs", known) == []
    assert unsupported_numbers("see [237,239]", known) == []
    assert unsupported_numbers("three levels, 2 layers", known) == []
    assert unsupported_numbers("at line 120", known, script_lines=364) == []
    assert unsupported_numbers("an RMSE of 2.5", known) == ["2.5"]


def test_only_verified_stated_problems_count() -> None:
    deviation = finding(
        kind="deviates_from_paper",
        aspect="preprocessing",
        severity="high",
        paper_quote="standard normalization or z-score normalization",
        script_lines=[2],
        script_snippet="scaler = StandardScaler()",
        explanation="The scaler is built but never applied.",
    )
    unverified = finding(kind="implementation_bug", severity="high", paper_quote="made up")
    review = review_of(deviation, unverified)
    assert [f.aspect for f in review.stated_problems()] == ["preprocessing"]

    wijaya = [{"claim_id": "c8", "metric": "RMSE", "reported_value": 3.02, "unit": ""}]
    judgement = judge(
        wijaya,
        {"claim_id": "c8", "value": 5.6, "higher_is_better": False, "num_eval_samples": 101},
        runner_status="success",
        metrics_mode="full",
        extra_run_values=[5.4, 5.8],
    )
    text = deviation_feedback(judgement, review)
    assert "The scaler is built but never applied." in text
    assert "made up" not in text
    assert "deviation fix:" in text
    assert "fidelity retry:" in unstated_feedback(judgement, review, ["batch size 32"])


def test_a_fail_with_stated_problems_routes_to_a_correctness_fix() -> None:
    common: dict[str, Any] = {
        "verdict": "fail",
        "recommendation": "guided_retry",
        "seeds_already_run": True,
        "can_run_seeds": True,
        "fidelity_retry_budget": 1,
        "retry_budget": 3,
        "has_unstated_choices": True,
    }
    fix = decide_after_critic(
        **common, fidelity_retries_used=1, retry_count=0, has_stated_problems=True
    )
    assert fix.action == "fix"  # not limited by the spent fidelity budget
    assert (
        decide_after_critic(
            **common, fidelity_retries_used=1, retry_count=3, has_stated_problems=True
        ).action
        == "accept"
    )
    assert decide_after_critic(**common, fidelity_retries_used=0, retry_count=0).action == "retry"
