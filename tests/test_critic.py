"""The Critic's verdicts on ReproBot's real phase-3 results, and on edge cases.

Claims and reproduced values are copied from the actual Reader outputs and metrics
files; the expected verdicts are the ones a person reached comparing them by hand.
"""

from typing import Any

import pytest

from critic.claims import group_claims, reporting_precision
from critic.judge import guided_retry_feedback, judge


def claim(
    claim_id: str,
    metric: str,
    value: float,
    unit: str,
    *,
    dataset: str = "D",
    variant: str | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "metric": metric,
        "dataset": dataset,
        "reported_value": value,
        "unit": unit,
        "model_variant": variant,
    }


def metrics(
    claim_id: str, value: float, *, hib: bool | None, n: int | None, **extra: Any
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "value": value,
        "higher_is_better": hib,
        "num_eval_samples": n,
        **extra,
    }


# Wijaya 2023's 14 claims, as the Reader extracted them.
BH = "Boston Housing"
WIJAYA = [
    claim("c1", "R-Squared", 0.948, "", dataset=BH, variant="training set"),
    claim("c2", "R-Squared", 0.911, "", dataset=BH, variant="testing set"),
    claim("c3", "MAE", 1.99, "", dataset=BH, variant="training set"),
    claim("c4", "MAE", 2.31, "", dataset=BH, variant="testing set"),
    claim("c5", "MSE", 7.24, "", dataset=BH, variant="training set"),
    claim("c6", "MSE", 9.16, "", dataset=BH, variant="testing set"),
    claim("c7", "RMSE", 2.69, "", dataset=BH, variant="training set"),
    claim("c8", "RMSE", 3.02, "", dataset=BH, variant="testing set"),
    claim("c9", "R2", 0.91, "", dataset=BH, variant="Proposed NN, Table 3"),
    claim("c10", "MSE", 9.16, "", dataset=BH, variant="Proposed NN, Table 3"),
    claim("c11", "RMSE", 3.02, "", dataset=BH, variant="Proposed NN, Table 3"),
    claim("c12", "MAE", 2.31, "", dataset=BH, variant="Proposed NN, Table 3"),
    claim("c13", "MSE", 9.16, "", dataset=BH, variant="Proposed NN Model, Figure 6"),
    claim("c14", "R-square", 0.91, "", dataset=BH, variant="Proposed NN Model, Figure 6"),
]  # fmt: skip


# --------------------------------------------------------------------------- #
# Claim merging
# --------------------------------------------------------------------------- #


def test_wijaya_fourteen_claims_are_eight_results() -> None:
    groups = group_claims(WIJAYA)
    merged = sorted(g.claim_ids for g in groups if len(g.claim_ids) > 1)
    assert len(groups) == 8
    assert merged == [["c2", "c9", "c14"], ["c4", "c12"], ["c6", "c10", "c13"], ["c8", "c11"]]


def test_merge_keeps_the_most_precise_value() -> None:
    group = next(g for g in group_claims(WIJAYA) if "c2" in g.claim_ids)
    assert group.canonical["reported_value"] == 0.911


def test_different_models_with_equal_values_stay_apart() -> None:
    # Fashion-MNIST: LinearSVC and LogisticRegression both reach 0.917 on MNIST.
    rows = [
        claim("c8", "Test Accuracy", 0.917, "", dataset="MNIST", variant="LinearSVC hinge C=1"),
        claim("c10", "Test Accuracy", 0.917, "", dataset="MNIST", variant="LogisticRegression C=1"),
    ]  # fmt: skip
    assert len(group_claims(rows)) == 2
    # Wide Residual Networks: two architectures that both report 5.78%.
    rows = [
        claim("c2", "test error", 5.78, "%", dataset="CIFAR-10", variant="B(3,1), depth 40, k=2"),
        claim("c6", "test error", 5.78, "%", dataset="CIFAR-10", variant="B(3,1,3), depth 22, k=2"),
    ]  # fmt: skip
    assert len(group_claims(rows)) == 2


def test_train_and_test_values_never_merge() -> None:
    rows = [
        claim("a", "RMSE", 3.02, "", variant="training set"),
        claim("b", "RMSE", 3.02, "", variant="testing set"),
    ]
    assert len(group_claims(rows)) == 2


def test_reporting_precision() -> None:
    assert reporting_precision(96.9) == pytest.approx(0.05)
    assert reporting_precision(0.873) == pytest.approx(0.0005)
    assert reporting_precision(3.02) == pytest.approx(0.005)


# --------------------------------------------------------------------------- #
# Verdicts on the real phase-3 results
# --------------------------------------------------------------------------- #


def test_tang_passes_within_binomial_noise() -> None:
    claims = [claim("c5", "test error", 0.87, "%", dataset="MNIST", variant="DLSVM")]
    j = judge(
        claims,
        metrics("c5", 0.82, hib=False, n=10000),
        runner_status="success",
        metrics_mode="full",
    )
    assert j.verdict == "pass" and not j.exceeds_claim
    assert j.tolerance == pytest.approx(0.1857, abs=1e-3)


@pytest.mark.parametrize("reproduced", [96.925, 96.625])
def test_svm_guide_passes_both_runs(reproduced: float) -> None:
    claims = [claim("c1", "Accuracy by our procedure", 96.9, "%", dataset="Astroparticle")]
    j = judge(
        claims,
        metrics("c1", reproduced, hib=True, n=4000),
        runner_status="success",
        metrics_mode="full",
    )
    assert j.verdict == "pass"


def test_random_forest_mean_of_five_exceeds_claim() -> None:
    claims = [claim("c17", "Test Accuracy", 0.873, "", dataset="Fashion-MNIST")]
    runs = [0.8773, 0.8774, 0.8753, 0.8792, 0.8775]
    j = judge(
        claims,
        metrics("c17", 0.8773, hib=True, n=10000, num_runs=5, run_values=runs),
        runner_status="success",
        metrics_mode="full",
    )
    assert j.verdict == "pass" and j.exceeds_claim
    assert j.evidence.startswith("measured spread of 5 runs")


def test_soft_tree_above_claim_passes() -> None:
    claims = [claim("c1", "test accuracy", 94.45, "%", dataset="MNIST")]
    j = judge(
        claims,
        metrics("c1", 95.11, hib=True, n=10000),
        runner_status="success",
        metrics_mode="full",
    )
    assert j.verdict == "pass" and j.exceeds_claim


def test_wijaya_single_run_is_inconclusive_and_asks_for_seeds() -> None:
    j = judge(
        WIJAYA,
        metrics("c11", 4.48182, hib=False, n=101),
        runner_status="success",
        metrics_mode="full",
    )
    assert (j.verdict, j.recommendation) == ("inconclusive", "run_more_seeds")
    assert j.merged_claim_ids == ["c8", "c11"]


def test_wijaya_with_seeds_that_agree_fails_and_asks_for_a_retry() -> None:
    # Constructed: seeds that agree with the old 4.48 run (no real run did this).
    j = judge(
        WIJAYA,
        metrics("c11", 4.48, hib=False, n=101),
        runner_status="success",
        metrics_mode="full",
        extra_run_values=[4.31, 4.62],
    )
    assert (j.verdict, j.recommendation) == ("fail", "guided_retry")
    assert len(j.run_values) == 3


def test_wijaya_real_three_seed_run_passes() -> None:
    # The Orchestrator's end-to-end run on 2026-09-14: full (seed 42) gave 3.7516, the
    # Critic judged it inconclusive, and seed2/seed3 (each also redraws the unstated
    # split) gave 2.8350 and 3.4009.
    j = judge(
        WIJAYA,
        metrics("c8", 3.7516193831964477, hib=False, n=101),
        runner_status="success",
        metrics_mode="full",
        extra_run_values=[2.835043960493163, 3.400903107780038],
    )
    assert j.verdict == "pass" and not j.exceeds_claim
    assert j.reproduced == pytest.approx(3.3292, abs=1e-4)
    assert j.tolerance == pytest.approx(1.0680, abs=1e-3)
    assert j.merged_claim_ids == ["c8", "c11"]


# --------------------------------------------------------------------------- #
# Nothing to judge
# --------------------------------------------------------------------------- #


def test_halted_or_crashed_runs_are_not_evaluated() -> None:
    for status in ("halted", "error", "timeout"):
        j = judge(
            WIJAYA,
            metrics("c11", 4.48, hib=False, n=101),
            runner_status=status,
            metrics_mode="full",
        )
        assert j.verdict == "not_evaluated"


def test_cheap_stage_numbers_are_not_evaluated() -> None:
    j = judge(
        WIJAYA,
        metrics("c11", 6.11, hib=False, n=101),
        runner_status="success",
        metrics_mode="capped",
    )
    assert j.verdict == "not_evaluated"


def test_unknown_claim_is_not_evaluated() -> None:
    j = judge(
        WIJAYA,
        metrics("c99", 3.0, hib=False, n=101),
        runner_status="success",
        metrics_mode="full",
    )
    assert j.verdict == "not_evaluated"


def test_guided_retry_feedback_lists_only_unstated_choices() -> None:
    j = judge(
        WIJAYA,
        metrics("c11", 4.48, hib=False, n=101),
        runner_status="success",
        metrics_mode="full",
        extra_run_values=[4.31, 4.62],
    )
    text = guided_retry_feedback(j, ["split method unstated - used a seeded random split"])
    assert "seeded random split" in text
    assert "Never change a value the paper states" in text
    assert "fidelity retry:" in text
