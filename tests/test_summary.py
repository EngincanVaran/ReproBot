"""Aggregating many runs of many papers, using this project's own measured spreads.

The numbers are real: Tang's two single-shot generations from the identical prompt
(0.85% pass, 1.17% fail), the SVM guide's four identical-to-3dp runs, and Wijaya's
scatter across generations (2.58 / 2.62 / 3.71).
"""

from typing import Any

from report.summary import collect, render_summary


def run(
    paper: str,
    verdict: str,
    reproduced: float | None,
    claimed: float | None = None,
    metric: str = "test error",
    unit: str = "%",
    execution: str = "success",
) -> tuple[str, dict[str, Any], str]:
    critic: dict[str, Any] = {"verdict": verdict, "metric": metric, "unit": unit}
    if reproduced is not None:
        critic["reproduced"] = reproduced
    if claimed is not None:
        critic["claimed"] = claimed
    state = {"verdict": execution, "critic_output": critic}
    return (paper, state, f"{paper}/report.md")


TANG = [
    run("Tang 2013", "pass", 0.85, 0.87),
    run("Tang 2013", "fail", 1.17, 0.87),
]
SVM = [
    run("SVM guide", "pass", v, 96.9, metric="accuracy") for v in (96.95, 96.625, 96.625, 96.625)
]
WIJAYA = [
    run("Wijaya 2023", "pass", 2.58, 3.02, metric="RMSE", unit=""),
    run("Wijaya 2023", "pass", 2.619, 3.02, metric="RMSE", unit=""),
    run("Wijaya 2023", "inconclusive", 3.709, 3.02, metric="RMSE", unit=""),
    run(
        "Wijaya 2023",
        "not_evaluated",
        None,
        None,
        metric="RMSE",
        unit="",
        execution="retry_budget_exhausted",
    ),
]


def test_runs_group_by_paper_not_by_state_file() -> None:
    papers = collect([*TANG, *SVM, *WIJAYA])
    assert [p.paper for p in papers] == ["Tang 2013", "SVM guide", "Wijaya 2023"]
    assert [p.runs for p in papers] == [2, 4, 4]


def test_a_paper_that_disagrees_with_itself_reports_a_pass_rate() -> None:
    tang = collect(TANG)[0]
    assert (tang.passes, tang.failures, tang.runs) == (1, 1, 2)
    spread = tang.spread()
    assert spread is not None
    low, mean, high = spread
    assert (low, high) == (0.85, 1.17)
    assert mean == 1.01
    # the paper's 0.87% sits inside what the system produced: the open question is
    # variance, not fidelity
    assert tang.claim_inside_spread() is True


def test_a_stable_paper_shows_a_narrow_spread() -> None:
    svm = collect(SVM)[0]
    assert svm.passes == 4
    spread = svm.spread()
    assert spread is not None
    low, _, high = spread
    assert round(high - low, 4) == 0.325  # three identical runs and one 0.325 above
    # 96.9 lies between 96.625 and 96.95: the runs bracket the paper's number, which
    # is the strongest thing a spread can say
    assert svm.claim_inside_spread() is True


def test_runs_with_no_number_are_counted_separately() -> None:
    wijaya = collect(WIJAYA)[0]
    assert (wijaya.passes, wijaya.inconclusive, wijaya.unjudged) == (2, 1, 1)
    assert len(wijaya.values) == 3  # the unjudged run contributes no value


def test_a_single_run_reports_no_spread_claim() -> None:
    once = collect([run("Solo", "pass", 1.0, 1.0)])[0]
    assert once.claim_inside_spread() is None  # one run cannot bracket anything


def test_the_summary_leads_with_pass_rates_and_names_the_failures() -> None:
    text = render_summary([*TANG, *SVM, *WIJAYA])
    assert "3 paper(s), 10 run(s)" in text
    assert "**1/2**" in text and "**4/4**" in text and "**2/4**" in text
    assert "0.85% – 1.17%" in text
    assert "## Runs that produced no passing number" in text
    assert "1 produced no comparable number" in text
    assert "1 judged `fail`" in text
    assert "## Every run" in text


def test_an_empty_report_set_still_renders() -> None:
    assert "0 paper(s), 0 run(s)" in render_summary([])


def test_identical_runs_report_no_spread_question() -> None:
    """Fashion-MNIST's three runs all returned 0.87734; "inside the spread" has no answer."""
    runs = [
        (
            "Fashion-MNIST",
            {
                "verdict": "success",
                "critic_output": {
                    "verdict": "pass",
                    "metric": "Test Accuracy",
                    "claimed": 0.873,
                    "reproduced": 0.87734,
                    "unit": "",
                    "exceeds_claim": True,
                },
            },
            f"run{i}/report.md",
        )
        for i in (1, 2, 3)
    ]
    papers = collect(runs)
    assert papers[0].passes == 3
    assert papers[0].claim_inside_spread() is None
    text = render_summary(runs)
    assert "identical across 3 runs" in text
    assert "| no |" not in text


def test_two_different_values_still_answer_it() -> None:
    """The SVM guide's runs disagree, so the question applies and the claim is inside."""

    def run(value: float, i: int) -> tuple[str, dict[str, object], str]:
        return (
            "SVM guide",
            {
                "verdict": "success",
                "critic_output": {
                    "verdict": "pass",
                    "metric": "Accuracy by our procedure",
                    "claimed": 96.9,
                    "reproduced": value,
                    "unit": "%",
                },
            },
            f"run{i}/report.md",
        )

    runs = [run(96.625, 1), run(96.925, 2), run(96.625, 3)]
    assert collect(runs)[0].claim_inside_spread() is True
