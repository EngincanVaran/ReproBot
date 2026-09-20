"""The report, rendered from a state shaped exactly like Wijaya's real one.

Numbers are the ones the Orchestrator actually recorded on 2026-09-14: RMSE 3.33 from
three runs against a claimed 3.02, tolerance 1.068 from their spread.
"""

import json
from pathlib import Path
from typing import Any

from report.curves import read_curve, render_curve
from report.pipeline import judged_mode
from report.render import ReportInputs, render, render_index

STATE: dict[str, Any] = {
    "paper_id": "2023-10 - Multi Level Dense Layer Neural Network Model",
    "source_pdf": "extra-papers/2023-10 - Multi Level Dense Layer.pdf",
    "verdict": "success",
    "retry_count": 1,
    "retry_budget": 3,
    "fidelity_retry_count": 0,
    "reader_output": {
        "claims": {
            "claims": [
                {
                    "claim_id": "c8",
                    "metric": "RMSE",
                    "dataset": "Boston Housing",
                    "reported_value": 3.02,
                    "unit": "",
                    "model_variant": "testing set",
                },
                {
                    "claim_id": "c11",
                    "metric": "RMSE",
                    "dataset": "Boston Housing",
                    "reported_value": 3.02,
                    "unit": "",
                    "model_variant": "Proposed NN, Table 3",
                },
                {
                    "claim_id": "c7",
                    "metric": "RMSE",
                    "dataset": "Boston Housing",
                    "reported_value": 2.69,
                    "unit": "",
                    "model_variant": "training set",
                },
            ]
        }
    },
    "coder_output": {
        "script_path": "coder/output/paper/train.py",
        "script_version": 2,
        "bookkeeping": {
            "task_type": "regression",
            "model_family": "neural",
            "dataset_used": "Boston Housing via OpenML",
            "assumptions": [
                "deviation fix: the split was unpacked backwards; fixed the order",
                "unstated_details: batch size is not stated - used 32",
            ],
        },
    },
    "runner_output": {"status": "success", "logs_path": "orchestrator/output/paper/logs"},
    "critic_output": {
        "verdict": "pass",
        "metric": "RMSE",
        "unit": "",
        "claimed": 3.02,
        "reproduced": 3.3995,
        "gap": 0.3795,
        "relative_gap": 0.1257,
        "tolerance": 1.046,
        "run_values": [3.92, 3.186, 3.093],
        "run_spread": 0.4525,
        "test_noise": None,
        "evidence": "measured spread of 3 runs",
        "exceeds_claim": False,
        "merged_claim_ids": ["c8", "c11"],
        "metrics_mode": "full",
        "reason": "the reproduced value is within tolerance of the claim",
        "review": {
            "model": "claude-opus-5",
            "method_fidelity": "faithful",
            "summary": "The script implements the paper's architecture.",
            "findings": [
                {
                    "kind": "matches_paper",
                    "aspect": "optimizer",
                    "severity": "low",
                    "paper_quote": "Adam with a learning rate equal to 0.001",
                    "script_lines": [237],
                    "script_snippet": "torch.optim.Adam(",
                    "explanation": "Adam at lr 0.001 as stated.",
                    "verified": True,
                },
                {
                    "kind": "unstated_choice",
                    "aspect": "batch size",
                    "severity": "low",
                    "paper_quote": "not stated",
                    "script_lines": [121],
                    "script_snippet": "--batch-size",
                    "explanation": "Keras default 32.",
                    "verified": True,
                },
                {
                    "kind": "deviates_from_paper",
                    "aspect": "init",
                    "severity": "low",
                    "paper_quote": "invented",
                    "script_lines": [9],
                    "script_snippet": "nope",
                    "explanation": "unverifiable",
                    "verified": False,
                },
            ],
            "hypotheses": [],
        },
    },
    "attempts": [
        {
            "version": 1,
            "stage_reached": "full",
            "runner_status": "success",
            "action": "retry",
            "critic_verdict": "fail",
            "wall_clock_seconds": 30.8,
            "reason": "the Critic judged a fail",
            "feedback_given": None,
        },
        {
            "version": 2,
            "stage_reached": "full",
            "runner_status": "success",
            "action": "done",
            "critic_verdict": "pass",
            "wall_clock_seconds": 73.2,
            "reason": "a stage succeeded",
            "feedback_given": "An independent review found the split unpacked backwards.",
        },
    ],
    "history": [
        {
            "timestamp": "2026-09-14T18:26:11+00:00",
            "stage": "orchestrator",
            "event": "loop started",
        },
        {
            "timestamp": "2026-09-14T18:37:32+00:00",
            "stage": "orchestrator",
            "event": "loop finished",
        },
    ],
}

HISTORY_RECORDS = [
    {
        "kind": "epoch",
        "step": s,
        "train_metric": t,
        "eval_metric": e,
        "metric": "RMSE",
        "unit": "",
        "higher_is_better": False,
        "target_value": 3.02,
        "chance_metric": 8.68,
    }
    for s, t, e in [(1, 20.6, 20.8), (50, 5.1, 5.4), (500, 2.2, 3.4), (1000, 1.9, 3.2)]
]


def report_text() -> str:
    return render(ReportInputs(state=STATE))


def test_the_headline_states_both_verdicts_and_the_tolerance() -> None:
    text = report_text()
    assert "**reproduced**" in text
    assert "3.3995" in text and "3.02" in text
    assert "tolerance ±1.046" in text
    assert "| Fidelity verdict | `pass` |" in text
    assert "| Execution verdict | `success` |" in text


def test_untested_claims_are_listed_not_hidden() -> None:
    text = report_text()
    assert "`c8`, `c11`" in text  # merged into one result, and it was tested
    assert "`not tested`" in text  # the train-set claim was not
    assert "3 claims" in text and "2 distinct" in text


def test_a_wide_tolerance_is_called_out_on_a_pass() -> None:
    assert "Read this pass carefully" in report_text()  # 1.046 is 35% of 3.02


def test_only_verified_review_findings_appear() -> None:
    text = report_text()
    assert "Adam at lr 0.001 as stated." in text
    assert "unverifiable" not in text
    assert "1 further finding(s) failed the Critic's citation checks" in text


def test_the_loops_own_changes_are_separated_from_guesses() -> None:
    text = report_text()
    assert "### Changes the loop made" in text
    assert "deviation fix: the split was unpacked backwards" in text
    assert "Settings the paper never states (1)" in text


def test_attempts_and_rerun_commands() -> None:
    text = report_text()
    assert "| v1 |" in text and "| v2 |" in text
    assert "`fail`" in text and "31 s" in text
    assert "bash reproduce.sh full" in text
    assert "The feedback each retry was given" in text


def test_a_run_with_nothing_comparable_still_renders() -> None:
    state = {**STATE, "critic_output": {"verdict": "not_evaluated", "reason": "capped stage only"}}
    text = render(ReportInputs(state=state))
    assert "**not judged**" in text and "capped stage only" in text
    assert "No progress history was recorded" in text


def test_curve_svg_draws_both_series_and_both_references(tmp_path: Path) -> None:
    path = tmp_path / "metrics.full.history.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in HISTORY_RECORDS), encoding="utf-8")
    curve = read_curve(path)
    assert curve is not None and len(curve.steps) == 4
    svg = render_curve(curve, "test")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<path") == 2  # train and eval
    assert "the paper&#x27;s claim 3.02" in svg or "the paper's claim 3.02" in svg
    assert "chance 8.68" in svg
    assert "stroke-dasharray" in svg


def test_a_history_file_that_is_missing_or_too_short_yields_no_curve(tmp_path: Path) -> None:
    assert read_curve(tmp_path / "absent.jsonl") is None
    short = tmp_path / "metrics.full.history.jsonl"
    short.write_text(json.dumps(HISTORY_RECORDS[0]), encoding="utf-8")
    assert read_curve(short) is None


def test_judged_mode_prefers_what_the_critic_judged() -> None:
    assert judged_mode(STATE) == "full"
    assert judged_mode({"attempts": [{"stage_reached": "capped"}]}) == "capped"
    assert judged_mode({}) == "full"


def test_the_index_lists_every_paper_with_its_band() -> None:
    index = render_index([("paper-a", STATE, "paper-a/report.md")])
    assert "[paper-a](paper-a/report.md)" in index
    assert "`pass`" in index and "±1.046" in index
