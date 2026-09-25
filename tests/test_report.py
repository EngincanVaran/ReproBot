import json
from pathlib import Path
from typing import Any
from xml.dom import minidom

import pytest

from report.build import _cell, build_index, build_report, caveats, slug
from report.curve import curve_svg
from report.sources import Roots, Run, list_papers, load_run

CLAIM = {"claim_id": "c1", "metric": "Test error", "dataset": "CIFAR-10", "reported_value": 4.0}


def critic(**over: Any) -> dict[str, Any]:
    base = {
        "verdict": "pass",
        "reason": "within tolerance",
        "claim_id": "c1",
        "metric": "Test error",
        "dataset": "CIFAR-10",
        "unit": "%",
        "claimed": 4.0,
        "reproduced": 3.8,
        "gap": -0.2,
        "relative_gap": -0.05,
        "tolerance": 0.4,
        "evidence": "binomial noise",
        "run_values": [3.8],
        "merged_claim_ids": ["c1"],
        "checks": [],
    }
    return {**base, **over}


def run(**over: Any) -> Run:
    base: dict[str, Any] = {
        "paper": "Paper - One",
        "origin": "files",
        "reader": {"claims": {"claims": [CLAIM]}},
        "coder": {},
        "runner": {"reproduced_metrics": {"metric": "Test error", "unit": "%", "value": 3.8}},
        "critic": critic(),
    }
    return Run(**{**base, **over})


def finding(kind: str, verified: bool = True) -> dict[str, Any]:
    return {
        "kind": kind,
        "aspect": "scaling",
        "severity": "high",
        "verified": verified,
        "explanation": "e",
    }


def test_a_single_run_is_called_a_single_run() -> None:
    notes = caveats(run())
    assert any("single run" in n for n in notes)


def test_a_wide_tolerance_band_is_called_out() -> None:
    assert any("of the paper's value" in n for n in caveats(run(critic=critic(tolerance=1.0))))
    assert not any("of the paper's value" in n for n in caveats(run()))  # 10% is not wide


def test_a_pass_with_a_verified_bug_is_flagged() -> None:
    review = {"findings": [finding("implementation_bug"), finding("unstated_choice", False)]}
    notes = caveats(run(critic=critic(review=review)))
    assert any("verified implementation bug" in n for n in notes)
    assert any("1 of 2 review finding(s)" in n for n in notes)


def test_checks_and_retries_are_reported() -> None:
    check = {"kind": "eval_count_mismatch", "message": "16 fewer"}
    notes = caveats(run(critic=critic(checks=[check]), attempts=[{}, {}]))
    assert any("eval_count_mismatch" in n for n in notes)
    assert any("regenerated 1 time" in n for n in notes)


def test_a_failed_verdict_is_never_dressed_up() -> None:
    report = build_report(run(critic=critic(verdict="fail")), generated="now").markdown
    assert "**FAIL**" in report
    assert "not a reproduction" in report


def test_a_run_nobody_judged_still_reports() -> None:
    report = build_report(run(critic=None), generated="now").markdown
    assert "NOT JUDGED" in report
    assert report.startswith("# Replication report: Paper - One")


def test_cells_cannot_break_a_table() -> None:
    assert "|" not in _cell("a | b\nc").replace("\\|", "")
    assert "\n" not in _cell("a\nb")
    assert _cell("x" * 500, 50).endswith("…") and len(_cell("x" * 500, 50)) == 50


def test_only_the_judged_claim_has_a_reproduced_value() -> None:
    other = {**CLAIM, "claim_id": "c2", "reported_value": 9.0, "model_variant": "other"}
    reader = {"claims": {"claims": [CLAIM, other]}}
    report = build_report(run(reader=reader), generated="now").markdown
    assert "reproduced |" in report and "not evaluated |" in report


def test_earlier_attempts_show_what_was_wrong() -> None:
    old = critic(
        verdict="fail",
        script_version=1,
        review={"method_fidelity": "major_deviations", "findings": [finding("implementation_bug")]},
    )
    final = critic(script_version=2, judgements=[old, critic(script_version=2)])
    report = build_report(run(critic=final), generated="now").markdown
    assert "What earlier attempts got wrong" in report
    assert "implementation_bug (high)" in report


def curve(n: int = 5, kind: str = "epoch") -> list[dict[str, Any]]:
    return [
        {"step": i + 1, "eval_metric": 40.0 / (i + 1), "kind": kind, "target_value": 4.0}
        for i in range(n)
    ]


def test_a_learning_curve_is_well_formed_svg() -> None:
    svg = curve_svg(curve(), metric="Test error", unit="%")
    assert svg is not None
    minidom.parseString(svg)  # raises if the XML is broken
    assert "paper 4%" in svg


def test_no_curve_when_there_is_nothing_to_draw() -> None:
    assert curve_svg(curve(2), metric="m") is None
    assert curve_svg(curve(6, kind="fold"), metric="m") is None  # a grid trace is not a curve


def test_a_report_embeds_its_curve_as_a_file_link() -> None:
    report = build_report(run(history=curve()), generated="now")
    name = f"{slug('Paper - One')}.curve.svg"
    assert name in report.assets and f"]({name})" in report.markdown


def test_slugs_are_safe_filenames() -> None:
    assert slug("2016-05 - Wide Residual Networks") == "2016-05-Wide-Residual-Networks"


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_run_loads_from_orchestrator_state(tmp_path: Path) -> None:
    roots = Roots(*(tmp_path / n for n in ("r", "c", "n", "k", "o")))
    state = {
        "reader_output": {"claims": {"claims": [CLAIM]}},
        "runner_output": {"status": "success", "reproduced_metrics": {"claim_id": "c1"}},
        "critic_output": critic(),
        "attempts": [{"stage_reached": "full", "version": 1}],
        "verdict": "success",
        "retry_count": 0,
        "retry_budget": 3,
    }
    write(roots.orchestrator / "P" / "state.json", state)
    loaded = load_run("P", roots)
    assert loaded.origin == "orchestrator" and loaded.runner["metrics_mode"] == "full"
    assert loaded.loop_verdict == "success" and list_papers(roots) == ["P"]


def test_a_run_loads_from_separate_files(tmp_path: Path) -> None:
    roots = Roots(*(tmp_path / n for n in ("r", "c", "n", "k", "o")))
    write(roots.runner / "P" / "runner_output.json", {"status": "success"})
    write(roots.reader / "P.json", {"claims": {"claims": [CLAIM]}})
    write(roots.critic / "P.json", critic())
    assert load_run("P", roots).origin == "files"


def test_a_paper_with_no_run_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run("missing", Roots(*(tmp_path / n for n in ("r", "c", "n", "k", "o"))))


def test_the_index_links_each_report() -> None:
    index = build_index([run()], generated="now")
    assert "(Paper-One.md)" in index and "PASS" in index
