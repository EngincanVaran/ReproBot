"""Build one paper's Markdown replication report, and the cross-paper index.

Everything here is a pure function of a `Run`: no I/O, no clock unless `generated` is left
out, no model call. The caveats are derived from the run's own numbers, so a report cannot
claim more than the Critic did: a pass is shown with its tolerance next to it, and a single
run is called a single run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from critic.claims import group_claims
from report.curve import curve_svg
from report.sources import Run

ORIGINS = {
    "orchestrator": "the orchestrator's saved state",
    "files": "the Reader, Runner and Critic output files",
}
WIDE_BAND = 0.10
CELL_LIMIT = 200
VERDICT_WORDS = {
    "pass": "PASS",
    "fail": "FAIL",
    "inconclusive": "INCONCLUSIVE",
    "not_evaluated": "NOT EVALUATED",
}


@dataclass
class Report:
    markdown: str
    assets: dict[str, str] = field(default_factory=dict)


def slug(paper: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", paper)).strip("-")


def _cell(value: object, limit: int = CELL_LIMIT) -> str:
    text = " ".join(str(value if value is not None else "-").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _num(value: object, unit: str = "") -> str:
    if not isinstance(value, int | float):
        return "-"
    suffix = unit if unit == "%" else (f" {unit}" if unit else "")
    return f"{value:.4g}{suffix}"


def _percent(value: object) -> str:
    return f"{value * 100:+.1f}%" if isinstance(value, int | float) else "-"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def verdict_word(run: Run) -> str:
    verdict = (run.critic or {}).get("verdict")
    return VERDICT_WORDS.get(str(verdict), "NOT JUDGED")


def caveats(run: Run) -> list[str]:
    """What a reader should know before trusting the number, from the run's own data."""
    critic = run.critic or {}
    out: list[str] = []
    if critic.get("verdict") not in ("pass", None):
        out.append(f"The Critic's verdict is {verdict_word(run)}, not a reproduction.")
    values = critic.get("run_values") or []
    if critic.get("claimed") is not None and len(values) <= 1:
        out.append(
            "This is a single run, so run-to-run spread was not measured; the tolerance "
            f"comes from {critic.get('evidence') or 'the evaluation set'} alone."
        )
    tolerance, claimed = critic.get("tolerance"), critic.get("claimed")
    if isinstance(tolerance, int | float) and isinstance(claimed, int | float) and claimed:
        share = tolerance / abs(claimed)
        if share > WIDE_BAND:
            out.append(
                f"The tolerance is {share:.0%} of the paper's value, so a pass means "
                '"consistent with the claim", not "matched closely".'
            )
    for check in critic.get("checks") or []:
        out.append(f"Check `{check.get('kind')}`: {check.get('message')}")
    review = critic.get("review") or {}
    findings = review.get("findings") or []
    unverified = [f for f in findings if not f.get("verified", True)]
    if unverified:
        out.append(
            f"{len(unverified)} of {len(findings)} review finding(s) failed the Critic's "
            "own checks and are marked unverified."
        )
    bugs = [
        f for f in findings if f.get("kind") == "implementation_bug" and f.get("verified", True)
    ]
    if bugs and critic.get("verdict") == "pass":
        out.append(
            "The review reported a verified implementation bug although the number passed: "
            + "; ".join(_cell(b.get("aspect"), 60) for b in bugs)
            + ". The loop accepts a passing verdict, so this was not fixed."
        )
    if len(run.attempts) > 1:
        out.append(
            f"The script was regenerated {len(run.attempts) - 1} time(s). Each retry rewrites "
            "the whole script, so the final script is not a patch of the first."
        )
    return out


def _outcome(run: Run) -> list[str]:
    critic, runner = run.critic or {}, run.runner or {}
    metrics = runner.get("reproduced_metrics") or {}
    unit = str(critic.get("unit") or metrics.get("unit") or "")
    reason = critic.get("reason") or "the Critic never judged this run"
    rows = [["Critic verdict", f"**{verdict_word(run)}**: {_cell(reason, 240)}"]]
    if runner.get("status") not in (None, "success"):
        failed = f"failed at `{runner.get('failed_stage')}`, exit {runner.get('exit_code')}"
        rows.append(["Runner", f"{runner['status']} ({failed})"])
        triage = runner.get("triage") or {}
        if triage:
            detail = f"{triage.get('category')}: {_cell(triage.get('reasoning'), 400)}"
            rows.append(["Triage", detail])
    if run.loop_verdict:
        rows.append(
            [
                "Loop",
                f"{run.loop_verdict} after {len(run.attempts)} attempt(s), "
                f"{run.retry_count} of {run.retry_budget} retries used",
            ]
        )
    if critic.get("claimed") is not None:
        target = f"{critic.get('claim_id')}: {critic.get('metric')} on {critic.get('dataset')}"
        rows += [
            ["Targeted claim", _cell(target)],
            ["Paper reports", _num(critic.get("claimed"), unit)],
            ["Reproduced", _num(critic.get("reproduced"), unit)],
            [
                "Gap",
                f"{_num(critic.get('gap'), unit)} "
                f"({_percent(critic.get('relative_gap'))} of the claim)",
            ],
            [
                "Tolerance",
                f"+/- {_num(critic.get('tolerance'), unit)} from {critic.get('evidence') or '-'}",
            ],
        ]
    elif metrics:
        rows.append(["Reproduced", _num(metrics.get("value"), unit)])
    seconds = sum(float(a.get("wall_clock_seconds") or 0) for a in run.attempts) or float(
        runner.get("wall_clock_seconds") or 0
    )
    if seconds >= 1:
        rows.append(
            ["Compute time", f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"]
        )
    return ["## Outcome", "", *_table(["", ""], rows)]


def _gap_analysis(run: Run) -> list[str]:
    critic = run.critic or {}
    if critic.get("claimed") is None:
        return []
    gap, tol = critic.get("gap"), critic.get("tolerance")
    unit = str(critic.get("unit") or "")
    lines = ["## Gap analysis", ""]
    if isinstance(gap, int | float) and isinstance(tol, int | float):
        side = "beyond" if abs(gap) > tol else "within"
        lines.append(
            f"The reproduced value differs from the paper's by {_num(gap, unit)}, {side} the "
            f"{_num(tol, unit)} tolerance ({verdict_word(run)})."
        )
    if critic.get("exceeds_claim"):
        lines.append("The reproduced value is better than the paper's.")
    review = critic.get("review") or {}
    stated = [
        f
        for f in review.get("findings") or []
        if f.get("kind") in ("implementation_bug", "deviates_from_paper")
        and f.get("verified", True)
    ]
    if stated:
        lines += ["", "Verified problems the review found in the script:", ""]
        for f in stated:
            head = f"**{f['kind']}** ({f['severity']}), {_cell(f['aspect'], 80)}"
            lines.append(f"- {head}: {_cell(f['explanation'], 320)}")
    hypotheses = review.get("hypotheses") or []
    if hypotheses:
        lines += ["", "Hypotheses for a remaining gap:", ""]
        lines += [f"{h.get('rank')}. {_cell(h.get('hypothesis'), 300)}" for h in hypotheses]
    return [*lines, ""]


def _claims(run: Run) -> list[str]:
    raw = run.reader.get("claims", [])
    claims = raw.get("claims", []) if isinstance(raw, dict) else raw
    if not claims:
        return []
    groups = group_claims(list(claims))
    judged = set((run.critic or {}).get("merged_claim_ids") or [])
    judged.add((run.critic or {}).get("claim_id"))
    rows = []
    for group in groups:
        d = group.to_dict()
        hit = bool(judged & set(d["merged_claim_ids"]))
        rows.append(
            [
                ", ".join(d["merged_claim_ids"]),
                _cell(d["metric"], 60),
                _cell(d["dataset"], 30),
                _cell(d.get("model_variant"), 50),
                _num(d["reported_value"], d.get("unit", "")),
                _num((run.critic or {}).get("reproduced"), d.get("unit", "")) if hit else "-",
                "reproduced" if hit else "not evaluated",
            ]
        )
    intro = (
        f"The Reader extracted {len(claims)} claim(s), {len(groups)} after merging duplicates. "
        "A run targets one claim, so only that one has a reproduced value."
    )
    header = ["Claim", "Metric", "Dataset", "Variant", "Paper reports", "Reproduced", "Status"]
    shown = [row for row in rows if row[-1] == "reproduced"]
    lines = ["## Claims", "", intro, ""]
    if shown:
        lines += _table(header, shown)
    lines += [
        f"<details><summary>All {len(rows)} claim groups</summary>",
        "",
        *_table(header, rows),
        "</details>",
        "",
    ]
    return lines


def _attempts(run: Run) -> list[str]:
    if not run.attempts:
        return []
    rows = []
    for a in run.attempts:
        note = a.get("feedback_given") or a.get("reason") or ""
        rows.append(
            [
                f"v{a.get('version')}",
                str(a.get("stage_reached") or "-"),
                str(a.get("runner_status") or "-"),
                str(a.get("critic_verdict") or "-"),
                str(a.get("action") or a.get("verdict") or "-"),
                _num(a.get("wall_clock_seconds")),
                f"{a['plateau_ratio']:.2f}" if isinstance(a.get("plateau_ratio"), float) else "-",
                _cell(note, 160),
            ]
        )
    header = ["Attempt", "Reached", "Runner", "Critic", "Next", "Seconds", "Similarity", "Note"]
    return ["## Attempts", "", *_table(header, rows)]


def _earlier(run: Run) -> list[str]:
    """Problems the Critic found in attempts that were later replaced."""
    judgements = (run.critic or {}).get("judgements") or []
    if len(judgements) < 2:
        return []
    rows = []
    for entry in judgements[:-1]:
        review = entry.get("review") or {}
        problems = [
            f"{f['kind']} ({f['severity']}), {_cell(f['aspect'], 50)}: "
            f"{_cell(f['explanation'], 180)}"
            for f in review.get("findings") or []
            if f.get("kind") in ("implementation_bug", "deviates_from_paper")
            and f.get("verified", True)
        ]
        rows.append(
            [
                f"v{entry.get('script_version')}",
                verdict_word(Run(run.paper, run.origin, {}, {}, None, entry)),
                _num(entry.get("reproduced"), str(entry.get("unit") or "")),
                str(review.get("method_fidelity") or "-"),
                "<br>".join(problems) or "-",
            ]
        )
    header = ["Script", "Critic", "Reproduced", "Review", "Verified problems found"]
    intro = "The Critic replaced these scripts. What it found wrong with each:"
    return ["## What earlier attempts got wrong", "", intro, "", *_table(header, rows)]


def _review(run: Run) -> list[str]:
    review = (run.critic or {}).get("review")
    if not review:
        return ["## Critic review", "", "No model review was run for this result.", ""]
    lines = ["## Critic review", "", f"**Method fidelity:** {review.get('method_fidelity')}", ""]
    lines += [_cell(review.get("summary"), 1200), ""]
    findings = review.get("findings") or []
    rows = [
        [
            _cell(f.get("kind"), 24),
            _cell(f.get("aspect"), 50),
            str(f.get("severity")),
            "yes" if f.get("verified", True) else "**no**",
            _cell(f.get("explanation"), 220),
        ]
        for f in findings
    ]
    if rows:
        lines += _table(["Kind", "Aspect", "Severity", "Verified", "Explanation"], rows)
    lines += [f"Reviewed by `{review.get('model')}`.", ""]
    return lines


def _assumptions(run: Run) -> list[str]:
    assumptions, used = (
        run.coder.get("assumptions") or [],
        run.coder.get("hyperparameters_used") or [],
    )
    if not assumptions and not used:
        return []
    lines = ["## What the Coder assumed", ""]
    lines.append(
        "These are the Coder's own disclosures. They are visible, not verified: a value taken "
        "from the model's memory rather than the paper appears here but was not checked."
    )
    lines.append("")
    if used:
        rows = [[_cell(h.get("name"), 40), _cell(h.get("value_used"), 200)] for h in used]
        lines += _table(["Setting", "Value used"], rows)
    lines += [f"- {_cell(a, 420)}" for a in assumptions]
    return [*lines, ""]


def _details(run: Run) -> list[str]:
    runner, metrics = run.runner or {}, (run.runner or {}).get("reproduced_metrics") or {}
    image = runner.get("image")
    rows = [
        ["Hardware", str(image) if image else "not recorded"],
        ["Stage of the reported value", str(runner.get("metrics_mode") or "-")],
        ["Script version", str(run.coder.get("script_version") or "-")],
        ["Epochs completed", _num(metrics.get("epochs_completed"))],
        ["Training samples", _num(metrics.get("num_train_samples"))],
        ["Evaluation samples", _num(metrics.get("num_eval_samples"))],
        ["Loaded from", run.origin],
    ]
    return ["## Reproduction details", "", *_table(["", ""], rows)]


def _script(run: Run) -> list[str]:
    if not run.script:
        return []
    lines = [
        "## Final script",
        "",
        f"<details><summary>train.py ({len(run.script.splitlines())} lines)</summary>",
        "",
        "```python",
        run.script.rstrip(),
        "```",
        "",
        "</details>",
        "",
    ]
    diff = run.coder.get("diff_from_previous")
    if diff:
        lines += [
            "<details><summary>Change from the previous attempt</summary>",
            "",
            "```diff",
            str(diff).rstrip(),
            "```",
            "",
            "</details>",
            "",
        ]
    return lines


def build_report(run: Run, *, generated: str | None = None) -> Report:
    when = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    summary = run.reader.get("method_summary") or {}
    lines = [
        f"# Replication report: {run.paper}",
        "",
        f"*Generated {when} from {ORIGINS.get(run.origin, run.origin)}. No model wrote this text.*",
        "",
    ]
    lines += _outcome(run)
    notes = caveats(run)
    if notes:
        lines += ["## Read this before trusting the result", ""] + [f"- {n}" for n in notes] + [""]
    lines += _gap_analysis(run)
    if summary.get("summary"):
        lines += ["## Method", "", str(summary["summary"]).strip(), ""]
    lines += _claims(run)
    lines += _attempts(run)
    lines += _earlier(run)
    lines += _review(run)

    assets: dict[str, str] = {}
    metrics = (run.runner or {}).get("reproduced_metrics") or {}
    svg = curve_svg(
        run.history,
        metric=str(metrics.get("metric") or "metric"),
        unit=str(metrics.get("unit") or ""),
    )
    if svg:
        name = f"{slug(run.paper)}.curve.svg"
        assets[name] = svg
        lines += ["## Learning curve", "", f"![{metrics.get('metric')} per epoch]({name})", ""]
    lines += _assumptions(run) + _details(run) + _script(run)
    return Report("\n".join(lines).rstrip() + "\n", assets)


def build_index(runs: list[Run], *, generated: str | None = None) -> str:
    when = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for run in runs:
        critic = run.critic or {}
        unit = str(critic.get("unit") or "")
        rows.append(
            [
                f"[{_cell(run.paper, 70)}]({slug(run.paper)}.md)",
                verdict_word(run),
                _num(critic.get("reproduced"), unit),
                _num(critic.get("claimed"), unit),
                str(len(run.attempts) or "-"),
                str((critic.get("review") or {}).get("method_fidelity") or "-"),
            ]
        )
    header = ["Paper", "Verdict", "Reproduced", "Paper reports", "Attempts", "Review"]
    return "\n".join(
        [
            "# ReproBot replication reports",
            "",
            f"*Generated {when}. One row per paper with a finished run.*",
            "",
            *_table(header, rows),
        ]
    )
