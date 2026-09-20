"""Turn one paper's shared-memory state into the replication report (project plan §2.6).

**Every number here is copied, never computed by a model.** The Critic already decided
the verdict and the tolerance behind it; this stage's whole job is to lay that out so a
reader can check it: what the paper claims, what the run produced, how far apart they
are, how much of that gap is noise, and what the code did differently from the paper.

Three principles the layout follows:

1. **The tolerance travels with the verdict.** A `pass` at +/-1.07 on a claim of 3.02
   means "consistent with the paper", not "matched", and a report that prints only the
   word `pass` hides that. Every verdict line carries its band and where the band came
   from.
2. **The claims the run did not test are listed, not omitted.** One run targets one
   claim; a paper states many. Showing the other claims as `not tested` is the honest
   denominator.
3. **The review's unverified findings stay out.** The Critic's deterministic checks
   already decide which findings are citable; the report shows those, and says how many
   were dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from critic.claims import ClaimGroup, group_claims, normalise_unit
from report.curves import Curve

VERDICT_WORDS: dict[str, str] = {
    "pass": "reproduced",
    "fail": "did not reproduce",
    "inconclusive": "undecided",
    "not_evaluated": "not judged",
}


@dataclass(frozen=True)
class ReportInputs:
    """Everything the report draws on, already read from disk."""

    state: dict[str, Any]
    curve: Curve | None = None
    curve_href: str | None = None


def render(inputs: ReportInputs) -> str:
    state = inputs.state
    paper = str(state.get("paper_id") or "unknown paper")
    critic = state.get("critic_output") or {}
    sections = [
        _headline(paper, state, critic),
        _claims(state, critic),
        _curve(inputs),
        _gap(critic),
        _review(critic),
        _setup(state),
        _attempts(state),
        _rerun(state),
        _provenance(state),
    ]
    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _headline(paper: str, state: dict[str, Any], critic: dict[str, Any]) -> str:
    verdict = str(critic.get("verdict") or "not_evaluated")
    execution = str(state.get("verdict") or "unknown")
    lines = [
        f"# {paper}",
        "",
        f"**{VERDICT_WORDS.get(verdict, verdict)}** — {_one_line(critic)}",
        "",
    ]
    lines += [
        "| | |",
        "|---|---|",
        f"| Fidelity verdict | `{verdict}` |",
        f"| Execution verdict | `{execution}` |",
        f"| Attempts | {len(state.get('attempts') or [])} "
        f"({state.get('retry_count', 0)} retries of {state.get('retry_budget', 0)} allowed) |",
    ]
    if state.get("fidelity_retry_count"):
        lines.append(f"| Guided fidelity retries | {state['fidelity_retry_count']} |")
    review = critic.get("review") or {}
    if review.get("method_fidelity"):
        lines.append(f"| Method fidelity (reviewed) | `{review['method_fidelity']}` |")
    return "\n".join(lines)


def _one_line(critic: dict[str, Any]) -> str:
    if not critic or critic.get("claimed") is None or critic.get("reproduced") is None:
        return str(critic.get("reason") or "no comparable result was produced")
    unit = normalise_unit(critic.get("unit"))
    runs = len(critic.get("run_values") or [])
    over = f" (mean of {runs} runs)" if runs > 1 else ""
    band = (
        f", tolerance ±{critic['tolerance']:.4g}"
        if isinstance(critic.get("tolerance"), float)
        else ""
    )
    better = " and better than the claim" if critic.get("exceeds_claim") else ""
    return (
        f"{critic.get('metric')} {critic['reproduced']:.6g}{unit}{over} against the paper's "
        f"{critic['claimed']:.6g}{unit}{band}{better}."
    )


def _claims(state: dict[str, Any], critic: dict[str, Any]) -> str:
    reader = state.get("reader_output") or {}
    claims = list((reader.get("claims") or {}).get("claims") or [])
    if not claims:
        return ""
    groups = group_claims(claims)
    targeted = set(critic.get("merged_claim_ids") or [])
    rows = [
        "## Claims",
        "",
        f"The Reader extracted **{len(claims)} claims**, which merge into **{len(groups)} distinct "
        f"results**. One run targets one of them; the rest are listed so the untested remainder is "
        f"visible.",
        "",
        "| Claim | Metric | Dataset | Paper | ReproBot | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for group in groups:
        rows.append(_claim_row(group, targeted, critic))
    return "\n".join(rows)


def _claim_row(group: ClaimGroup, targeted: set[str], critic: dict[str, Any]) -> str:
    claim = group.canonical
    ids = ", ".join(f"`{i}`" for i in group.claim_ids)
    unit = normalise_unit(claim.get("unit"))
    claimed = claim.get("reported_value")
    claimed_text = f"{claimed:.6g}{unit}" if isinstance(claimed, int | float) else str(claimed)
    variant = str(claim.get("model_variant") or "").strip()
    metric = f"{claim.get('metric')}"
    if variant:
        metric += f"<br><sub>{variant}</sub>"
    if set(group.claim_ids) & targeted and critic.get("reproduced") is not None:
        runs = len(critic.get("run_values") or [])
        over = f"<br><sub>mean of {runs} runs</sub>" if runs > 1 else ""
        got = f"**{critic['reproduced']:.6g}{unit}**{over}"
        verdict = f"`{critic.get('verdict')}`"
        if critic.get("exceeds_claim"):
            verdict += " ↑"
    else:
        got, verdict = "—", "`not tested`"
    return f"| {ids} | {metric} | {claim.get('dataset')} | {claimed_text} | {got} | {verdict} |"


def _curve(inputs: ReportInputs) -> str:
    if inputs.curve is None or not inputs.curve_href:
        return (
            "## Learning curve\n\nNo progress history was recorded for the judged run, so no "
            "curve can be drawn. Scripts generated before the progress contract (2026-09-14) "
            "write none."
        )
        # A missing curve is a real state, not an error: say so rather than omitting the section.
    curve = inputs.curve
    reference = []
    if curve.target is not None:
        reference.append(f"the paper's claim ({curve.target:.6g})")
    if curve.chance is not None:
        reference.append(f"chance ({curve.chance:.6g})")
    note = f" Dashed lines mark {' and '.join(reference)}." if reference else ""
    return (
        f"## Learning curve\n\n![Learning curve]({inputs.curve_href})\n\n"
        f"{len(curve.steps)} records from the judged run, {curve.metric} per {curve.kind}.{note}"
    )


def _gap(critic: dict[str, Any]) -> str:
    if not critic or critic.get("claimed") is None:
        return ""
    lines = ["## The gap, and what is noise", "", "| | |", "|---|---|"]
    # States written before 2026-09-20 can carry a pseudo-unit; normalise on read too.
    unit = normalise_unit(critic.get("unit"))
    rows: list[tuple[str, str]] = [
        ("Paper claims", f"{critic['claimed']:.6g}{unit}"),
        ("ReproBot", f"{critic['reproduced']:.6g}{unit}"),
    ]
    if isinstance(critic.get("gap"), float):
        relative = critic.get("relative_gap")
        rel = f" ({relative * 100:+.1f}%)" if isinstance(relative, float) else ""
        rows.append(("Gap", f"{critic['gap']:+.4g}{unit}{rel}"))
    if isinstance(critic.get("tolerance"), float):
        rows.append(("Tolerance", f"±{critic['tolerance']:.4g} — {critic.get('evidence')}"))
    values = critic.get("run_values") or []
    if len(values) > 1:
        rows.append(("Runs", ", ".join(f"{v:.6g}" for v in values)))
    if isinstance(critic.get("run_spread"), float):
        rows.append(("Spread (σ)", f"{critic['run_spread']:.4g}"))
    if isinstance(critic.get("test_noise"), float):
        rows.append(("Test-set noise", f"{critic['test_noise']:.4g}"))
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", f"**Why this verdict:** {critic.get('reason')}"]
    if isinstance(critic.get("tolerance"), float) and isinstance(critic.get("claimed"), float):
        share = abs(critic["tolerance"] / critic["claimed"]) if critic["claimed"] else 0.0
        if share > 0.1 and critic.get("verdict") == "pass":
            lines += [
                "",
                f"> **Read this pass carefully.** The tolerance is {share * 100:.0f}% of the "
                f"claim, so it means the result is *consistent with* the paper, not that it "
                f"matched closely.",
            ]
    return "\n".join(lines)


def _review(critic: dict[str, Any]) -> str:
    review = critic.get("review") or {}
    if not review:
        return ""
    findings = [f for f in review.get("findings") or [] if f.get("verified")]
    dropped = len(review.get("findings") or []) - len(findings)
    lines = [
        "## Does the code match the paper?",
        "",
        f"Reviewed by `{review.get('model')}`: **{review.get('method_fidelity')}**. "
        f"{review.get('summary')}",
    ]
    problems = [
        f for f in findings if f.get("kind") in ("deviates_from_paper", "implementation_bug")
    ]
    unstated = [f for f in findings if f.get("kind") == "unstated_choice"]
    matches = [f for f in findings if f.get("kind") == "matches_paper"]
    if problems:
        lines += ["", "### Deviations from the paper", ""]
        for f in problems:
            lines += [
                f"- **{f.get('aspect')}** ({f.get('severity')}, `{f.get('kind')}`) — "
                f"{f.get('explanation')}",
                f'  - paper: *"{f.get("paper_quote")}"*',
                f"  - script lines {f.get('script_lines')}: `{_flatten(f.get('script_snippet'))}`",
            ]
    if matches:
        lines += ["", "### Matches the paper", ""]
        lines += [f"- **{f.get('aspect')}** — {f.get('explanation')}" for f in matches]
    if unstated:
        lines += ["", "### Choices the paper never states", ""]
        lines += [f"- **{f.get('aspect')}** — {f.get('explanation')}" for f in unstated]
    hypotheses = [h for h in review.get("hypotheses") or [] if h.get("verified")]
    if hypotheses:
        lines += ["", "### Ranked explanations for the gap", ""]
        lines += [
            f"{h.get('rank')}. {h.get('hypothesis')} **Change:** {h.get('change')} "
            f"(`{h.get('change_type')}`)"
            for h in hypotheses
        ]
    if dropped:
        lines += [
            "",
            f"*{dropped} further finding(s) failed the Critic's citation checks and are omitted; "
            f"they are kept in `critic_output.review` with the check that failed.*",
        ]
    return "\n".join(lines)


def _setup(state: dict[str, Any]) -> str:
    coder = state.get("coder_output") or {}
    bookkeeping = coder.get("bookkeeping") or {}
    lines = ["## What was run", "", "| | |", "|---|---|"]
    for label, key in (
        ("Task", "task_type"),
        ("Model family", "model_family"),
        ("Dataset", "dataset_used"),
        ("Targeted claim", "claim_targeted"),
    ):
        if bookkeeping.get(key):
            lines.append(f"| {label} | {_flatten(str(bookkeeping[key]))} |")
    if coder.get("script_version"):
        lines.append(
            f"| Final script | version {coder['script_version']}, `{coder.get('script_path')}` |"
        )
    assumptions = [str(a) for a in bookkeeping.get("assumptions") or []]
    if assumptions:
        fixes = [
            a for a in assumptions if a.lower().startswith(("deviation fix", "fidelity retry"))
        ]
        guesses = [a for a in assumptions if a not in fixes]
        if fixes:
            lines += ["", "### Changes the loop made", ""] + [f"- {a}" for a in fixes]
        if guesses:
            lines += [
                "",
                f"### Settings the paper never states ({len(guesses)})",
                "",
                "<details><summary>The Coder recorded every one of these</summary>",
                "",
            ]
            lines += [f"- {a}" for a in guesses]
            lines += ["", "</details>"]
    return "\n".join(lines)


def _attempts(state: dict[str, Any]) -> str:
    attempts = state.get("attempts") or []
    if not attempts:
        return ""
    lines = [
        "## How it got there",
        "",
        "| # | Ran to | Status | Critic | Time | Why it stopped or retried |",
        "|---|---|---|---|---|---|",
    ]
    for a in attempts:
        seconds = a.get("wall_clock_seconds")
        time = f"{seconds:.0f} s" if isinstance(seconds, int | float) else "—"
        reason = _flatten(str(a.get("reason") or ""))
        if a.get("checkup_rule"):
            reason = f"halted by `{a['checkup_rule']}` — {reason}"
        elif a.get("triage_category"):
            reason = f"`{a['triage_category']}` — {reason}"
        lines.append(
            f"| v{a.get('version')} | `{a.get('stage_reached') or '—'}` | "
            f"`{a.get('runner_status') or a.get('action')}` | "
            f"`{a.get('critic_verdict') or '—'}` | {time} | {reason[:160]} |"
        )
    feedback = [
        (a.get("version"), a.get("feedback_given")) for a in attempts if a.get("feedback_given")
    ]
    if feedback:
        lines += ["", "<details><summary>The feedback each retry was given</summary>", ""]
        for version, text in feedback:
            lines += [
                f"**→ v{version}**",
                "",
                "```",
                _flatten(str(text), keep_newlines=True),
                "```",
                "",
            ]
        lines += ["</details>"]
    return "\n".join(lines)


def _rerun(state: dict[str, Any]) -> str:
    coder = state.get("coder_output") or {}
    script = coder.get("script_path")
    if not script:
        return ""
    paper_dir = Path(str(script)).parent
    return (
        "## Reproduce this reproduction\n\n"
        "```bash\n"
        f"cd {paper_dir}\n"
        "bash reproduce.sh full          # the paper's own setup\n"
        "bash reproduce.sh seed2         # the same run, another seed\n"
        "```\n\n"
        "Or drive the whole loop again, from the Reader's output:\n\n"
        "```bash\n"
        f"uv run python -m orchestrator.pipeline \\\n"
        f'    --input "reader/output/{state.get("paper_id")}.json" --max-stage full --force\n'
        "```"
    )


def _provenance(state: dict[str, Any]) -> str:
    history = state.get("history") or []
    started = history[0].get("timestamp") if history else None
    finished = history[-1].get("timestamp") if history else None
    runner = state.get("runner_output") or {}
    lines = ["---", "", "<sub>"]
    if started and finished:
        lines.append(f"Run started {started}, finished {finished}. ")
    if state.get("source_pdf"):
        lines.append(f"Paper: `{state['source_pdf']}`. ")
    if runner.get("logs_path"):
        lines.append(f"Container logs: `{runner['logs_path']}`. ")
    lines.append(
        "Generated by `report/` from the Orchestrator's state object; every number is copied "
        "from it, none computed here.</sub>"
    )
    return "".join(lines[:1] + [""] + ["".join(lines[2:])])


def _flatten(text: object, keep_newlines: bool = False) -> str:
    """Markdown tables break on newlines and pipes; prose keeps its shape elsewhere."""
    value = str(text or "")
    if keep_newlines:
        return value.strip()
    return " ".join(value.split()).replace("|", "\\|")


def render_index(reports: list[tuple[str, dict[str, Any], str]]) -> str:
    """One table over every paper reported, newest verdicts first."""
    count = f"{len(reports)} paper" + ("" if len(reports) == 1 else "s")
    lines = [
        "# Replication reports",
        "",
        f"{count}, generated by `report/`. Each row links to the full report.",
        "",
        "| Paper | Fidelity | Claim | ReproBot | Tolerance | Execution |",
        "|---|---|---|---|---|---|",
    ]
    for paper, state, href in sorted(reports):
        critic = state.get("critic_output") or {}
        unit = normalise_unit(critic.get("unit"))
        claimed = critic.get("claimed")
        reproduced = critic.get("reproduced")
        tolerance = critic.get("tolerance")
        lines.append(
            f"| [{paper}]({quote(href)}) | `{critic.get('verdict') or 'not_evaluated'}` | "
            f"{f'{claimed:.6g}{unit}' if isinstance(claimed, int | float) else '—'} | "
            f"{f'{reproduced:.6g}{unit}' if isinstance(reproduced, int | float) else '—'} | "
            f"{f'±{tolerance:.4g}' if isinstance(tolerance, int | float) else '—'} | "
            f"`{state.get('verdict') or '—'}` |"
        )
    return "\n".join(lines) + "\n"
