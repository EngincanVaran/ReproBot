"""Many runs of many papers, summarised in one table.

`render.py` describes a single run: one state object, one verdict, one curve. That
is the right unit for reading a reproduction, and the wrong unit for answering the
question an evaluation actually asks - *does this paper reproduce?* One run cannot
answer it. Measured on this project's own papers, two generations from the identical
prompt landed at 0.85% and 1.17% test error on Tang, which is the difference between
`pass` and `fail`, and Wijaya's RMSE moved between 2.58 and 3.71 across generations.

So this module groups runs by paper and reports the distribution rather than a point:

* **how often it reproduced** - the pass rate over runs, which is the honest headline
  when generations disagree;
* **the spread of what it produced** - min, mean and max of the reproduced value, so
  a reader can see whether the paper's number sits inside the system's own variation;
* **what went wrong in the rest** - runs that never produced a number at all, split by
  why (halted by a check-up, crashed, judged but failing), because "3 of 5 passed" and
  "3 of 5 passed, 2 never ran" describe very different systems.

Runs are grouped by `paper_id`, so any directory layout works: one state per run
directory, several run directories side by side, or a mixture.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from critic.claims import normalise_unit


@dataclass
class PaperRuns:
    """Every run of one paper, and what they add up to."""

    paper: str
    claimed: float | None = None
    unit: str = ""
    metric: str = ""
    values: list[float] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    executions: list[str] = field(default_factory=list)
    hrefs: list[str] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.verdicts)

    @property
    def passes(self) -> int:
        return sum(1 for v in self.verdicts if v == "pass")

    @property
    def unjudged(self) -> int:
        """Runs that produced no comparable number at all."""
        return sum(1 for v in self.verdicts if v == "not_evaluated")

    @property
    def failures(self) -> int:
        return sum(1 for v in self.verdicts if v == "fail")

    @property
    def inconclusive(self) -> int:
        return sum(1 for v in self.verdicts if v == "inconclusive")

    def spread(self) -> tuple[float, float, float] | None:
        """min, mean, max of the reproduced values, when there are any."""
        if not self.values:
            return None
        return min(self.values), statistics.fmean(self.values), max(self.values)

    def claim_inside_spread(self) -> bool | None:
        """Does the paper's own number fall within what this system produced?

        The most useful single fact when generations disagree: a claim inside the
        spread means the system brackets the paper, and the remaining question is
        variance, not fidelity.
        """
        spread = self.spread()
        if spread is None or self.claimed is None or len(self.values) < 2:
            return None
        return spread[0] <= self.claimed <= spread[2]


def collect(reports: list[tuple[str, dict[str, Any], str]]) -> list[PaperRuns]:
    """Group `(paper, state, href)` triples by paper, newest grouping order kept stable."""
    grouped: dict[str, PaperRuns] = {}
    order: list[str] = []
    for paper, state, href in reports:
        critic = state.get("critic_output") or {}
        entry = grouped.get(paper)
        if entry is None:
            entry = grouped[paper] = PaperRuns(paper=paper)
            order.append(paper)
        entry.verdicts.append(str(critic.get("verdict") or "not_evaluated"))
        entry.executions.append(str(state.get("verdict") or "unknown"))
        entry.hrefs.append(href)
        if isinstance(critic.get("reproduced"), int | float):
            entry.values.append(float(critic["reproduced"]))
        if entry.claimed is None and isinstance(critic.get("claimed"), int | float):
            entry.claimed = float(critic["claimed"])
            entry.unit = normalise_unit(critic.get("unit"))
            entry.metric = str(critic.get("metric") or "")
    return [grouped[p] for p in order]


def _fmt(value: float, unit: str) -> str:
    return f"{value:.6g}{unit}"


def render_summary(reports: list[tuple[str, dict[str, Any], str]]) -> str:
    """The cross-paper, multi-run summary: one row per paper, not per run."""
    papers = collect(reports)
    total_runs = sum(p.runs for p in papers)
    reproduced = sum(1 for p in papers if p.passes)
    lines = [
        "# Replication summary",
        "",
        f"**{len(papers)} paper(s), {total_runs} run(s).** "
        f"{reproduced} paper(s) reproduced their claim at least once.",
        "",
        "Each row is a paper, not a run. Where a paper was run several times, the "
        "pass rate and the spread are what matter: the same prompt does not produce "
        "the same script twice, and a single run can land on either side of the "
        "tolerance band on its own.",
        "",
        "| Paper | Metric | Paper's claim | Runs | Reproduced | ReproBot (min–max) | "
        "Claim inside spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in papers:
        spread = p.spread()
        if spread is None:
            got = "—"
        elif len(p.values) == 1:
            got = _fmt(spread[1], p.unit)
        else:
            got = (
                f"{_fmt(spread[0], p.unit)} – {_fmt(spread[2], p.unit)}"
                f"<br><sub>mean {_fmt(spread[1], p.unit)}</sub>"
            )
        inside = p.claim_inside_spread()
        inside_text = "—" if inside is None else ("**yes**" if inside else "no")
        claimed = _fmt(p.claimed, p.unit) if p.claimed is not None else "—"
        lines.append(
            f"| {p.paper} | {p.metric or '—'} | {claimed} | {p.runs} | "
            f"**{p.passes}/{p.runs}** | {got} | {inside_text} |"
        )

    notes = [p for p in papers if p.unjudged or p.failures or p.inconclusive]
    if notes:
        lines += ["", "## Runs that produced no passing number", ""]
        for p in notes:
            parts = []
            if p.unjudged:
                parts.append(f"{p.unjudged} produced no comparable number")
            if p.failures:
                parts.append(f"{p.failures} judged `fail`")
            if p.inconclusive:
                parts.append(f"{p.inconclusive} `inconclusive`")
            lines.append(f"- **{p.paper}** — {', '.join(parts)} (of {p.runs} run(s)).")

    lines += [
        "",
        "## Every run",
        "",
        "| Paper | Run | Fidelity | Execution | Report |",
        "|---|---|---|---|---|",
    ]
    counters: dict[str, int] = defaultdict(int)
    for paper, state, href in reports:
        counters[paper] += 1
        critic = state.get("critic_output") or {}
        lines.append(
            f"| {paper} | {counters[paper]} | `{critic.get('verdict') or 'not_evaluated'}` | "
            f"`{state.get('verdict') or '—'}` | [report]({quote(href)}) |"
        )
    return "\n".join(lines) + "\n"
