# report/ — the replication report

The last stage of the project plan's pipeline (§2.6). It takes the Orchestrator's
shared-memory object and lays it out as Markdown: what the paper claims, what ReproBot
got, how far apart they are, how much of that gap is noise, and where the code differed
from the paper.

**It calls no model.** A report is a rearrangement of facts the loop already established,
so every number is copied from `state.json` and the learning curve is drawn as plain SVG.
No API key, no Docker, no network, no plotting dependency — the same reasoning that keeps
the Critic's verdict arithmetic.

```
report/
├── curves.py    metrics.<mode>.history.jsonl → curve.svg (train, eval, claim, chance)
├── render.py    state.json → report.md, and the cross-paper index
└── pipeline.py  CLI over one state, one paper directory, or a directory of them
```

## What a report contains

| Section | What it answers |
|---|---|
| Headline | Did it reproduce? Both verdicts — execution and fidelity — plus the review's fidelity rating |
| Claims | Every claim the paper makes, merged into distinct results, with the tested one marked and the rest shown as `not tested` |
| Learning curve | Did it learn, and where did it land relative to the claim and to chance? |
| The gap | Claim, result, gap, tolerance and where the tolerance came from; each seed's value |
| Does the code match the paper? | The Critic's review: deviations, matches and unstated choices, each with its paper quote and script lines |
| What was run | Library, dataset, targeted claim, the loop's own fixes, and every setting the paper never states |
| How it got there | One row per attempt: stage reached, status, Critic verdict, wall clock, why it retried — with the exact feedback each retry was given |
| Reproduce this reproduction | The commands to run the final script again |

## Three decisions worth knowing

**The tolerance travels with the verdict.** A bare `pass` hides its own strength. Wijaya
passes at ±1.046 on a claim of 3.02, so the report prints the band everywhere the verdict
appears, and adds an explicit warning when the band is more than 10% of the claim:

> **Read this pass carefully.** The tolerance is 35% of the claim, so it means the result
> is *consistent with* the paper, not that it matched closely.

**Untested claims are listed, not omitted.** One run targets one claim; Wijaya states 14,
which merge into 8 distinct results. The claim table shows all 8 with `not tested` against
seven of them, because the honest denominator belongs in the report rather than in a
footnote. That table is also the clearest argument for running one paper once per claim.

**Only verified review findings appear.** The Critic's deterministic checks already decide
which findings carry a real paper quote and real script lines. The report prints those and
says how many were dropped; the full set, with the check each one failed, stays in
`critic_output.review`.

## The curve

Drawn from the same `metrics.<mode>.history.jsonl` the Runner's live check-ups read, so the
report and the check-ups always describe the same run. Two reference lines make it readable
without knowing the dataset: the **paper's claim**, and the **chance** level the script
recorded — a regression RMSE of 3.3 means nothing until you know that predicting the mean
gives 8.7. Colours are literal, not theme tokens, because the SVG is read on GitHub, in an
editor and in a browser, none of which pass a theme in.

A script generated before the progress contract (2026-09-14) writes no history. The report
then says so in place of the curve rather than dropping the section.

## Usage

```bash
# every paper the Orchestrator has run
uv run python -m report.pipeline --input orchestrator/output

# one paper, rewriting an existing report
uv run python -m report.pipeline --input "orchestrator/output/<paper>" --force
```

Output goes to `report/output/<paper>/report.md` with `curve.svg` beside it, plus
`report/output/README.md` as the index across papers (gitignored, like every other stage's
output). Papers already reported are skipped unless `--force` is given, and one paper's
failure never stops the batch.

## Status

- Generated from the real Wijaya state (the Critic v2 loop: fail 4.76 → fix → pass 3.40),
  including its 101-record curve and its Opus review.
- `tests/test_report.py` replays that state's shape: verdicts, the wide-band warning,
  untested claims, verified-only findings, the attempt table, and a curve with both series
  and both reference lines.

## Not built yet

- **One report per run, not per paper.** A paper run several times (different claims,
  different seeds) gets one report per state file; nothing merges them.
- **No cross-paper narrative.** The index is a table, not an evaluation summary.
- **No LaTeX or HTML output.** Markdown renders on GitHub and in Mert's viewer; a progress
  report still gets written by hand.
