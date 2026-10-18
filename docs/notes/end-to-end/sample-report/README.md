# A generated report, committed as a sample

`report/` writes one report per run into `report/output-e2e/`, which is gitignored like
every other stage's output. These two are copied here so the report generator's real
output is readable in the repo without rerunning anything. The content is byte-for-byte
what `report/` wrote; the only change is a trailing newline the repo's `end-of-file-fixer`
hook appends to each `curve.svg`, which no renderer notices.

Both are Wijaya 2023 (claim: Boston Housing test RMSE 3.02) from the 2026-10-09 end-to-end
batch. They are the same paper and the same claim, and they show opposite halves of the
system:

| | What it shows |
|---|---|
| [`wijaya-run1/report.md`](wijaya-run1/report.md) | A pass on the first full run, 2.763 — better than the claim, so the tolerance is only the claim's own reporting precision (±0.005). Carries the **Opus code review**, rated `faithful`: 12 findings under *Matches the paper* and *Choices the paper never states*, each with its paper quote and script lines. Those 12 are the ones that passed the citation guards; the Critic checked 15 and the three that failed a guard stay in `critic_output.review` rather than reaching the page. The review names the unstated split assignment as the likely reason the result landed below 3.02 — without being told the result. |
| [`wijaya-run3/report.md`](wijaya-run3/report.md) | The **seed escalation**. The first value was 3.1285, worse than the claim with no noise model, so the Critic returned `inconclusive` rather than ruling; two extra seeds gave 3.0283 and 2.9520 and the re-judgement was `pass` at a mean of 3.03629, tolerance ±0.2044 now measured from the spread. Review was off for runs 2 and 3, so this one is the arithmetic half alone. |

Each `curve.svg` is drawn from the same `metrics.full.history.jsonl` the Runner's live
check-ups read, so the curve and the check-ups always describe the same run. The dashed
reference lines are the paper's claim (3.02) and the chance level the script recorded
(8.48, what predicting the mean gives) — an RMSE means nothing without it.

Note what both reports do with the claims the run did not test: Wijaya's other three
extracted results are listed as `not tested` rather than omitted. One run targets one
claim, and the denominator belongs in the report.

Regenerate all nine with:

```bash
uv run python -m report.pipeline --input orchestrator/output-e2e \
    --coder-output coder/output-e2e --output report/output-e2e --force
```
