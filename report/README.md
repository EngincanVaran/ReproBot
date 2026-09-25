# report/ - replication reports

Turns a finished run into one Markdown report per paper, plus an `index.md` across papers.
Deterministic: it reads what the earlier stages wrote and makes **no model call**, so it
needs no API key, costs nothing, and can be regenerated at any time.

```
uv run python -m report.pipeline --paper "2016-05 - Wide Residual Networks"
uv run python -m report.pipeline --all              # every paper with a finished run
```

Output goes to `report/output/` (gitignored): `<paper>.md`, `<paper>.curve.svg` when the
run has a per-epoch curve, and `index.md`.

## Inputs

A paper is loaded from whichever pipeline produced it:

- `orchestrator/output/<paper>/state.json` - the whole loop: attempts, every Critic
  judgement, the final review. Preferred when it exists.
- otherwise the separate files: `reader/output/<paper>.json`, `runner/output/<paper>/
  runner_output.json`, `critic/output/<paper>.json`, `coder/output/<paper>/`.

The learning curve comes from `metrics.full.history.jsonl`, or is rebuilt from the run's
HF `Trainer` log. A history that is not per-epoch (a grid-search trace) is not drawn.

## What a report contains

Outcome (verdict, claimed vs reproduced, gap, tolerance) - **Read this before trusting the
result** - gap analysis - method - claims - attempts - what earlier attempts got wrong -
Critic review - learning curve - what the Coder assumed - reproduction details - final script.

The caveats section is derived from the run's own numbers: a single run is called a single
run, a tolerance wider than 10% of the claim is called out, the Critic's `checks` are
repeated, unverified review findings are counted, and a passing verdict alongside a
verified implementation bug is flagged (the loop accepts such a pass).

## Limits

- **A run targets one claim,** so only that claim has a reproduced value; every other claim
  is listed as "not evaluated". Judging every claim a run could speak to is not built.
- **No model-written narrative.** The plan (§2.6) had a Sonnet gap analysis; this version
  builds it from the Critic's facts instead. A model paragraph could be added as an option.
- Compute time for an orchestrated run is summed over attempts; hardware is shown only when
  the Runner recorded the image.
