# How repeatable is a reproduction? Five generations per paper

The ablation (`docs/notes/ablation/`) compared the loop against single-shot and ran
each paper **once per arm**. That turned out to be too few: Tang's two single-shot
generations landed at 0.85% and 1.17% test error from the identical prompt, which is
the difference between `pass` and `fail`. A comparison built on one draw per paper
cannot distinguish the thing being measured from the noise.

So: two cheap papers, **five independent generations each**, nothing shared between
runs. Every run regenerates the script from the same Reader output, pinned to the
same claim, loop off (`--retry-budget 0 --no-critic`) so this measures the **Coder's
own variance** rather than the loop's repair rate. Each run was then judged by the
same arithmetic.

```bash
uv run --extra orchestrator python -m orchestrator.pipeline \
    --input "reader/output/<paper>.json" --max-stage full --force --no-build \
    --claim-id <id> --retry-budget 0 --no-critic \
    --output "orchestrator/output-consistency/run<N>" \
    --coder-output "coder/output-consistency/run<N>"
uv run --extra orchestrator python -m critic.pipeline \
    --state "orchestrator/output-consistency/run<N>/<paper>" \
    --coder-output "coder/output-consistency/run<N>" \
    --output "critic/output-consistency/run<N>"
```

## Result

| Paper | Claim | Passes | Reproduced, across 5 generations | sd | Claim inside the spread |
|---|---|---|---|---|---|
| Hsu/Chang/Lin, SVM guide (c1) | 96.9% | **5 / 5** | 96.625, 96.625, 96.625, 96.625, **96.95** | 0.145 | **yes** |
| Wijaya 2023 (c3) | RMSE 3.02 | **1 / 5** | 2.619, 3.703, 3.709, 4.003, *(one run never produced a number)* | 0.609 | **yes** |

Per-run detail, including exit codes and wall clock, is in `data/runs.json`; each
run's full judgement is in `data/critic/run<N>/`.

## What it says

**1. The variance tracks how precisely the paper specifies itself.** The SVM guide
states its exact procedure — scale to [-1,1], grid search C and γ, these settings —
and four of five generations produced *the identical number to three decimals*. Wijaya
never states how its 405/101 split was drawn, so every generation invents one, and the
reproduced RMSE swings by 1.38 between generations. This is not the Coder being
unreliable on one paper and reliable on another: it is the paper's own
under-specification, made visible.

**2. A single run is a weak verdict, and the pass rate shows it.** Wijaya's 1/5 is not
"ReproBot fails to reproduce this paper four times in five". The spread **contains** the
paper's 3.02; what varies is which side of the tolerance band one draw lands on. The
honest summary of Wijaya is "2.6 to 4.0 depending on the split, and the paper is inside
that" — which is also exactly why the Critic asks for extra seeds instead of ruling on
one RMSE.

**3. One generation in five was born broken.** Wijaya's run 2 produced a script that
called `.numpy()` on something that was not a tensor and died at `probe` in 6.5 s.
Triage classified it `recoverable_error` — a bug a retry could fix — and with the retry
budget at zero it simply ended. That is the population the retry loop serves, and it is
the honest companion to the ablation's "the loop changed one verdict in five": about
20% of generations here needed a repair the loop could have made.

**4. Verdict stability differs from value stability.** All five SVM runs agreed on both
the number and the verdict. Wijaya's runs agreed on no verdict at all: one `pass`, three
`inconclusive`, one `not_evaluated`. A system reporting a single verdict per paper hides
that; the summary table in `report/SUMMARY.md` now reports pass rate and spread instead.

## Honest limits

- **Two papers, both cheap.** The SVM guide is a deterministic library procedure and
  Wijaya is a small regression net. Neither says anything about the image papers.
- **Five is enough to see a spread, not to estimate it precisely.** The standard
  deviations above are five-sample estimates; treat them as an order of magnitude.
- **Loop off throughout.** This measures generation variance only. How much of it the
  loop would absorb is a separate experiment — the one broken generation suggests some,
  the ablation suggests not much on papers that already work.
- **Five of the fifteen attempted runs were lost to infrastructure** (four to API
  connection errors and timeouts, one to a container killed by the OS at exit 137) and
  were re-run. Those are not in the table: they say nothing about the system under test.
  The broken generation in run 2 **is** in the table, because that failure was the
  system's own.
