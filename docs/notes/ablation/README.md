# Single-shot vs. iterative loop — the headline ablation

**Date:** 2026-10-08 · **Papers:** 5 CPU-sized targets · **Status:** complete

The project plan calls this "the headline result" (§5 Month 3, §6). The question is narrow
and answerable: **does the Coder↔Runner↔Critic loop actually improve replication over a
single-shot generation?**

Two arms, same five papers, same pinned claim, same arithmetic judge:

- **loop-off** — one generation, no retries, no Critic, no extra seeds (`--retry-budget 0
  --no-critic`). Produced by this note on 2026-10-08.
- **loop-on** — the existing evidence in `orchestrator/output/`, `coder/output/` and
  `runner/output/`, produced 2026-09-13 → 2026-09-20.

## The short answer

**The loop changed the outcome on 1 of 5 papers. On the other 4 it changed nothing.**

Single-shot reproduced **4 of 5** claims; the loop-on arm reproduced **5 of 5**. The single
difference is the soft decision tree, where the single-shot script was dead on arrival and
the run ended with no number at all.

That is a thin result, and it is weaker than the project's own narrative implies. It must
not be dressed up: **on four of five CPU-sized papers, the iterative loop bought nothing
measurable.** It cost one extra Critic call each and left every verdict unchanged. Worse for
the loop's case, two of the five loop-on numbers were **human-assisted** and so are not
evidence of the loop working at all (see
[caveat 1](#1-two-of-the-five-loop-on-numbers-are-human-assisted)).

The one genuinely robust finding is not about the loop's retries but about its **guard
rails**: see [what the loop actually contributed](#what-the-loop-actually-contributed).

---

## Table of contents

- [Exact commands](#exact-commands)
- [Per-paper results](#per-paper-results)
- [Totals](#totals)
- [What the loop actually contributed](#what-the-loop-actually-contributed)
- [Methodology, asymmetries and caveats](#methodology-asymmetries-and-caveats)
- [Every raw number and where it came from](#every-raw-number-and-where-it-came-from)

---

## Exact commands

Claims are **pinned** with `--claim-id`, because target-claim selection is not reproducible
across runs (`TODO.md`, `coder/` open bugs: Fashion-MNIST picked `c17` on 2026-09-13 and
`c21` on 2026-09-20 from the same prompt). Both arms must target the same claim or the
comparison is meaningless.

Every ablation artefact goes to **separate directories** so the loop-on evidence is never
overwritten:

```bash
# The loop-off arm, one paper at a time (never two: the Runner sets no --cpus, so two
# containers each claim every core and roughly double wall clock).
uv run --extra orchestrator python -m orchestrator.pipeline \
    --input "reader/output/<paper>.json" \
    --max-stage full --force --no-build \
    --claim-id <id> \
    --retry-budget 0 --no-critic \
    --output orchestrator/output-ablation \
    --coder-output coder/output-ablation

# Judge it with the SAME arithmetic the loop-on arm was judged with. The verdict is pure
# arithmetic (no LLM, no Docker, no network), so judging afterwards cannot contaminate it.
uv run --extra orchestrator python -m critic.pipeline \
    --state "orchestrator/output-ablation/<paper>" \
    --coder-output coder/output-ablation \
    --output critic/output-ablation
```

Two loop-on numbers had no stored Critic verdict (Tang, soft tree), so they were judged
offline with the same arithmetic, from a metrics file:

```bash
uv run --extra orchestrator python -m critic.pipeline \
    --reader-json "reader/output/<paper>.json" \
    --metrics-json "coder/output/<paper>/metrics.full.json" \
    --coder-output coder/output \
    --output critic/output-ablation-loopon
```

Driver and collector (both in this directory):

```bash
./docs/notes/ablation/run-ablation.sh "<paper stem>" <claim-id>   # one paper, both steps, timed
uv run python docs/notes/ablation/collect-evidence.py             # copies JSON into data/, prints the table
```

The five invocations, with the claim each was pinned to:

| Paper stem | `--claim-id` | Claim |
|---|---|---|
| `2013-06 - Deep Learning using Linear Support Vector Machines` | `c5` | MNIST test error 0.87% (DLSVM) |
| `2023-10 - Multi Level Dense Layer Neural Network Model for Housing Price Prediction` | `c3` | Boston Housing test RMSE 3.02 |
| `2003 - A Practical Guide to Support Vector Classification` | `c1` | Astroparticle accuracy 96.9% |
| `2017-08 - Fashion-MNIST - a Novel Image Dataset for Benchmarking Machine Learning Algorithms` | `c17` | Test accuracy 0.873 (RF, n=100, entropy, depth 100) |
| `2017-11 - Distilling a Neural Network Into a Soft Decision Tree` | `c1` | MNIST test accuracy 94.45% (soft tree depth 8) |

Each `claim_id` was read from the loop-on run's own record — `critic_output.claim_id` in
`orchestrator/output/<paper>/state.json` where one exists, otherwise `claim_targeted` in
`coder/output/<paper>/coder_output.json`. The two agree wherever both are present.

---

## Per-paper results

Fidelity verdicts are the Critic's arithmetic: `pass` within `max(2 × uncertainty,
reporting precision)`, `fail` worse than that with an uncertainty estimate,
`not_evaluated` when the run produced no comparable number. Wall clock is the `full` stage
itself; per-paper end-to-end times including script generation are in `data/timings.psv`.

| Paper | Claim | Loop-off result + verdict | Loop-on result + verdict | Retries (off / on) | Wall clock (off / on) | What the loop actually changed |
|---|---|---|---|---|---|---|
| **Tang 2013** | MNIST test error **0.87%** | **0.85%** — `pass` (±0.1857) | **0.82%** — `pass` (±0.1857) | 0 / n/a | 1139.99 s / 3222.67 s | **Nothing.** Single-shot passed unaided and slightly beat the claim. But see the variance note below — a second single-shot generation of this same paper scored 1.17% and *failed* |
| **Wijaya 2023** | Boston test RMSE **3.02** | **2.5802** — `pass`, exceeds claim (±0.005) | **2.8843** — `pass`, exceeds claim (±0.005) | 0 / 0 | 135.41 s / 140.07 s | **Nothing.** Single-shot passed first time; the loop never retried. Both beat the claim; the 0.30 spread is the paper's unstated split, not the loop |
| **Hsu/Chang/Lin SVM guide 2003** | Astroparticle **96.9%** | **96.625%** — `pass` (±0.5481) | **96.625%** — `pass` (±0.5481) | 0 / 0 | 89.42 s / 91.95 s | **Nothing.** Identical number to three decimals from a script only 35% similar — `SVC` wraps LIBSVM, so the library is the method |
| **Fashion-MNIST 2017** | Test accuracy **0.873** | **0.8773** (mean of 5) — `pass`, exceeds claim (±0.0018) | **0.87648** (mean of 5) — `pass`, exceeds claim (±0.0008) | 0 / 0 | 147.94 s / 177.97 s | **Nothing.** Single-shot passed first time; both exceed the claim |
| **Frosst & Hinton 2017** (soft tree) | MNIST test accuracy **94.45%** | **no number** — execution `retry_budget_exhausted`, fidelity `not_evaluated` | **95.11%** — `pass`, exceeds claim (±0.4579) | 0 / 1 | 6.18 s (halted) / 4190.75 s | **Everything.** Single-shot implemented the paper's misprinted Eq. 3 literally, the loss went negative, a check-up halted it at probe epoch 1, and with no retry budget the run ended there |

### The soft tree — the one paper where the arms diverge

The single-shot Coder hit the known landmine and fell in. The paper prints Eq. 3 as `−log`
of a quantity that is never positive; the generated script's own comments show it noticing
and deciding to obey anyway:

```
# L(x) = -log(weighted); weighted should be negative-ish since inner_term <= 0,
# ... (since log of a negative is undefined) we clamp ...
# effectively a negative quantity and -log(-weighted) is used.
loss = -torch.log(torch.clamp(-weighted, min=eps))
```
— `coder/output-ablation/2017-11 - Distilling a Neural Network Into a Soft Decision Tree/train.py`, lines 152–167

That objective is unbounded below, so training walks away from accuracy. The Runner's live
check-up caught it in **6.18 s**:

```
HALTED by live check-up 'loss_below_lower_bound' at epoch 1: train_loss reached -0.8361,
below the loss's own lower bound of 0 (epoch 1/1 train_loss=-0.8361 eval_loss=-0.8361
test accuracy(eval)=7.812% chance=11.37%)
```
— `orchestrator/output-ablation/2017-11 .../state.json`, `attempts[0].checkup_message`

With `--retry-budget 0` the Orchestrator's own log states the consequence plainly: *"a live
check-up halted the run because training was not healthy, and the retry budget (0) is
spent"*. Verdict `retry_budget_exhausted`; no number to judge.

**What this does and does not isolate.** The halt happens in *both* arms — check-ups are
Runner behaviour, not loop behaviour. What the retry budget changes is whether a
regeneration follows the halt. So this paper does isolate the retry mechanism's value, but
only *conditionally*: it shows single-shot stops with nothing, not that a regeneration would
have fixed it. The loop-on 95.11% came from a script whose Eq. 3 a **human** had already
corrected, so it cannot settle the question. `TODO.md` records that a 2026-09-14
regeneration under the current prompt did implement the correct loss unaided — but that run
was only verified to `capped`.

### Tang — single-shot is a coin flip on this paper

Tang was run twice in the loop-off arm (the first leg was discarded for an unrelated reason;
see [caveat 4](#4-one-tang-leg-was-discarded-and-re-run)). Both legs were single-shot, same
pinned claim `c5`, same prompt, no retries. They disagree on the verdict:

| Tang loop-off leg | Reproduced | Verdict | `full` wall clock |
|---|---|---|---|
| **kept** (22:24–22:44, stable code) | **0.85%** | **`pass`** (gap −0.02, within ±0.1857) | 1139.99 s |
| **discarded** (21:57–22:21, no check-ups) | **1.17%** | **`fail`** (gap +0.30, recommendation `guided_retry`) | 1311.75 s |

The discarded leg's missing check-ups cannot explain this: check-ups halt runs, they do not
change a completed run's number, and that leg completed all 400 epochs and was judged
`healthy` after exit. The difference is **generation variance** — two different scripts
(similarity to the loop-on script 0.2834 and 0.2356 respectively) landing either side of the
tolerance band. **One single-shot generation per paper is therefore a noisy measurement, and
a 5-paper, 1-generation-per-arm design cannot resolve differences this size.** Tang's
loop-off "pass" in the table above is one draw from a distribution that also contains a fail.

---

## Totals

**Reproduced the claim (fidelity verdict):**

| Arm | `pass` | `fail` | `not_evaluated` |
|---|---|---|---|
| **loop-off** (single-shot) | **4 of 5** | 0 | 1 (soft tree) |
| **loop-on** | **5 of 5** | 0 | 0 |

**Failed to execute at all (loop-off):** **1 of 5** — the soft tree, halted at `probe` by a
check-up with no retry budget to repair it. The other four all reached `full` and produced a
number. No loop-off run crashed, timed out, or failed a syntax gate.

**Retries consumed in the loop-on arm:** 0 (Wijaya), 0 (SVM guide), 0 (Fashion-MNIST),
1 (soft tree), n/a (Tang — no orchestrated run exists). So on the artefacts available,
**the loop's retry machinery actually fired on exactly one of five papers.**

**Cost of the loop where it did nothing:** one Critic call per paper, plus the extra seed
runs it is entitled to spend. Four of five papers paid that and got the verdict single-shot
had already reached.

---

## What the loop actually contributed

Stated plainly, including where it contributed nothing.

**1. On 4 of 5 papers the loop contributed nothing measurable.** Tang, Wijaya, the SVM guide
and Fashion-MNIST all passed in both arms with zero retries in both. The single-shot script
was already good enough, so the loop had nothing to repair. **This is the main result of the
experiment and it is a negative one.** If ReproBot only ever met papers like these four, the
loop would be dead weight — it would add API cost, wall clock and machinery for no change in
outcome.

**2. On 1 of 5 the loop was the difference between a number and nothing.** The soft tree's
single-shot script was dead on arrival, and it failed in exactly the way this project already
knows about: the Coder independently reproduced the misprinted-Eq.-3 defect that cost a
hand-debugged run in September. Single-shot ends there with no result. Whether the loop would
have *repaired* it is untested here (see the soft tree section above).

**3. The most robust contribution is the guard rails, not the retries.** Note what did
*not* happen on the soft tree: the pipeline did not report 7.8% accuracy as a replication
result. The check-up refused to let a dead network produce a number, and the Critic refused
to judge a halted run, returning `not_evaluated`. That machinery fired with retries switched
**off** — so it is a property of the Runner and Critic, not of the loop. On the evidence
here, **ReproBot's main defence against publishing a wrong number is its guards; the retry
loop is a comparatively rare repair path.**

**4. The loop is variance insurance, and the variance is real.** The two arms' scripts are
very different texts — SequenceMatcher similarity **0.2834** (Tang), **0.3092** (Wijaya),
**0.3514** (SVM guide), **0.4079** (Fashion-MNIST) — yet four of five land on the same
verdict. The Coder's output varies enormously in *form* and, on easy papers, barely at all in
*outcome*. But Tang's two legs (0.85% pass vs 1.17% fail) show the tail is not empty even on
a paper that "passes". The loop's value is in catching that tail, which on this sample was
1-to-2 papers in 5.

**5. What would make this measurement actually conclusive.** The design is too thin to
support a strong claim either way: n=5, one generation per arm, two loop-on numbers
human-assisted, three loop-on states captured on a zero-retry day. To settle it, run **k
independent single-shot generations per paper** (k≈5) and report pass rates with intervals,
against a loop-on arm captured fresh under the same code. Tang's coin flip is the direct
evidence that single draws are not enough.

---

## Methodology, asymmetries and caveats

Six things a reader should not have to discover for themselves.

### 1. Two of the five loop-on numbers are human-assisted

Neither is a clean orchestrated `full` run:

- **Tang** has no `orchestrator/output/.../state.json` at all. Its 0.82% comes from a
  `runner/`-only invocation (`runner/output/.../runner_output.json`), on a script whose SVM
  `C` a human changed from 1.0 to 0.1 after a six-config ablation
  (`docs/notes/tang-2013-ablation/`). No loop ran.
- **The soft tree's** 95.11% comes from an orchestrated `full` run on 2026-09-14 00:59
  (`orchestrator/output/.../logs/attempt-1/full.stdout.log`) whose `state.json` was later
  **overwritten** by a 2026-09-14 13:14 rerun that only reached `capped`. That script's
  misprinted Eq. 3 had been fixed by hand.

So the loop-on arm is better described as **"loop plus the human attention this project
actually gave these papers"**. That inflates it relative to a pure loop-on arm — and it is
why the soft-tree divergence cannot be credited to the loop's retries.

### 2. The three loop-on states on disk show zero retries — they are a best case

Wijaya, the SVM guide and Fashion-MNIST were all re-run on 2026-09-20 during
report-generator work, and all three happened to succeed on the first generation
(`retry_count: 0`, one attempt, no extra seeds). `TODO.md` credits *earlier* runs of the same
papers with real repairs (the SVM guide's auto-repair; Wijaya's "first repair of a bug nobody
planted"; Wijaya's inconclusive → 2 seeds → pass). **Those repairs are not in the current
artefacts**, so this ablation cannot measure them. A loop-on arm captured on 2026-09-14 would
likely have scored the loop higher. Cuts both ways: it also means the loop-on arm's 5/5 is
not a demonstration of the loop earning its keep.

### 3. The two arms did not run under byte-identical Runner code

An unrelated agent was editing `runner/` for a different feature during this batch:

| Papers | `runner/checkups.py` | `runner/docker_runner.py` | Check-up rules |
|---|---|---|---|
| SVM guide, Wijaya, Fashion-MNIST | `e33602ca9f952cd0` (committed `d87f0f9`) | `cea0fd256b9c597a` | 6 |
| Tang (kept leg), soft tree | `9b2fa0c104cd0af3` (working tree) | `3438ad6f062ed2cd` | 7 (adds `no_first_record`) |

The new rule halts a stage that stays completely silent past a per-stage deadline (5400 s on
`full`, 900 s on `capped`). **It did not fire for either paper** — the deadlines were logged
and never reached. The asymmetry is small: six of the seven rules are unchanged, and the
three papers that ran under the older code were never close to any halt. But it is real, and
readers should know the arms were not frozen against one commit.

### 4. One Tang leg was discarded and re-run

The first Tang loop-off attempt (21:57–22:21) started while `runner/` was mid-rename and
picked up an inconsistent import contract: its check-up watcher thread died on every stage
with `TypeError: evaluate() got an unexpected keyword argument 'elapsed_seconds'`, so that
run had **no live check-ups**. Since check-ups are Runner behaviour present in both arms —
and Tang is the paper that collapsed into a dead network on its first-ever `full` run — a
loop-off leg without them is not comparable to the loop-on result. It was discarded and
re-run under stable code, with zero tracebacks.

The discarded leg's artefacts are kept at `data/discarded/` and its number is reported in
the [Tang variance table](#tang--single-shot-is-a-coin-flip-on-this-paper), because the
disagreement between the two legs is itself a finding. Nothing from it is used in the
headline table or the totals.

### 5. The archived scripts are no longer byte-identical to what ran

A repo-wide `ruff format` by the other agent reformatted the generated scripts in place after
they executed, so `script_chars` in the loop-off `state.json` files no longer matches the
file on disk:

| Paper | `script_chars` recorded | On disk now | Δ |
|---|---|---|---|
| SVM guide | 12,906 | 12,941 | +35 |
| Wijaya | 16,939 | 17,449 | +510 |
| Fashion-MNIST | 12,360 | 12,335 | −25 |

The code is semantically identical and **no reported number is affected**, but the archived
scripts are formatting-only variants of the exact text that ran. (Tang and the soft tree ran
after the reformat and are unaffected: Tang's recorded 16,432 matches disk.)

### 6. No stage code was modified for this experiment

The only writes outside the `*-ablation` output directories are this note,
`run-ablation.sh`, `collect-evidence.py`, `data/`, and four paths added to `.gitignore` so
the new ablation output directories are ignored exactly as `coder/output/` and
`orchestrator/output/` already are. Without that, `ruff check` descends into the generated
training scripts and reports 178 style errors in them.

---

## Every raw number and where it came from

Paths are repo-relative. The output directories are gitignored, so the key JSON is copied
into `data/` as committed evidence; `data/summary.json` is the machine-readable table.

### Loop-off arm (produced 2026-10-08)

| Number | Value | Source file |
|---|---|---|
| Tang execution verdict / retries | `success`, 0 retries, 1 attempt | `orchestrator/output-ablation/2013-06 .../state.json` → `verdict`, `retry_count` (copy: `data/loop-off/2013-06.state.json`) |
| Tang reproduced | 0.8499999999999952% test error | `critic/output-ablation/2013-06 ....json` → `reproduced` (copy: `data/loop-off/2013-06.critic.json`) |
| Tang verdict / tolerance / evidence | `pass`, 0.18573432639121937, `binomial noise of the evaluation set` (gap −0.02000000000000479) | same file |
| Tang `full` wall clock | 1139.99 s (training 1099.6 s) | `state.json` → `attempts[0].wall_clock_seconds`; stage log line |
| Tang script chars | 16,432 | `state.json` → `attempts[0].script_chars` |
| Wijaya reproduced | 2.5801572246107285 | `critic/output-ablation/2023-10 ....json` → `reproduced` (copy: `data/loop-off/2023-10.critic.json`) |
| Wijaya verdict / tolerance / evidence | `pass`, exceeds claim, 0.005, `reporting precision only (single run, no noise estimate)` | same file |
| Wijaya `full` wall clock | 135.41 s | `orchestrator/output-ablation/2023-10 .../state.json` → `attempts[0].wall_clock_seconds` |
| SVM guide reproduced | 96.625 | `critic/output-ablation/2003 ....json` → `reproduced` (copy: `data/loop-off/2003.critic.json`) |
| SVM guide verdict / tolerance / evidence | `pass`, 0.5480784615363017, `binomial noise of the evaluation set` | same file |
| SVM guide `full` wall clock | 89.42 s | `orchestrator/output-ablation/2003 .../state.json` |
| Fashion-MNIST reproduced | 0.8773, mean of `[0.8775, 0.8773, 0.8752, 0.8792, 0.8773]` | `critic/output-ablation/2017-08 ....json` → `reproduced`, `run_values` (copy: `data/loop-off/2017-08.critic.json`) |
| Fashion-MNIST verdict / tolerance / evidence | `pass`, exceeds claim, 0.0017955500549970756, `measured spread of 5 runs` (run spread 0.0008977750274985378) | same file |
| Fashion-MNIST `full` wall clock | 147.94 s | `orchestrator/output-ablation/2017-08 .../state.json` |
| Soft tree execution verdict | `retry_budget_exhausted` | `orchestrator/output-ablation/2017-11 .../state.json` → `verdict` (copy: `data/loop-off/2017-11.state.json`) |
| Soft tree halt rule / message | `loss_below_lower_bound`; `train_loss reached -0.8361, below the loss's own lower bound of 0 (epoch 1/1 ... test accuracy(eval)=7.812% chance=11.37%)` | same file → `attempts[0].checkup_rule`, `.checkup_message` |
| Soft tree fidelity verdict | `not_evaluated` — *the run did not succeed (runner status 'halted')* | `critic/output-ablation/2017-11 ....json` (copy: `data/loop-off/2017-11.critic.json`) |
| Soft tree stage wall clock | 6.18 s (`probe`, never reached `full`) | `state.json` → `attempts[0].wall_clock_seconds` |
| End-to-end per paper (incl. generation) | 1220 s (Tang), 232 s (Wijaya), 161 s (SVM guide), 213 s (Fashion-MNIST), 104 s (soft tree) | `data/timings.psv`, written by `run-ablation.sh` |

### Discarded Tang leg (not used in the table or totals)

| Number | Value | Source file |
|---|---|---|
| Reproduced / verdict | 1.1700000000000044% test error, `fail`, recommendation `guided_retry` | `data/discarded/2013-06.critic.DISCARDED-no-checkups.json` |
| Tolerance / gap | 0.18573432639121937, gap +0.3000000000000044 (relative +0.3448) | same file |
| `full` wall clock | 1311.75 s | `data/discarded/2013-06.orchestrator.DISCARDED-no-checkups.log` |
| Watcher failure | `TypeError: evaluate() got an unexpected keyword argument 'elapsed_seconds'`, twice | same log |

### Loop-on arm (produced 2026-09-13 → 2026-09-20)

| Number | Value | Source file |
|---|---|---|
| Tang reproduced | 0.8199999999999985% test error | `runner/output/2013-06 .../runner_output.json` → `reproduced_metrics.value`; also `coder/output/2013-06 .../metrics.json` |
| Tang verdict (judged here, same arithmetic) | `pass`, tolerance 0.18573432639121937, binomial | `critic/output-ablation-loopon/2013-06 ....json` (copy: `data/loop-on/2013-06.critic.json`) |
| Tang wall clock | 3222.67 s (training 3207.70 s) | `runner/output/2013-06 .../runner_output.json` → `wall_clock_seconds` |
| Wijaya reproduced / verdict | 2.8842813120953834, `pass`, exceeds claim, ±0.005 | `orchestrator/output/2023-10 .../state.json` → `critic_output` (copy: `data/loop-on/2023-10.state.json`) |
| Wijaya wall clock / retries | 140.07 s, 0 retries, 1 attempt | same file → `attempts[0]`, `retry_count` |
| SVM guide reproduced / verdict | 96.625, `pass`, ±0.5480784615363017, binomial | `orchestrator/output/2003 .../state.json` → `critic_output` (copy: `data/loop-on/2003.state.json`) |
| SVM guide wall clock / retries | 91.95 s, 0 retries, 1 attempt | same file |
| Fashion-MNIST reproduced / verdict | 0.8764799999999999, mean of `[0.8773, 0.8763, 0.8756, 0.8764, 0.8768]`, `pass`, exceeds claim, ±0.0007969943538068267 | `orchestrator/output/2017-08 .../state.json` → `critic_output` (copy: `data/loop-on/2017-08.state.json`) |
| Fashion-MNIST wall clock / retries | 177.97 s, 0 retries, 1 attempt | same file |
| Soft tree reproduced | 95.11% | `coder/output/2017-11 .../metrics.full.json` → `value`; identical in `orchestrator/output/2017-11 .../logs/attempt-1/full.stdout.log` |
| Soft tree verdict (judged here, same arithmetic) | `pass`, exceeds claim, tolerance 0.4578 (binomial, test noise 0.22895359791887962) | `critic/output-ablation-loopon/2017-11 ....json` (copy: `data/loop-on/2017-11.critic.json`) |
| Soft tree wall clock | 4190.75 s | `coder/output/2017-11 .../metrics.full.json` → `wall_clock_seconds` |
| Soft tree retries (later `capped` rerun) | 1 retry, 2 attempts: halted (`loss_below_lower_bound`) → regenerated (plateau 0.2324) → success at `capped` | `orchestrator/output/2017-11 .../state.json` (copy: `data/loop-on/2017-11.state.json`) |

### Claim values (the paper's own numbers)

All from `reader/output/<paper>.json` → `claims`, the entry whose `claim_id` matches:

| Paper | `claim_id` | Metric | Dataset | Reported |
|---|---|---|---|---|
| Tang 2013 | `c5` | test error | MNIST | 0.87% (DLSVM) |
| Wijaya 2023 | `c3` | RMSE | Boston Housing | 3.02 |
| SVM guide 2003 | `c1` | Accuracy by our procedure | Astroparticle | 96.9% |
| Fashion-MNIST 2017 | `c17` | Test Accuracy | Fashion-MNIST | 0.873 (RF n=100, entropy, depth 100) |
| Soft tree 2017 | `c1` | test accuracy | MNIST | 94.45% (depth 8, true targets) |

### Script similarity, loop-off vs loop-on

`SequenceMatcher(None, on_lines, off_lines, autojunk=False).ratio()` — the same measure and
the same `autojunk=False` the Orchestrator's plateau guard uses:

| Paper | Similarity | Loop-off lines | Loop-on lines |
|---|---|---|---|
| Tang (kept leg) | 0.2834 | 400 | 334 |
| Tang (discarded leg) | 0.2356 | 396 | 334 |
| Wijaya | 0.3092 | 454 | 387 |
| SVM guide | 0.3514 | 381 | 319 |
| Fashion-MNIST | 0.4079 | 373 | 338 |

### Verification

```
uv run ruff check                             # All checks passed!
uv run ruff format --check                    # 55 files already formatted
uv run --extra orchestrator pytest tests -q   # 92 passed
```

92, not the 81 of the previous baseline: the concurrent `runner/` work added tests to
`tests/test_checkups.py`. **This note changed no stage code** — `git status` shows no
modification under `ocr/`, `reader/`, `coder/`, `runner/`, `orchestrator/`, `critic/` or
`report/` attributable to it.
