# critic/ — does the reproduced number match the paper?

`runner/` says whether a script **ran** and its live check-ups say whether it was
**learning**. Neither says whether the result **matches the paper**. That is this
stage's job. It reads the paper's claims (`reader/`) and the reproduced value
(`metrics.full.json`) and returns a verdict. Every verdict comes with the numbers and
the rule behind it.

The Critic has two halves (project plan §2.5):

- **The verdict** (`judge.py`, v1) is arithmetic. It decides `pass` / `fail` /
  `inconclusive` / `not_evaluated`, and no model can change it.
- **The review** (`review.py`, v2) is one model call (Opus 5 by default). It reads the paper, the script
  and the run, and explains the verdict: is this the paper's method, what deviates, and
  what should the Coder change? Its output passes deterministic checks before anything
  reaches the Coder. See "Critic v2: the review" below.

**The verdict is arithmetic, not a model.** It makes no API call, runs no Docker and needs no
network. Project plan §2.5 is explicit that numeric comparison must not be left to
an LLM, which may read 0.82 and 0.87 as "close" one time and "different" the next.
Every threshold below is a formula you can check by hand.

```
critic/
├── claims.py    merge duplicate claims (one result printed in prose, table and figure)
├── judge.py     tolerance + verdict for one claim (v1 guided-retry text as a fallback)
├── review.py    v2: model review of script vs paper, guards, feedback to the Coder
└── pipeline.py  CLI: judge (and --review) an Orchestrator state, or reader JSON + metrics
```

## Verdicts

| Verdict | When | What the Orchestrator does |
|---|---|---|
| `pass` | \|gap\| ≤ tolerance, **or** the result is better than the claim beyond it (`exceeds_claim: true`) | accepts |
| `fail` | worse than the claim beyond tolerance, **with** an uncertainty estimate behind the tolerance | one guided fidelity retry, then final |
| `inconclusive` | worse beyond tolerance, but **no** uncertainty estimate exists (one run of RMSE, say) | runs two extra seeds if the full run was cheap, then judges again |
| `not_evaluated` | nothing comparable: the run did not succeed, produced no number, came from a stage cheaper than `full`, or targeted an unknown claim | accepts |

Two verdicts are deliberately *not* failures:

- **A result better than the claim passes.** A reproduction that beats the paper
  shows the claimed result is achievable with this implementation, which is the
  question a replication answers. The flag is kept so a report can say so.
- **Only `full` counts.** `probe`, `smoke` and `capped` are execution checks: a
  probe is two optimizer steps, and comparing its 92% error with a claimed 4% would
  give a confident but meaningless `fail`.

Execution verdicts (`state.verdict`: `success`, `timeout`, …) and fidelity verdicts
(`critic_output.verdict`) stay separate fields, so a `success` can never be misread
as "replicated".

## Tolerance

```
tolerance = max(2 × uncertainty, reporting precision)
```

**Reporting precision** is the floor. A paper that prints 96.9 cannot be matched more
closely than ±0.05, which is half a unit of its last digit. It is read from the
number's shortest repr. JSON drops trailing zeros, so 0.910 arrives as 0.91 and the
floor is conservative (wider, never narrower).

**Uncertainty** comes from the best evidence available, in this order:

1. **Measured run spread.** Used when several runs exist, either because the claim
   is itself a mean over N runs (Fashion-MNIST's random forest, mean of 5) or
   because the Orchestrator ran extra seeds.
   - When the claim is a mean over the same number of runs, both sides carry
     `std/√n`, so the comparison uses `std·√(2/n)`.
   - When the claim is a single run, it is one draw from the spread, compared with
     our mean: `std·√(1+1/n)`.
2. **Binomial test-set noise.** For accuracy-like metrics (a `%` unit, or a 0–1
   accuracy or error) it is `√(p(1−p)/n)` on the evaluation set's size. This stands
   in for run-to-run variation when only one run exists. On MNIST's 10,000 test
   images, a 0.87% error has a noise of 0.093 points.
3. **None.** One run of a metric with no sampling model (RMSE, MSE, R²). The
   tolerance is the reporting precision alone. A gap in the losing direction is then
   `inconclusive` rather than `fail`: it may be one unlucky split.

The factor 2 is a roughly 95% band under a normal model. `SIGMA_MULTIPLIER` in
`judge.py` is the one place to change it.

## The real results, judged

These are the five full-fidelity replications, re-judged by this stage. Each verdict
matches the one reached by hand. They are the replay cases in `tests/test_critic.py`.

| Paper | Claimed | Reproduced | Tolerance (evidence) | Verdict |
|---|---|---|---|---|
| Tang 2013, DLSVM on MNIST | 0.87% error | 0.82% | ±0.186 (binomial, n=10,000) | **pass** |
| Hsu/Chang/Lin, SVM guide | 96.9% | 96.925% | ±0.548 (binomial, n=4,000) | **pass** |
| Fashion-MNIST, random forest | 0.873 | 0.8773 (mean of 5) | ±0.0018 (spread of 5 runs) | **pass**, exceeds claim |
| Frosst & Hinton, soft decision tree | 94.45% | 95.11% | ±0.458 (binomial, n=10,000) | **pass**, exceeds claim |
| Wijaya 2023, Boston Housing (old script) | RMSE 3.02 | 4.48 (one run) | ±0.005 (precision only) | **inconclusive** → run seeds |
| Wijaya 2023, **end to end through the Orchestrator** | RMSE 3.02 | 3.33 (mean of 3.75, 2.84, 3.40) | ±1.07 (spread of 3 runs) | **pass** |

Wijaya is the interesting one. A single RMSE on a 101-row test split carries no noise
estimate, and the paper never says how its split was drawn. A lone number could come
from a bad split as easily as from a bad model, so the Critic asks for more seeds
instead of calling it a fail.

That is exactly what happened on the first end-to-end run (2026-09-14).
1. A regenerated script trained healthily through all four stages. Its full run
   (182 s) gave RMSE 3.75: `inconclusive`.
2. The full run was well under the 30-minute seed budget, so the Orchestrator ran
   `seed2` and `seed3` (about 2 minutes each). Each seed also redraws the unstated
   split, and they gave 2.84 and 3.40.
3. `seed2` beats the paper's 3.02, and `seed3` falls between the paper and the first
   run. Three runs, mean 3.33, standard deviation 0.46: `pass`, with no retry and no
   human.

The spread is the finding. The paper's 3.02 is well inside the variation that a split
choice alone produces on this dataset. The single 4.48 from the earlier script was
never evidence of a failed replication.

## Duplicate claims

A paper often prints one result several times: in the prose, in a table and in a
figure caption. The Reader extracts each occurrence as its own claim. Wijaya's 14
claims are 8 results. Test RMSE is both `c8` ("testing set") and `c11` ("Proposed NN,
Table 3"). Test R² appears as 0.911, 0.91 and 0.91 under three spellings (R-Squared,
R2, R-square).

`group_claims` merges two claims only when all of these hold:

- **Same result:** same dataset, same unit, and the same normalised metric
  (R-Squared / R2 / R-square → `r2`).
- **Same value:** the values agree within the coarser of their two reporting
  precisions.
- **Same split:** the variants do not name different splits. Train vs test never
  merge.
- **Same model:** the variants do not name different models. A variant's "model
  words" are what remains after removing split words, filler words ("proposed",
  "model", "set") and location references with their numbers ("Table 3",
  "Figure 6"). Other numbers stay, since "depth 40" and "depth 22" are different
  networks.

The group keeps its **most precise** member as the canonical claim, so R² is judged
against 0.911, not 0.91. **The Reader's output is never edited.** The merge happens
at judging time and every verdict lists `merged_claim_ids`.

Checked against all 9 papers' extractions (243 claims): only Wijaya's four groups
merge. Two near-misses were caught and now stay apart, and both are regression
tests:

- Fashion-MNIST: LinearSVC and LogisticRegression both reach 0.917 on MNIST.
- Wide Residual Networks: depth 40 and depth 22 both report 5.78%.

## Inside the Orchestrator

The Critic runs automatically after any attempt whose execution finished
successfully (`--no-critic` turns it off). The routing is a pure function,
`decide_after_critic` in `orchestrator/loop.py`, unit-tested in
`tests/test_loop_critic.py`:

```
success ──► judge ──► pass / not_evaluated ─────────────────────► accept
              │
              ├─► inconclusive ──► full run ≤ --seed-budget (1800 s) and
              │                    reproduce.sh has seed2|seed3?
              │                      yes ─► run seed2, seed3 ─► judge again (once)
              │                      no  ─► accept as inconclusive
              │
              └─► fail ──► fidelity retries left (--fidelity-retry-budget, 1),
                           retry budget left, and the Coder recorded unstated choices?
                             yes ─► regenerate with guided feedback ─► run ─► judge
                             no  ─► accept as fail
```

**Extra seeds** run as `reproduce.sh seed2` and `reproduce.sh seed3`. That is the
same full command with `--seed 2`/`--seed 3` and `metrics.seed<N>.json`, generated by
`coder/`'s template, so the Runner still never builds a command itself. They run only
when one full run took at most 30 minutes. A 22-day WRN run will never silently
triple.

**The guided retry** is deliberately narrow:

- It happens **once per paper**.
- The feedback states the gap and the tolerance, then lists the Coder's own
  recorded `assumptions` (the settings the paper leaves unstated) as the **only**
  things it may revisit. It also says to "never change a value the paper states".
- Every change must be recorded as a `fidelity retry:` assumption, with old value,
  new value and a justification from the paper, the library's defaults or standard
  practice. Wanting the test number to move is not a justification.
- It is skipped when the Coder recorded no unstated choices, because then a retry
  could only tune toward the test set.

**Not yet exercised live.** No real run has produced a `fail` since the Critic was
built, so the guided retry is verified only by the routing tests and the feedback-text
test.

This is the Tang lesson made mechanical: a human changed one unstated guess (SVM `C`
1.0 → 0.1) after an ablation, and it replicated. One retry, limited to guesses, keeps
that kind of fix while preventing a loop that tunes hyperparameters against the test
split.

Every judgement is kept in `state.critic_output`: the latest at the top level, with
`seed_run_values` and `script_version`, and the full list under `judgements`. Each
attempt also records its `critic_verdict`. The fidelity retry count is
`state.fidelity_retry_count`.

## Critic v2: the review

**Why.** The arithmetic says *whether* a number matches, never *why not*, and it can't
tell whether a passing script is the paper's method at all. Every real defect found in
this project so far was caught by a person reading code or logs:
- the soft tree implemented the misprinted Eq. 3 and trained in the wrong direction;
- Tang's `C=1.0` collapsed the network;
- the old NIN script used gradient clipping the paper never mentions.

**What it reads.** After a judged full run, one call (`claude-opus-5` by default; see below
for why not Sonnet) gets:
- the verdict facts (claimed, reproduced, run values, gap, relative gap, tolerance);
- the run evidence: final metrics and a compact learning curve (evenly spaced records, the
  best eval point, chance level, target);
- the Coder's bookkeeping (`hyperparameters_used`, `assumptions`);
- the Reader extraction (method summary, architecture notes with equations,
  hyperparameters, data pipeline);
- the full stage's runner log with per-epoch lines removed;
- the line-numbered script and the paper's Markdown.

**What it returns** (forced tool use, `review_replication`):
- `method_fidelity`: `faithful` / `minor_deviations` / `major_deviations`.
- `findings`, each one of `matches_paper`, `deviates_from_paper`, `unstated_choice` or
  `implementation_bug`, with a severity, an exact paper quote (or "not stated"), script
  lines and an exact code snippet.
- `hypotheses`, ranked, each with the concrete change it implies and a `change_type`
  (`fix_stated_deviation`, `fix_implementation_bug`, `revisit_unstated_choice`,
  `run_more`).
- A short `summary` and `curve_assessment`.

**Guards (deterministic, `verify`).** A finding or hypothesis that fails any of these is
kept in the record with its problems, marked unverified, and **never fed to the Coder**:
1. **Numbers.** Every number in its prose must appear in the material it was given, at the
   precision it was written ("3.0" matches 3.02).
   - Integers up to 10 are exempt.
   - "1,000" and "[237,239]" are both read correctly.
   - A number after "line" passes if the script has that line.
2. **Quotes.** A paper quote must occur in the paper, the extraction or the Coder's
   bookkeeping (case, whitespace and quote marks ignored; `...` joins fragments). A
   `matches_paper` or `deviates_from_paper` finding cannot cite "not stated".
3. **Script references.** Lines must exist, and the snippet must occur in the script
   (`...` allowed).

**Routing** (`decide_after_critic` in `orchestrator/loop.py`):

| Verdict | Review | Action |
|---|---|---|
| `fail` | verified high/medium `deviates_from_paper` or `implementation_bug` | **`fix`**: a correctness retry on the normal retry budget. Feedback lists the verified problems with quotes and lines plus the fix hypotheses; changes are recorded as `deviation fix:` assumptions |
| `fail` | none | the one guided fidelity retry, now with the review's verified `revisit_unstated_choice` hypotheses (`fidelity retry:` assumptions) |
| `pass` | stated problems found | accepted (a pass is final). The problems are logged loudly and recorded for the report |
| any | the call fails | logged, recorded in history, and the loop continues with v1 behaviour |

Making the script do what the paper states is never tuning, which is why `fix` does not
use the one fidelity retry.

### The first Critic loops (Wijaya, 2026-09-14)

**Setup.** The correct Wijaya script passes, so nothing loops. To exercise the loop, one
classic bug was injected, like the earlier injected-runtime-bug test of the Orchestrator:
the validation split's `train_test_split` return order was swapped. The model then trains
on 81 rows instead of 324, while still learning and never tripping a check-up. The run
started with `--use-existing-script --max-stage full`, and every retry regenerates normally.

**Same bug, two reviewer models:**

| | Sonnet 5 reviewer | **Opus 5 reviewer** |
|---|---|---|
| Attempt 1 verdict | fail: 4.76 vs 3.02 (4.95, 4.45, 4.87; ±0.62) | same numbers (the run is deterministic) |
| Review of attempt 1 | `minor_deviations`. **Missed the swap.** Filed "trains the stated 1000 epochs with no early stopping" as a high-severity bug | `major_deviations`. **Found the swap**: cited the log line "81 train_fit, 324 val" and lines 178–180, plus the scaler fitted on the wrong block. 12/12 findings verified |
| Route | `fix` retry, with the wrong diagnosis | `fix` retry, with the right one |
| Attempt 2 script | split fixed (only because it was rewritten from scratch), **plus checkpoint selection the paper never uses** | split and scaler fixed as two `deviation fix:` entries, nothing else changed |
| Attempt 2 verdict | pass: 2.96 | **pass: 3.40** (3.92, 3.19, 3.09; ±1.05), in line with the correct script's 3.33 |
| Review of attempt 2 | flagged the checkpoint selection its own advice caused | `faithful`, 8 matches, 5 unstated choices, 0 problems, 13/13 verified |

The Opus loop took 11 minutes and one retry, with no human involved.

**What this changed:**
1. **The default review model is `claude-opus-5`.** Sonnet also missed the bug in two
   offline re-reviews: once with a data-tracing prompt, and once with the runner log in
   front of it saying "81 train_fit, 324 val". Opus found it on its first try.
   `--review-model claude-sonnet-5` is the cheaper option.
2. **The review reads the runner log** (the full stage's own lines, with per-epoch
   progress removed), and the prompt asks it to trace the data flow.
3. **A deterministic add-on-technique guard.** Checks on citations prove a quote is real,
   not that the reasoning is right. So a fix may not add early stopping, checkpoint
   selection, dropout, weight decay, clipping, a learning-rate schedule, augmentation or
   ensembling unless the paper itself (not the extraction, which lists absences) mentions
   it. A "bug" also cannot be the absence of such a technique. This would have blocked the
   Sonnet misdiagnosis.
4. **Snippet checks go line by line.** Opus stitched non-adjacent lines into one snippet,
   and every line was real.

**Honest limits:**
- One injected bug and one paper is a demonstration, not a measurement.
- With Opus the fix loop worked; with Sonnet the arithmetic still ended in `pass`, but on
  a script that no longer matched the paper. The review's own next pass caught that, and a
  report must show it.
- Every retry still regenerates the whole script, so "the fix" is partly luck until loop
  controls add patching.

**Seen on the first real review (Wijaya's correct script, 2026-09-14).**
- It rated the script `faithful`, with 5 `matches_paper` and 5 `unstated_choice` findings
  (OpenML loader, batch size 32, PyTorch init vs Keras Glorot, Adam ε 1e-7, validation
  split usage).
- **The model leaked its whole `findings` array inside `curve_assessment`.** It was the
  same failure the Coder had hit before. `reader/tooluse.py::recover_leaked_fields` now
  splits it back out. The raw payload is the replay fixture in `tests/test_critic_review.py`.
- After recovery, 9 of 10 findings passed the checks. The 10th had shortened a code snippet
  with `...`, and elided snippets are now allowed.

## Usage

```bash
# judge every Orchestrator state (no API key, no Docker)
uv run python -m critic.pipeline --state orchestrator/output

# judge one and store the verdict in its state.json
uv run python -m critic.pipeline --state "orchestrator/output/<paper>" --write

# add the model review of the script against the paper (one API call, Opus 5)
uv run --extra orchestrator python -m critic.pipeline --state "orchestrator/output/<paper>" --review

# judge a run made without the Orchestrator, optionally with extra seeds
uv run python -m critic.pipeline --reader-json "reader/output/<paper>.json" \
    --metrics-json "coder/output/<paper>/metrics.full.json" \
    --extra-metrics "coder/output/<paper>/metrics.seed2.json" "coder/output/<paper>/metrics.seed3.json"
```

Output goes to `critic/output/<paper>.json` (gitignored). With `--reader-json` it also
includes every claim group.

## Limits

- **It judges one claim per run**, the one the Coder targeted. A claim-by-claim table
  over every claim is the Report Generator's job.
- **Binomial noise is a proxy.** It measures test-set sampling, not training
  variance. When real run spread exists it replaces the proxy, but a single-run
  accuracy result is judged on the proxy.
- **Higher-is-better** comes from `metrics.json` and is inferred from the metric name
  only when missing. A metric whose name says neither ("score") can pass within
  tolerance but can never be flagged as better or worse beyond it.
- **A pass is only as strong as its tolerance.** Wijaya passes with ±1.07 on a claim
  of 3.02, a band of 35%. That is honest: the three splits really do vary that much.
  But it means "the claim is consistent with this implementation", not "matched
  closely". A report must print the tolerance next to every verdict, and should call
  out passes whose band is wide relative to the claim.
- **Two seeds are a small sample.** Three values give a rough spread. They are
  enough to tell "one bad split" apart from "consistently off", but not to estimate
  a tight interval.
