# Three papers, three runs each, PDF to report

The ablation (`docs/notes/ablation/`) asked whether the retry loop earns its place, and the
consistency note (`docs/notes/consistency/`) asked how far one generation varies from the
next. Both ran stages in isolation. This one asks the question a demo has to answer:
**start at the PDF, finish at a replication report, three times per paper, and show the
combined result.**

Nine runs on 2026-10-09, three papers chosen to be CPU-sized and to span three model
families. Every paper was re-read from its PDF the same day, so the Reader output these
runs consumed is not inherited from an earlier session.

```bash
uv run --extra orchestrator python -m orchestrator.pipeline \
    --input "reader/output/<paper>.json" --max-stage full --force --no-build \
    --claim-id <id> \
    --output "orchestrator/output-e2e/run<N>" --coder-output "coder/output-e2e/run<N>"
# review on run 1 of each paper only; runs 2 and 3 add --no-review
uv run python -m report.pipeline --input orchestrator/output-e2e \
    --coder-output coder/output-e2e --output report/output-e2e --force
```

The full loop was on: retries, live check-ups, the Critic's verdict, its seed escalation and
its one guided fidelity retry. The Opus code review ran once per paper (run 1), because the
review reads the script, not the run, and three generations of one paper do not need three
reviews to establish that the method is faithfully implemented.

## Result

**9 of 9 runs reproduced their claim. Every run reached `full` on its first attempt** — no
retry was needed, which is itself a change from the consistency experiment, where one
generation in five was born broken.

| Paper | Claim | Passes | Reproduced, across 3 runs | Claim inside the spread |
|---|---|---|---|---|
| Hsu/Chang/Lin, SVM guide (c2) | 96.9% accuracy | **3 / 3** | 96.625, 96.925, 96.625 | **yes** |
| Fashion-MNIST, random forest (c41) | 0.873 accuracy | **3 / 3** | 0.87734, 0.87734, 0.87734 | n/a — identical |
| Wijaya 2023 (c3) | RMSE 3.02 | **3 / 3** | 2.763, 2.863, 3.036 | **yes** |

Per-run detail — wall clock, stage reached, tolerance, seed values, review findings — is in
`data/runs.json`; the generated summary table is `data/SUMMARY.md`. End-to-end wall clock per
run was 77–238 s, so the whole nine-run batch cost about 35 minutes of laptop CPU.

## What it says

**1. Every disagreement between runs traced to one unstated detail — and we could name it
each time.** This is the finding worth keeping, because it reframes variance as a property of
the paper rather than a defect of the system:

- **Fashion-MNIST states its protocol completely** — official 60k/10k split, raw pixels,
  `n_estimators=100`, `criterion=entropy`, `max_depth=100`, five repetitions with shuffled
  training data. Three independent generations - 303, 274 and 291 lines, with 299 and 290
  changed lines against run 1 - produced **bit-identical** values:
  `[0.8773, 0.8774, 0.8753, 0.8792, 0.8775]` every time. All three independently chose base
  seed 42, per-repetition `seed + i`, and `random_state=rep_seed` on both the shuffle and the
  forest. Run 2 additionally spelled out the library defaults (`min_samples_split=2`,
  `max_features='sqrt'`, `bootstrap=True`) — which *are* the defaults, so behaviour was
  unchanged.
- **The SVM guide says "5-fold cross-validation" and never says whether the folds are
  shuffled.** Runs 1 and 3 used `StratifiedKFold(shuffle=True, random_state=42)`, selected
  (C=8, γ=8) at CV 96.9894% and scored 96.625% on test — identically. Run 2 passed `cv=5`,
  which is scikit-learn's *unshuffled* contiguous folds, selected (C=2, γ=8) at a **lower** CV
  of 96.7952%, and scored **96.925%** on test. The grid search optimises cross-validation
  accuracy, and here the CV winner was the test loser: run 1's choice won CV by 0.19 points
  and lost on test by 0.30. The paper's own answer is (C=2, γ=2) → 96.9%.
- **Wijaya gives the 405/101 split sizes and not how the rows are assigned**, so each
  generation draws its own and the RMSE moves continuously: 2.763, 2.863, 3.036.

**2. The seed escalation fired on its own, and it mattered.** Wijaya's run 3 produced RMSE
3.1285 — worse than the claimed 3.02 with no noise model — so the Critic returned
`inconclusive` and asked for seeds rather than ruling. Two extra full runs gave 3.0283 and
2.9520, and the re-judgement was **pass at a mean of 3.03629 against 3.02, tolerance ±0.2044
now measured rather than assumed**. A system that ruled on one number would have called this
paper a failure at 3.13, or a success at 2.76, depending only on which run you saw.

**3. Not every pass is equally strong, and the report says which is which.** Wijaya's runs 1
and 2 passed by *beating* the claim on a single run, with a tolerance of ±0.005 that is only
the claim's own reporting precision — the Critic's `exceeds_claim` path. Run 3's pass is the
load-bearing one: three seeds, a measured spread, and a mean within 0.5% of the paper.
Pooling every Wijaya seed from the day gives 2.763, 2.863, 3.128, 3.028, 2.952 — a
distribution that straddles 3.02 instead of sitting beside it.

**4. The code review agreed with the arithmetic, independently.** All three run-1 reviews
returned `method_fidelity: faithful` with no `deviates_from_paper` or `implementation_bug`
finding, and every citation passed the deterministic guards (SVM 10/10 verified,
Fashion 8/8, Wijaya 12/15 — the three dropped are reported as dropped). The Wijaya review
also predicted its own result's direction without being told it: the unstated split
assignment "can move test RMSE by several tenths, which plausibly explains the reproduced
2.763 landing below the claimed 3.02."

## Honest limits

- **One claim per paper.** Fashion-MNIST has 74 extracted claims and this tested one; the SVM
  guide 27, Wijaya 4. Each report lists the untested remainder as `not tested`, so the
  denominator is visible, but nine runs is three claims, not three papers replicated.
- **Three runs is enough to see whether a paper varies, not to estimate how much.** The
  Fashion row says "identical", which is a real finding; the Wijaya row's three values are a
  three-sample sketch of a distribution.
- **All three papers already worked.** They were picked because they reproduce on CPU in
  minutes, so a 9/9 pass rate is a statement about the pipeline running end to end
  unattended, not about ReproBot's success rate on an arbitrary paper. The CIFAR-10 set still
  has no fidelity verdict at all, because `full` there needs a GPU.
- **Review on run 1 only.** Runs 2 and 3 were judged by arithmetic alone, so a deviation that
  appeared in exactly those generations would not have been caught by the reviewer.

---

# Addendum, 2026-10-10: a paper the pipeline had never seen

The three papers above were the papers this project debugged against. None of them answers
the question a reviewer actually asks — *does it work on a paper you have not tuned on?* So
the same protocol was run on a fourth: **Isolation Forest** (Liu, Ting & Zhou, ICDM 2008),
never OCR'd, never read, never coded before today. Target claim **c9: AUC 0.67 on Pima**,
three runs, full loop, review on run 1.

Pima was chosen deliberately: 768 rows on OpenML so it costs seconds, a **mid-range** value
that cannot be hit by luck (Breastw 0.99 and Shuttle 1.00 are near-ceiling and would pass
almost anything), and the paper's own **weak** row — iForest's 0.67 loses to ORCA's 0.71,
which is boldfaced as best. The Reader noticed that on its own and annotated the claim.

## Result: 3 / 3 pass, first attempt every time, no retries

| Run | Reproduced | How it passed | Tolerance | Evidence behind the tolerance |
|---|---|---|---|---|
| 1 | 0.682799 | `exceeds_claim` — better than 0.67 beyond tolerance | ±0.005 | reporting precision only |
| 2 | 0.678728 | **within tolerance** | ±0.011 | **measured spread of 10 runs** |
| 3 | 0.674582 | within tolerance (gap 0.0046) | ±0.005 | reporting precision only |

Spread 0.6746 – 0.6828, mean 0.6787. End-to-end wall clock 20–33 s per run. The claim 0.67
sits just **below** the spread: all three runs overshoot slightly, so this is a consistent
small positive bias rather than noise around the claim. `sklearn`'s `IsolationForest` is the
same authors' algorithm but not the same code, and the paper never says whether its 0.67 is
one draw or a mean — either would explain a few thousandths.

## What this one added that the first three did not

**1. A task shape nothing else in the project has.** No train/test split: the forest is fit
unsupervised on the whole contaminated set and the labels are used only afterwards, to score
AUC. The Coder produced a working script for it on the first attempt, with no retry, having
never seen the paper.

**2. The same unstated-detail pattern, found again and predicted in advance.** Runs 1 and 3
reproduced a **single seeded run**; run 2 implemented the paper's **ten-repeat protocol**.
That is the entire difference in evidentiary quality between the rows above — and the Opus
review on run 1 had already named the ambiguity before run 2 demonstrated it:

> Table 3 does not state whether its AUC is a single run or an average (the ten-repeat
> protocol is stated only for Section 5.4). The script reports one seeded run, so the 0.6828
> vs 0.67 margin carries no variance estimate.

So the reviewer identified the unstated choice, and an independent generation then resolved
it the other way. Four papers, four different mechanisms, the same finding each time: the
spread is the paper's under-specification, not the system's noise.

**3. AUC exercised a path through the Critic that nothing else had.** AUC gets no binomial
noise model, so a single run's tolerance is the claim's reporting precision alone (±0.005) —
which is why run 3's pass, at a gap of 0.0046, is *just* inside. Run 2's ten repetitions gave
the Critic a measured spread instead, and a ±0.011 band. Same claim, same verdict, two very
different strengths of evidence, and the report records which is which.

## Two bugs it surfaced, both silent

**The VLM refused 3 of its 10 pages** — reproducibly, at two render scales, with
`Output blocked by content filtering policy`. The paper's datasets are KDD Cup 99 intrusion
traffic, which is the likeliest trigger. Worse, `ocr/vlm_extract.py` wrote nothing until every
page succeeded, so one refusal produced **no output at all**. It now records a marker where a
refused page belongs and continues; the 7 surviving pages carried Table 3 complete, so all 12
AUC claims were extractable. Without that fix this paper could not be read.

**`model_family` was misclassified on run 3**, which degraded a stage with no error anywhere.
Runs 1 and 2 declared `classical`, so their `capped` stage scaled *data*; run 3 declared
`neural`, whose `capped` scales *epochs* and keeps smoke's caps. Epochs mean nothing to a
forest, so run 3's `capped` was a byte-identical repeat of its `smoke` — same 0.61669921875,
same 512 train / 256 eval — while reporting PASSED and printing a comment about a net
overfitting. The judged `full` number was unaffected.

Both are recorded in `TODO.md`. Per-run evidence: `data/iforest-runs.json`, combined table
`data/iforest-SUMMARY.md`.

## Honest limits of this addendum

- **One claim of 24.** The Reader extracted 24 claims (12 AUC, 12 processing times) and left
  **7 flags unresolved after its 3-pass cap**, two of them substantive: Section 5.4's prose
  AUC values and Figure 7's high-dimensional results were never captured at all.
- **The ladder collapsed.** Pima has 768 rows and `capped` caps at 5,000/2,000, so for runs 1
  and 2 `capped` *was* `full` — identical to 16 decimals. The escalation saved nothing here.
- **Still a small CPU paper**, chosen for the same reason the other three were.
- **It passed.** The project still has no paper that ReproBot honestly reports as *not*
  replicating; every `fail` on record comes from a deliberately injected bug.
