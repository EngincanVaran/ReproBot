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
