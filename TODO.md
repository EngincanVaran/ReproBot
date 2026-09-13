# ReproBot — status and open work

Living tracker. Updated by hand as things land; the narrative history with full detail
lives in [`docs/agent-log.md`](docs/agent-log.md), and each stage's own README holds the
deep version of its known issues.

**Last updated:** 2026-09-13 · `main`

---

## Resume here

**The two-paper generalization test is finished (2026-09-13)** and produced ReproBot's
**first two fidelity measurements**, compared by hand since the Critic doesn't exist:

| Paper | Claim | Reproduced (`full`) | Reading |
|---|---|---|---|
| **Tang 2013** — MNIST, L2-SVM loss | test error **0.87%** | **0.82%** | Replicates — gap is ~½ a binomial SE. **But** only after a human changed one unstated guess (`C` 1.0 → 0.1, from the ablation). |
| **Wijaya 2023** — Boston Housing | test RMSE **3.02** | **4.48** (train 2.94 vs 2.69) | Does not replicate on this split. Validation RMSE sat at ~3.2–3.5 all run; the paper never says how its 405/101 split was drawn, so one seed can't separate "wrong model" from "unlucky split". |

- **Wijaya** went through the Orchestrator: attempt 1 crashed (`fetch_openml` given both
  `data_id` and `version` — my prompt's wording invited it, now fixed), triage returned the
  right one-line fix, attempt 2 passed probe → smoke → capped → full. **The retry loop's
  first repair of a bug nobody planted.**
- **Tang** detail, epoch log and caveats: [`docs/notes/tang-2013-ablation/`](docs/notes/tang-2013-ablation/README.md).
- Logs: `orchestrator/output/2023-10 - .../logs/attempt-{1,2}/`, `runner/output/2013-06 - .../logs/`.

**Classical-ML run (later 2026-09-13).** `coder/` gained scikit-learn support; three more papers went
through the Orchestrator (`--max-stage full --retry-budget 2`):

| Paper | Claim | Reproduced | Reading |
|---|---|---|---|
| **Hsu/Chang/Lin SVM guide** (c1) | svmguide1 96.9% (appendix 96.875%) | **96.625%** | `success` after 1 retry. Implementation exact: `SVC` at the paper's C=γ=2 gives 66.925 / 96.15 / 96.875% exactly; the grid search picked C=γ=8 (CV 96.99% vs paper's 96.89%) because fold assignment differs. |
| **Fashion-MNIST random forest** (c17) | 0.873, mean of 5 | **0.8773** (runs 0.8753–0.8792) | `success`, 0 retries, 200 s. All 5 runs above the claim; likely library version, not isolated. |
| **Soft decision tree** (c1) | MNIST 94.45% | none | Retry fixed an in-place autograd crash, then every check stage **passed** with accuracy 5–9% (below chance) and a **negative, constant loss** — broken objective. Full 40-epoch run was left running; it cannot learn. |

**Third progress report written (13.09.2026)** —
`docs/progress-reports/third-progress-report/`, single-column (14 pp) and two-column (11 pp),
both built from `generate.py`. Covers everything above plus Mert's `viewer/` branch. Every page
was rendered and checked by eye; wide floats set with `\raggedbottom` so two-column pages don't
stretch. Regenerate rather than hand-editing either `.tex`.

---

## Where the pipeline is

**CIFAR-10 set** (`dataset/`, curated by someone else):
```
8 PDFs  →  6 OCR'd  →  4 read  →  2 coded  →  2 executed  →  2 orchestrated
```
**Generalization test set** (`extra-papers/`): Tang 2013 and Wijaya 2023 — both OCR'd,
read, coded and run through `full` (Wijaya via the Orchestrator). Results above.

| Stage | Status | Verified by |
|---|---|---|
| `ocr/` | ✅ built | 8 papers extracted (6 CIFAR + 2 new) |
| `reader/` | ✅ complete vs. §1.2 | 6 papers, all 5 fields |
| `coder/` | ✅ built, **task-agnostic** since `dc5a541` | NIN regression test; Tang + Wijaya generated |
| `runner/` | ✅ built | real containers through every stage, including `full` |
| `orchestrator/` | ✅ built | NIN + WRN `success`; repaired an injected bug on 1st retry |
| **`critic/`** | ❌ **not started** | — |
| **report generator** | ❌ **not started** | — |

**The honest summary:** the machinery works end to end, now for more than image
classification, and two reproduced numbers exist — one matching its claim, one not. Both
comparisons were done by hand. The first `full` run (Tang, `C=1.0`) collapsed with **no
stage noticing** and would have been reported `success`; the Critic is still the most
important missing piece.

---

## Next up

- [ ] **Nine new diversity papers in `extra-papers/`** (2026-09-13) — SVM, scikit-learn model zoo,
      GCN, text CNN, TCN, soft decision tree, XGBoost, CatBoost (LightGBM baselines), XGBoost-vs-RF-vs-GB;
      details in `extra-papers/README.md`. Five need **classical-ML support**: scikit-learn in
      `coder/`, and xgboost/lightgbm/catboost in `runner/`'s image (plan proposed to Engincan,
      awaiting go-ahead). None is OCR'd yet (~124 VLM page calls for all nine).
- [ ] **Build `critic/`** — compare `metrics.json` `value` against `claims[].reported_value`
      under an explicit tolerance, using `higher_is_better` from the new metrics contract.
      Deliberately arithmetic, not an LLM judging numbers (project plan §2.5). Tang's
      collapse is the concrete case it must catch. **Wijaya shows a single run is not
      enough:** the tolerance has to account for seed/split variance, so the Critic likely
      needs several seeds before it can call a gap a failure.
- [ ] **Make the Runner check learning, not just exit codes** — `capped` claims to answer
      "does it actually learn?" but nothing reads `train_metric`. Smaller than the Critic,
      and would have flagged a non-learning run early.
- [ ] **Secure GPU compute** — still gates fidelity for the CIFAR-10 papers (~22 days per
      real WRN run on this CPU). Small targets like Tang and Wijaya sidestep it.
- [ ] **Build the report generator** — `orchestrator/state.py` already carries everything.
- [ ] **Single entry point** — one command, PDF through all five stages. Blocked in part by
      the exit-code bug below.
- [ ] **Fix the progress report's rendering issues** — noted 2026-08-22, not catalogued.
      Regenerate via `docs/progress-reports/second-progress-report/generate.py`.
- [ ] **Review and merge Mert's `viewer/` branch** (`origin/mert/runner-agent`, a Streamlit
      dashboard) — described in the third report at Engincan's request, but not reviewed or
      merged. It predates `dc5a541`, so reconcile `pyproject.toml`/`uv.lock` first; it shows
      OCR/Reader/Coder output only, not Runner/Orchestrator.

---

## Open bugs and gaps

Anything marked **cost** is actively wasting money or time on every run.

### Across stages
- [ ] **`ocr/`, `reader/` and `coder/` exit 0 even when every paper fails.** Only `runner/`
      and `orchestrator/` set a failing exit code. This is how a Pillow import error went
      unnoticed on the first OCR attempt of 2026-09-13. It must be fixed before anything
      chains stages together.

### `ocr/`
- [ ] **2 of 8 CIFAR papers cannot be extracted** — *Stochastic Depth* and *DenseNet* fail on
      page 2 with a deterministic Anthropic content-filter false positive (model-side).
- [ ] **Per-page resilience missing** — one bad page discards the whole paper.
- [ ] **Duplicated figure captions** — *All Convolutional Net*, Figures 5 and 6.
- [ ] **Hallucinated heading** — *AutoAugment* page 5, `# Page Content`.

### `reader/`
- [ ] **cost — validation never converges.** Every paper ends with flags left after the
      3-pass cap (NIN 4, All-Conv 5, ResNet 8, WRN 6, Tang 5, Wijaya 5). Many are the
      validator raising a concern and dismissing it inside the same description.
- [ ] **No field records a MISSING hyperparameter.** `unstated_details` only covers
      architecture, so values like Tang's momentum and SVM `C` — both unstated, and `C`
      sits in the very loss being reproduced — are flagged nowhere. They were exactly the
      two guesses that collapsed Tang's run. Extend the gap-recording discipline to
      `hyperparameters`.
- [ ] **Dataset/benchmark papers have no "own method"** — Fashion-MNIST's first claims pass kept 0 of 124
      table rows and found no architecture; only the validator's flag recovered 26 claims.
- [ ] **The same result is claimed two or three times** when a paper states it in prose, a
      table and a figure (Wijaya: 14 claims for 8 distinct results, at mixed precision
      0.911 vs 0.91). Needs dedup before the Critic compares against them.
- [ ] **WRN ImageNet hyperparameters gap** — 14 ImageNet claims, zero hyperparameters.
- [ ] **Intermittent all-empty tool payload** — caught and retried; root cause unknown.

### `coder/`
- [ ] **Generated scripts time `wall_clock_seconds` from the training loop, not the start of
      the run** — the NIN regression script missed ~22 s. The contract says whole-run.
- [ ] **Intermittent tool-field leak** — worked around in `_recover_leaked_fields`, not fixed.
- [ ] **Bookkeeping can disagree with the code** — NIN reported stage 3 as `192→192→10`, the
      code built `192→10→10`. No static gate catches this.
- [ ] **Priors still supply unstated numbers** — disclosed, so *visible*, not *verified*.
- [ ] **Tang's generated setup is borderline unstable** — momentum 0.9 + `C=1.0` at lr 0.1
      collapses in a seed-dependent way. Each guess was a reasonable default; the pair was
      not. Longer term this argues for disclosing *combinations* of guesses, or a cheap
      stability probe, not just listing each guess. (Worked around by hand for the rerun:
      `C=0.1` in the gitignored script — a regeneration would bring `C=1.0` back.)
- [ ] **The Keras-defaults rule is only partly followed.** Wijaya's script matched Keras's
      Adam ε (1e-7) but used PyTorch's BatchNorm defaults (momentum 0.1, ε 1e-5) and default
      Kaiming init instead of Keras's (0.99 / 1e-3, Glorot) — disclosed in `assumptions`,
      but disclosed *instead of* followed. Could contribute to its RMSE gap; untested.
- [ ] **Regeneration silently changes guessed hyperparameters.** The soft tree's retry was asked to fix an
      in-place op and also moved lr 0.01→0.1, batch 128→32, λ 0.1→0.01.
- [ ] **Soft-tree objective generated wrong — root cause: the paper misprints Eq. 3.** It prints
      `-log(Σ P·Σ T log Q)`, the log of a never-positive number. The Coder implemented it faithfully and
      negated inside the log, so it minimized `-log(cross-entropy)` = **maximized** cross-entropy: full run
      test accuracy fell to 0.15%, loss settled at -log(20.7) (the 1e-9 clamp floor). One-line fix
      (`per_example_loss = -inner_sum`, expected CE) learned at once: 54% after 3 epochs on 5k images
      (tested outside the pipeline; script in the session scratchpad, not in the repo). No stage can tell a
      misprinted equation from a correct one.
- [ ] **Prompt wording is copied literally.** The phrase "OpenML data_id and version" in
      the prompt produced `fetch_openml(data_id=531, version=1)`, which scikit-learn rejects.
      Fixed; worth auditing the rest of the prompt for the same kind of example.

### `runner/`
- [ ] **Success is decided on exit code alone** — see *Next up*.
- [ ] **The ladder is uninformative for non-iterative models** — for the random forest and SVM, `smoke`
      and `capped` gave identical numbers (they differ only in epochs). Scale sample size instead.
- [ ] **The escalation ladder cannot catch at-scale instability** — Tang learned through
      probe → smoke → capped, then collapsed after thousands of steps in `full`.
- [ ] **The image is not reproducible byte-for-byte** — `python:3.11-slim` is a moving tag and
      torch's transitive deps are unpinned. Pin the base image by digest.
- [ ] **Container runs as root** — hidden on macOS, visible on a Linux host.
- [ ] **`--memory` / `--cpus` exposed but unset** — deliberate; exit 137 looks like a crash.
      Cost of leaving it unset, measured 2026-09-13: two containers at once each claim every
      core. Wijaya's 1000 epochs over 405 rows took **1906 s** (~1.9 s/epoch for 13 tiny
      batches — thread oversubscription, not model cost) and Tang's `full` took 3223 s vs a
      ~28 min estimate. Run one container at a time, or cap threads for small models.

### `orchestrator/`
- [ ] **Retries regenerate rather than patch** — a fix can introduce a new defect elsewhere.
      Measured on a real bug: triage's fix was *delete one keyword argument*, and the
      regenerated Wijaya script was only **22% similar** to the one it replaced.
- [ ] **3 of 7 verdicts never seen in the wild** — `timeout`, `untriaged_error`,
      `coder_failed`. (`environment_error` was hit for real on 2026-09-13 — wrongly, see
      the triage fix.)
- [ ] **Plateau threshold has two data points** — 0.98, observed ratios 0.4079 (injected
      bug) and 0.2207 (real bug, Wijaya). Neither came near it.
- [ ] **Skip-if-done also skips failed papers** — rerunning a failed paper needs `--force`.

---

## Findings worth keeping

- **HuggingFace `Trainer` was hiding a default that made results look better.** The old NIN
  script never set `max_grad_norm`, so it inherited `Trainer`'s gradient clipping at 1.0
  (verified inside the image) — which the paper never uses. The faithful plain-PyTorch loop
  has no clipping, and its regression probe loss hit ~947 where the old script reported
  2.30. The old "working" result was partly an artifact of a hidden default.
- **A remembered identifier is worse than a missing one.** While fixing the dead Boston URL,
  a triage model confidently recalled Boston Housing as OpenML `data_id=506` — actually
  `analcatdata_gsssexsurvey` (Boston is `531`). A wrong `data_id` doesn't fail; it trains on
  the wrong data. Both prompts now forbid recalled identifiers and require verifying the
  fetched dataset in code.

---

## Blockers and constraints

- **GPU compute gates fidelity for the CIFAR-10 papers** — ~22 days per real WRN run on this
  CPU. Tang and Wijaya are small enough to measure on CPU: measured `full` 3223 s and
  1906 s respectively, both while sharing the CPU with each other (alone, less).
- **The dev machine cannot run generated scripts natively** (Intel Mac torch trap) — this is
  *why* `runner/` uses Docker. See CLAUDE.md's Tooling section.
- **Docker Desktop isn't always running** — `open -a Docker` starts it, unless it's waiting
  on a macOS privileged-access password prompt only Engincan can clear.
- **Two sessions on one working tree is hazardous** — keep to one while a batch is in flight.

---

## Done since the first progress report

- [x] `ocr/` — 4 backends, VLM verified; `vlm` extra now declares Pillow directly
- [x] `reader/` — 5 extractors + validation retry loop; complete against project plan §1.2
- [x] `reader/tooluse.py` — shared guards for three silent tool-use failure modes
- [x] `coder/` — training script + `reproduce.sh`, two deterministic gates
- [x] Coder builds from `architecture_notes` rather than pretrained priors
- [x] `runner/` — Docker sandbox, escalating gates, Haiku triage, timeout-and-kill
- [x] `orchestrator/` — §1.2 shared-memory state, 7 verdicts, plateau guard
- [x] Retry loop proven to repair a real runtime bug on the first retry
- [x] 2 CIFAR papers carried end to end, both `success`
- [x] Second progress report, both column formats; one-page handouts; Coder slide diagram
- [x] **Coder and Runner generalized beyond image classification** — plain PyTorch loop,
      task-agnostic metrics contract, OpenML-first data loading, tabular deps in the image
- [x] **First `full`-stage run** (Tang 2013), diagnosed via ablation
- [x] Triage and data-source prompts hardened after the first real tabular failure
- [x] **First two fidelity measurements** — Tang 0.82% vs 0.87% (after a manual `C` fix),
      Wijaya RMSE 4.48 vs 3.02; first real (non-injected) bug repaired by the retry loop
- [x] `reproduce.sh full` now writes `metrics.full.json` — it wrote `metrics.json`, so the
      Runner only found `full` metrics through its stdout fallback (template + 4 outputs)
