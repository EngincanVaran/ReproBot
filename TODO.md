# ReproBot — status and open work

Living tracker. Updated by hand as things land; the narrative history with full detail
lives in [`docs/agent-log.md`](docs/agent-log.md), and each stage's own README holds the
deep version of its known issues.

**Last updated:** 2026-09-14 · `main` · **4th progress-report period** (everything since 2026-09-14)

---

## Resume here

**Phase 3 is done.** Five papers across five model families were replicated at full fidelity,
the third progress report is written (`docs/progress-reports/third-progress-report/`), and
everything is pushed. **The next phase is the three priorities below, in this order** —
agreed with Engincan on 2026-09-13/14. Each one grew directly out of something this phase's
runs showed.

1. **Runner live check-ups** — watch training while it runs, not just its exit code.
2. **The Critic agent** — judge each reproduced number against its claim.
3. **Loop controls** — make the Orchestrator's retries targeted, bounded and auditable.

**Priority 1 is done (2026-09-14)** — Runner live check-ups, the learning-curve history
contract, and the classical-model ladder, verified by replaying real curves and live in Docker
(details below and in `runner/README.md`). **Next: priority 2, the Critic.** The history
files and halt evidence it needs now exist. Present a plan before building, as usual.

---

## Results so far

| Paper | Model family | Claim | ReproBot | Notes |
|---|---|---|---|---|
| Tang 2013 | MLP + L2-SVM loss | MNIST test error 0.87% | **0.82%** | `full`, 54 min; needed manual `C` 1.0 → 0.1 ([ablation](docs/notes/tang-2013-ablation/README.md)) |
| Wijaya 2023 | Dense regression net (Keras) | Boston test RMSE 3.02 | **4.48** (train 2.94 vs 2.69) | Orchestrated, 1 auto-repair; gap most likely the paper's unstated split |
| Hsu/Chang/Lin SVM guide | RBF SVM | svmguide1 96.9% | **96.625%** (rerun 2026-09-14 with check-ups: **96.925%**) | Orchestrated, 1 auto-repair, 73 s; `SVC` at the paper's C=γ=2 reproduces 66.925 / 96.15 / 96.875% **exactly**; the grid search's pick varies with fold assignment |
| Fashion-MNIST (Xiao 2017) | Random forest | 0.873, mean of 5 | **0.8773** (0.8753–0.8792) | Orchestrated, 0 retries, 3.3 min |
| Frosst & Hinton 2017 | Soft decision tree | MNIST 94.45% | **95.11%** | Orchestrated, 0 retries, 70 min; needed manual fix of the paper's misprinted loss (Eq. 3). Regenerated 2026-09-14 under the new prompt, the Coder implemented the correct loss on its own (verified to `capped` only) |
| Network In Network, WRN | CIFAR-10 CNNs | — | smoke only | Full runs need a GPU (~22 days/WRN run on CPU) |

All comparisons are by hand; no stage judges a number yet.

---

## Where the pipeline is

**CIFAR-10 set** (`dataset/`, curated by someone else):
```
8 PDFs  →  6 OCR'd  →  4 read  →  2 coded  →  2 executed (smoke)  →  2 orchestrated
```
**CPU-sized set** (`extra-papers/`, see its README):
```
11 PDFs  →  5 OCR'd  →  5 read  →  5 coded  →  5 run at full fidelity (4 orchestrated)
```

| Stage | Status | Verified by |
|---|---|---|
| `ocr/` | ✅ built | 11 papers extracted |
| `reader/` | ✅ complete vs. §1.2 | 9 papers, all 5 fields; first clean convergences (SVM guide, Fashion-MNIST) |
| `coder/` | ✅ built — PyTorch loop **or scikit-learn** by model family | 5 families generated and run |
| `runner/` | ✅ built — **live check-ups** judge training health while it runs (`halted` status) | every stage incl. `full`, 7 papers; live kill verified |
| `orchestrator/` | ✅ built | 6 papers to `success` (NIN, WRN at smoke; Wijaya, SVM guide, Fashion-MNIST, soft tree at full); 3 real defects auto-repaired |
| **`critic/`** | ❌ **not started** — Next phase #2 | — |
| **report generator** | ❌ not started | — |
| `viewer/` | 🟡 Mert's branch `origin/mert/runner-agent`, not merged | — |

---

## Next phase (agreed priorities)

### 1. Runner live check-ups — ✅ done 2026-09-14
- [x] **Learning-curve history contract** — every generated script writes
      `metrics.<mode>.history.jsonl` (+ `REPROBOT_PROGRESS` stdout lines) with `chance_metric`,
      `target_value`, `loss_lower_bound`; Coder gate 3 rejects scripts without it.
- [x] **Watcher + six rules** (`runner/checkups.py`) — kills the container and ends the stage
      `halted`; Orchestrator retries with the evidence as feedback. 28 pytest replays of real
      curves; live Docker kill at epoch 1 of 40; full halt → regenerate → `success` loop.
- [x] **Ladder for models without epochs** — `model_family` → classical `capped` uses 5,000/2,000
      rows (RF capped 0.8435 vs smoke 0.776, previously identical).
- [x] Coder: capped subsets seeded random + stratified (svmguide1 first-N-rows single-class crash).
- [ ] **Follow-ups:** regenerate the older scripts (NIN, WRN, Tang, Wijaya) so they write history;
      consider a "no record for too long" hang rule; keep calibrating thresholds from every halt's
      logged evidence (one false positive already found and fixed end to end).

### 2. The Critic agent (`critic/`)
- [ ] **Compare `value` with `claims[].reported_value`** using `higher_is_better`, as explicit
      arithmetic (project plan §2.5), emitting pass / retry / fail with a reason the Coder can act on.
- [ ] **Tolerance matched to how the claim was produced:** near-exact for deterministic library
      procedures (SVM guide), a seed-variance band for stochastic training (Wijaya: one split can't
      decide — run several seeds), the reported spread when the claim is a mean (Fashion-MNIST).
- [ ] **Claim dedup first** — the Reader extracts the same result from prose, table and figure
      (Wijaya: 14 claims for 8 results, mixed precision).
- [ ] Feed the verdict into `ReproState.critic_output` (field already reserved) and the retry loop.

### 3. Loop controls (`orchestrator/`)
- [ ] **Patch, don't regenerate** — real one-line repairs rewrote 73–78% of a script
      (plateau similarities 0.22, 0.25, 0.27). Line-range edits, as AutoReproduce does.
- [ ] **Freeze guessed hyperparameters across retries** unless the feedback targets them — the soft
      tree's crash fix also moved lr 0.01→0.1, batch 128→32, λ 0.1→0.01.
- [x] **Abort a running stage** when its live check-up fails — done with the check-ups (the watcher
      kills the container; the Orchestrator retries with the evidence).
- [ ] **Per-paper time and API-cost budgets.**
- [ ] **Record human corrections in the state object** — Tang's `C` and the soft tree's loss were
      fixed by hand in gitignored scripts; a regeneration would silently undo both.

---

## Backlog

- [ ] **Run the remaining diversity targets** in `extra-papers/`: GCN (Cora), Kim text CNN (TREC),
      TCN adding problem, XGBoost-vs-RF-vs-GB (Spambase), CatBoost/LightGBM/XGBoost (Adult),
      XGBoost Higgs-1M. The boosting three need **xgboost, lightgbm, catboost (+hyperopt, libgomp1)
      in the Runner image** — PyPI wheels checked compatible with numpy 2.1.3 / py3.11.
- [ ] **Sanity-check own-method equations** before implementing — domain checks (log/sqrt
      arguments, probabilities in [0,1]) to flag misprints like the soft tree's Eq. 3.
- [ ] **Reader records missing hyperparameters** — `unstated_details` covers architecture only;
      Tang's momentum and `C` were flagged nowhere.
- [ ] **Reader handles dataset/benchmark papers** — Fashion-MNIST's first pass kept 0 of 124 rows
      (no "own method"); only the validator's retry recovered 26 claims.
- [ ] **Secure GPU compute** for the CIFAR-10 set.
- [ ] **Report generator** — `orchestrator/state.py` already carries everything.
- [ ] **Single entry point** — PDF through all stages; blocked by the exit-code bug below.
- [ ] **Review and merge Mert's `viewer/`** (`origin/mert/runner-agent`) — reconcile
      `pyproject.toml`/`uv.lock` first (it predates `dc5a541`); extend it to Runner/Orchestrator
      output and learning curves.
- [ ] **Refresh the handouts and claude.ai artifacts** — all predate the generalization work.
- [ ] Second progress report's uncatalogued rendering issues (low priority).

---

## Open bugs and gaps

Anything marked **cost** is actively wasting money or time on every run.

### Across stages
- [ ] **`ocr/`, `reader/` and `coder/` exit 0 even when every paper fails.** Only `runner/`
      and `orchestrator/` set a failing exit code. Must be fixed before chaining stages.

### `ocr/`
- [ ] **2 of 8 CIFAR papers cannot be extracted** — *Stochastic Depth* and *DenseNet* fail on
      page 2 with a deterministic Anthropic content-filter false positive (model-side).
- [ ] **Per-page resilience missing** — one bad page discards the whole paper.
- [ ] **Duplicated figure captions** — *All Convolutional Net*, Figures 5 and 6.
- [ ] **Hallucinated heading** — *AutoAugment* page 5, `# Page Content`.

### `reader/`
- [ ] **cost — validation rarely converges.** Flags left after the 3-pass cap: NIN 4, All-Conv 5,
      ResNet 8, WRN 6, Tang 5, Wijaya 5, soft tree 4 (SVM guide and Fashion-MNIST converged to 0).
- [ ] **WRN ImageNet hyperparameters gap** — 14 ImageNet claims, zero hyperparameters.
- [ ] **Intermittent all-empty tool payload** — caught and retried; root cause unknown.
- [ ] **SVM guide: took the rounded 96.9% from the summary table**, not the appendix's 96.875%,
      and did not record the appendix's default-parameter accuracies as claims.

### `coder/`
- [ ] **`wall_clock_seconds` timed from the training loop, not the run start** — prompt made explicit
      2026-09-14 ("first statement of `main()`"); verify on the next regenerated scripts.
- [ ] **Intermittent tool-field leak** — worked around in `_recover_leaked_fields`, not fixed.
- [ ] **Bookkeeping can disagree with the code** — NIN reported `192→192→10`, code built `192→10→10`.
- [ ] **Priors still supply unstated numbers** — disclosed, so *visible*, not *verified*.
- [ ] **Unstated guesses can fail in combination** — Tang's momentum 0.9 + `C=1.0`; argues for a
      cheap stability probe over unstated values.
- [ ] **Keras-defaults rule only partly followed** — Wijaya matched Adam ε but used PyTorch
      BatchNorm/init defaults; disclosed instead of followed; untested effect on its RMSE gap.
- [ ] **Audit the prompt for example wording copied literally** — "OpenML data_id and version"
      produced an invalid call (fixed); similar phrasing may remain.

### `runner/`
- [ ] **Check-ups judge health, not fidelity** — a healthy run far from the paper's number
      (Wijaya 4.48 vs 3.02) passes. That is the Critic's job (Next phase #2).
- [ ] **Scripts from before the progress contract** fall back to exit-code-only judgement until
      regenerated.
- [ ] **The image is not reproducible byte-for-byte** — pin `python:3.11-slim` by digest.
- [ ] **Container runs as root** — hidden on macOS, visible on a Linux host.
- [ ] **`--memory` / `--cpus` unset** — deliberate (exit 137 looks like a crash), but two containers
      at once each claim every core and roughly double wall clock. Run one at a time.

### `orchestrator/`
- [ ] **Retries regenerate rather than patch** — Next phase #3.
- [ ] **3 of 7 verdicts never seen in the wild** — `timeout`, `untriaged_error`, `coder_failed`.
- [ ] **Plateau threshold has four data points** — 0.98 vs observed 0.41 (injected), 0.22, 0.25,
      0.27 (real). None came near it.
- [ ] **Skip-if-done also skips failed papers** — rerunning a failed paper needs `--force`.

---

## Findings worth keeping

- **Watching a run beats waiting for it.** Both bad `full` runs of phase 3 were diagnosed from
  live `docker logs` long before they would have finished, while the pipeline reported success.
- **Where the library is the method, exactness is achievable.** scikit-learn's `SVC` wraps LIBSVM
  and reproduces the SVM guide's three accuracies to three decimals; the residual gap came only
  from which near-equal (C, γ) the randomly folded grid search picked.
- **A faithful implementation of a misprinted equation trains to be wrong.** The soft-tree paper's
  Eq. 3 is −log of a never-positive quantity; implemented literally it maximized cross-entropy.
- **HuggingFace `Trainer` hid a default that flattered results** — gradient clipping at 1.0 that
  the NIN paper never uses.
- **A remembered identifier is worse than a missing one** — a triage model recalled Boston Housing
  as OpenML `data_id=506` (a different dataset; Boston is 531). Prompts now forbid recalled
  identifiers and scripts assert the fetched dataset.

---

## Blockers and constraints

- **GPU compute gates fidelity for the CIFAR-10 papers** — ~22 days per real WRN run on this CPU.
  CPU-sized papers run in 73 s to 70 min.
- **The dev machine cannot run generated scripts natively** (Intel Mac torch trap) — this is *why*
  `runner/` uses Docker. See CLAUDE.md's Tooling section.
- **Docker Desktop isn't always running** — `open -a Docker` starts it, unless it's waiting on a
  macOS privileged-access password prompt only Engincan can clear.
- **Two sessions on one working tree is hazardous** — keep to one while a batch is in flight.

---

## Done

### 4th-report period (from 2026-09-14)
- [x] **Runner live check-ups** — history contract, watcher, six rules, `halted` status,
      Orchestrator halt routing, classical ladder, `pytest` suite (28 tests from real curves)
- [x] End-to-end in Docker: SVM guide `success` (96.925% vs 96.875%), Fashion-MNIST RF `success`
      (0.8773), sign-flipped soft tree killed live at epoch 1/40, halt → regenerate → `success`

### Phase 3 (2026-09-13 → 14)
- [x] Coder and Runner generalized beyond image classification — plain PyTorch loop, task-agnostic
      metrics contract, OpenML-first data loading with in-code dataset verification
- [x] **scikit-learn support in `coder/`** — library chosen by model family, paper-era defaults,
      repeated-run protocols (`num_runs`, `run_values`), nullable loss/epoch metrics
- [x] Triage and data-source prompts hardened after real failures; `metrics.full.json` fix
- [x] **Five full-fidelity replications** across five model families (table above), three defects
      auto-repaired by the retry loop
- [x] Nine CPU-sized diversity papers added to `extra-papers/` with verified claims and data URLs
- [x] **Third progress report** — self-contained, results-first, five-panel learning-curve figure,
      both column formats
- [x] Everything pushed to `origin/main`

### Phases 1–2
- [x] `ocr/` — 4 backends, VLM verified; `vlm` extra declares Pillow directly
- [x] `reader/` — 5 extractors + validation retry loop; complete against project plan §1.2
- [x] `reader/tooluse.py` — shared guards for three silent tool-use failure modes
- [x] `coder/` — training script + `reproduce.sh`, two deterministic gates; builds from `architecture_notes`
- [x] `runner/` — Docker sandbox, escalating gates, Haiku triage, timeout-and-kill
- [x] `orchestrator/` — §1.2 shared-memory state, 7 verdicts, plateau guard; repaired an injected bug
- [x] 2 CIFAR papers carried end to end to `smoke`
- [x] Second progress report, both column formats; one-page handouts; Coder slide diagram
