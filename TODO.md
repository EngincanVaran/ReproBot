# ReproBot — status and open work

Living tracker. Updated by hand as things land; the narrative history with full detail
lives in [`docs/agent-log.md`](docs/agent-log.md), and each stage's own README holds the
deep version of its known issues.

**Last updated:** 2026-09-13 · `main`

---

## Resume here

Work stopped mid-way through **testing the generalized pipeline on two non-CIFAR papers**.
Both runs are paused, each with a known next step, and **both wait on Engincan's go-ahead**
because they cost API calls or long CPU runs.

1. **Wijaya 2023 (house prices, tabular regression) — rerun.** Its first attempt failed in
   5 s: the generated script fetched Boston Housing from CMU StatLib, which now answers
   HTTP 403 to every client. Triage then called it `environment_error`, so the Orchestrator
   stopped with no retries. **Both prompts are fixed and committed**: the Coder prefers
   OpenML and must verify the dataset it fetched; triage now calls a dead URL
   `recoverable_error`. Rerun (it regenerates the script, ~2 min):
   ```bash
   uv run --extra orchestrator python -m orchestrator.pipeline \
     --input "reader/output/2023-10 - Multi Level Dense Layer Neural Network Model for Housing Price Prediction.json" \
     --claim-id c11 --max-stage full --retry-budget 2 --force
   ```
   `--force` is needed because a finished (failed) state already exists. Claimed: **test RMSE
   3.02**. Expect noise — the paper gives a 405/101 split but never says how rows were chosen.
   Check its `assumptions` for the Keras-default translations: batch size 32, Glorot init,
   BatchNorm momentum 0.99 / ε 1e-3, Adam ε 1e-7.

2. **Tang 2013 (MNIST, custom L2-SVM loss) — rerun `full` with `C = 0.1`.** Its first `full`
   run collapsed into a dead network (loss sitting exactly on the input-ignoring optimum,
   ~89% error vs. claimed **0.87%**). An ablation isolated the cause — full write-up in
   [`docs/notes/tang-2013-ablation/`](docs/notes/tang-2013-ablation/README.md). Apply as a
   controlled one-line change, **not** a Coder regeneration:
   - edit `coder/output/2013-06 - Deep Learning using Linear Support Vector Machines/train.py`
     line 174: `svm_C = 1.0` → `svm_C = 0.1`
   - `uv run --extra runner python -m runner.pipeline --input "coder/output/2013-06 - Deep Learning using Linear Support Vector Machines" --mode full` (~28 min CPU)

3. **Compare each reproduced number to its claim by hand** — the Critic doesn't exist yet.
   These would be ReproBot's **first real fidelity measurements**.

---

## Where the pipeline is

**CIFAR-10 set** (`dataset/`, curated by someone else):
```
8 PDFs  →  6 OCR'd  →  4 read  →  2 coded  →  2 executed  →  2 orchestrated
```
**Generalization test set** (`extra-papers/`): Tang 2013 and Wijaya 2023 — both OCR'd,
read and coded; Tang has run through `full`; Wijaya is waiting on a rerun (above).

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
classification. Nothing compares a reproduced number against a claim, and the first run long
enough to produce a comparable number (Tang `full`) collapsed with **no stage noticing** —
it would have been reported `success`. The Critic is now the most important missing piece.

---

## Next up

- [ ] **Finish the generalization test** — the three steps under *Resume here*.
- [ ] **Build `critic/`** — compare `metrics.json` `value` against `claims[].reported_value`
      under an explicit tolerance, using `higher_is_better` from the new metrics contract.
      Deliberately arithmetic, not an LLM judging numbers (project plan §2.5). Tang's
      collapse is the concrete case it must catch.
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
- [ ] **Review Mert's `viewer/` branch** (`origin/mert/runner-agent`, a Streamlit dashboard)
      — **deferred at Engincan's request**, 2026-09-13; focus is on our own stages for now.

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
      stability probe, not just listing each guess.

### `runner/`
- [ ] **Success is decided on exit code alone** — see *Next up*.
- [ ] **The escalation ladder cannot catch at-scale instability** — Tang learned through
      probe → smoke → capped, then collapsed after thousands of steps in `full`.
- [ ] **The image is not reproducible byte-for-byte** — `python:3.11-slim` is a moving tag and
      torch's transitive deps are unpinned. Pin the base image by digest.
- [ ] **Container runs as root** — hidden on macOS, visible on a Linux host.
- [ ] **`--memory` / `--cpus` exposed but unset** — deliberate; exit 137 looks like a crash.

### `orchestrator/`
- [ ] **Retries regenerate rather than patch** — a fix can introduce a new defect elsewhere.
- [ ] **3 of 7 verdicts never seen in the wild** — `timeout`, `untriaged_error`,
      `coder_failed`. (`environment_error` was hit for real on 2026-09-13 — wrongly, see
      the triage fix.)
- [ ] **Plateau threshold has one data point** — 0.98, observed ratio 0.4079.
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
  CPU. Tang (~28 min `full`) and Wijaya (~2 min) are small enough to measure on CPU.
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
