# ReproBot — status and open work

Living tracker. Updated by hand as things land; the narrative history with full detail
lives in [`docs/agent-log.md`](docs/agent-log.md), and each stage's own README holds the
deep version of its known issues.

**Last updated:** 2026-08-22 · `main` @ `4befad5`

---

## Where the pipeline is

```
8 PDFs  →  6 OCR'd  →  4 read  →  2 coded  →  2 executed  →  2 orchestrated
```

| Stage | Status | Verified by |
|---|---|---|
| `ocr/` | ✅ built | 6/8 papers extracted end to end |
| `reader/` | ✅ built, complete vs. §1.2 | 4 papers, all 5 fields |
| `coder/` | ✅ built | 2 papers; NIN architecture came from extraction, not priors |
| `runner/` | ✅ built | real Docker container, real training run |
| `orchestrator/` | ✅ built | 2 papers `success`; repaired an injected bug on 1st retry |
| **`critic/`** | ❌ **not started** | — |
| **report generator** | ❌ **not started** | — |

**The honest summary:** the machinery works end to end on two papers. Nothing has yet
compared a reproduced number against a paper's claim, so no replication-fidelity claim is
possible. That is what the Critic is for.

---

## Next up

Roughly in the order that unblocks the most.

- [ ] **Build `critic/`** — compare `metrics.json` `value` against `claims[].reported_value`
      under an explicit numeric tolerance. Deliberately simple arithmetic, not an LLM
      eyeballing numbers (project plan §2.5). Buildable and testable now against synthetic
      metrics; verdicts only become *meaningful* once a real run exists (see Blockers).
- [ ] **Secure GPU compute** — gates every fidelity result. Worth starting in parallel,
      not after.
- [ ] **Extend coverage** — 4 papers have Reader output, only 2 have been coded/run.
      Costs API calls, not new code, and turns single demonstrations into distributions.
- [ ] **Build the report generator** — the promised deliverable and the cheapest remaining
      piece: `orchestrator/state.py` already carries everything it needs, including
      `history`.
- [ ] **Single entry point** — one command taking a PDF through all five stages.
      `orchestrator/` already drives coder+runner in-process; `ocr/` and `reader/` are
      still manual steps.
- [ ] **Fix the report's rendering issues** — noted 2026-08-22, not yet catalogued.
      Regenerate via `docs/progress-reports/second-progress-report/generate.py`; never
      hand-edit the `.tex` files, they are generated and would drift apart.

---

## Open bugs and gaps

Ordered by stage. Anything marked **cost** is actively wasting money or time on every run.

### `ocr/`
- [ ] **2 of 8 papers cannot be extracted at all** — *Deep Networks with Stochastic Depth*
      and *Densely Connected Convolutional Networks*, both failing on page 2 with an
      Anthropic content-filter false positive. Diagnosed as model-side, not ours: the same
      page fails identically at a different render scale and under a one-line prompt.
      Blocks those two papers from the entire pipeline.
- [ ] **Per-page resilience missing** — one bad page discards the whole paper, including
      pages that already transcribed fine. Proposed fix: catch per-page, insert a
      placeholder, continue.
- [ ] **Duplicated figure captions** — *All Convolutional Net*, Figures 5 and 6: the
      caption is restated as a bare paragraph after the bracketed description block.
- [ ] **Hallucinated heading** — *AutoAugment* page 5 opens with a `# Page Content`
      heading that is not in the source PDF.

### `reader/`
- [ ] **cost — validation never converges.** All four papers finish with flags remaining
      after the 3-pass cap (NIN 4, All-Conv 5, ResNet 8, WRN 6). A substantial share are
      the validator raising a concern and dismissing it inside its own description. Every
      non-converged pass is a wasted retry round. Needs either a tighter validator prompt
      or a "does this flag assert a real defect" gate.
- [ ] **WRN ImageNet hyperparameters gap** — 14 ImageNet claims (c52–c65) but zero
      hyperparameter entries for that dataset, not even a "not stated" placeholder, unlike
      every other gap in the file. Suggests multi-dataset papers are under-covered.
- [ ] **Intermittent all-empty tool payload** — a well-formed response with every required
      field absent. Caught by `reader/tooluse.py` and retried, but the root cause is
      unknown and not reproducible on demand.

### `coder/`
- [ ] **Intermittent tool-field leak** — the model serialises later tool fields as literal
      `<parameter name="...">` text inside an earlier field, twice swallowing
      `script_content` entirely. Worked around deterministically in
      `_recover_leaked_fields`; it is a workaround for a model quirk, not a fix.
- [ ] **Bookkeeping can disagree with the code it describes** — the NIN run reports stage 3
      as `192→192→10` while the code builds `192→10→10`. The script runs and the choice is
      a defensible reading of an unstated dimension; what is wrong is that the record
      describes a different network. No static gate can catch this.
- [ ] **Priors still supply unstated numbers** — NIN's channel widths are the canonical
      configuration from pretrained memory, since the paper genuinely omits them. This is
      the designed behaviour and it *is* disclosed in `assumptions`, but it means those
      values are **visible, not verified**. The Critic will need to weight a disclosed
      prior below a paper-sourced value.

### `runner/`
- [ ] **Container runs as root** — macOS Docker Desktop hides this; a Linux host will
      leave root-owned files in `coder/output/`. Fix is `--user $(id -u):$(id -g)` plus a
      writable `HOME` in the container; unverified, so not applied blind.
- [ ] **`--memory` / `--cpus` exposed but unset** — deliberate. An arbitrary cap produces
      exit 137, which looks exactly like a script crash and would be triaged
      `recoverable_error`, sending the Coder to fix a bug that does not exist. Revisit if
      runaway resource use ever actually bites.

### `orchestrator/`
- [ ] **Retries regenerate rather than patch** — a repair rewrites the whole file, so a
      successful fix can introduce a new defect elsewhere, and nothing diffs semantics
      between attempts. AutoReproduce's line-range `EDIT` mechanism is the natural remedy.
- [ ] **4 of 7 verdicts never seen in the wild** — `environment_error`, `timeout`,
      `untriaged_error` and `coder_failed` are verified against test doubles only.
- [ ] **Plateau threshold has one data point** — 0.98, with a single real observation
      (0.4079, a genuine repair). The ratio is logged every retry so it can be tuned from
      data; it has not been yet.

---

## Blockers and constraints

- **GPU compute is a precondition for any fidelity result.** Measured 5.256 training
  samples/sec for WRN-28-10 on this CPU; the paper's setup is 50,000 samples × 200 epochs
  ≈ **22 days for one paper, one claim**. Everything verified so far runs at
  `probe`/`smoke` scale, which answers "does it run", never "is the number right".
- **The dev machine cannot run generated scripts natively.** Intel Mac: no PyTorch release
  supports Python 3.13 *and* Intel macOS. Python 3.11 gets torch 2.2.2, but current
  `transformers` needs ≥2.5 and silently disables its PyTorch backend rather than raising.
  This is *why* `runner/` uses Docker. See CLAUDE.md's Tooling section for the one narrow
  legacy pin-set that works for throwaway local testing.
- **Two sessions on one working tree is hazardous.** It happened once on 2026-08-22 and
  landed on docs only. Keep to one session per repo while a batch is in flight.

---

## Doc debt

- [ ] **`coder/README.md` claims the generated script has never been executed.** Stale — it
      has now run twice in Docker, via `orchestrator/`.
- [ ] **Audit artifact** ([pipeline audit](https://claude.ai/code/artifact/44a768fb-cd35-4638-b70c-5b3864d640ad))
      predates the second paper's orchestrated run; its funnel shows `2 coded → 1 executed
      → 0 orchestrated`, now `2 → 2 → 2`.

---

## Done since the first progress report

- [x] `ocr/` — 4 backends, VLM verified on 6 papers
- [x] `reader/` — 5 extractors + validation retry loop; complete against project plan §1.2
- [x] `reader/tooluse.py` — shared guards for three silent tool-use failure modes
- [x] `coder/` — training script + `reproduce.sh`, two deterministic gates
- [x] Coder wired to build from `architecture_notes` rather than pretrained priors
- [x] `runner/` — Docker sandbox, escalating gates, Haiku triage, timeout-and-kill
- [x] `orchestrator/` — §1.2 shared-memory state, 7 verdicts, plateau guard
- [x] Retry loop proven to repair a real runtime bug on the first retry
- [x] 2 papers carried end to end, both `success`
- [x] Second progress report, both column formats
