# ReproBot — Detailed Project Plan

> inzva AI Projects #10. Companion to `ReproBot.pdf` (the original proposal) and `Papers/Introduction_and_Literature_Review.md`. This document expands the proposal into an implementation-ready plan: a feasibility assessment grounded in comparable systems' actual numbers, an architecture deep dive, per-agent specifications (tools, models, prompting notes), a de-risked timeline, and a cost/compute budget.

---

## 0. Feasibility Assessment

**Verdict: doable in 4 months, but only with disciplined scope control.** The proposal's six objectives are all individually reasonable; the risk is in taking on the full combination — a 4-agent pipeline, VLM figure parsing, Docker sandboxing, a Critic-driven retry loop, *and* a 20-paper benchmark with an ablation study — as a fixed spec for two people working part-time alongside an educational program.

### 0.1 Reality check from comparable systems

| System | Team / effort | What they achieved | Cost |
|---|---|---|---|
| PaperBench BasicAgent (Starace et al., 2025) | Dedicated OpenAI team | 21.0% avg replication score (best backbone, Claude 3.5 Sonnet), across 20 ICML papers / 12 topics | ~$400/paper (12h agent) + $66/paper (grading) |
| PaperBench IterativeAgent, extended | Same team | 26.0% (o1-high, 36h budget) | Same order of magnitude, longer runs |
| PaperCoder (Seo et al., 2025) | Research team, months | 45.1% on PaperBench Code-Dev (**lightweight — no execution**); only 28.5% on the harder 10-paper execution+result-match subset | Not reported, but no GPU training cost since code is never run |
| AutoReproduce (Zhao et al., 2025) | Tsinghua research team | 48.5% on PaperBench Code-Dev with full paper-lineage engineering; 77–95% *execution rate* (not full replication) on their own easier ReproduceBench | Not reported |
| Human ML PhDs (PaperBench baseline) | 8 PhDs, 48h each | ~41.4% best-of-3 | N/A |

The takeaway is not "this is impossible" — it's that **no system reviewed reaches anywhere near 100% faithful replication**, and the ones that get furthest spent significant engineering effort on exactly the narrow techniques (sampling-based dry-run testing, sequential dependency-ordered code generation, explicit retry loops) that ReproBot's proposal already plans to use. ReproBot is not competing to beat these numbers in 4 months; it's demonstrating the same category of system in a deliberately narrower domain.

### 0.2 Why ReproBot's scope is more tractable than the table above

- **Single domain.** Image classification means a small, well-known vocabulary of architectures (ResNet, ViT, EfficientNet-style CNNs, plain MLPs/CNN baselines) and datasets (CIFAR-10/100, ImageNet or class-balanced subsets, MNIST-family) — the Reader and Coder don't need to handle RL, generative models, or theory papers.
- **Fixed code target.** PaperCoder and AutoReproduce generate arbitrary multi-file repositories; ReproBot's Coder always targets one shape — a self-contained HuggingFace `Trainer` script. Constraining the output space this much should make both generation *and* automated verification meaningfully easier.
- **Single-number claims.** Matching one headline top-1/top-5 accuracy is a much smaller verification target than PaperBench's 8,316 binary sub-outcomes per 20 papers.

### 0.3 What has to be true for the timeline to hold

1. **Compute must be cheap per paper.** Image classification at CIFAR/small-ImageNet scale trains in minutes-to-hours on a single GPU, not days — this is what makes a 12h-style time cap (à la PaperBench) affordable to run repeatedly during development, unlike papers requiring large-scale pretraining.
2. **The 20-paper benchmark is a stretch goal, evaluated at the end, not a moving target during development.** Build and validate against 3–5 papers first (see §6).
3. **The retry loop, not paper-count coverage, is the thing to optimize for and report on.** This is both the cheapest thing to demonstrate convincingly (MLR-Copilot showed a 0%→50% success swing from adding iteration, using only 5 tasks and 8 trials each) and the actual novel contribution per the literature review's gap analysis.
4. **API/compute budget must be planned, not discovered.** See §9.

---

## 1. Architecture Deep Dive

### 1.1 Pipeline and shared memory

```
                              PDF paper
                                  │
                                  ▼
        ┌───────────────────────────────────────────────────────────┐
        │                     ORCHESTRATOR                            │
        │   owns: shared memory state, retry budget, confidence      │
        │   threshold, stage transitions                              │
        └───────────────────────────────┬─────────────────────────────┘
                                         │
      ┌──────────┐   plan   ┌──────────┐   script  ┌──────────┐  metrics  ┌──────────┐
      │  READER  │ ───────► │  CODER   │ ────────► │  RUNNER  │ ────────► │  CRITIC  │
      └──────────┘          └──────────┘           └──────────┘           └────┬─────┘
           ▲                      ▲                                            │
           │                      │            verdict = retry (with feedback) │
           │                      └────────────────────────────────────────────┘
           │                                          │
           │                              verdict = pass / fail (exhausted)
           │                                          ▼
           │                                 ┌─────────────────────┐
           └── (rare: Reader re-parse on ─── │  REPORT GENERATOR    │
                repeated Coder failures)      │  claim-by-claim table│
                                              │  + gap analysis        │
                                              └─────────────────────┘
```

### 1.2 Shared memory schema (what the Orchestrator actually stores)

A single structured object per paper, threaded through every agent call — this is the concrete artifact behind the proposal's "shared memory state":

```jsonc
{
  "paper_id": "resnet-2015",
  "source_pdf": "path/to/paper.pdf",
  "reader_output": {
    "method_summary": "...",
    "claims": [
      {"claim_id": "c1", "metric": "top-1 accuracy", "dataset": "CIFAR-10",
       "reported_value": 93.6, "unit": "%", "source": "Table 2, page 6"}
    ],
    "hyperparameters": {"lr": 0.1, "batch_size": 128, "epochs": 164, "optimizer": "SGD+momentum"},
    "architecture_notes": "..."
  },
  "coder_output": {
    "script_path": "generated/resnet-2015/train.py",
    "script_version": 3,
    "diff_from_previous": "..."
  },
  "runner_output": {
    "status": "success | error | timeout",
    "reproduced_metrics": {"top-1 accuracy": 91.2},
    "logs_path": "...",
    "error_trace": null
  },
  "critic_output": {
    "verdict": "pass | retry | fail",
    "comparison": [{"claim_id": "c1", "reported": 93.6, "reproduced": 91.2,
                     "gap_pct": 2.6, "within_tolerance": false}],
    "feedback_to_coder": "Accuracy is 2.6pp below claim; check LR schedule — paper uses step decay at epochs 82/123, current script uses cosine annealing."
  },
  "retry_count": 1,
  "retry_budget": 5,
  "history": ["...append-only log of every stage transition for audit..."]
}
```

Keeping this as one versioned object (not scattered state) is what lets the Orchestrator answer "have we already tried this fix" and lets the Report Generator produce a claim-by-claim table without re-deriving anything.

### 1.3 Orchestration framework recommendation

The proposal lists **LangChain** as the agent framework. For this specific shape of pipeline — a small number of named roles, a shared mutable state object, and a conditional retry loop — **LangGraph** (built on LangChain, but modeling the pipeline as an explicit state graph rather than a generic agent loop) is a better fit than a general LangChain agent: it makes the Critic→Coder retry edge, the retry-budget-exhausted→fail edge, and the pass→Report-Generator edge explicit, inspectable graph transitions rather than implicit control flow buried in prompts. This also makes the retry loop directly testable in isolation (a unit test can assert "given this critic verdict, does the graph route to Coder or to Report Generator" without invoking any LLM).

---

## 2. Agent-by-Agent Specification

For each agent: responsibilities, inputs/outputs (referencing the shared-memory schema above), required tools, recommended model, and specific risks pulled from the literature review with mitigations.

### 2.1 Orchestrator

- **Responsibilities:** own the shared-memory object; decide stage transitions; enforce the retry budget and confidence threshold; decide when to escalate a `fail` verdict to a human (per Agent Laboratory's human-escalation pattern, already noted in the literature review).
- **Inputs/Outputs:** reads/writes the full shared-memory object; no direct LLM calls needed for the majority of transitions (they're deterministic given `critic_output.verdict` and `retry_count`).
- **Tools:** none beyond the state store itself (a LangGraph graph + a simple DB/JSON store per paper is sufficient — no need for a message queue at this scale).
- **Model:** mostly **no model call required** (deterministic routing). Where a judgment call is genuinely needed — e.g., deciding whether a Runner *timeout* (vs. a hard error) still counts against the retry budget — a single cheap call to **Claude Haiku 4.5** is enough; this is not a task requiring deep reasoning.
- **Key risk:** silently looping forever or burning the whole retry budget on the same mistake. *Mitigation:* the shared-memory `history` log must be checked before each retry — if the Coder's last two attempts produced near-identical scripts, escalate to `fail` early rather than exhausting the budget on a plateaued fix, echoing PaperBench's finding that agents plateau rather than making steady long-horizon progress.

### 2.2 Reader

- **Responsibilities:** parse the PDF (text + figures + tables); extract method summary, dataset(s), hyperparameters, and — critically — the exact claimed metric(s) with their source location (table/figure/page) so the Critic has something concrete to compare against later.
- **Inputs/Outputs:** raw PDF → `reader_output` (see schema above).
- **Tools:** `pdfplumber` (text/table extraction, as proposed) + `pdf2image` (render pages to images for VLM figure parsing) + a "read specific page range" tool for the VLM step, since full-paper VLM parsing is expensive and usually unnecessary beyond the method/results sections.
- **Model:** **Claude Sonnet 5** (native vision) as the primary Reader backbone. Rationale: among tested backbones in PaperBench's BasicAgent evaluation, Claude 3.5 Sonnet was the clear leader (21.0% vs. GPT-4o's 4.1%, o1's 13.2%) on exactly this kind of paper-comprehension-driven task, and image-classification papers' key figures (architecture diagrams, results tables) are within a VLM's comfortable range. Use **Claude Opus 4.8** selectively for papers where the Reader's first-pass extraction is flagged low-confidence (e.g., an architecture diagram with unusual notation) rather than as the default, to control cost.
- **Key risk (from PaperCoder's own ablation):** data-processing/preprocessing details are the single weakest-covered aspect of paper→code extraction across every system reviewed (PaperCoder: 56% coverage vs. 79–92% elsewhere). *Mitigation:* give the Reader an explicit, separate extraction pass for "data pipeline" (normalization, augmentation, train/val/test split convention) rather than folding it into the general method summary — treat it as a first-class field in `reader_output`, not an afterthought.

### 2.3 Coder

- **Responsibilities:** translate `reader_output` into a single, self-contained HuggingFace `Trainer`-based training script; on retry, apply the Critic's targeted feedback rather than regenerating from scratch.
- **Inputs/Outputs:** `reader_output` (+ `critic_output.feedback_to_coder` on retries) → `coder_output`.
- **Tools:** file write; a HuggingFace Hub search tool (model architectures / dataset loaders) so the Coder can start from an existing `datasets.load_dataset`/`AutoModelForImageClassification` call rather than hand-rolling data loading — this mirrors MLR-Copilot's retrieval-before-generation pattern, which the literature review already flagged as directly transferable.
- **Model:** **Claude Sonnet 5** as the default backbone — strong general coding ability at a reasonable cost, and this task (single-file, fixed-framework script generation) is narrower than the open-ended multi-file repo generation PaperCoder targets, so it should not need Opus-tier reasoning by default. Reserve **Claude Opus 4.8** for scripts that have failed 2+ retries, where the extra reasoning budget is worth the cost.
- **Key risk:** single-shot generation essentially never works on non-trivial tasks — MLR-Copilot's own ablation showed a single-prompt baseline at **0% success across every one of 5 tasks and 8 trials**, while the iterative loop reached up to 50%. *Mitigation:* this is exactly why the Critic↔Coder retry loop is not optional polish — budget real engineering time for it in M2/M3 rather than treating it as a stretch feature.

### 2.4 Runner

- **Responsibilities:** execute the Coder's script inside a Docker sandbox; capture metrics, stdout/stderr, and error traces; write results back to shared memory.
- **Inputs/Outputs:** `coder_output.script_path` → `runner_output`.
- **Tools:** Docker (isolated container per run, no host access — see AI Scientist v1's documented sandbox-escape incidents as the concrete cautionary example for why this must not be skipped); a log-parsing/truncation tool so error traces fit in context without flooding the Critic's prompt.
- **Model:** the Runner itself does not need an LLM to execute code — but a cheap model call is useful to **triage** the outcome before handing it to the Critic (classify: clean success / recoverable error / environment error / timeout). Use **Claude Haiku 4.5** for this triage step; it's a high-volume, low-complexity classification task and doesn't warrant Sonnet-tier cost.
- **Key risk / high-value technique to adopt:** AutoReproduce's **sampling-based unit testing** — validating tensor shapes and dry-running the full pipeline (with an early-`break` before the real training loop) on a small batch *before* committing to a full run — was the single largest driver of their execution-rate improvement (2.6–23% for non-lineage baselines up to 77–95% for AutoReproduce) and their performance-gap reduction (removing it inflated their gap metric from 31.62% to 88.78% in ablation). This is directly reusable and should be implemented as the Runner's default behavior, not an optional step: always dry-run on a mini-batch first, only proceed to a full training run once the dry run passes.
- **Compute discipline:** cap wall-clock time per training run (PaperBench uses 12h; given ReproBot's CIFAR/small-ImageNet scope, a much shorter cap — e.g., 1–2 hours per attempt — should be sufficient and keeps iteration cheap).

### 2.5 Critic

- **Responsibilities:** compare `runner_output.reproduced_metrics` against `reader_output.claims`; produce a structured verdict (`pass` / `retry` / `fail`) with a numeric gap and, on `retry`, specific and actionable feedback (not "try again" — a concrete hypothesis, as in the AutoReproduce-inspired example in §1.2).
- **Inputs/Outputs:** `reader_output.claims` + `runner_output` → `critic_output`.
- **Tools:** none beyond arithmetic/statistical comparison — deliberately kept simple and auditable (a hard numeric tolerance check, not an LLM "eyeballing" the numbers). This is a direct lesson from Agent Laboratory's own finding: their automated self-reviewer overestimated quality by 2.3 points (out of 10) relative to real human judges. ReproBot's core value proposition is exactly the opposite of that failure mode — an *objective*, number-vs-number check — so the Critic's pass/fail arithmetic should stay simple and transparent even if an LLM is used to draft the natural-language feedback.
- **Model:** **Claude Sonnet 5** for the standard case (compare numbers, draft feedback). For borderline cases — e.g., is a 1.5-point accuracy gap within normal training-seed variance, or a real bug? — use **Claude Opus 4.8 with extended thinking**, since this is the one place in the pipeline where genuinely nuanced judgment (not just arithmetic) is needed.
- **Verdict structure — borrow PaperBench's three-way split rather than a flat pass/fail:** internally track whether the script even *executed* (Execution), whether it appears to correctly implement the described method (Code Development), and whether the final number matches (Result Match) — this partial-credit structure is what lets the Report Generator show meaningful progress even on papers that don't fully pass, and it's what lets the Orchestrator distinguish "close, keep retrying" from "fundamentally broken, escalate."

### 2.6 Report Generator

- **Responsibilities:** once the Orchestrator ends the loop (pass, fail, or budget exhausted), produce the structured Markdown replication report: final script, claim-by-claim comparison table, and gap analysis.
- **Inputs/Outputs:** full shared-memory object (including `history`) → Markdown report.
- **Tools:** file write only.
- **Model:** **Claude Sonnet 5** — this is a structured-writing task, not a reasoning-heavy one.

---

## 3. Model Selection Summary

| Agent | Primary model | Escalation / alternate | Why not always use the strongest model |
|---|---|---|---|
| Orchestrator | none (deterministic) / Claude Haiku 4.5 for edge-case routing | — | Routing logic should be cheap and auditable, not a black-box LLM decision |
| Reader | Claude Sonnet 5 (vision) | Claude Opus 4.8 for low-confidence figure/table extractions | Sonnet matched or beat every backbone tested in PaperBench's closest analogous task |
| Coder | Claude Sonnet 5 | Claude Opus 4.8 after 2+ failed retries | Single-file, fixed-framework generation is narrower than open-ended repo generation; save Opus budget for genuinely stuck cases |
| Runner (triage only) | Claude Haiku 4.5 | — | High-volume, low-complexity classification (success/error/timeout) |
| Critic | Claude Sonnet 5 | Claude Opus 4.8 (extended thinking) for borderline numeric calls | Most comparisons are simple arithmetic; reserve deep reasoning for genuine ambiguity |
| Report Generator | Claude Sonnet 5 | — | Structured writing task |

This deviates from the proposal's "OpenAI or open-source API usage, LLaVA/GPT-4V" language by standardizing on one vendor/model family — recommended for consistency of behavior across agents and because the closest empirical evidence available (PaperBench's own backbone comparison) favors Claude Sonnet for exactly this kind of paper-comprehension-and-agentic-coding task. If cost or access constraints make Claude infeasible, GPT-4o is the next-best evidenced fallback per the same PaperBench comparison table.

---

## 4. Paper & Dataset Selection Criteria

To keep the 20-paper (or fewer, per the staged plan below) benchmark tractable on realistic student compute:

- **Datasets:** CIFAR-10/CIFAR-100, MNIST-family, and class-balanced ImageNet subsets (e.g., Imagenette/Imagewoof-style subsets) — all trainable end-to-end on a single consumer/cloud GPU within a bounded time cap.
- **Architectures:** canonical, well-documented families only for the initial set — ResNets, plain CNNs, small Vision Transformers, EfficientNet-style models. Avoid papers requiring custom CUDA kernels, distributed multi-GPU training, or non-standard training infrastructure (these introduce Runner-environment failure modes unrelated to the replication question being tested).
- **Claim structure:** prefer papers with a single, clearly-tabulated headline metric (e.g., "Table 2: Top-1 accuracy on CIFAR-10") over papers whose main claims are qualitative or spread across many ablations — this keeps the Critic's core comparison task well-defined.
- **Avoid at launch:** papers requiring pretraining at scale (ImageNet-1k from scratch, multi-day runs), proprietary datasets, or non-public code dependencies.

---

## 5. Revised Timeline

The proposal's 4 monthly milestones are directionally right; below is a more granular breakdown with an explicit pilot checkpoint before committing to the full 20-paper scope.

**Month 1 — Reader + pilot paper set**
- Weeks 1–2: PDF ingestion (pdfplumber + pdf2image), Reader prompt/schema design, shared-memory object definition.
- Weeks 3–4: Reader validated end-to-end on **3–5 pilot papers** (hand-checked extraction quality against the papers directly) before touching the Coder.

**Month 2 — Coder + Runner + Orchestrator skeleton**
- Weeks 1–2: Coder generating scripts for the pilot set; Docker sandbox + sampling-based dry-run validation (§2.4) built first, before full-run execution — this is the cheapest way to catch generation bugs.
- Weeks 3–4: Orchestrator skeleton (LangGraph state machine) wired end-to-end Reader→Coder→Runner on the pilot set, with a *hard-coded* single retry (no Critic yet) just to validate the plumbing.

**Month 3 — Critic + retry loop + report generation**
- Weeks 1–2: Critic verdict logic (pass/retry/fail + three-way Execution/Code-Dev/Result-Match split); wire the real retry loop with targeted feedback.
- Weeks 3–4: Report Generator; run the **single-shot vs. iterative-loop ablation on the pilot set** — this is the headline result and should exist well before the full benchmark run, both to validate the loop is actually helping and as a fallback deliverable if time runs short.

**Month 4 — Scale-up, evaluation, demo, report**
- Weeks 1–2: Expand from the pilot set toward the full paper set as compute/time allows; this is where "20 papers" is a target, not a guarantee — report actual coverage honestly, the way every paper reviewed in the literature review does (none claim 100%).
- Weeks 3–4: Gradio demo, ablation write-up, final project report (using `Papers/Introduction_and_Literature_Review.md` as the literature review section).

**Explicit checkpoint:** if, by the end of Month 2, the Reader→Coder→Runner plumbing is not working end-to-end on the pilot set, de-scope the Critic to a simpler single-pass comparison (no retry loop) for Month 3, and prioritize a working demo on fewer papers over an unfinished retry loop on more papers.

---

## 6. Evaluation & Success Metrics

Borrowing directly from the literature review's gap analysis, report all three of the following — not just an aggregate "replication rate":

1. **Coverage:** how many of the target papers produced *any* executing script (Execution, in PaperBench's terms).
2. **Fidelity:** of those, how many reproduced the claimed metric within a stated tolerance (Result Match) — report the actual gap percentage per paper, not just pass/fail, the way AutoReproduce's Performance Gap metric does.
3. **The ablation (headline result):** replication rate/fidelity with the Critic-driven retry loop enabled vs. disabled (single-shot Coder only) — this isolates and demonstrates the specific contribution the literature review identifies as ReproBot's actual gap-filling feature, and is achievable even on a small pilot set.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Retry loop never converges / burns budget on repeated identical failures | Medium | High | Orchestrator checks `history` for near-duplicate Coder attempts before retrying (§2.1) |
| Data-processing/preprocessing details silently wrong (papers under-specify this most) | High | Medium | Dedicated Reader extraction pass for data pipeline details (§2.2) |
| Docker sandbox escape / runaway resource usage | Low but high-severity if it happens (documented in AI Scientist v1) | High | No host access from container; hard wall-clock + storage caps per run |
| API/compute cost exceeds available budget before 20-paper target is reached | Medium-High | Medium | Pilot-first staged plan (§5); tight time caps per run; report partial coverage honestly (§0.3, §5) |
| Critic's numeric tolerance is either too strict (rejects valid noisy-but-correct runs) or too loose (passes real bugs) | Medium | Medium | Escalate borderline cases to Opus with extended thinking (§2.5) rather than a fixed hard threshold alone |

---

## 8. Compute & Cost Budget (rough estimate)

Numbers below are order-of-magnitude estimates for planning, not quotes:

- **LLM API cost per paper:** Reader (1–2 vision calls) + Coder (1 call per retry attempt, budget ~5 retries) + Critic (1 call per attempt) + Report Generator (1 call) ≈ 8–15 model calls per paper at Sonnet-tier pricing — comparable in shape to Agent Laboratory's reported $2–$8/paper full-pipeline cost, well below PaperBench's ~$400/paper (ReproBot's shorter time caps and narrower scope are what keep this low).
- **GPU compute per paper:** CIFAR/small-ImageNet-scale training runs, capped at 1–2 hours per attempt × up to 5 retries ≈ 2–10 GPU-hours/paper — feasible on a single cloud GPU instance or free-tier compute (Colab/Kaggle) for the pilot set; budget cloud credits if scaling to the full 20-paper set.
- **Recommendation:** track actual $ and GPU-hour spend from the very first pilot paper in Month 1, not just at evaluation time — this is the concrete number needed to decide, by the Month-4 checkpoint, how much of the 20-paper stretch goal is realistically affordable.
