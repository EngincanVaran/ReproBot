# Code Generation & Debug/Repair Loops in AutoReproduce and AutoP2C

> Working note, not part of the polished literature-review stack (`docs/literature-review/`). Companion to `reader-agent-precedents.md`, same method (read the actual PDFs in `papers/`, not summaries), but scoped to the **next** pipeline stage: the Coder. Focuses narrowly on code generation granularity, the debug/repair loop mechanics, pre-full-run validation, and what the ablations say about whether any of it actually matters — not the paper-parsing stages, already covered in `reader-agent-precedents.md`.

## TL;DR

Both systems reject single-shot code generation and both gate expensive execution behind something cheaper first — but the two mechanisms are shaped differently, and the difference maps directly onto an open ReproBot design question (repo-scale vs. single-script Coder):

- **AutoReproduce**: single-script, line-range-precise edits. A structured `EDIT N M\n<new code>` command replaces exactly lines *N* through *M* of the current file — never a full-file rewrite. Retries are triggered by **two independent things**: execution errors (debugged by the Code Agent) *and* semantic misalignment against the paper (flagged by the Research Agent, even when the code runs fine) — this second trigger already anticipates a Critic-like role. Pre-full-run validation is a genuine cheap dry run: generate-and-execute small analysis code on a sampled mini-batch to learn tensor shapes/dtypes, then run the full script with an early `break` swapped in before the real training loop. Ablating debug+refine barely moves the LLM-judged alignment score (69.97→68.32) but **explodes the Performance Gap metric from 31.62% to 88.78%** — code that "reads right" but was never actually run is badly wrong.
- **AutoP2C**: multi-file repository, file-granularity edits. Code is generated **progressively, file by file**, in dependency-graph order (`Code_i = LLM_ImplementCode(τ_i, {Code_j|j<i}, P_distilled)`). The debug loop is a two-phase **localize-then-correct** process (`LLM_LocalizeError` → `LLM_CorrectError`), operating on whole files identified by the localization step, not arbitrary line ranges. Pre-execution validation is *static/semantic*, not a dry run: an LLM cross-checks the generated code against the paper's text/diagrams/equations on three specific axes — architecture, loss function, optimizer/update rule — before anything is executed at all. Ablating the iterative feedback-driven implementation stage entirely doesn't just hurt scores, it produces **0% executable code across every test paper** (confirmed, exact framing below).
- Neither reports a debug loop that's still improving by turn 20 — AutoReproduce's own numbers show convergence (or failure) happening in 5–8 iterations in practice, and AutoP2C's HPO step (`ray.tune`, wired in only *after* the code first executes cleanly) is evidence that "does it run" and "is it well-tuned" are treated as sequential, not simultaneous, concerns in both systems.

---

## 1. AutoReproduce's EDIT-based debug/refine loop

AutoReproduce (Zhao et al., 2025, Tsinghua) has no separate "Coder" role — the **Code Agent** (paired with the Research Agent throughout) does all implementation and debugging inside the third pipeline phase, **Code Development**, which itself has three sequential stages: Data Acquisition → Method Replication → Experiments Execution, ending in a final refactor pass.

```
reader_output-equivalent (paper summaries) + Paper Lineage K_lineage
                      │
                      ▼
      ┌───────────────────────────────────┐
      │ Code Agent: initial implementation    │
      │ (Data Acquisition → Method Replication)│
      └──────────────────┬───────────────────────┘
                          ▼
      ┌───────────────────────────────────────────┐
      │ PRE-RUN VALIDATION (sampling-based, cheap):   │
      │  • proactive inference: generate + EXECUTE      │
      │    small analysis code on a sampled mini-batch  │
      │    to learn real tensor shape/dtype attributes  │
      │  • full experiment script assembled with early- │
      │    exit `break` swapped in → fast structural     │
      │    dry run before committing to a real training  │
      │    loop                                            │
      └──────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴────────────┐
              │ execution error?         │  research agent also independently
              ▼ yes                     │  checks: does code semantically
      ┌─────────────────────┐          │  align with the paper summary?
      │ Research Agent:         │          │  (can fire even if code RUNS fine)
      │ diagnose error trace,    │          ▼ misaligned
      │ produce localized         │  ┌─────────────────────┐
      │ diagnosis (analysis only,│  │ Research Agent updates   │
      │ no code written here)    │  │ the paper summary to     │
      └──────────┬────────────────┘  │ steer the fix, flags     │
                 │                      │ the discrepancy           │
                 ▼                      └──────────┬─────────────────┘
      ┌─────────────────────────────────────────────▼─────┐
      │ Code Agent: EDIT command                              │
      │   ```EDIT\n N M\n<new code>\n```                      │
      │   replaces lines N..M of the CURRENT file only —       │
      │   never a full-file regeneration. Single command per   │
      │   inference turn.                                       │
      └──────────────────────┬─────────────────────────────────┘
                             │
                loop back to pre-run validation
                (cap: 20 debug tries per subphase;
                 in practice converges — or gives up —
                 in 5-8 iterations; further tries rarely
                 help)
                             │
                             ▼ (both checks pass)
                  research agent submits code
                  → proceed to next subphase / full run
```

**The EDIT command, verbatim from the paper (§3.2.3 + Figure 7):**
> "In response to execution errors, the code agent first diagnoses the error traceback and refines the script using the `EDIT` command, structured as ` ```EDIT\n N M\n<new code>\n``` `. We decouple error analysis and code editing into two distinct steps, as we observe that conducting a preliminary analysis to guide the debugging process significantly improves the success rate. The `EDIT` command facilitates targeted updates by replacing lines N through M with the generated code segment, rather than regenerating the entire file. This granular approach significantly reduces token generation overhead, and is employed consistently throughout all subsequent phases of our framework."

**Two independent retry triggers, not one.** This is easy to miss on a skim: the Method Replication stage explicitly runs *two* checks in parallel, not just "did it crash":
1. The Code Agent debugs **execution/data-flow errors** — "the code agent debugs potential errors in model computations by inspecting data flow properties."
2. The Research Agent independently **validates semantic alignment against the paper summary** — "the research agent validates the code against the paper summary, updating the summary to guide the code agent in resolving any identified discrepancies. The research agent chooses to submit the code once it fully aligns with the paper" — this can fire *even when the code runs without error*, i.e. it's a proto-Critic check baked into the Coder-equivalent stage, not deferred to a separate agent.

Table 6 reports these as two separate categories of `EDIT`-command usage, **Debugging** vs. **Refinement**, each with its own turn/line averages — confirming they're tracked as distinct loop types, not folded together. Representative numbers (Claude-3.5-Sonnet backbone): Method-stage debugging averages 5.78 turns / 33.52 edited lines per turn; Method-stage refinement averages 2.16 turns / 40.53 lines. o3-mini and Gemini-2.5-Pro both need noticeably *fewer* debug/refine turns than Claude-3.5-Sonnet to converge — the paper's own reading is that the weaker-seeming backbones are actually more debug-efficient here.

**Retry budget:** "We set the maximum debug tries to 20 for all the subphases in the code development phase. However, further debug attempts generally do not produce more executable code; bugs are typically fixed within 5–8 iterations." No explicit statement of what happens exactly at cap-exhaustion (paper doesn't describe a hard-fail branch), but the framing is clear: 20 is a safety ceiling, not a target — real convergence is an order of magnitude cheaper.

**Ablation evidence the loop matters (Table 4, Claude-3.5-Sonnet backbone):**

| Ablation | Mixed-Level (alignment, LLM-judged) | Perf. Gap % (↓ better) |
|---|---|---|
| Full AutoReproduce | 69.97 | 31.62 |
| `w/o Refine` (drop the semantic-alignment check only) | 65.78 | 36.37 |
| `w/o Debug+Refine` (drop both retry mechanisms) | 68.32 | **88.78** |

The headline finding: dropping Debug+Refine barely moves the *alignment* score (69.97→68.32, a 1.65-point drop) but **more than doubles the Performance Gap** (31.62%→88.78%) — the single largest degradation anywhere in the paper's ablation table, worse even than removing MinerU OCR entirely (47.81%) or Paper Lineage (39.59%). The code can look like a faithful implementation of the paper to an LLM judge and still be numerically catastrophic if it was never actually run and fixed.

**No hyperparameter-search / post-success refinement step is described anywhere in AutoReproduce** — confirmed absent by reading the full Code Development section; the pipeline ends at the final refactor pass ("remove unnecessary debug settings and clean up the generated code").

**Cost:** $1.87/paper average, o3-mini backbone, across the full pipeline (not debug-loop-only). Main experiments use `claude-3-5-sonnet-20240620` as the primary backbone.

---

## 2. AutoP2C's localize-then-correct debug loop

AutoP2C (Lin et al., 2025) has four stages: (1) Repository Blueprint Extraction (offline, paper-agnostic), (2) Multimodal Content Parsing (the Reader-equivalent — covered in `reader-agent-precedents.md`), (3) Hierarchical Task Decomposition, (4) **Iterative Feedback-Driven Implementation** — this last stage is the Coder-equivalent and where all code generation, validation, and debugging live.

```
P_distilled (unified multimodal paper repr.) + repository blueprint T
+ per-file task descriptions {τ_1 ... τ_n}, dependency-graph ordered
                      │
                      ▼
      ┌───────────────────────────────────────────┐
      │ PROGRESSIVE FILE-BY-FILE GENERATION:            │
      │  Code_i = LLM_ImplementCode(τ_i,                  │
      │             {Code_j | j < i}, P_distilled)         │
      │  — one file at a time, dependency order,            │
      │    each new file sees all previously-generated       │
      │    files as context (model: o1-mini)                 │
      └──────────────────────┬─────────────────────────────────┘
                             ▼
      ┌───────────────────────────────────────────┐
      │ PRE-RUN VALIDATION (static / semantic, NOT     │
      │ an execution dry run):                            │
      │  V = LLM_validate({Code_i}, P_distilled)          │
      │  cross-modal check of code against the paper's    │
      │  own text / diagrams / equations, on 3 axes:       │
      │   1. model architecture matches the paper's spec   │
      │   2. loss function: mathematical correctness         │
      │   3. optimization strategy: update rule alignment    │
      │  (model: o1)                                          │
      └──────────────────────┬─────────────────────────────────┘
                             ▼
                execute {Code_i}  (actual testing)
                             │
                  ┌──────────┴──────────┐
                  │ error E?              │
                  ▼ yes                   ▼ no
      ┌─────────────────────────┐   proceed to
      │ PHASE 1 — Localize:         │   post-success HPO
      │  L = LLM_LocalizeError(       │   (below)
      │        E, {Code_i})            │
      │  identifies WHICH files/       │
      │  components are implicated —   │
      │  minimizes scope of change,     │
      │  reduces risk of side effects   │
      └────────────┬─────────────────────┘
                   ▼
      ┌─────────────────────────────┐
      │ PHASE 2 — Correct:              │
      │  {Code'_i} = LLM_CorrectError(     │
      │       L, P_distilled)                │
      │  targeted modifications to the        │
      │  localized files — may adjust          │
      │  function interfaces, data              │
      │  structures, implementation              │
      │  details for cross-file compatibility    │
      │  (model: o3-mini)                          │
      └────────────┬───────────────────────────────┘
                   │
        loop back to execute {Code'_i}
        (no explicit retry cap stated in the paper)
                   │
                   ▼ (once it executes cleanly)
      ┌───────────────────────────────────────┐
      │ POST-SUCCESS HPO:                         │
      │  wire ray.tune into the main execution      │
      │  file; extract hyperparameters + search      │
      │  ranges from the paper's experimental          │
      │  section, not open-ended                        │
      └───────────────────────────────────────────────┘
```

**Model split across the loop is deliberate, not incidental**: GPT-4o for repository blueprint / multimodal parsing / task decomposition, **o1-mini** for initial per-file generation, **o1** for the pre-run validation step (architecture/loss/optimizer check), **o3-mini** for the debug loop's correction phase — four different models for four different cognitive demands, contrasted with AutoReproduce's single Claude-3.5-Sonnet backbone doing everything.

**Validation is explicitly separate from, and prior to, the debug loop.** The paper is precise about this: "Despite the validation step, producing a fully executable pipeline in a single pass remains challenging due to complex inter-file interactions and interface consistency requirements. To address this, AutoP2C implements an iterative debugging mechanism... where `E` denotes execution errors encountered during testing." So `LLM_validate` is a static, pre-execution semantic check (does the code *look* architecturally/mathematically right against the paper), and the debug loop only exists to handle what that check can't catch — genuine runtime execution failures.

**Ablation — the 0% executability claim, confirmed and exact framing (Table 3):**
> "Without the iterative feedback-driven implementation, AutoP2C relies entirely on single-pass generation, resulting in **completely non-executable implementations across all test cases** and yielding the lowest average class completeness of 15.9%. Moreover, the function completeness falls to 25.1%."

This is the single most damaging ablation of the four AutoP2C stages tested — worse than removing the repository blueprint (83.5%→51.4% perf), worse than removing multimodal parsing (→53.5%), worse than removing hierarchical task decomposition (→forces single-pass generation too, but that variant still isn't reported as 0%-executable the way removing Feedback specifically is). The per-modality Table 4 ablation (image/table removal) is a separate, smaller-scope experiment on the *parsing* stage, not this one — don't conflate the two ablations when citing this number.

**Post-success HPO, confirmed exact trigger condition:** "After successful debugging, AutoP2C integrates hyperparameter optimization (HPO) through an automated process following prior work. Specifically, we incorporate `ray.tune` into the main execution file, enabling exploration of the hyperparameter space defined in the paper. The HPO implementation extracts relevant hyperparameters from the experimental sections of the paper and configures appropriate search spaces based on reported values and ranges." Two things worth being precise about: (1) this fires strictly *after* the code first executes without error — it is not concurrent with debugging; (2) the search space is bounded by what the paper itself reports (values/ranges), not an open-ended sweep — HPO here is "recover the paper's own hyperparameters if the Reader's extraction was imprecise," not "find better hyperparameters than the paper."

**Cost/iteration data:** AutoP2C does not report retry-count or turn statistics the way AutoReproduce's Table 6 does. What it reports instead is token consumption per paper (Table 1) — e.g. 852K in / 120K out, 1,177K in / 145K out, up to that range across the 8-paper benchmark — consistent with the progressive file-by-file approach's context accumulating across a growing repository (each new file's prompt includes every previously generated file). No explicit statement of a debug-loop retry cap or "attempts plateau around N" finding — this is a real gap relative to AutoReproduce's much more instrumented reporting.

---

## 3. Side-by-side

| | AutoReproduce | AutoP2C |
|---|---|---|
| Code generation target | Single script, three sub-phases (Data Acquisition → Method Replication → Experiments Execution) | Multi-file repository, dependency-graph ordered |
| Generation granularity | Whole-script initial pass, then **line-range** edits | **Whole-file** progressive generation, one file at a time |
| Edit command / mechanism | `EDIT\n N M\n<new code>` — replaces lines N..M of current file, one command per turn | No named syntax; `LLM_CorrectError(L, P_distilled)` regenerates the files identified by localization |
| What triggers a retry | (1) execution error, debugged by Code Agent; (2) semantic misalignment vs. paper, flagged independently by Research Agent even if code runs | Execution error `E` only, encountered during post-validation testing |
| What's fed back into the next attempt | Diagnosis-first, code-second: error trace is analyzed before any edit is written (decoupled on purpose — improves success rate per the paper) | Two-phase: `LocalizeError` (which files) → `CorrectError` (targeted fix guided by `P_distilled`, the paper's own distilled content) |
| Pre-full-run validation | **Dynamic** — actually execute small analysis code on a sampled mini-batch (learn tensor shapes/dtypes); full script assembled with early `break` for a structural dry run | **Static** — LLM cross-checks code against paper's text/diagrams/equations on 3 axes (architecture, loss, optimizer), no execution involved |
| Retry budget / cap | Explicit: 20 max per subphase; converges (or doesn't) in practice within 5–8 | Not stated in the paper |
| Ablation: removing the debug/repair loop | Alignment score barely moves (69.97→68.32); Perf. Gap explodes 31.62%→88.78% | Executability collapses to **0% across every test paper**; class completeness 61.6%→15.9%, function completeness 43.4%→25.1% |
| Post-success HPO step | Not present | Yes — `ray.tune`, wired in only after first clean execution, search space bounded by paper's own reported hyperparameter values/ranges |
| Backbone model(s) for the coding/debug loop | `claude-3-5-sonnet-20240620` for everything | o1-mini (initial gen) → o1 (validation) → o3-mini (debug correction) — task-split |
| Cost/iteration reporting | Turn + line-count stats per phase per backbone (Table 6); $1.87/paper full-pipeline avg. (o3-mini) | Token counts per paper only (300K–1.1M in / 77–145K out); no turn-count or plateau data reported |

---

## 4. Implications for ReproBot's Coder

Grounding in the project plan's existing spec (`docs/project-plan/ReproBot_Project_Plan.md` §2.3 Coder, §2.4 Runner, §1.2 shared-memory schema, §1.3 LangGraph recommendation):

**(a) What should trigger the Coder→Runner→back-to-Coder loop at the smallest useful first slice, before Critic exists at all.**
Both systems' loops are cleanly separable into two trigger types — "did it execute" and "does it semantically/numerically match the paper" — and in both papers the *execution* trigger is the one that requires no LLM judgment call: it's a boolean (or classified: clean success / error / timeout, exactly as the project plan's Runner triage step already proposes with Haiku 4.5). AutoP2C's debug loop is triggered by execution error `E` alone; even AutoReproduce's dual-trigger design keeps the execution-error path structurally separate from its semantic-alignment path (different agent, different check). This maps directly onto the project plan's own staged build order (§5, Month 2: "hard-coded single retry, no Critic yet, just to validate the plumbing") — **the first-slice loop trigger should be exactly "Runner reports non-zero exit / exception / timeout," nothing more**, deferring the "does the reproduced number match the claim" trigger to when the Critic actually exists. This is not a compromise; it's what both precedent systems structurally do anyway — the semantic-alignment check (AutoReproduce's Research Agent re-validation, AutoP2C's pre-execution `LLM_validate`) is architecturally a *different* mechanism from the execution-error debug loop, not a fancier version of it, so building the execution-error loop first doesn't paint ReproBot into a corner when the Critic is added later.

**(b) What edit granularity ReproBot's Coder should use.**
ReproBot's Coder targets one fixed shape — a single self-contained HuggingFace `Trainer` script (§2.3) — which is architecturally much closer to AutoReproduce's single-script setting than AutoP2C's multi-file repository. AutoP2C's file-level localize-then-correct machinery exists to solve a problem ReproBot's Coder doesn't have (which of N files does this bug live in) — that complexity buys nothing here and shouldn't be imported. **Adopt AutoReproduce's `EDIT N M` line-range mechanism directly**: a structured tool call (`start_line`, `end_line`, `new_code`) is simple to implement as a Claude tool-use schema, is exactly the mechanism the paper credits for reducing token overhead on retries (avoiding full-script regeneration every attempt), and — concretely — the shared-memory schema's `coder_output.diff_from_previous` field (§1.2) is already shaped for exactly this: an `EDIT`-style diff is a natural fit for that field, a full-file overwrite is not. Also worth replicating: the **decoupled diagnose-then-edit** structure both papers converge on independently (AutoReproduce explicitly, AutoP2C via its two-phase Localize/Correct split) — feed the Coder a *diagnosis* of the Runner's error trace first (a cheap, possibly Haiku-tier localization step, consistent with the Runner's already-planned triage call), not the raw stack trace directly into the same call that's asked to also write the fix.

**(c) Whether ReproBot needs its own sampling-based dry-run validation before a full training run, and why.**
Yes, and AutoReproduce's ablation is the concrete evidence, not just intuition: removing the debug/refine loop barely dents the LLM-judged *alignment* score (a ~2.4% relative drop) while nearly tripling the *performance gap* metric (31.62%→88.78%) — meaning code can pass every static "does this look like a faithful implementation" check and still be numerically wrong or non-executing the first time it's actually run. AutoP2C's own error-analysis appendix independently confirms the failure mode this catches: "the majority of issues stemmed from incorrect data shapes during internal model calculations" — exactly what AutoReproduce's proactive-inference dry run (execute small analysis code on a real mini-batch to learn tensor shapes *before* generating the full script) is built to catch cheaply. Concretely, ReproBot's Runner (§2.4, already speccing "sampling-based dry-run testing" as default behavior) should implement the *two specific* mechanisms AutoReproduce actually uses, not a generic "run on a small batch": (1) a shape/dtype probe — generate and execute small analysis code against one real mini-batch before committing to the full script, and (2) a full-script structural dry run with an early-exit swapped in before the real training loop, to catch integration bugs (data loading → model → loss → optimizer step) end-to-end, cheaply. AutoP2C's complementary `LLM_validate` (static architecture/loss/optimizer alignment against the paper, zero execution cost) is worth adding too, but as an *earlier*, even-cheaper gate before the dry run — not a replacement for it, since it's exactly the kind of check AutoReproduce's ablation shows isn't sufficient on its own.

**One more finding worth flagging even though it wasn't in the original scoping questions:** AutoP2C's post-success `ray.tune` step (search space bounded by the paper's own reported hyperparameter values/ranges, triggered only once the script first executes cleanly) is a plausible answer to a risk the project plan already names but doesn't fully resolve — §2.5's "is a 1.5-point accuracy gap within normal training-seed/hyperparameter variance, or a real bug?" Rather than routing every borderline gap to Opus-with-extended-thinking, a bounded post-success HPO pass (search only within the paper's own stated ranges, not open-ended) could let the Critic distinguish "the implementation is wrong" from "the implementation is right but under-tuned within the paper's own documented hyperparameter space" — worth keeping as a candidate post-MVP Coder/Runner feature once the core execution-error loop is working, not something to build in the first slice.
