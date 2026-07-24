# Evaluation & Metrics Comparison — PaperBench, Paper2Code, AutoP2C, AutoReproduce

> inzva AI Projects #10 | ReproBot
> Companion to `PaperAgent_LiteratureReview.md` and `Paper_Summaries.md`. Each of the four replication-focused papers reviewed here invented its own evaluation vocabulary; this document inventories those metrics, maps them onto a shared taxonomy, checks judge/grader trustworthiness, and proposes a combined rubric for ReproBot's own Critic.

---

## Table of Contents

1. [Table A — Raw Metric Inventory Per Paper](#table-a--raw-metric-inventory-per-paper)
2. [Table B — Unified Dimension Taxonomy](#table-b--unified-dimension-taxonomy)
3. [Table C — Judge / Grader Trustworthiness](#table-c--judge--grader-trustworthiness)
4. [Proposed Combined Rubric for ReproBot](#proposed-combined-rubric-for-reprobot)
5. [Scale Note](#scale-note)

---

## Table A — Raw Metric Inventory Per Paper

| Paper | Metric | Definition / Formula | Scale | Grader | Granularity |
|---|---|---|---|---|---|
| **PaperBench** | Replication Score | Weighted bottom-up average of binary leaf scores across a hierarchical rubric tree (8,316 leaves / 20 papers) | 0–100% | LLM judge (SimpleJudge, default o3-mini-high) | Per-leaf-node → paper-level |
| | ↳ leaf types | Code Development / Execution / Result Match (each 0/1) | binary | same | per requirement |
| | PaperBench Code-Dev | Same rubric, restricted to Code-Dev leaves only (no execution needed) — cheap proxy | 0–100% | same | paper-level |
| | JudgeEval | Meta-metric: F1 of the judge itself vs. human gold labels | F1 | — | judge-level |
| **PaperCoder** | Reference-based score | 5-pt Likert vs. gold author repo + paper, by component severity (high/med/low) | 1–5 | LLM judge (o3-mini-high), 8 samples avg | paper-level |
| | Reference-free score | Same Likert, judged vs. paper only (no gold repo) | 1–5 | same | paper-level |
| | Human score/rank | Original paper's first authors rank system outputs | rank → 5/3/1 | human | paper-level |
| | Component coverage | % of Helpfulness / Method / Eval / Data-Processing components present | % | LLM judge | per-component |
| | LOC-fix rate | % of generated lines needing manual fix to execute | % | manual (5 papers) | file-level |
| | *(borrowed)* PaperBench Rep. Score | reported when evaluated on PaperBench Code-Dev / full | % | SimpleJudge | as PaperBench |
| **AutoP2C** | Executability rate | # papers producing runnable code / total | % | automated (run/no-run) | paper-level |
| | Absolute performance | Actual reproduced metric value | metric-native | automated | paper-level |
| | Relative performance | `P_agent / P_paper × 100%` — **no threshold defined** | % | automated | paper-level |
| | COMP_class / COMP_func | LLM-judged % of reference classes/functions faithfully reproduced | % | LLM judge | per-file |
| **AutoReproduce** | Align-Score (Paper-Level) | o1 extracts 5 critical components from paper, judges code coverage | 0–100 | LLM judge (o1) | paper-level |
| | Align-Score (Code-Level) | LLM judge compares code vs. cleaned reference across 4 dimensions | 0–100 | LLM judge | code-level |
| | Align-Score (Mixed-Level) | Combines paper-objectives + reference-code context | 0–100 | LLM judge | hybrid |
| | Execution Rate | % of generated code that runs | % | automated | paper-level |
| | Performance Gap | `(1/n)Σ |P_ref − P_agent| / max(P_ref, P_agent)`; non-executable scored as gap = 1.0 | 0–1 (lower better) | automated | paper-level (avg over n metrics) |
| | Human eval | Method / Parameter / Experiment / Overall, scored out of 10/5/5/20 | points | human | paper-level |
| | Lineage Recall@k / Hits@N | Retrieval quality of the paper-lineage component itself, vs. expert gold references | % / count | human-curated gold set | component-level |

---

## Table B — Unified Dimension Taxonomy

Underneath the different names, every metric falls into one of five dimensions. The four papers *converge* on the same underlying questions even though their vocabulary doesn't match:

| Dimension | PaperBench | PaperCoder | AutoP2C | AutoReproduce |
|---|---|---|---|---|
| **Executability** ("does it run?") | Execution leaf nodes | LOC-fix rate (weak proxy — not primary metric) | Executability rate | Execution Rate (Exec-Score) |
| **Structural/code fidelity** ("does the code look right?") | Code Development leaf nodes | Ref-based/free Likert, component coverage | COMP_class / COMP_func | Align-Score (Code-Level) |
| **Numeric result fidelity** ("does the number match?") | Result Match leaf nodes | **none — never executes** | Absolute/relative performance (no threshold) | Performance Gap (continuous, no threshold either) |
| **Paper-level semantic fidelity** ("did it capture the core idea?") | Root score (aggregated) | Reference-free Likert | — | Align-Score (Paper-Level) |
| **Human-validated correlation** | JudgeEval (F1 = 0.83) + 8-PhD baseline | Author ranking, r = 0.71–0.78 | **none reported** | Human eval, small study |

**The gap that matters:** only PaperBench's leaf nodes carry an explicit, human-authored pass/fail criterion. AutoP2C's "relative performance" and AutoReproduce's "Performance Gap" are both continuous numbers with **no stated tolerance** — a paper at 89.8% relative performance and one at 122% are reported identically, with nothing in either method saying which one "passed."

---

## Table C — Judge / Grader Trustworthiness

The meta-layer most comparisons skip — worth checking since LLM-judge reliability is a direct risk for ReproBot's own Critic design.

| System | Judge model | F1 / Correlation vs. human | Cost |
|---|---|---|---|
| PaperBench (JudgeEval) | Random baseline | F1 = 0.49 | — |
| | GPT-4o-mini | F1 = 0.59 | $8/paper |
| | GPT-4o | F1 = 0.78 | $120/paper |
| | o1-mini | F1 = 0.73 | $72/paper |
| | o1 | F1 = 0.84 | $830/paper |
| | **o3-mini** (chosen) | F1 = 0.83 | $66/paper |
| PaperCoder | o3-mini-high | r = 0.74–0.78 (ref-based), 0.71–0.73 (ref-free) vs. human | — |
| AutoP2C | (not validated against humans) | — | — |
| AutoReproduce | o1 (Align-Score judge) | Mixed-Level claimed most human-aligned; no explicit correlation coefficient reported | — |

---

## Proposed Combined Rubric for ReproBot

Rather than inventing a sixth metric family, the strongest move is to **adopt PaperBench's three-tier taxonomy as the skeleton** (it's the most validated — author-co-developed rubrics, JudgeEval meta-check) but **fill each tier with the cheaper, more mechanistic techniques the other three papers proved work**, and add the one thing none of them have: an explicit numeric tolerance gate.

| Tier | What it checks | Borrowed technique | Source | Output |
|---|---|---|---|---|
| **1. Code Development** | Does the script structurally implement the paper's method? | Component-coverage check by section (method / data-processing / eval) — cheap, no GPU needed | PaperCoder's coverage breakdown + AutoP2C's COMP_class/func idea | partial-credit % |
| **2. Execution** | Does it run, cheaply verified before committing to full training | Sampling-based dry-run (mini-batch shape/dtype check + early-`break` full-pipeline dry run) — proven to be the single biggest lever (Perf Gap 31.62% → 88.78% without it) | AutoReproduce ablation (Table 4) | binary + triage class (clean / recoverable / env-error / timeout) |
| **3. Result Match** | Does the reproduced metric match the paper's claim? | Continuous gap signal (`Performance Gap` formula) **feeding an explicit numeric tolerance threshold** → pass/retry/fail verdict, not just a reported percentage | Formula from AutoReproduce/AutoP2C, **but** with the threshold none of them define | pass / retry / fail |
| **Meta-layer** | Is the Critic's own judgment trustworthy? | Validate the Critic's LLM-drafted feedback against a small hand-graded gold set, same pattern as JudgeEval | PaperBench | judge F1 / correlation |

---

## Scale Note

PaperBench's 8,316-leaf rubric took "many tens of hours" per paper to author — not viable for a 4-month, 20-paper project. ReproBot's simpler claim structure (headline top-1/top-5 accuracy per paper) is much closer to AutoReproduce's ReproduceBench scale (5 metrics × 13 papers, execution-verified ground truth) than to PaperBench's granularity — this is the scope justification to state explicitly if asked.

---

*ReproBot evaluation & metrics comparison — inzva AI Projects #10*
*Companion reference for the PaperBench / Paper2Code / AutoP2C / AutoReproduce presentation.*
