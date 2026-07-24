# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository is currently a **research/proposal phase** for ReproBot — there is no application code yet (no `src/`/`app/` package, no `tests/`, no Python tooling files). What exists today is the project proposal, progress reports, and a literature review built from 9 related papers, organized under `docs/`. When implementation starts, expect a standard Python project layout (package + tests + `pyproject.toml`) to be scaffolded fresh rather than assumed from anything currently in the repo.

## What ReproBot is

ReproBot (`docs/proposal/ReproBot.pdf` — the original project proposal) is a four-agent pipeline, coordinated by a central **Orchestrator** over **shared memory**, that reads an ML paper PDF and produces a verified replication report:

```
PDF paper
    │
    ▼
Orchestrator (shared memory state)
    │
    Reader (pdfplumber + VLM)   → method summary, claims, datasets, hyperparameters
    Coder (HuggingFace Trainer) → self-contained training script
    Runner (Docker sandbox)     → executes script, captures metrics + error traces
    Critic                      → compares reproduced metrics vs. paper's claimed numbers,
                                   emits pass/retry/fail verdict, feeds targeted fixes back to Coder
    │
    ▼
Structured Markdown replication report (claim-by-claim comparison + gap analysis)
```

The Orchestrator drives the Critic↔Coder retry loop until results clear a confidence threshold or the retry budget is exhausted. Initial evaluation scope is deliberately narrow: **20 image-classification papers** (not a broad multi-topic benchmark) — this is a legibility trade-off, not a limitation, since a single domain allows a much stricter numeric tolerance in the Critic's pass/fail logic. Proposed 4-month timeline: Reader (M1) → Coder+Runner+Orchestrator skeleton (M2) → Critic+refinement loop+report generation (M3) → evaluation+ablation+demo (M4). Planned stack per the proposal: LangChain (agent framework), pdfplumber/pdf2image (PDF parsing), LLaVA/GPT-4V (vision-language), Docker (sandboxing), Gradio (demo), Weights & Biases (tracking).

`docs/project-plan/ReproBot_Project_Plan.md` expands this into an implementation-ready plan — feasibility assessment, per-agent tool/model specs, a de-risked timeline, risk register, and a compute/cost budget. It supersedes the original proposal's rougher edges (e.g. recommends LangGraph over plain LangChain, standardizes on Claude models across agents rather than the proposal's OpenAI/open-source language) — treat it as the primary reference for implementation decisions, and the proposal PDF as historical context for *why* the project exists.

## docs/ — project documents

- **`docs/proposal/ReproBot.pdf`** — the original project proposal.
- **`docs/progress-reports/`** — dated progress reports, one subfolder per report (e.g. `first-progress-report/`, single-column and two-column LaTeX + PDF variants).
- **`docs/project-plan/ReproBot_Project_Plan.md`** — the implementation-ready plan (see above); the most detailed and current architectural reference in the repo.
- **`docs/literature-review/`** — three documents forming a deliberate three-layer stack; check which one to update/reference based on what's being asked:
  - **`PaperAgent_LiteratureReview.md`** — the cross-paper comparison document: landscape diagram (goal × execution-depth axes), a 3-wave chronological timeline, one "deep dive" subsection per paper (core idea, ASCII architecture diagram, feature-support table, explicit "Relevance to ReproBot" framing), a capability gap matrix, and a "how to cite ReproBot against each competitor" table. This is where cross-paper *positioning* lives. (Filename still says "PaperAgent" — that was the project's working title before it was renamed to ReproBot; the content itself already says "ReproBot" throughout.)
  - **`Paper_Summaries.md`** — one deep, template-consistent summary per paper (Problem/Motivation → Proposed Solution → Architecture → Method Details → Experimental Setup → Results-with-real-numbers → Strengths → Limitations → Takeaways for ReproBot). This is the reference to pull *exact reported numbers* from (success rates, replication scores, costs) — the other two documents intentionally stay qualitative/comparative.
  - **`Introduction_and_Literature_Review.md`** — the polished, prose/academic-register draft of the actual Introduction + Literature Review sections for the project report, with in-text citations and a numbered references list. This is the one meant to be copy-edited into the final report; the other two are internal working references it was synthesized from.
  - **`Evaluation_Metrics_Comparison.md`** — inventories the different evaluation vocabularies used by PaperBench/PaperCoder/AutoP2C/AutoReproduce, maps them onto a shared taxonomy, checks judge/grader trustworthiness against human gold labels, and proposes a combined rubric (Code Development / Execution / Result Match + a numeric tolerance gate) for ReproBot's own Critic. Reference this when designing the Critic's verdict logic specifically.

## papers/ — reference PDFs

The 9 papers analyzed in `docs/literature-review/` are stored as PDFs directly in `papers/`, renamed to their full titles (not arXiv IDs) for readability:
- `MLAgentBench - Evaluating Language Agents on Machine Learning Experimentation.pdf` (Huang et al., 2023) — foundational agentic-ML-experimentation benchmark; motivates ReproBot's Reader/Coder/Runner/Critic role separation.
- `PaperBench - Evaluating AI's Ability to Replicate AI Research.pdf` (Starace et al., 2025, OpenAI) — the gold-standard replication benchmark (20 ICML papers, 8,316-leaf-node rubric); primary target for evaluating ReproBot.
- `Paper2Code (PaperCoder) - Automating Code Generation from Scientific Papers in Machine Learning.pdf` (Seo et al., 2025) — closest architectural cousin (Planning→Analysis→Coding maps to Reader→Coder); generates code but never executes it.
- `AutoP2C - An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers.pdf` (Lin et al., 2025) — closest single system to ReproBot's full pipeline shape (multimodal parsing + execution + iterative debug loop); its debug loop stops at "runs and looks aligned," with no numeric-tolerance Critic verdict.
- `AutoReproduce - Automatic AI Experiment Reproduction with Paper Lineage.pdf` (Zhao et al., 2025) — most sophisticated direct competitor; the "paper lineage" (mine cited papers/repos for implicit detail) idea is an open extension point for ReproBot's Reader/Coder.
- `Agent Laboratory - Using LLM Agents as Research Assistants.pdf` (Schmidgall et al., 2025) — new-research-generation system; architectural inspiration (Literature Review→Experimentation→Report Writing), complementary not competing.
- `The AI Scientist - Towards Fully Automated Open-Ended Scientific Discovery.pdf` (Lu et al., 2024) and `The AI Scientist-v2 - ....pdf` (Yamada et al., 2025) — new-research-generation, complementary; v2's VLM figure-feedback loop and tree-search retry strategy are relevant design references.
- `MLR-Copilot - Autonomous Machine Learning Research based on Large Language Models Agents.pdf` (Li et al., 2024) — new-research-generation; its retrieval-augmented (HuggingFace prototype code/model/dataset) implementation pattern is a candidate technique for ReproBot's Coder.

When adding a new paper to this set: (1) download/rename the PDF to its full title in `papers/`, (2) add a deep-dive subsection to `docs/literature-review/PaperAgent_LiteratureReview.md` (and wire it into that file's TOC, capability gap matrix, and citation-differentiator table), (3) add a template-consistent entry to `docs/literature-review/Paper_Summaries.md` with real extracted numbers, (4) update `docs/literature-review/Introduction_and_Literature_Review.md`'s prose and references list if it changes the positioning narrative, (5) update `docs/literature-review/Evaluation_Metrics_Comparison.md` if the new paper introduces its own evaluation metric worth folding into the shared taxonomy.
