# ReproBot

**ReproBot** is a multi-agent LLM pipeline that reads a machine learning paper (PDF), extracts its method and claimed results, generates and executes a training script to reproduce them, and produces a structured replication report comparing the reproduced metrics against the paper's stated numbers — retrying the implementation when a Critic agent detects a mismatch.

> inzva AI Projects #10

## Project status

This repository is currently in the **research / proposal phase** — there is no implementation yet. What exists today is the project proposal, progress reports, and a literature review built from 9 related papers. Implementation is planned to follow the pipeline described in [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md).

## Repository structure

```
ReproBot/
├── docs/
│   ├── proposal/              # Original project proposal (ReproBot.pdf)
│   ├── progress-reports/      # Dated progress reports, one subfolder each
│   │   └── first-progress-report/
│   ├── project-plan/          # Detailed implementation-ready project plan
│   └── literature-review/     # Cross-paper comparison, per-paper summaries,
│                               # and the polished Introduction/Lit-Review draft
└── papers/                    # The 9 reference papers (PDFs), full titles as filenames
```

See [`CLAUDE.md`](CLAUDE.md) for a more detailed map of what's in each document and how they relate to each other.

## The pipeline (planned)

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

Initial evaluation scope is deliberately narrow: **20 image-classification papers**, trading topic breadth for a stricter numeric tolerance in the Critic's pass/fail logic. See [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md) for the full feasibility assessment, architecture deep dive, timeline, and cost budget.
