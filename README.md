# ReproBot

**ReproBot** is a multi-agent LLM pipeline that reads a machine learning paper (PDF), extracts its method and claimed results, generates and executes a training script to reproduce them, and produces a structured replication report comparing the reproduced metrics against the paper's stated numbers — retrying the implementation when a Critic agent detects a mismatch.

> inzva AI Projects #10

## Project status

Implementation has started: **`ocr/` (PDF extraction), `reader/` (claims, hyperparameters, and data-pipeline extraction with a validation retry loop), `coder/` (training-script generation), `runner/` (Docker-sandboxed execution), and `orchestrator/` (the Coder↔Runner retry loop) are built.** `reader/` is now complete against its spec — method summary, claims, hyperparameters, architecture notes, and data pipeline. Still design-only, per [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md): the **Critic** (nothing yet compares a reproduced number against the paper's claim) and the Report Generator. The chain is proven end to end and is now a loop — a failing script is triaged, fed back to the Coder, and regenerated. Each pipeline stage lives in its own top-level folder built one small, verified increment at a time — see [`CLAUDE.md`](CLAUDE.md) for the full convention and current state.

## Repository structure

```
ReproBot/
├── ocr/                        # PDF → Markdown extraction (4 backends; pdfplumber + Claude VLM verified)
├── reader/                     # Markdown → structured method/architecture/claims/hyperparameters/data JSON
├── coder/                      # reader JSON + paper Markdown → HuggingFace Trainer training script
├── runner/                     # executes a generated script in a Docker sandbox, triages the outcome
├── orchestrator/               # shared-memory state + the Coder↔Runner retry loop
├── dataset/                    # 8 CIFAR-10 papers, ReproBot's first replication targets
├── papers/                     # 9 agent-framework reference papers (literature review)
├── docs/
│   ├── proposal/                    # Original project proposal (ReproBot.pdf)
│   ├── progress-reports/            # Dated progress reports, one subfolder each
│   ├── project-plan/                # Detailed implementation-ready project plan
│   ├── literature-review/           # Cross-paper comparison, per-paper summaries,
│   │                                 # the CIFAR-10 shortlist, and the polished Intro/Lit-Review draft
│   ├── notes/                       # Narrower working notes (e.g. Reader-agent precedents)
│   └── agent-log.md                 # Record of every delegated subagent task and result
├── TODO.md                     # status, open bugs, next steps — start here
└── pyproject.toml               # uv-managed; each stage installs via its own --extra
```

See [`CLAUDE.md`](CLAUDE.md) for a detailed map of what's in each document/stage and how they relate, and [`TODO.md`](TODO.md) for current status, open bugs, and what's next.

## How this gets built

Claude acts as orchestrator: it talks directly to the project owner, and delegates scoped tasks (research, code review, implementation) to specialized subagents — Research Agent, Review Agent, Coding Agent, Validator Agent — logging what each was asked and what it returned to [`docs/agent-log.md`](docs/agent-log.md). See `CLAUDE.md`'s "Orchestrator + agent delegation" section for the full convention.

## The pipeline

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

Initial evaluation scope is deliberately narrow: **CIFAR-10 image-classification papers** (see [`docs/literature-review/CIFAR10_Candidate_Replication_Targets.md`](docs/literature-review/CIFAR10_Candidate_Replication_Targets.md) for the 8-paper shortlist, ordered by publish date), trading topic breadth for a stricter numeric tolerance in the Critic's pass/fail logic. See [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md) for the full feasibility assessment, architecture deep dive, timeline, and cost budget.

## Setup

```bash
uv sync --extra pdfplumber --extra vlm --extra reader --extra coder --extra runner --extra orchestrator --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
uv run pre-commit install
```

See `ocr/README.md`, `reader/README.md`, `coder/README.md`, `runner/README.md`, and `orchestrator/README.md` for how to actually run each stage.
