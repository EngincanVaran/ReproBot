# ReproBot

**ReproBot** is a multi-agent LLM pipeline that reads a machine learning paper (PDF), extracts its method and claimed results, generates and executes a training script to reproduce them, and produces a structured replication report comparing the reproduced metrics against the paper's stated numbers — retrying the implementation when a Critic agent detects a mismatch.

> inzva AI Projects #10

## Project status

**Six of the seven stages are built:** `ocr/` (PDF extraction), `reader/` (method, architecture, claims, hyperparameters and data pipeline, with a validation retry loop), `coder/` (training-script generation), `runner/` (Docker-sandboxed execution with live training check-ups), `orchestrator/` (shared state and the retry loop) and `critic/` (fidelity verdicts plus a model review of the script against the paper). Only the **Report Generator** is still design-only, per [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md).

**Five papers reproduce at full fidelity on a CPU**, across five model families, and the Critic judges every one of them the same way a person did by hand:

| Paper | Claimed | ReproBot |
|---|---|---|
| Tang 2013 — MLP with an L2-SVM loss | MNIST 0.87% test error | **0.82%** |
| Hsu/Chang/Lin — RBF SVM guide | svmguide1 96.9% | **96.925%** |
| Xiao 2017 — Fashion-MNIST random forest | 0.873 (mean of 5) | **0.8773** |
| Frosst & Hinton — soft decision tree | MNIST 94.45% | **95.11%** |
| Wijaya 2023 — dense regression net | Boston RMSE 3.02 | **3.33** (mean of 3 seeds) |

The chain is a loop, not a line: a failing script is triaged and regenerated, a run that stops learning is halted mid-training with evidence, and a result that misses its claim is either re-run with more seeds or sent back to the Coder with a cited diagnosis. Each stage lives in its own top-level folder, built one verified increment at a time — see [`CLAUDE.md`](CLAUDE.md) for the full convention and current state.

## Repository structure

```
ReproBot/
├── ocr/                        # PDF → Markdown extraction (4 backends; pdfplumber + Claude VLM verified)
├── reader/                     # Markdown → structured method/architecture/claims/hyperparameters/data JSON
├── coder/                      # reader JSON + paper Markdown → self-contained PyTorch/scikit-learn script
├── runner/                     # runs a generated script in a Docker sandbox; live check-ups; triage
├── orchestrator/               # shared-memory state + the retry loop over every stage
├── critic/                     # reproduced number vs the paper's claim: verdict + cited review
├── tests/                      # pytest replays of real runs (curves, verdicts, reviews)
├── dataset/                    # 8 CIFAR-10 papers, ReproBot's first replication targets
├── papers/                     # 9 agent-framework reference papers (literature review)
├── docs/
│   ├── proposal/                    # Original project proposal (ReproBot.pdf)
│   ├── progress-reports/            # Dated progress reports, one subfolder each
│   ├── project-plan/                # Detailed implementation-ready project plan
│   ├── literature-review/           # Cross-paper comparison, per-paper summaries,
│   │                                 # the CIFAR-10 shortlist, and the polished Intro/Lit-Review draft
│   ├── handouts/                    # One-page explainers per stage (HTML + A4 PDF)
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
    Reader (Claude VLM)         → method summary, claims, architecture, hyperparameters, data
    Coder (PyTorch/scikit-learn)→ self-contained training script, three gates before it runs
    Runner (Docker sandbox)     → executes it, watches the curve live, triages a failure
    Critic                      → arithmetic verdict vs the paper's claim, then a cited review
                                   of the script; extra seeds, or a diagnosis back to the Coder
    │
    ▼
Structured Markdown replication report (claim-by-claim comparison + gap analysis)   ← not built yet
```

The main evaluation set is **CIFAR-10 image-classification papers** (see [`docs/literature-review/CIFAR10_Candidate_Replication_Targets.md`](docs/literature-review/CIFAR10_Candidate_Replication_Targets.md) for the 8-paper shortlist). Those need a GPU — one full WRN-28-10 run measures at ~22 days on this project's CPU — so the fidelity results above come from the CPU-sized papers in `extra-papers/`, which since 2026-09-13 deliberately span several model families rather than images alone. See [`docs/project-plan/ReproBot_Project_Plan.md`](docs/project-plan/ReproBot_Project_Plan.md) for the full feasibility assessment, architecture deep dive, timeline, and cost budget.

## Setup

```bash
uv sync --extra pdfplumber --extra vlm --extra reader --extra coder --extra runner --extra orchestrator --extra critic --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
uv run pre-commit install
```

Running the whole loop over one paper, Critic included:

```bash
uv run python -m orchestrator.pipeline --input "reader/output/<paper>.json" --max-stage full
uv run --extra orchestrator pytest tests          # replay tests, no Docker or API key needed
```

See `ocr/README.md`, `reader/README.md`, `coder/README.md`, `runner/README.md`, `orchestrator/README.md` and `critic/README.md` for how to run each stage on its own.
