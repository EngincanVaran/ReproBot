<div align="center">

# ReproBot

**A multi-agent pipeline that reads a machine learning paper and tries to reproduce its numbers — then tells you honestly whether it did.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![mypy strict](https://img.shields.io/badge/mypy-strict-2A6DB2)](https://mypy-lang.org/)
[![tests](https://img.shields.io/badge/tests-135%20passing-3FB950)](tests/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*inzva AI Projects #10*

</div>

---

A paper claims 96.9% accuracy. Does it? Answering that by hand means reading the method,
finding the hyperparameters it did and did not state, writing the training script, debugging
it, running it, and comparing the result against a number the paper rounded. ReproBot does
that end to end, in a Docker sandbox, and — the part that matters — **it knows the difference
between a result that matches and a result it cannot tell apart from noise.**

```
PDF paper
    │
    ▼  Orchestrator (shared memory, bounded retry loops)
    │
    ├─ Reader   (Claude VLM)          → method, architecture, claims, hyperparameters, data
    ├─ Coder    (PyTorch / sklearn)   → one self-contained script, three gates before it runs
    ├─ Runner   (Docker sandbox)      → executes it, watches the curve live, triages failures
    ├─ Critic   (arithmetic + review) → verdict vs the claim, then a cited code review
    └─ Report                         → claim table, learning curve, gap analysis, timeline
    │
    ▼
Markdown replication report, claim by claim
```

## What it found

**Across every repeated run, each paper's variation traced to exactly one detail the paper
never stated — and we could name it every time.** That is the project's actual result: run-to-run
disagreement is a property of the paper, not a defect of the system.

| Paper | What it never states | What that does |
|---|---|---|
| Fashion-MNIST | *nothing* — the protocol is complete | three independently written scripts produced **bit-identical** numbers |
| SVM guide | whether the 5 CV folds are shuffled | picks a different (C, γ), moving test accuracy 96.625 ↔ 96.925 |
| Wijaya 2023 | how the 405/101 split is drawn | RMSE moves continuously, 2.62 – 4.00 |
| Isolation Forest | whether its AUC is one run or an average | changes what the tolerance is even measured from |

## Results

**Six papers reproduced on a laptop CPU**, six model families, every verdict decided by
arithmetic that no model can overrule:

| Paper | Claimed | ReproBot | Verdict | Tolerance |
|---|---|---|---|---|
| Tang 2013 — MLP with an L2-SVM loss | MNIST 0.87% test error | **0.82%** † | `pass` | ±0.1857 |
| Hsu/Chang/Lin — RBF SVM guide | svmguide1 96.9% | **96.625 – 96.95%** (8 runs) | `pass 8/8` | ±0.5481 |
| Xiao 2017 — Fashion-MNIST random forest | 0.873, mean of 5 | **0.87734** ↑ | `pass` | ±0.00175 |
| Frosst & Hinton — soft decision tree | MNIST 94.45% | **95.11%** ↑ † | `pass` | ±0.4579 |
| Wijaya 2023 — dense regression net | Boston RMSE 3.02 | **3.03629**, mean of 3 seeds | `pass` | ±0.2044 |
| Liu/Ting/Zhou 2008 — isolation forest | Pima AUC 0.67 | **0.67873**, mean of 10 | `pass 3/3` | ±0.011 |

↑ better than the paper's own number.  † one documented human change (Tang's unstated SVM `C`;
the soft tree's misprinted Eq. 3) — the other four ran with no human input.

**And it reports failure.** A further eleven papers ran on a rented GPU box: four passed, **four
failed** (TCN, fastText, Tang's CIFAR-10 claim, XGBoost), two produced no judgeable number at all
— Network In Network was halted four times by the live check-ups, and Breiman's claims never
extracted — and one was still training when the batch was logged. A replication system that never
says *no* is not measuring anything.

## How a verdict is made

```
tolerance = max( 2 × uncertainty , reporting precision )
```

Allow twice the measured run-to-run noise, but never demand a closer match than the paper's own
rounding. Uncertainty comes from the measured spread when repeated runs exist, otherwise the
binomial noise of the test set, otherwise nothing — and *nothing* is a verdict of its own:

| Verdict | Meaning |
|---|---|
| `pass` | within tolerance, or better than the claim beyond it |
| `fail` | worse beyond a tolerance backed by a real noise estimate → one guided retry |
| `inconclusive` | worse, but with no way to tell the gap from one unlucky seed → run more seeds |
| `not_evaluated` | nothing comparable: crashed, halted, or stopped before the full stage |

Wijaya is the clearest case for the loop: the same paper, same claim, same machine —
**3 of 3 runs reproduce with the loop on, 1 of 5 with it off.**

## Repository

```
ReproBot/
├── ocr/            PDF → Markdown (4 backends; falls back to the text layer if a page is refused)
├── reader/         Markdown → structured claims, architecture, hyperparameters, data pipeline
├── coder/          reader JSON + paper → one self-contained PyTorch/scikit-learn script
├── runner/         Docker-sandboxed execution, live training check-ups, failure triage
├── orchestrator/   shared-memory state and the bounded retry loops
├── critic/         arithmetic verdict + a cited model review of the script against the paper
├── report/         the replication report: claim table, SVG curve, gap, attempt timeline
├── viewer/         Streamlit dashboard over every stage, with a one-click full run
├── tests/          135 replay tests built from real recorded runs
├── dataset/        8 CIFAR-10 papers (the GPU-scale evaluation set)
├── extra-papers/   15 CPU-sized papers spanning model families
└── docs/           proposal, project plan, literature review, progress reports,
                    the final report, handouts, and the experiment notes below
```

**The evidence** lives in [`docs/notes/`](docs/notes/): [`end-to-end/`](docs/notes/end-to-end/)
(three papers × three runs, plus a paper the pipeline had never seen),
[`ablation/`](docs/notes/ablation/) (does the retry loop earn its cost?),
[`consistency/`](docs/notes/consistency/) (five independent generations per paper), and
[`tang-2013-ablation/`](docs/notes/tang-2013-ablation/) (why the first full run collapsed).
Each carries its raw per-run JSON next to the write-up.

Start with [`TODO.md`](TODO.md) for current status and every open bug; [`CLAUDE.md`](CLAUDE.md)
is the architectural map.

## Setup

```bash
uv sync --extra pdfplumber --extra vlm --extra reader --extra coder --extra runner \
        --extra orchestrator --extra critic --group dev
cp .env.example .env     # then fill in ANTHROPIC_API_KEY
uv run pre-commit install
```

Run the whole loop over one paper, Critic included:

```bash
uv run --extra orchestrator python -m orchestrator.pipeline \
    --input "reader/output/<paper>.json" --max-stage full
```

Then render its report (no API key, no Docker, no plotting dependency):

```bash
uv run python -m report.pipeline --input orchestrator/output
```

Tests and coverage — neither needs Docker or an API key:

```bash
uv run --extra orchestrator pytest tests
uv run --extra orchestrator pytest tests --cov --cov-report=term
```

Coverage is **47%** overall with branch coverage on. The shape matters more than the total: the
logic that decides anything is 90–100% (`critic/judge.py` 93%, `runner/checkups.py` 93%,
`report/render.py` 90%, `report/summary.py` 96%), while the CLI entry points and the `reader/`
extractors sit near zero because each is a single API call or a Docker invocation.

Each stage has its own README — [`ocr/`](ocr/README.md), [`reader/`](reader/README.md),
[`coder/`](coder/README.md), [`runner/`](runner/README.md),
[`orchestrator/`](orchestrator/README.md), [`critic/`](critic/README.md),
[`report/`](report/README.md), [`viewer/`](viewer/README.md) — for running it alone.

## Team

Aleyna Kütük · Engincan Varan · Gül Korkut · Göktuğ Mert Özdoğan

## License

[MIT](LICENSE)
