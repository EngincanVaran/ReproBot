# reader/ — claims extraction

The Reader agent's first real slice: turn one paper's already-OCR'd Markdown
into a structured list of **claims** — the paper's own reported
metric/dataset/value results, source-grounded to a table and page, ready for
the Critic to compare against reproduced numbers later. This is deliberately
narrow: no method summary, hyperparameters, or architecture notes yet (see
`docs/agent-log.md`, "scope the first `reader/` increment", for why claims
extraction was picked as the first slice to validate in isolation).

## Input

`ocr/output/vlm/<paper>.md` — the Claude-VLM backend's output, **not**
pdfplumber's. pdfplumber's output has no inter-word spacing in body text and
renders every figure/table region unreliably; the VLM backend produces clean
Markdown with faithfully transcribed tables. See `ocr/README.md` for how
that Markdown is produced.

## What it does

`extract_claims.py` sends a paper's full Markdown to Claude in one call and
uses **tool-use** (a JSON-schema tool definition, with `tool_choice` forced
to it) to get back validated structured output — not free-form text that
then needs parsing. The model is prompted, AutoP2C-style, to:

1. Locate every results table and note which table/page it examined.
2. Classify each row in those tables as either the paper's **own proposed
   method** or a **baseline/prior-work** row (baselines cite another paper,
   e.g. `[14]`; own-method rows don't and are usually bolded).
3. Emit one claim per distinct own-method row — a paper reporting multiple
   headline configurations for the same metric+dataset (e.g. with/without
   data augmentation) gets one claim per configuration, distinguished by
   `model_variant`.
4. Drop baseline rows entirely, even when they sit in the same table as a
   kept claim.

Each claim has: `claim_id`, `metric`, `dataset`, `reported_value` (float),
`unit`, `source` (e.g. `"Table 1, transcribed page 5"`), and an optional
`model_variant`. This matches the project plan's shared-memory `Claim` shape
(`docs/project-plan/ReproBot_Project_Plan.md`, §1.2) plus the `model_variant`
addition.

## Logging

Detailed by design, via `loguru` (not `print`) — for each paper this logs
every results table examined, how many candidate rows were seen before
baseline-filtering, how many claims were kept, and each kept claim as it's
extracted (not just a final count), so a bad extraction is obvious from the
console output alone.

## Setup

```bash
uv sync --extra reader --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

```bash
# a single paper
uv run python -m reader.extract_claims --input "ocr/output/vlm/2013-12 - Network In Network.md"

# every Markdown file in a directory, skipping papers already extracted
uv run python -m reader.extract_claims --input ocr/output/vlm

# custom output location
uv run python -m reader.extract_claims --input ocr/output/vlm --output somewhere/else
```

Output goes to `reader/output/<paper>.json` by default (gitignored —
regenerate anytime): the claim list plus the `tables_examined` and
`candidates_considered` bookkeeping fields, so the log trail is also
recoverable from disk after the run.

## Dependency choice

`reader` is its own `pyproject.toml` extra (`anthropic` + `python-dotenv`),
not a reuse of `ocr`'s `vlm` extra — this stage never touches a PDF or
renders a page image (`pypdfium2`, which `vlm` needs), it only reads Markdown
`ocr/vlm_extract.py` already produced. Keeping the extras separate means
`uv sync --extra reader` doesn't pull in a dependency this stage never
imports, in line with the repo's one-folder-per-stage convention of keeping
each stage independently installable.

## Status

Verified against `ocr/output/vlm/2013-12 - Network In Network.md`: correctly extracts
both headline CIFAR-10 claims (**NIN + Dropout, 10.41%** and
**NIN + Dropout + Data Augmentation, 8.81%**, both `Table 1, transcribed page
5`) and correctly excludes every baseline row from the same table (e.g.
`Stochastic Pooling [11]`, `15.13%`).
