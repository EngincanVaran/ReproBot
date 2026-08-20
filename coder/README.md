# coder/ — training-script generation

Turns one paper's structured Reader output (`reader/output/<paper>.json`) plus
that paper's OCR Markdown (`ocr/output/vlm/<paper>.md`) into a **self-contained,
runnable HuggingFace `Trainer` training script** that targets one specific
numeric claim from the paper. One Claude tool-use call, two deterministic
zero-cost safety gates, one script on disk.

**This stage never executes the script and never imports torch.** It writes
code; the future `runner/` stage runs it in a Docker sandbox.

## Class architecture

```
coder/base.py
┌──────────────────────────────────────────────┐
│  CodeWriter[ResultT]  (ABC)                     │
│    name: ClassVar[str]                           │
│    write(reader_output, paper_markdown,           │
│          client, feedback: str | None) -> ResultT  │
└──────────────────────────────────────────────┘
                       ▲
                       │
        ┌──────────────┴──────────────────┐
        │ TrainingScriptWriter               │
        │ name="training_script"              │
        │ __init__(target_claim_id=None,       │
        │          max_attempts=3)              │
        │ coder/script_writer.py                 │
        └────────────────────────────────┘

coder/pipeline.py
┌──────────────────────────────────────────────┐
│  CoderPipeline(writer)                          │
│    run(reader_json_path, markdown_path,          │
│        output_dir, client) -> CoderOutput          │
│  Owns input resolution, the two gates, and         │
│  all file I/O. A second CodeWriter (an eval          │
│  script, a Dockerfile) plugs in without              │
│  touching the gates or the CLI.                        │
└──────────────────────────────────────────────┘
```

Same shape as `reader/`, and for the same reason: the pipeline should treat
every generation stage uniformly — call `.write()`, key by `.name`, and on a
retry hand back `feedback`. `script_writer.py` owns a prompt, a tool schema and
parsing; it does no file I/O at all.

## Why TWO inputs

This is the central design decision of the stage, not redundancy.

| Input | Role |
|---|---|
| `reader/output/<paper>.json` | **Authoritative** for claims, hyperparameters, data_pipeline. Structured, source-grounded (every entry cites a table/section and transcribed page), already cross-checked by `reader/validator.py`'s retry loop. |
| `ocr/output/vlm/<paper>.md` | **Fills gaps**, architecture above all. `reader/` has no `architecture_notes` stage yet, so the paper's own layer table, block structure, channel widths and BN-ReLU-conv ordering exist *only* here. |

The prompt states the precedence explicitly: the reader output wins wherever it
has data, the Markdown may **extend** it but never **override** it, and a
genuine disagreement between the two gets recorded in `assumptions`. Without the
Markdown the model would fall back on its pretrained recollection of a famous
architecture; with it, WRN-28-10 is built from the paper's actual Table 1
(`n = 6N+4`, widths `16, 16k, 32k, 64k`, downsampling in groups 3 and 4).

Both fit comfortably in one context window — the Wide Residual Networks run
sends ~33k input tokens total.

## Target-claim selection

`TrainingScriptWriter(target_claim_id=...)`, exposed as `--claim-id`:

- **Given** — the prompt pins that exact claim and refuses to substitute a
  better-looking one.
- **Omitted** — Claude picks the paper's headline claim itself and must justify
  the pick in `claim_selection_reasoning`, including what it rejected.

Either way the choice and its full reasoning are logged at `INFO`. A wrong pick
silently reproduces the wrong number, and until the Critic exists the console is
the only place that is visible.

### Matching the claim's regime — the worst failure mode

Papers report several regimes side by side, and the reader output tags them via
`model_variant`. Wide Residual Networks alone carries three axes of them:
CIFAR vs. SVHN learning-rate schedules, mean/std normalization vs. ZCA
whitening, dropout vs. no dropout. Taking the CIFAR claim but the SVHN schedule
would run fine and quietly reproduce a different number.

The prompt therefore states this as a general rule — *"papers often report
several regimes; match the one your claim came from"* — and requires the model
to work out the regime from the claim's `model_variant` and `source` before
picking hyperparameters. On the real run this worked: targeting `c34`
(`WRN 28-10, no dropout, mean/std normalization`) produced the CIFAR schedule
(`0.1`, ×0.2 at epochs 60/120/160, 200 epochs), not the SVHN one
(`0.01`, ×0.1 at 80/120, 160 epochs), plus mean/std normalization rather than
ZCA and no dropout.

## The two safety gates

Both are pure Python in `pipeline.py`, cost nothing, and catch failures that
would otherwise surface only inside the Runner's Docker sandbox.

```
writer.write()  ──►  script_content
                          │
                          ▼
              ┌───────────────────────┐
              │ GATE 1: ast.parse()     │
              └───────────┬───────────────┘
                 ▼ ok          ▼ SyntaxError
                 │           write train.py.invalid
                 │           + coder_output.failed.json
                 │           log msg/lineno/offending line
                 │           raise ScriptSyntaxError
                 ▼
              ┌───────────────────────┐
              │ GATE 2: required flags  │  warns, does not fail
              └───────────┬───────────────┘
                          ▼
              write train.py + coder_output.json
```

**Gate 1 — syntax.** Broken Python is never silently written as a `.py`. The
offending source is still persisted (as `train.py.invalid`) for debugging, the
`SyntaxError`'s message, line number and offending line are logged, and the run
is marked failed. Because the bookkeeping goes to `coder_output.failed.json`
rather than `coder_output.json`, a failed paper is **not** skipped on the next
run.

**Gate 2 — CLI flags.** Every flag in `REQUIRED_CLI_FLAGS` must literally appear
in the script text. Missing flags are warned about by name and recorded in
`missing_cli_flags`. The model's own `cli_flags_included` self-report is
cross-checked in both directions — a model can claim a flag it never wrote, and
can write one it forgot to report.

## The metrics.json contract — the Runner/Critic interface

**This is the interface the future `runner/` and `critic/` stages consume.** The
generated script writes this JSON to `--metrics-output` *and* prints the
identical object as its final single line of stdout (so the Runner can capture
it without a volume mount).

```json
{
  "claim_id": "c34",
  "metric": "test error",
  "unit": "%",
  "value": 4.32,
  "train_loss": 0.02,
  "train_accuracy": 98.1,
  "eval_loss": 0.71,
  "eval_accuracy": 61.4,
  "epochs_completed": 5,
  "num_train_samples": 512,
  "num_eval_samples": 256,
  "wall_clock_seconds": 142.3
}
```

`claim_id`, `metric` and `unit` are copied **verbatim** from the targeted claim —
never normalized, renamed or unit-converted. That is the whole point: the Critic
diffs `value` against the claim's `reported_value` with **no unit conversion at
all**, because `"test error"` / `"%"` on both sides means the numbers are
directly comparable. `eval_accuracy` is kept alongside as the raw measurement
`value` was derived from.

### The required CLI flags

The generated script must always define these, with defaults equal to the
paper's real values. The Runner uses them to force fast capped smoke runs before
committing to a full one.

| Flag | Default |
|---|---|
| `--epochs` | the paper's real epoch count for the regime |
| `--max-train-samples` | `None` (full training set) |
| `--max-eval-samples` | `None` (full evaluation set) |
| `--batch-size` | the paper's real batch size |
| `--lr` | the paper's real initial learning rate |
| `--output-dir` | a local directory |
| `--metrics-output` | a `metrics.json` path |
| `--seed` | a fixed integer, seeded through `torch`/`numpy`/`random` |

## `base.py` — `CodeWriter[ResultT]`

PEP 695 generic ABC mirroring `reader/base.py`'s `Extractor`. `write()` takes
both inputs plus an optional `feedback: str | None`.

**`feedback` is unused today and that is deliberate.** Nothing passes it —
`pipeline.py` is single-shot. It exists because the planned Coder↔Runner loop
will fold a Runner error trace (a traceback, or a metrics mismatch) back into
the prompt exactly the way `reader/pipeline.py` already routes a validation flag
into `Extractor.extract(feedback=...)`. `TrainingScriptWriter` honours it now, so
that increment is a pipeline change rather than a rewrite of every writer.

## `script_writer.py` — `TrainingScriptWriter`

One Claude call (`claude-sonnet-5`, `max_tokens=16384`), forced
`tool_choice` on `write_training_script`. The tool returns `claim_targeted`,
`claim_selection_reasoning`, `architecture_used`, `dataset_used`,
`hyperparameters_used`, `assumptions`, `cli_flags_included` and `script_content`.

`max_tokens` is **16384, double `reader/`'s 8192** — a full training script is
long-form output (the real WRN run emits ~9k output tokens), and this repo has
already lost runs to a silent `stop_reason: max_tokens` truncation twice
(`ocr/vlm_extract.py`, then `reader/claims.py`). `stop_reason` is now checked
explicitly and logged as an `ERROR` when it is `max_tokens`, so that failure can
never be silent again.

What the prompt requires of the generated script:

1. **A hand-rolled `nn.Module`**, never `AutoModelForImageClassification` — HF's
   built-in ResNets assume ImageNet's 224×224 stem (7×7 stride-2 conv +
   maxpool), which destroys CIFAR's 32×32 inputs before the first block.
2. **A thin `Trainer` adapter**, not a `PreTrainedModel` subclass: a plain
   `nn.Module` whose `forward(pixel_values, labels=None)` returns a dict with
   `logits`, plus `loss` when labels are passed. That dict is the entire
   `Trainer` contract — no config class, no `from_pretrained`.
3. **`torchvision.datasets.CIFAR10(download=True)`**, not
   `datasets.load_dataset` — the paper's augmentation (4px reflection pad →
   random 32×32 crop → horizontal flip) maps one-to-one onto
   `torchvision.transforms`, and it keeps pyarrow out of the Runner's Docker
   image.
4. **Never invent an unstated detail silently.** Unlike `reader/data_pipeline.py`,
   which may record "not stated" and stop, generated code has to actually run —
   so the rule here is *choose the canonical default, always disclose it in
   `assumptions`*.
5. **Guard the zero-batch edge case**: `--max-train-samples` below the batch size
   with `drop_last=True` silently yields zero batches and a meaningless
   "successful" run. The script must clamp (logging it) or fail loudly.
6. **stdlib `logging`, never `loguru`** — the script runs standalone in a Docker
   container and must not depend on this repo's tooling.

## Logging

Detailed by design, via `loguru` (not `print`) — input sizes, token usage,
`stop_reason`, the exact fields the tool returned, the target claim **and its
full reasoning**, every hyperparameter encoded, every assumption recorded, both
gates' verdicts, and every file written. A wrong claim pick or a mismatched
regime should be obvious from console output alone.

The generated script has its own separate stdlib-`logging` instrumentation
(configuration, device, sample counts, LR milestones, per-epoch metrics).

## Setup

```bash
uv sync --extra coder --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

```bash
# a single paper, Claude picks the headline claim
uv run python -m coder.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json"

# pin the claim
uv run python -m coder.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json" --claim-id c44

# every reader output, skipping papers already generated
uv run python -m coder.pipeline --input reader/output

# custom locations
uv run python -m coder.pipeline --input reader/output \
    --paper-markdown-dir ocr/output/vlm --output somewhere/else
```

Output per paper (gitignored):

```
coder/output/<paper>/
├── train.py             # the generated script (or train.py.invalid on gate 1 failure)
└── coder_output.json    # bookkeeping (or coder_output.failed.json)
```

`coder_output.json` deliberately does **not** duplicate `script_content` — the
script lives at `script_path` and the JSON stays readable. `script_version: 1`
and `diff_from_previous: null` are present but inert: the Coder↔Runner retry
increment will rewrite a script in place and needs somewhere to record which
revision it is and what changed, and emitting the fields now keeps the schema
stable rather than forcing consumers to handle two shapes later.

## Dependency choice

`coder` is its own `pyproject.toml` extra (`anthropic` + `python-dotenv`),
identical in shape to `reader`'s and, importantly, **torch-free**. This stage
only *writes* a training script; it never runs one. That is what lets it stay in
the managed `uv.lock` on the Intel-macOS dev machine — the generated script's
`torch`/`torchvision`/`transformers` dependencies belong to the future
`runner/` Docker image, not to this project (see the platform trap in
`CLAUDE.md` and `ocr/README.md`).

`coder/output/` is excluded from mypy in `pyproject.toml` for the same reason:
the generated scripts are standalone untyped artifacts targeting libraries that
are not installed here, and their gate is `ast.parse` plus the future Runner,
not `--strict`.

## Known issues

**The model intermittently leaks tool-field delimiters into another field
(worked around, not fixed).** In real runs Claude sometimes serializes its tool
fields as literal text *inside* an earlier field instead of as separate tool
inputs, in two observed shapes:

```
...model_variant='WRN 28-10'.</architecture_used>
<parameter name="dataset_used">CIFAR-10 (torchvision...)</dataset_used>
```

```
...is the headline result.</claim_selection_reasoning>
<architecture_used>Wide Residual Network WRN-28-10...</architecture_used>
<script_content>import torch ...
```

This produced a payload missing two schema-required fields on one run, and
missing `script_content` entirely on two others. Two mitigations are in place:

- `_recover_leaked_fields()` — deterministic and free. It finds a `</field>`
  closer inside a string value, truncates that field there, and re-homes each
  trailing block onto the key it names. Matching is restricted to the eight
  known `TOOL_FIELDS`, so a `<` or `>` operator inside `script_content` cannot
  trigger a false split. Verified against both shapes, including a
  `script_content` containing `if a < b and c > d:`.
- `max_attempts=3` in `TrainingScriptWriter` — covers the shape where the
  remaining fields were genuinely never emitted and nothing is recoverable. It
  is **not** a quality retry: a structurally valid but bad script is returned
  as-is for the gates, and later the Critic, to judge.

On the verified run, attempts 1 and 2 hit the unrecoverable shape and attempt 3
succeeded with all eight fields. Frequency is high enough to notice (roughly
half of observed calls leak in some form) and is worth revisiting — the recovery
now handles both shapes, so a future run that leaks shape B should succeed on
attempt 1 rather than burning calls.

**The gates check structure, not semantics — and a real bug slipped through.**
`ast.parse` proves the script *parses*, not that it *runs*. The verified WRN run
produced a `_NoOpScheduler` helper class whose `get_last_lr()` returns
`[g["lr"] for g in self.optimizer.param_groups]` — but `_NoOpScheduler` never
sets `self.optimizer`, so that call raises `AttributeError` when `Trainer` logs
the learning rate. This is exactly the class of defect the Runner→Critic→Coder
loop is meant to catch, and it is the clearest evidence available that the loop
is genuinely needed rather than nice-to-have. Adding a third semantic gate here
(an undefined-attribute check) was deliberately **not** done — it needs a real
static-analysis dependency and would duplicate what actually executing the
script tells you for free.

## Status

Verified end-to-end against **Wide Residual Networks**, with a real API call:

- Targeted claim `c34` — *test error 4.00% on CIFAR-10, WRN-28-10, no dropout,
  mean/std normalization* (Table 5) — chosen freely by Claude and correctly
  identified as the paper's headline result over the ZCA (Table 4), dropout
  (Table 6) and ImageNet/SVHN/COCO alternatives.
- 434-line script, `stop_reason=tool_use` (no truncation), both gates passed,
  all 8 required CLI flags present, 10 hyperparameters encoded, 8 assumptions
  recorded.
- Correct regime: the CIFAR LR schedule, mean/std normalization, no dropout —
  not the SVHN schedule or the ZCA regime.
- Real `WideResNet`: `n = (depth - 4) // 6` blocks per group, widths
  `16 / 16k / 32k / 64k`, pre-activation BN-ReLU-conv basic blocks, 1×1
  projection shortcuts on the downsampling blocks, `depth=28, widen_factor=10`.

**The generated script has never been executed** — no torch on the dev machine
(Python 3.13 + Intel macOS, per `CLAUDE.md`'s platform trap). It is
syntax-gated only. Actually running it is `runner/`'s job.

Not yet built: the Coder↔Runner retry loop (`feedback` plumbing,
`script_version`/`diff_from_previous` bookkeeping), and any second `CodeWriter`
(eval script, Dockerfile, requirements file).
