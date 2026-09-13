# coder/ — training-script generation

Turns one paper's structured Reader output (`reader/output/<paper>.json`) plus
that paper's OCR Markdown (`ocr/output/vlm/<paper>.md`) into a **self-contained,
runnable PyTorch training script — a plain, explicit training loop** — that
targets one specific numeric claim from the paper. One Claude tool-use call, two
deterministic zero-cost safety gates, one script on disk.

The paper can be about **any supervised task**: image classification,
classification of other inputs, tabular regression. The Coder infers which from
the claim itself and records it as `task_type`.

**This stage never executes the script and never imports torch.** It writes
code; `runner/` runs it in a Docker sandbox.

## What the generated script is — and why

This stage was first built for image classification only: it generated
HuggingFace `Trainer` scripts over `torchvision` datasets. Generalizing to other
paper types (a custom-loss MNIST classifier, a Keras tabular regression) changed
four things, each of which closes a path to a replication that runs fine and
quietly reproduces something else.

### 1. A plain PyTorch loop, not HuggingFace `Trainer`

The prompt forbids `Trainer`, `accelerate`, Lightning, Keras and any other
training wrapper. The model writes the loop itself — `DataLoader`s, the
optimizer with every argument passed by name, the scheduler stepped at the
granularity the paper describes, `model.train()` / `model.eval()` +
`torch.no_grad()` — and adds nothing the paper does not state (no gradient
clipping, warmup, label smoothing, mixed precision or early stopping unless the
reader output says so).

Two reasons:

- **`Trainer`'s model contract fits regression and custom losses badly.** It is
  built around a dict of `pixel_values`/`labels` in and `logits`/`loss` out.
  Every old script needed a hand-written adapter to satisfy it, and a regression
  head or a paper-defined objective (an L2-SVM loss in place of softmax) has to
  be contorted further still to fit.
- **`Trainer` carries its own defaults, which no paper states.** Optimizer
  choice and epsilon, LR schedule, warmup, weight-decay handling, logging and
  evaluation cadence — each a number the reproduction used that nobody chose,
  and none of them visible in the generated code. That is a silent divergence
  risk. A plain loop has no hidden defaults: every setting in it is one the model
  wrote down and can disclose.

### 2. Always PyTorch, whatever framework the paper used — with the differences disclosed

The Runner's image deliberately carries **no second framework**, so a paper
built in Keras/TensorFlow, JAX, Theano, Caffe or MATLAB is still reimplemented
in PyTorch. That translation is not neutral: "we used Adam" in a Keras paper
meant Keras's Adam. The prompt states the rule generally, with examples:

| Setting | Keras / TensorFlow default | PyTorch default |
|---|---|---|
| `Adam` epsilon | `1e-7` | `1e-8` |
| `Dense` / `nn.Linear` init | Glorot-uniform kernel, zero bias | Kaiming-uniform weight, non-zero uniform bias |
| `BatchNormalization` | `momentum=0.99`, `epsilon=1e-3` | `momentum=0.1` (the complement — Keras-equivalent is `0.01`), `eps=1e-5` |
| `Model.fit` batch size | `32` when unspecified | no default; you always pass one |

The rule: work out which framework defaults the paper's unstated settings
inherited; **where matching them is cheap and exact, do it explicitly in code**
(`eps=1e-7`, `nn.init.xavier_uniform_` + `nn.init.zeros_`); and **record every
difference as its own `assumptions` entry, matched or not**. A paper that names
no framework gets explicit, standard choices, disclosed as usual — never a
guessed framework.

### 3. Task type is inferred, not extracted

There is no Reader field for it. The Coder infers `task_type` from the targeted
claim's `metric` and `dataset`, `method_summary` and `data_pipeline`, and returns
it as a tool field recorded in `coder_output.json`. Expected values are
`"classification"` and `"regression"`; the schema does **not** restrict it, and
an unexpected label is accepted but logged as a warning (`KNOWN_TASK_TYPES` in
`script_writer.py`). The prompt's rule is to decide by what the model
*predicts*, not how it is trained — a paper that replaces softmax with an SVM
loss on MNIST is still `classification`. The task type drives the output layer,
the default loss, how the per-split metric is computed, and `higher_is_better`.

### 4. Data from a stable, documented source, cached under `--data-dir`

- **Image datasets** `torchvision.datasets` provides: loaded with it, with
  augmentation via `torchvision.transforms`, train split only.
- **Tabular and other datasets, OpenML first**: `sklearn.datasets.fetch_openml(
  data_home=args.data_dir, ...)` pinned by `data_id` or by `name` plus an explicit
  `version`; a direct URL **only** if the dataset is on neither torchvision nor
  OpenML. The order is explicit because the first real tabular run proved it
  necessary: the Coder chose CMU StatLib's Boston Housing URL — the one most
  tutorials cite — and it now returns **HTTP 403 to every client**. A remembered
  URL looks exactly as plausible as a working one. (scikit-learn's `load_boston`
  is likewise gone since 1.2; Boston Housing is OpenML `data_id=531`.)
- **Verify the fetched data is the intended dataset, in code, before training.**
  An OpenML `data_id` that is off by a digit does not fail — it downloads a
  different dataset and the script trains on it, reporting a plausible metric for
  the wrong problem. This is not hypothetical: while fixing the 403, a model
  confidently recalled Boston Housing as `data_id=506`, which is actually
  `analcatdata_gsssexsurvey`. The prompt therefore requires asserting
  `details["name"]` and the row/column counts right after the fetch.
- **Every download lands under `--data-dir`**, passed explicitly as the
  library's root/cache argument, so the Runner's shared cache mount serves it.
- **When the paper names no source**, the chosen one is recorded in
  `assumptions`.
- **When the paper states split sizes or a ratio but not how rows were
  assigned**, the script uses a fixed split seeded from `--seed` and records in
  `assumptions` that the split method is unstated and that the metric depends on
  it. On a dataset of a few hundred rows that dependence is not small.
- **Data-dependent preprocessing** (normalization statistics, scaling, PCA,
  target scaling) is fit on the training split only, unless the paper explicitly
  says otherwise.

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
| `reader/output/<paper>.json` | **Authoritative**, across all five extraction stages. Structured, source-grounded (every entry cites a table/section/figure and transcribed page), already cross-checked by `reader/validator.py`'s retry loop. |
| `ocr/output/vlm/<paper>.md` | **Corroborates and fills gaps.** A component's one-line `specification` is often only interpretable next to the prose it was lifted from, and `unstated_details` is only trustworthy if the text it claims is silent is there to be read. |

The prompt states a **three-level** precedence, and the third level is the one
that matters:

1. **Reader output** wins wherever it has data.
2. **Paper Markdown** may *extend* it, never *override* it; a genuine
   disagreement is recorded in `assumptions`.
3. **The model's own recollection** of a famous architecture ranks **last**, and
   using it at all must be disclosed in `assumptions`.

Level 3 is not theoretical. These are well-known papers, and an unguided model
rebuilds them from the third-party reimplementations it has memorized — complete
with modernizations the paper never had. That failure is silent: the wrong
network still trains and still reports a number.

Both inputs fit comfortably in one context window — the Wide Residual Networks
run sends ~33k input tokens, Network In Network ~30k.

## How the five reader fields are consumed

`reader/` emits five stages, and the prompt gives each a distinct job. Getting
these roles wrong is how a replication silently reproduces the wrong thing.

| Field | Role in the prompt |
|---|---|
| `method_summary` | **Context only.** What the method is for and what is novel about it, so the implementation preserves the paper's point. Explicitly **not** a source of numbers, architecture, or hyperparameters — those have their own fields. |
| `architecture_notes` | **The primary architecture source**, outranking both the Markdown and the model's own knowledge. See below. |
| `claims` | The single targeted claim, quoted verbatim into the metrics contract. |
| `hyperparameters` | Used verbatim, for the regime the targeted claim came from. |
| `data_pipeline` | Preprocessing, augmentation and splits, for that same regime. |

### `architecture_notes` — field by field

- **`overall_structure`** is read once as the specification for `forward()`;
  **`components`** become one piece of code each, built in that order, using
  each entry's `specification` as the build detail and its `source` as the
  pointer back into the Markdown.
- **Baseline components are skipped.** `components` can carry blocks belonging
  to a prior-work or comparison architecture, marked in the entry's `name` or
  `role` (`"[baseline, not own method] ..."`). Network In Network has two: the
  §4.6 fully-connected head and the Hinton et al. conventional CNN. Building one
  because it looked like an ordinary layer is a total, silent failure.
- **`key_equations` — implement only `is_own_method: true`.** Entries marked
  `false` are the conventional formulation or a rival's, printed for contrast.
  Implementing one silently builds the wrong layer, and the resulting network
  still trains and still reports a number. Network In Network is the live case:
  three equations, and only eq. (2) — the mlpconv per-patch MLP — is NIN's own.
  Eq. (1) is a plain convolution and eq. (3) is maxout, both decoys. The prompt
  requires the implemented equation label(s) to be named in `architecture_used`.
- **`depth_or_scale` — never invent a formula it says is absent.** If the field
  reports only a naming convention, there is no formula to encode. This has gone
  wrong for real: the verified WRN script asserts
  `assert (depth - 4) % 6 == 0, "depth must satisfy depth = 6N + 4 ..."`, but
  Wide Residual Networks never states that relation — it gives only Table 1's
  symbolic `N` and the `WRN-n-k` naming convention. The formula came purely from
  pretrained recall. The rule now: derive the structure from `components`
  instead, and disclose the derivation in `assumptions`.
- **`unstated_details` is a required checklist, not background reading.** It is
  the explicit list of what the paper never states but working code needs —
  filter counts, kernel sizes, strides, initialization. The Coder walks it entry
  by entry: choose a standard, era-appropriate value **and** record it in
  `assumptions`, naming the gap it came from. That is the whole mechanism by
  which a paper's gaps stay visible; once an invented channel count is in the
  code it is indistinguishable from a stated one. Entries irrelevant to the
  targeted script (a gap in a baseline, or in a dataset not being targeted) may
  be skipped.

`script_writer.py` logs what actually reached the prompt — component count,
own-method vs. contrast-only equation counts with each own-method equation's
`defines` line, and the number of unstated details awaiting an `assumptions`
entry. When a generated script comes out looking like a stock reimplementation,
that line is what separates "the extraction was thin" from "the model ignored
it".

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

**This is the interface `runner/` and the future `critic/` consume.** The
generated script writes this JSON to `--metrics-output` *and* prints the
identical object as its final single line of stdout (so the Runner can recover
it even if the file write never happened). It is task-agnostic; this example is
a regression claim:

```json
{
  "claim_id": "c1",
  "metric": "RMSE",
  "unit": "",
  "value": 3.02,
  "higher_is_better": false,
  "task_type": "regression",
  "train_loss": 7.1,
  "eval_loss": 9.2,
  "train_metric": 2.69,
  "eval_metric": 3.02,
  "epochs_completed": 1000,
  "num_train_samples": 405,
  "num_eval_samples": 101,
  "wall_clock_seconds": 104.0
}
```

The same literal lives in `script_writer.METRICS_CONTRACT`, which the prompt
embeds — this block and the prompt cannot describe different shapes.

| Field | Rule |
|---|---|
| `claim_id`, `metric`, `unit` | Copied **verbatim** from the targeted claim — never normalized, renamed or unit-converted; an empty `unit` stays `""`. |
| `value` | The reproduced number, in the claim's metric and unit, on the split the claim reports — so it equals `eval_metric`. The Critic diffs it against `reported_value` with **no conversion at all**. |
| `higher_is_better` | Boolean from the metric's *meaning*: `false` for error rates, RMSE, MSE, MAE; `true` for accuracy, R², F1, AUC. Tells a consumer whether a gap is a shortfall or an improvement. |
| `task_type` | The same label as `coder_output.json`'s `task_type`. |
| `train_metric`, `eval_metric` | The claim's **own** metric, computed on the training and evaluation splits. For a "test error %" claim, `train_metric` is training error %; for RMSE, both are RMSEs in the target's original units (scaled targets are inverted first). |
| `train_loss`, `eval_loss` | The paper's training objective on each split, from the same post-training eval-mode passes as the two metrics — each loss/metric pair describes the same model on the same data. |
| `epochs_completed`, `num_train_samples`, `num_eval_samples` | Sizes are those actually used, after any `--max-*-samples` cap. |
| `wall_clock_seconds` | Measured from the start of the run. |

**`train_metric`/`eval_metric` replace `train_accuracy`/`eval_accuracy`.** The
old keys meant nothing for a regression claim, and for an error-rate claim they
forced every consumer to know that `value` was `100 - eval_accuracy`. Metrics
files already on disk in the old shape — the Network In Network and Wide Residual
Networks runs, and every archived attempt under `orchestrator/output/` — stay
valid: nothing in the repo reads either shape strictly (`runner/` logs whichever
keys are present, and its stdout fallback keys on `claim_id`/`metric`/`value`,
which both shapes share).

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
| `--data-dir` | **exactly** `"./data"` — the Runner's shared dataset cache is mounted there (see the comment on `REQUIRED_CLI_FLAGS` for the 2000 s vs 169 s incident that made this a contract) |

**Every flag is required even when it does not apply to the paper's method** — a
method with no epochs, no learning rate, or no mini-batches. `reproduce.sh` is
the same template for every paper and always passes `--epochs`,
`--max-train-samples`, `--max-eval-samples` and `--metrics-output`, so a flag
that is not meaningful stays in the parser as a documented no-op (its `help`
says so, and the script logs it at startup). Dropping one is flagged by gate 2,
and for those four it would also make `argparse` reject the Runner's very first
`probe` invocation.

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
`claim_selection_reasoning`, `task_type`, `architecture_used`, `dataset_used`,
`hyperparameters_used`, `assumptions`, `cli_flags_included` and `script_content`.

`max_tokens` is **16384, double `reader/`'s 8192** — a full training script is
long-form output (the real WRN run emits ~9k output tokens; the plain-loop NIN
regeneration ~9.9k), and this repo has
already lost runs to a silent `stop_reason: max_tokens` truncation twice
(`ocr/vlm_extract.py`, then `reader/claims.py`). `stop_reason` is now checked
explicitly and logged as an `ERROR` when it is `max_tokens`, so that failure can
never be silent again.

What the prompt requires of the generated script:

1. **An inferred `task_type`** (see "What the generated script is" above).
2. **A hand-rolled `nn.Module`**, built from `architecture_notes` (see above),
   never a stock or pretrained model from any library — HF's built-in ResNets,
   for example, assume ImageNet's 224×224 stem (7×7 stride-2 conv + maxpool),
   which destroys CIFAR's 32×32 inputs before the first block. Equally, never a
   generic network that merely resembles the paper: when the paper's
   contribution *is* a custom layer **or a custom loss**, that piece is the thing
   being reproduced, and a plain layer stack or a stock `torch.nn` loss in its
   place is a failed replication at any number. The `is_own_method` rule applies
   to loss equations exactly as it does to layer equations.
3. **A plain PyTorch training loop**, never `Trainer` or another framework, with
   every optimizer argument explicit and nothing added that the paper does not
   state.
4. **PyTorch regardless of the paper's framework**, with framework-default
   differences matched in code where cheap and always disclosed.
5. **Data from a stable, documented source** under `--data-dir` — `torchvision`
   for image datasets, OpenML or a direct URL for tabular ones, never the HF
   `datasets` library (not in the image). Source and split-method choices are
   disclosed; preprocessing is fit on the training split only.
6. **Never invent an unstated detail silently.** Unlike `reader/data_pipeline.py`,
   which may record "not stated" and stop, generated code has to actually run —
   so the rule here is *choose the canonical default, always disclose it in
   `assumptions`*.
7. **Guard the zero-batch edge case**: `--max-train-samples` below the batch size
   with `drop_last=True` silently yields zero batches and a meaningless
   "successful" run. The script must clamp (logging it) or fail loudly.
8. **stdlib `logging`, never `loguru`** — the script runs standalone in a Docker
   container and must not depend on this repo's tooling. Third-party imports are
   limited to `torch` and `numpy`, plus `torchvision` for image data and
   `pandas`/`scikit-learn` for tabular data (never as the model).

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
├── reproduce.sh         # runnable wrapper: ./reproduce.sh [full|smoke]
└── coder_output.json    # bookkeeping (or coder_output.failed.json)
```

### `reproduce.sh` — the human-readable handle on a generated script

Written **deterministically by `coder/pipeline.py`**, not by the model. A shell
wrapper is pure templating from data already in hand (claim, hyperparameters,
script path), so generating it in Python costs no tokens, cannot invent a flag
`train.py` does not define, and cannot be mangled by the tool-field leak
described under "Known issues".

Its header is the audit trail: the targeted claim, every hyperparameter with the
paper value it came from, and every assumption the Coder had to make because the
paper did not state something. Reading it is the fastest way to answer "what did
this actually decide, and where did each number come from" without opening the
JSON.

**This file is the only interface `runner/` uses.** Runner picks a mode; it never
builds a `python` command and never passes a `--flag`. That is what keeps Runner
paper-agnostic — a future paper whose script needs entirely different arguments
changes only its own `reproduce.sh`, and `runner/` is untouched. The contract
narrows from "nine CLI flags spelled exactly right" to "four mode names".

```bash
./reproduce.sh probe    # 2 optimizer steps  — does anything run at all?
./reproduce.sh smoke    # 1 full epoch       — does it reach eval + write metrics?
./reproduce.sh capped   # 5 epochs, 512 imgs — does it actually learn?
./reproduce.sh full     # no training flags  — the paper's real setup
```

The modes are cumulative gates: run them in order, stop at the first non-zero
exit. Each writes its **own** metrics file (`metrics.probe.json`,
`metrics.smoke.json`, `metrics.capped.json`, `metrics.full.json`) so a cheap
stage's numbers can never be mistaken for a real run's. `full` passes no
*training* flags but does still set `--metrics-output`: until 2026-09-13 it
passed nothing, so the script wrote its default `metrics.json`, the Runner
found no `metrics.full.json`, and the first two completed `full` runs were
parsed only through the Runner's stdout fallback.

`capped` is the one that carries real signal on a CPU: on a few hundred examples
a network should overfit fast, so **`train_metric`** moving well past a trivial
baseline (10% accuracy for CIFAR-10 chance; predicting the mean for a
regression) means learning is wired up correctly. Its `eval_metric` is not
comparable to the paper's claim — only `full` is, and for the image papers only
`full` needs a GPU. (A small tabular paper's `full` may well finish on a CPU;
the escalation ladder is the same either way.)

All modes need torch (plus torchvision or pandas/scikit-learn, depending on the
dataset), deliberately absent from this repo's lock — run inside `runner/`'s
image, or a throwaway venv (the script's own header carries the exact commands).
The modes, their caps and the template itself are unchanged by the move away
from `Trainer`: only the comments that named `transformers` or assumed CIFAR-10
were reworded.

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
`torch`/`torchvision`/`pandas`/`scikit-learn` dependencies belong to the
`runner/` Docker image, not to this project (see the platform trap in
`CLAUDE.md` and `ocr/README.md`).

`coder/output/` is excluded from mypy in `pyproject.toml` for the same reason:
the generated scripts are standalone untyped artifacts targeting libraries that
are not installed here, and their gate is `ast.parse` plus the Runner, not
`--strict`.

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
  trailing block onto the key it names. Matching is restricted to the nine
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

**The bookkeeping can disagree with the code it describes.** The NIN run's
`hyperparameters_used` reports stage 3 as `3x3 conv 192->192, 1x1 conv
192->192, 1x1 conv 192->10`, but the code builds
`MLPConvLayer(192, 192, num_classes, num_classes, ...)` — i.e. `192→192`,
`192→10`, `10→10`, narrowing to ten channels one layer early. The script is
valid, trains, and is a defensible reading of a dimension the paper never
states; what is wrong is that the *self-report* does not match the *code*.
Nothing deterministic catches this, because both halves are internally
plausible. It is the same class of defect as the `_NoOpScheduler` bug above:
evidence for the Runner→Critic→Coder loop, not for another gate here.

## Status

### Plain-PyTorch generalization — regression-tested on Network In Network

The move away from `Trainer` was regression-tested on the image-classification
case before any non-image paper was attempted: Network In Network regenerated
with the new prompt, pinned to the **same claim as before** (`--claim-id c1`),
into a scratch output directory, then executed in Docker.

- **Coder call:** attempt 1 of 3, `stop_reason=tool_use`, **34,100 input /
  9,925 output tokens**, all 9 tool fields returned (no leak), **both gates
  passed**, 0 missing CLI flags. `task_type: "classification"`. 348-line script,
  8 hyperparameters, 12 assumptions (5 citing an `unstated_details` gap by name).
- **It is a plain PyTorch loop.** Imports are `torch`, `torchvision`, `numpy` and
  stdlib only; no `transformers`, `Trainer`, `accelerate` or `pixel_values`
  anywhere in the file. Explicit `torch.optim.SGD(lr=..., momentum=0.9,
  weight_decay=1e-4, nesterov=False)`, `MultiStepLR`, `model.train()` /
  `model.eval()` + `torch.no_grad()`.
- **The architecture survived the rewrite.** A real `MLPConvLayer` (k×k conv →
  two 1×1 convs, each with bias and ReLU — eq. (2)); three stages, max pooling
  after each, dropout after the first two; global average pooling straight into
  the loss. **Zero `nn.Linear`**. Its docstring records that eq. (1) and eq. (3)
  are not implemented and the `[baseline, not own method]` components are not
  built.
- **`runner.pipeline --mode probe`: PASSED, exit 0, 76.3 s**, `triage: null`,
  warm shared CIFAR-10 cache hit ("Files already downloaded and verified").
  `metrics.probe.json` is in the new shape — `task_type: "classification"`,
  `higher_is_better: false`, `train_metric: 90.625`, `eval_metric: 89.84375`
  (= `value`), `metric: "Test Error"` / `unit: "%"` verbatim from the claim.

What that run also showed, recorded rather than tuned away:

- `wall_clock_seconds` (29.3) is timed from the start of the training loop, not
  the start of the run as the contract says — it omits ~22 s of dataset loading
  and mean/std computation. The prompt already stated the rule; this generation
  did not follow it.
- The prompt originally defined `train_loss` as a train-mode average over the
  final epoch, which would disagree with `train_metric` (a post-training
  eval-mode pass). The script computed both from the eval-mode pass, which is the
  consistent reading, so the contract text was aligned to it.
- Loss reached ~900 after two SGD steps at `lr=0.1` with Kaiming-normal init and
  no normalization layer. A `probe`'s numbers are not meaningful, but that
  magnitude suggests this configuration may diverge in a longer run — `capped`
  is the stage that would show it.

### Earlier runs (HuggingFace `Trainer` era)

Both runs below predate the plain-loop prompt; the scripts on disk under
`coder/output/` still import `Trainer`, which is why `runner/`'s image keeps
`transformers`/`accelerate`.

Verified end-to-end against **Network In Network**, with a real API call — the
paper chosen deliberately to test whether the `architecture_notes` wiring beats
pretrained priors. NIN is the hard case: its `mlpconv` layer *is* the paper's
contribution, the paper states none of its dimensions, and its equation list
carries two decoys. (A famous ResNet variant would pass on priors alone and
prove nothing.)

- Targeted claim `c1` — *test error 10.41% on CIFAR-10, NIN + Dropout*
  (Table 1) — chosen freely, over the augmented `c2` and the §4.6 ablation rows.
- 343-line script, `stop_reason=tool_use` (no truncation), no tool-field leak
  (all 8 fields on attempt 1), **both gates passed**, all 9 required CLI flags
  present, 8 hyperparameters encoded, **12 assumptions recorded, 6 of them
  explicitly citing an `unstated_details` gap by name**.
- A real `MLPConvLayer`: `k×k` spatial convolution → two 1×1 convolutions, each
  with bias and ReLU, i.e. eq. (2)'s cascaded cross-channel parametric pooling —
  not a generic CNN.
- The decoys were avoided: the script's own docstring records that eq. (1)
  (conventional convolution) and eq. (3) (maxout) are *not* implemented, and the
  two `[baseline, not own method]` components are not built.
- Structure matches `overall_structure`: three mlpconv stages, max pooling +
  dropout after the first two, then `AdaptiveAvgPool2d(1)` → flatten → softmax.
  **Zero `nn.Linear` in the file** — the FC head really is gone, which is the
  paper's second contribution.

Also verified against **Wide Residual Networks** (before the `architecture_notes`
wiring landed):

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

This stage itself never executes a script — there is no torch on the dev
machine (Python 3.13 + Intel macOS, per `CLAUDE.md`'s platform trap), so its own
gates are syntax and flags only. Execution is `runner/`'s job, and the retry
loop that feeds a Runner failure back through `feedback` is `orchestrator/`'s.

Not yet built: any second `CodeWriter` (eval script, Dockerfile, requirements
file). Not yet attempted with the plain-loop prompt: any non-image paper.
