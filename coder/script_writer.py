"""Write a self-contained HuggingFace `Trainer` training script for one paper.

Input is a paper's *structured* Reader output (`reader/output/<paper>.json`)
plus that same paper's VLM Markdown (`ocr/output/vlm/<paper>.md`), sent to
Claude in one tool-use call. Output is a complete, runnable Python script -
this module never executes it and never imports torch.

Why both inputs (this is deliberate, not redundancy):

- `reader/output/<paper>.json` is AUTHORITATIVE, across all FIVE of its
  extraction stages: `method_summary`, `architecture_notes`, `claims`,
  `hyperparameters`, `data_pipeline`. It is structured, source-grounded (every
  entry cites a table/section/figure and a transcribed page) and has already
  been cross-checked by `reader/validator.py`'s retry loop.
- The paper Markdown is CORROBORATION and gap-fill. It used to carry the whole
  architecture burden - before `reader/architecture_notes.py` existed, the
  paper's own layer tables and figure descriptions lived only there. They are
  now extracted, so the Markdown's job narrowed to: confirming a component's
  `specification` against the surrounding prose, and supplying wording the
  extraction did not need to keep.

The prompt states a three-level precedence explicitly: the reader output wins
wherever it has data, the Markdown may EXTEND but never OVERRIDE it, and the
model's own recollection of a famous architecture ranks LAST and must be
disclosed in `assumptions` whenever it is used. That last level is the point of
the `architecture_notes` wiring: these are well-known papers, and an unguided
model rebuilds them from third-party reimplementations it has memorized -
complete with modernizations (batch-norm the paper never had, channel counts
from a later codebase) that quietly reproduce a DIFFERENT network than the one
the targeted claim came from.

Both inputs fit comfortably in one context window (~8k + ~10k tokens for Wide
Residual Networks; ~13k + ~6k for Network In Network).

`max_tokens` is 16384 here, double the 8192 used across `reader/`. A full
training script is long-form output, and this repo has already been bitten by
a silent `stop_reason: max_tokens` truncation twice (once in
`ocr/vlm_extract.py`, once in `reader/claims.py` - see reader/README.md's
"Known issues, fixed"). `stop_reason` is checked explicitly below so that
failure mode can never be silent again.

This module is an importable code-generation step, not a standalone script -
`coder/pipeline.py` is the entry point that resolves both inputs, runs the
deterministic gates, and writes `train.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger

from coder.base import CodeWriter

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16384

# The interface contract with the future `runner/` stage: the Runner forces
# fast, capped, reproducible runs by passing these flags, so every generated
# script must define all of them. `coder/pipeline.py` gates on this exact
# tuple, and the prompt below lists it verbatim.
REQUIRED_CLI_FLAGS: tuple[str, ...] = (
    "--epochs",
    "--max-train-samples",
    "--max-eval-samples",
    "--batch-size",
    "--lr",
    "--output-dir",
    "--metrics-output",
    "--seed",
    # `--data-dir` is required for a reason that only showed up under the retry
    # loop. `runner/` bind-mounts a shared CIFAR-10 cache at the container's
    # `/workspace/data`, which only lands where the script looks if the script
    # defaults its data directory to `./data`. That used to be a convention -
    # whatever the prompt happened to produce - and `runner/README.md` correctly
    # called it "a convention, not a contract".
    #
    # Then a real regeneration defaulted it to `./<output-dir>/data` instead,
    # missed the mount, and re-downloaded 170 MB: the same `probe` stage took
    # **2000 s instead of 169 s**. Nothing broke, which is what makes it nasty -
    # it just silently costs twelve minutes. In a retry loop every regeneration
    # re-rolls that dice, and the stage budgets are calibrated for a warm cache.
    # So the convention is now a contract, enforced by the same literal check as
    # every other flag here.
    "--data-dir",
)

# The exact metrics shape the future Critic diffs against the paper's claim.
# Kept as a literal string so the prompt and the README show the same thing.
METRICS_CONTRACT = """{
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
}"""

PROMPT = (
    """You are the Coder agent of an automated ML-paper replication \
pipeline. You are given TWO inputs describing ONE machine-learning paper, and \
you must write ONE self-contained HuggingFace `Trainer` training script that \
attempts to reproduce ONE specific numeric claim from that paper.

## Input precedence - read this before anything else

1. READER OUTPUT (the JSON below) is AUTHORITATIVE. An earlier pipeline stage \
extracted it from this paper with FIVE separate extractors, and a validation \
pass cross-checked them against the paper text and against each other:
   - `method_summary` - what problem the method attacks, its core idea, what \
is novel about it.
   - `architecture_notes` - the paper's OWN model: `overall_structure`, \
`components`, `key_equations`, `depth_or_scale`, and `unstated_details`.
   - `claims` - the paper's own reported numbers.
   - `hyperparameters` - the paper's own training settings.
   - `data_pipeline` - per-dataset preprocessing, augmentation and splits.
   Use their values VERBATIM. Do not "correct" them from memory, and do not \
substitute numbers you happen to remember about this paper or this \
architecture.
2. PAPER MARKDOWN (below the JSON) is CORROBORATION AND GAP-FILL. Use it to \
confirm what a component's `specification` means in context, to read the \
surrounding prose of a cited section/table/figure, and to supply wording the \
extraction did not need to keep. It may EXTEND the reader output; it may never \
OVERRIDE it. If the two appear to disagree, the reader output wins and you \
note the disagreement in `assumptions`.
3. YOUR OWN PRETRAINED KNOWLEDGE of this architecture ranks LAST - after both \
inputs, not alongside them. Several of these papers are famous and you have \
seen many third-party reimplementations of them; those carry accumulated \
modernizations (a normalization layer the paper never had, channel counts from \
a later codebase, a stem borrowed from a different paper) that do not fail \
loudly. They produce a network that trains happily and reports a number for \
the WRONG model. Whenever you fall back on that knowledge, you MUST record it \
as an `assumptions` entry saying so explicitly.

## Building the architecture - `architecture_notes` is the primary source

The reader output's `architecture_notes` object, NOT the Markdown and NOT your \
memory, is where the model definition comes from. Work through it field by \
field:

A. STRUCTURE. Read `overall_structure` once as the specification for \
`forward()` - it describes how the pieces compose end to end, in order, from \
the input image to the output. Then write one piece of code per `components` \
entry, using that entry's `specification` as the build detail and its `source` \
as the pointer into the Markdown when you need the surrounding prose. Build \
EVERY component the paper's own model is made of, and build them in the order \
`overall_structure` gives. If a component's presence or placement contradicts \
what you remember about this architecture, follow the extraction.

B. IGNORE BASELINE COMPONENTS. `components` can include blocks that belong to \
a prior-work or comparison architecture the paper evaluates against - they are \
marked as such in the entry's `name` or `role` (e.g. a name prefixed \
"[baseline, not own method]", or a role saying it is used only in an \
ablation/comparison section). Those are NOT part of the model you build. \
Building one because it looked like a normal layer is a silent, total failure.

C. EQUATIONS - implement ONLY `is_own_method: true`. Each `key_equations` \
entry carries `latex`, `label`, `defines` and a boolean `is_own_method`. \
Entries with `is_own_method: true` define THIS paper's proposed computation: \
when a paper's contribution is a new layer, that equation IS the layer's \
implementation specification, so your code must compute exactly what it says. \
Entries with `is_own_method: false` are the conventional formulation or a \
rival method's, printed only for contrast - implementing one of those silently \
builds the wrong layer, and the resulting network still trains and still \
reports a number. Check the flag on every equation before you write a line of \
it; read `defines` to see which is which. State in `architecture_used` which \
equation label(s) you implemented.

D. `depth_or_scale` - never invent a formula it says is absent. This field \
holds whatever depth/width formula, naming convention or scaling rule the \
paper actually gives. If it states a closed-form relation, encode that \
relation. If it says the paper gives ONLY a naming convention and NO formula, \
then there is no formula: do not write one, do not assert one, and do not \
`assert` on one. This has already gone wrong for real - a generated script \
asserted `depth = 6N + 4` for Wide Residual Networks, which that paper does \
not state anywhere (it gives only Table 1's symbolic N and the WRN-n-k naming \
convention); the formula came purely from pretrained recall. In that situation, \
derive the concrete structure from `components` and `overall_structure` \
instead, hard-code the depths/widths that produces, and record the derivation \
as an `assumptions` entry.

E. `unstated_details` is a REQUIRED CHECKLIST, not background reading. It \
lists, explicitly, every architectural detail the paper does NOT state but that \
working code needs - filter counts, kernel sizes, strides, padding, pooling \
windows, MLP widths, initialization, whether a normalization layer exists. \
Go through it ENTRY BY ENTRY. For each one that your script has to decide in \
order to run: choose the standard, era-appropriate value, AND record it as an \
`assumptions` entry that names the gap it came from - phrase it like \
"unstated_details: <the gap, in the extraction's own words> - used <what you \
chose> because <why>". Entries that genuinely do not affect your script (a gap \
in a baseline architecture you are not building, a dataset you are not \
targeting) may be skipped, but say so rather than dropping them silently. This \
is the mechanism that keeps a paper's gaps VISIBLE instead of papered over: an \
invented-but-plausible channel count is indistinguishable from a stated one \
once it is in the code, unless it is written down here.

F. `method_summary` is CONTEXT ONLY. Use it to understand what the method is \
trying to achieve and what is novel about it, so your implementation preserves \
the point of the paper rather than an incidental resemblance to it. It is NOT \
a source of numbers, architecture detail, or hyperparameters - those have their \
own fields, and a number appearing in prose there does not override them.

## What to do

1. TARGET EXACTLY ONE CLAIM. Pick it from the reader output's `claims.claims` \
list, and quote its `metric`, `dataset`, `reported_value`, `unit` and \
`model_variant` back in `architecture_used` / `dataset_used` / \
`claim_selection_reasoning` so it is unambiguous which number the script is \
chasing. Put its `claim_id` in `claim_targeted`.

2. MATCH THE CLAIM'S EXPERIMENTAL REGIME. Papers routinely report several \
regimes side by side - different preprocessing (mean/std normalization vs. \
ZCA whitening), different learning-rate schedules per dataset, with-dropout \
vs. without-dropout, different augmentation settings - and the reader output \
tags them via each entry's `model_variant`, `source`, or the wording of its \
value. Work out which regime your targeted claim actually came from (its \
`model_variant` and `source` table/section tell you), then take the \
hyperparameters and data-pipeline entries belonging to THAT regime. Picking \
the wrong regime silently reproduces a different number than the one you are \
targeting, which is the single worst failure mode of this stage. If the \
reader output gives a regime-specific and a regime-neutral value for the same \
setting, prefer the regime-specific one and say so in `assumptions`.

3. USE THE HYPERPARAMETERS VERBATIM. Learning rate, its full schedule \
(milestones and decay factor, not just the initial value), optimizer and its \
momentum/weight-decay/nesterov settings, batch size, epoch count, dropout \
rate, augmentation - all straight from the reader output's `hyperparameters` \
entries for the matched regime. Record each one you actually used in \
`hyperparameters_used` as `{name, value_used}`, where `value_used` is the \
concrete value the script encodes (e.g. "0.1, x0.2 at epochs 60/120/160").

4. WRITE A HAND-ROLLED `nn.Module`, built from `architecture_notes` exactly as \
the section above prescribes, from `torch.nn` primitives. Do NOT use \
`AutoModelForImageClassification` or any pretrained/stock HuggingFace vision \
model: HuggingFace's built-in ResNets assume ImageNet's 224x224 stem (7x7 \
stride-2 convolution followed by max-pooling), which destroys CIFAR's 32x32 \
inputs before the first block. Equally, do NOT emit a generic CNN that merely \
resembles the paper: if the paper's contribution is a custom layer, the custom \
layer is the thing being reproduced, and a plain convolution stack in its place \
is a failed replication no matter what accuracy it reaches. Name your classes \
after the paper's own component names so the mapping is readable. \
`architecture_used` must describe what you built concretely (depth, widths, \
block structure, layer ordering) and tie each part back to the \
`architecture_notes` component or equation label it came from.

5. ADAPT IT TO `Trainer` WITHOUT SUBCLASSING `PreTrainedModel`. Wrap the raw \
`nn.Module` in a thin `nn.Module` adapter whose `forward(self, pixel_values, \
labels=None)` returns a dict containing `logits`, plus `loss` when `labels` \
is not None. That dict is the whole contract `transformers.Trainer` needs - \
no config class, no `PreTrainedModel`, no `from_pretrained`.

6. LOAD DATA WITH `torchvision.datasets`, e.g. \
`torchvision.datasets.CIFAR10(root=..., train=..., download=True, \
transform=...)`. Do NOT use `datasets.load_dataset`. Two reasons: the paper's \
augmentation (reflection-pad by 4 pixels, random 32x32 crop, random \
horizontal flip) maps one-to-one onto `torchvision.transforms`, and it keeps \
a pyarrow dependency out of the Runner's future Docker image. Apply train-time \
augmentation to the training split only; the evaluation split gets \
normalization only. Provide a `data_collator` that stacks the torchvision \
`(image, label)` tuples into `{"pixel_values": ..., "labels": ...}`.

7. NEVER INVENT AN UNSTATED DETAIL SILENTLY. If neither input states \
something the script cannot run without (weight initialization, the exact \
normalization constants, `num_workers`, the eval batch size, ...), choose the \
canonical standard default AND record it as one entry in `assumptions`, \
phrased as "<what was missing> - used <what you chose> because <why>". \
`assumptions` is where BOTH kinds of gap land: the ones \
`architecture_notes.unstated_details` already named for you (walk that list, \
per rule E above) and any further one you hit while writing the code. An empty \
`assumptions` array means you genuinely needed nothing beyond the two inputs; \
with a non-empty `unstated_details` that is impossible, so never empty it just \
to look tidy. This differs deliberately from the Reader's rule: the Reader is \
allowed to record "not stated" and stop, but generated code has to actually \
run, so the rule here is CHOOSE SENSIBLY, ALWAYS DISCLOSE.

8. ALWAYS DEFINE THESE ARGPARSE FLAGS, spelled exactly like this:
   `--epochs`            default = the paper's real epoch count for this regime
   `--max-train-samples` default None, meaning use the full training set
   `--max-eval-samples`  default None, meaning use the full evaluation set
   `--batch-size`        default = the paper's real batch size
   `--lr`                default = the paper's real initial learning rate
   `--output-dir`        default = a sensible local directory
   `--metrics-output`    default = a `metrics.json` path
   `--seed`              default = any fixed integer, seeded through `torch`, \
`numpy` and `random`
   `--data-dir`          default = EXACTLY `"./data"` - not a path derived from \
`--output-dir`, not any other name. The Runner bind-mounts a shared, \
pre-populated dataset cache at that exact relative location, and the dataset \
download must be pointed at it. Getting this wrong does not fail: it silently \
re-downloads ~170 MB and makes the run about twelve times slower.
   This is a load-bearing interface contract with the pipeline's Runner stage, \
which uses these flags to force fast capped smoke runs before a full one. \
List every flag your `argparse` actually defines in `cli_flags_included`.

9. EMIT THE METRICS JSON in exactly this shape - write it to the \
`--metrics-output` path AND print the identical JSON as the FINAL single line \
on stdout:

"""
    + METRICS_CONTRACT
    + """

   `claim_id`, `metric` and `unit` must be copied VERBATIM from the targeted \
claim - do not normalize, rename, or unit-convert them (if the claim says \
"test error" and "%", the script emits "test error" and "%", never "accuracy" \
or "error_rate"). `value` is the reproduced number expressed in that same \
metric and unit, so a future Critic can diff `value` against `reported_value` \
with no conversion at all. Keep `eval_accuracy` alongside it as the raw \
measurement it was derived from. `wall_clock_seconds` is measured from the \
start of the run.

10. GUARD THE ZERO-BATCH EDGE CASE. A capped smoke run with \
`--max-train-samples` smaller than `--batch-size` combined with \
`drop_last=True` silently yields zero batches, "succeeds", and reports \
meaningless numbers. The script must detect this and either fail loudly with \
a clear message or clamp the batch size down (logging the clamp) - it must \
never silently train on nothing. Apply the same reasoning to the evaluation \
split.

11. USE STDLIB `logging` (or `print`) IN THE GENERATED SCRIPT - NOT `loguru`. \
The script runs standalone inside a future Docker container and must not \
depend on this repository's tooling. Its only third-party imports should be \
`torch`, `torchvision`, `transformers`, `numpy` (all standard in an ML image).

12. THE SCRIPT MUST BE COMPLETE AND RUNNABLE AS WRITTEN. No placeholders, no \
`...`, no `TODO`, no "fill in your own", no omitted function bodies, no \
truncation. Include a module docstring, `if __name__ == "__main__": main()`, \
and enough stdlib logging that a reader of the console output can tell what \
configuration ran.

Call the `write_training_script` tool with the results."""
)

CLAIM_SELECTION_FREE = """
## Claim selection

No specific claim was requested, so YOU choose which claim to target. Pick the \
paper's HEADLINE result - the single number the paper is best known for and \
most prominently argues for (typically its best-performing configuration on \
its primary dataset, the one quoted in the abstract/conclusion), not an \
ablation row and not a secondary dataset. Prefer a claim whose training regime \
is fully specified by the reader output's hyperparameters. Explain the choice \
in `claim_selection_reasoning`, including what you rejected and why."""

CLAIM_SELECTION_FIXED = """
## Claim selection

Target EXACTLY the claim whose `claim_id` is "{claim_id}". Do not substitute a \
different claim even if another looks more impressive or better specified. \
Find it in the reader output's `claims.claims` list, put "{claim_id}" in \
`claim_targeted`, and use `claim_selection_reasoning` to state what that claim \
is (metric/dataset/value/unit/model_variant) and which experimental regime it \
belongs to. If no claim with that id exists in the reader output, say so \
plainly in `claim_selection_reasoning` and target the closest headline claim \
instead."""

SCRIPT_TOOL: dict[str, Any] = {
    "name": "write_training_script",
    "description": (
        "Record a complete, self-contained HuggingFace Trainer training script "
        "targeting one specific claim from the paper, plus the bookkeeping "
        "explaining which claim it targets, which architecture/dataset/"
        "hyperparameters it encodes, what had to be assumed, and which CLI "
        "flags it defines."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claim_targeted": {
                "type": "string",
                "description": (
                    "The `claim_id` of the single claim this script reproduces, "
                    "copied from the reader output's claims list, e.g. 'c34'."
                ),
            },
            "claim_selection_reasoning": {
                "type": "string",
                "description": (
                    "Why this claim is the right target - what it is (metric, "
                    "dataset, reported_value, unit, model_variant), why it is the "
                    "paper's headline/most representative result if it was freely "
                    "chosen, and which experimental regime (preprocessing, "
                    "schedule, dropout setting) it belongs to."
                ),
            },
            "architecture_used": {
                "type": "string",
                "description": (
                    "The architecture the script implements, described concretely "
                    "(depth, width, block structure, layer ordering) and tied back "
                    "to where the paper's Markdown describes it."
                ),
            },
            "dataset_used": {
                "type": "string",
                "description": (
                    "The dataset the script trains and evaluates on, plus how it is "
                    "loaded, normalized, augmented and split."
                ),
            },
            "hyperparameters_used": {
                "type": "array",
                "description": (
                    "Every hyperparameter the script actually encodes, taken from "
                    "the reader output's entries for the targeted claim's regime."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "e.g. 'learning rate schedule'.",
                        },
                        "value_used": {
                            "type": "string",
                            "description": (
                                "The concrete value encoded in the script, e.g. "
                                "'0.1 initial, x0.2 at epochs 60/120/160'."
                            ),
                        },
                    },
                    "required": ["name", "value_used"],
                },
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One entry per detail that neither input stated and the script "
                    "had to decide anyway, each naming what was missing, the "
                    "standard default chosen, and why. Empty array only if nothing "
                    "at all had to be assumed."
                ),
            },
            "cli_flags_included": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Every command-line flag the script's argparse parser actually "
                    "defines, spelled exactly as on the command line, e.g. "
                    "'--epochs'. Must include all required contract flags."
                ),
            },
            "script_content": {
                "type": "string",
                "description": (
                    "The COMPLETE Python script as plain source text - no Markdown "
                    "code fences, no placeholders, no ellipses, no TODOs, no "
                    "omitted bodies. It must be syntactically valid Python 3 and "
                    "runnable as written."
                ),
            },
        },
        "required": [
            "claim_targeted",
            "claim_selection_reasoning",
            "architecture_used",
            "dataset_used",
            "hyperparameters_used",
            "assumptions",
            "cli_flags_included",
            "script_content",
        ],
    },
}


@dataclass
class HyperparameterUsed:
    name: str
    value_used: str


@dataclass
class TrainingScript:
    claim_targeted: str
    claim_selection_reasoning: str
    architecture_used: str
    dataset_used: str
    hyperparameters_used: list[HyperparameterUsed]
    assumptions: list[str]
    cli_flags_included: list[str]
    script_content: str


def _tool_input(message: Message) -> dict[str, object]:
    for block in message.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Claude did not call the write_training_script tool")


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _parse_hyperparameter(raw: object) -> HyperparameterUsed:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a hyperparameter object, got {raw!r}")
    return HyperparameterUsed(
        name=str(raw.get("name", "<unnamed>")),
        value_used=str(raw.get("value_used", "<not reported>")),
    )


# Field names of SCRIPT_TOOL, used to re-home leaked blocks onto the key they
# name. Restricted to this known set on purpose: a bare `<...>` regex would
# happily match a comparison operator or a type annotation inside script_content.
TOOL_FIELDS: tuple[str, ...] = (
    "claim_targeted",
    "claim_selection_reasoning",
    "architecture_used",
    "dataset_used",
    "hyperparameters_used",
    "assumptions",
    "cli_flags_included",
    "script_content",
)

# Two leak shapes have been observed in real runs, so both are matched:
#   <parameter name="dataset_used">...</dataset_used>
#   <architecture_used>...</architecture_used>
_LEAKED_PARAMETER = re.compile(
    r'<(?:parameter name=")?(' + "|".join(TOOL_FIELDS) + r')"?>(.*?)(?:</\1>|\Z)',
    re.DOTALL,
)


def _coerce_leaked(raw: str) -> object:
    """A leaked value arrives as text; array-valued fields arrive as JSON text."""
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _recover_leaked_fields(payload: dict[str, object]) -> dict[str, object]:
    """Split apart tool fields the model serialized *inside* another field.

    Observed for real on the first two Wide Residual Networks runs: instead of
    emitting `dataset_used` and `hyperparameters_used` as separate tool inputs,
    the model appended them to the end of `architecture_used` using the literal
    delimiter syntax it sees internally::

        ...model_variant='WRN 28-10'.</architecture_used>
        <parameter name="dataset_used">CIFAR-10 (torchvision...)</dataset_used>
        <parameter name="hyperparameters_used">[{"name": "optimizer", ...}]

    The result is a payload that is missing two schema-required fields while one
    other field carries 1700 characters of the wrong content. This recovery is
    deterministic and costs nothing: find a `</field>` closer inside a string
    value, truncate that field there, and re-home each trailing
    `<parameter name="...">` block onto the key it names. Only keys genuinely
    absent from the payload are filled, so a properly emitted field is never
    overwritten by a leaked duplicate.
    """
    recovered = dict(payload)
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        closer = f"</{key}>"
        head, marker, tail = value.partition(closer)
        if not marker:
            continue
        logger.warning(
            f"  [training_script] field '{key}' contains a leaked '{closer}' delimiter - "
            f"the model serialized other tool fields inside it; splitting them back out"
        )
        recovered[key] = head.strip()
        found = 0
        for match in _LEAKED_PARAMETER.finditer(tail):
            leaked_key = match.group(1)
            found += 1
            if leaked_key in payload:
                logger.warning(
                    f"  [training_script] ignoring leaked duplicate of '{leaked_key}' - "
                    f"the model also emitted it properly"
                )
                continue
            recovered[leaked_key] = _coerce_leaked(match.group(2).strip())
            logger.warning(f"  [training_script] recovered leaked field '{leaked_key}'")
        if not found:
            # The delimiter leaked but no re-homeable blocks followed it, so the
            # remaining fields were never emitted at all. Show the tail rather
            # than failing blind - this is the only view of what came back.
            logger.warning(
                f"  [training_script] nothing recoverable after the leaked '{closer}' "
                f"({len(tail)} chars of tail): {tail[:500]!r}"
            )
    return recovered


def _optional_str(payload: dict[str, object], key: str) -> str:
    """Read a schema-required *bookkeeping* string, tolerating its absence.

    Every field in `SCRIPT_TOOL` is marked required, but a model can still omit
    one, and a real run has: the first Wide Residual Networks run came back with
    a complete 460-line script and no `architecture_used`. Hard-failing there
    would have thrown away a correct, expensive generation over a descriptive
    field nothing downstream computes on. `script_content` is NOT read through
    this helper - that one is genuinely fatal when missing.
    """
    value = payload.get(key)
    if value is None:
        logger.warning(
            f"  [training_script] model omitted the schema-required bookkeeping field "
            f"'{key}'; recording it as not reported (the script itself is unaffected)"
        )
        return "<not reported by the model>"
    return str(value)


def _strip_code_fences(script: str) -> str:
    """Defensively unwrap a Markdown code fence around the script.

    The tool schema forbids fences, but a fenced value would otherwise fail the
    pipeline's `ast.parse` gate for a purely cosmetic reason. Only strips when
    the whole value is fenced - a fence inside a docstring is left alone.
    """
    stripped = script.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return script
    lines = stripped.splitlines()
    if len(lines) < 2:
        return script
    logger.warning("  [training_script] script_content came wrapped in a code fence; stripping it")
    return "\n".join(lines[1:-1]) + "\n"


def _log_architecture_inputs(reader_output: dict[str, Any]) -> None:
    """Log what the `architecture_notes` stage actually handed this call.

    Purely observational - the whole `reader_output` dict is serialized into the
    prompt either way. It exists because the prompt now leans on three specific
    sub-fields (`components`, `key_equations[].is_own_method`,
    `unstated_details`), and when a generated script comes out looking like a
    stock reimplementation, the first question is whether the extraction was
    thin or whether the model ignored it. Without this line the console cannot
    tell those two apart.
    """
    notes = reader_output.get("architecture_notes")
    if not isinstance(notes, dict):
        logger.warning(
            "  [training_script] reader output carries NO `architecture_notes` - this is a "
            "pre-architecture_notes reader run, so the model must fall back on the paper "
            "Markdown for structure. Re-run reader/pipeline.py for this paper."
        )
        return

    components = _as_list(notes.get("components"))
    equations = _as_list(notes.get("key_equations"))
    own = [eq for eq in equations if isinstance(eq, dict) and eq.get("is_own_method")]
    unstated = _as_list(notes.get("unstated_details"))
    logger.info(
        f"  [training_script] architecture_notes in prompt: "
        f"model_name={notes.get('model_name', '<none>')!r}, "
        f"{len(components)} component(s), {len(equations)} equation(s) "
        f"({len(own)} own-method, {len(equations) - len(own)} contrast-only), "
        f"{len(unstated)} unstated detail(s) to be answered in `assumptions`"
    )
    for equation in own:
        logger.info(f"    - implement eq {equation.get('label', '?')}: {equation.get('defines')}")
    if equations and not own:
        logger.warning(
            "  [training_script] every extracted equation is marked contrast-only - there is "
            "no equation for the script to implement; structure must come from `components`"
        )
    if not unstated:
        logger.warning(
            "  [training_script] `unstated_details` is empty - the paper is claimed to fully "
            "specify its architecture, which is rare; expect few architectural assumptions"
        )


class TrainingScriptWriter(CodeWriter[TrainingScript]):
    """Writes one self-contained HuggingFace `Trainer` script targeting one claim."""

    name: ClassVar[str] = "training_script"

    def __init__(self, target_claim_id: str | None = None, max_attempts: int = 3) -> None:
        """`target_claim_id` pins which claim the script must reproduce. Left
        as None, Claude picks the paper's headline claim itself and justifies
        the pick in `claim_selection_reasoning`.

        `max_attempts` bounds retries of one specific, observed, intermittent
        failure: the model serializing its tool fields as literal
        `<parameter name="...">` text inside another field instead of as
        separate tool inputs, sometimes swallowing `script_content` itself.
        `_recover_leaked_fields` repairs the recoverable shape; this retry
        covers the shape where the remaining fields were never emitted at all.
        It is NOT a quality retry - a structurally valid but bad script is
        returned as-is for the gates (and later the Runner/Critic) to judge.
        """
        self.target_claim_id = target_claim_id
        self.max_attempts = max_attempts

    def _build_prompt(self, feedback: str | None) -> str:
        if self.target_claim_id is None:
            prompt = f"{PROMPT}\n{CLAIM_SELECTION_FREE}"
        else:
            prompt = f"{PROMPT}\n{CLAIM_SELECTION_FIXED.format(claim_id=self.target_claim_id)}"
        if feedback:
            prompt = (
                f"{prompt}\n\n"
                f"## Feedback on your previous attempt\n\n"
                f"A prior run of this script failed or was rejected for this "
                f"specific reason: {feedback}\nAddress it specifically in this "
                f"attempt - do not simply regenerate the same script."
            )
        return prompt

    def write(
        self,
        reader_output: dict[str, Any],
        paper_markdown: str,
        client: Anthropic,
        feedback: str | None = None,
    ) -> TrainingScript:
        """Send one paper's Reader output plus its VLM Markdown to Claude and get
        back a complete training script. No file I/O here - the pipeline owns
        reading both inputs, gating the result, and writing the script to disk."""
        prompt = self._build_prompt(feedback)
        paper = str(reader_output.get("paper", "<unknown paper>"))
        if self.target_claim_id is None:
            logger.info(f"  [training_script] no --claim-id given; Claude will pick for '{paper}'")
        else:
            logger.info(f"  [training_script] pinned to claim '{self.target_claim_id}'")

        user_content = (
            f"{prompt}\n\n"
            f"--- READER OUTPUT (AUTHORITATIVE: method_summary, "
            f"architecture_notes, claims, hyperparameters, data_pipeline) ---"
            f"\n\n{json.dumps(reader_output, indent=2)}\n\n"
            f"--- PAPER MARKDOWN (corroborates and fills gaps; never overrides "
            f"the reader output) ---\n\n{paper_markdown}"
        )
        logger.info(
            f"  [training_script] prompting {MODEL} (max_tokens={MAX_TOKENS}) with "
            f"{len(json.dumps(reader_output))} chars of reader output + "
            f"{len(paper_markdown)} chars of paper Markdown"
        )
        _log_architecture_inputs(reader_output)

        payload: dict[str, object] = {}
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"  [training_script] API attempt {attempt}/{self.max_attempts}")
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=[cast(ToolParam, SCRIPT_TOOL)],
                tool_choice=cast(
                    ToolChoiceToolParam, {"type": "tool", "name": "write_training_script"}
                ),
                messages=[{"role": "user", "content": user_content}],
            )

            # Never let a truncated script look like a successful one: this repo
            # has already lost two runs to a silent `max_tokens` stop
            # (ocr/vlm_extract.py and reader/claims.py). Log it loudly; the
            # pipeline's ast.parse gate then catches the half-written file.
            if message.stop_reason == "max_tokens":
                logger.error(
                    f"  [training_script] stop_reason=max_tokens - the script was TRUNCATED "
                    f"at the {MAX_TOKENS}-token cap and is almost certainly incomplete. "
                    f"Raise MAX_TOKENS in coder/script_writer.py."
                )
            else:
                logger.info(f"  [training_script] stop_reason={message.stop_reason}")
            logger.info(
                f"  [training_script] tokens: {message.usage.input_tokens} in / "
                f"{message.usage.output_tokens} out"
            )

            payload = _recover_leaked_fields(_tool_input(message))
            logger.info(f"  [training_script] tool fields returned: {', '.join(sorted(payload))}")
            if "script_content" in payload:
                break
            logger.warning(
                f"  [training_script] malformed tool call - no `script_content` field "
                f"(got only {sorted(payload)}). This is the known parameter-leak glitch; "
                f"retrying the call."
            )
        else:
            raise RuntimeError(
                f"Claude called write_training_script without a `script_content` field on "
                f"all {self.max_attempts} attempts - got only {sorted(payload)}. "
                f"Nothing to write."
            )

        result = TrainingScript(
            claim_targeted=_optional_str(payload, "claim_targeted"),
            claim_selection_reasoning=_optional_str(payload, "claim_selection_reasoning"),
            architecture_used=_optional_str(payload, "architecture_used"),
            dataset_used=_optional_str(payload, "dataset_used"),
            hyperparameters_used=[
                _parse_hyperparameter(raw) for raw in _as_list(payload.get("hyperparameters_used"))
            ],
            assumptions=[str(item) for item in _as_list(payload.get("assumptions"))],
            cli_flags_included=[str(flag) for flag in _as_list(payload.get("cli_flags_included"))],
            script_content=_strip_code_fences(str(payload["script_content"])),
        )

        # Logged loudly and in full: a wrong claim pick silently reproduces the
        # wrong number, and the console output is the only place that is visible
        # before the Critic stage exists.
        logger.info(f"  [training_script] TARGET CLAIM: {result.claim_targeted}")
        logger.info(f"  [training_script] reasoning: {result.claim_selection_reasoning}")
        logger.info(f"  [training_script] architecture: {result.architecture_used}")
        logger.info(f"  [training_script] dataset: {result.dataset_used}")
        logger.info(
            f"  [training_script] hyperparameters encoded ({len(result.hyperparameters_used)}):"
        )
        for hyperparameter in result.hyperparameters_used:
            logger.info(f"    - {hyperparameter.name} = {hyperparameter.value_used}")
        logger.info(f"  [training_script] assumptions recorded ({len(result.assumptions)}):")
        for assumption in result.assumptions:
            logger.info(f"    - {assumption}")
        logger.info(
            f"  [training_script] CLI flags self-reported "
            f"({len(result.cli_flags_included)}): {', '.join(result.cli_flags_included)}"
        )
        logger.info(
            f"  [training_script] script generated: "
            f"{len(result.script_content.splitlines())} lines, "
            f"{len(result.script_content)} chars"
        )
        return result
