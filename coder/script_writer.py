"""Write a self-contained training script for one paper: PyTorch or scikit-learn.

Input is a paper's *structured* Reader output (`reader/output/<paper>.json`)
plus that same paper's VLM Markdown (`ocr/output/vlm/<paper>.md`), sent to
Claude in one tool-use call. Output is a complete, runnable Python script -
this module never executes it and never imports torch.

The script targets ANY supervised task the paper's claim measures - image
classification, classification of other inputs, tabular regression - and the
model infers which one (`task_type`) from the claim's metric and dataset plus
`method_summary`; there is no Reader field for it.

Three decisions shape the prompt, and each one closes a silent-divergence path:

- A PLAIN PYTORCH LOOP, NOT HUGGINGFACE `Trainer`. This stage used to generate
  `Trainer` scripts, which only fit image classification: `Trainer`'s model
  contract is a dict of `pixel_values`/`labels` in and `logits`/`loss` out,
  which a regression head or a paper-defined loss has to be contorted into. And
  `Trainer` brings its own optimizer, schedule, warmup and loop defaults that no
  paper states - every one of them a number the reproduction used that nobody
  chose. A hand-written loop has no hidden defaults: every setting in it is one
  the model wrote down and can disclose.
- THE MODEL FAMILY PICKS THE LIBRARY. Classical estimators (SVMs, trees,
  forests, k-NN, logistic regression, ...) are built with scikit-learn, whose
  `SVC` wraps LIBSVM itself - re-implementing one in PyTorch would change the
  optimizer and the solution. Library versions matter too: a 2017 paper's
  "default" SVC, random forest or logistic regression is not scikit-learn
  1.5.2's, so paper-era defaults are set explicitly and disclosed. Neural
  models, including trees trained by gradient descent, stay in PyTorch.
- NEURAL MODELS ALWAYS IN PYTORCH, WHATEVER FRAMEWORK THE PAPER USED. The
  Runner's image has no second deep-learning framework, deliberately.
  Translating a Keras paper is not neutral, though -
  "we used Adam" meant Keras's Adam (`epsilon=1e-7`, not PyTorch's `1e-8`), and
  an unmarked `Dense` layer meant Glorot-uniform init (not `nn.Linear`'s
  Kaiming-uniform). The prompt requires matching such defaults in code where
  that is cheap, and recording every difference in `assumptions` either way.
- DATA FROM A STABLE, DOCUMENTED SOURCE, cached under `--data-dir`:
  `torchvision.datasets` for image datasets, OpenML (`fetch_openml`) or a
  direct URL for tabular ones. When the paper names no source, or states split
  sizes without saying how rows were assigned, the choice is disclosed - the
  reproduced metric depends on it.

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
training script is long-form output - longer now that the training and
evaluation loops are written out rather than delegated to `Trainer` - and this
repo has already been bitten by a silent `stop_reason: max_tokens` truncation
twice (once in `ocr/vlm_extract.py`, once in `reader/claims.py` - see
reader/README.md's "Known issues, fixed"). `stop_reason` is checked explicitly
below so that failure mode can never be silent again.

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
    # loop. `runner/` bind-mounts a shared dataset cache (CIFAR-10 today, OpenML
    # and URL downloads for tabular papers too) at the container's
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

# Task types the prompt names as expected. Deliberately NOT a closed set: the
# model may return another label for a task that is genuinely neither, and that
# is accepted as-is - only logged, so an unusual label is visible, not rejected.
KNOWN_TASK_TYPES: tuple[str, ...] = ("classification", "regression")

# The exact metrics shape the future Critic diffs against the paper's claim.
# Kept as a literal string so the prompt and the README show the same thing.
#
# Task-agnostic on purpose. The previous shape carried `train_accuracy` /
# `eval_accuracy`, which has no meaning for a regression claim (RMSE) and was a
# silent trap for an error-rate claim (the script had to remember to convert).
# `train_metric` / `eval_metric` are the claim's OWN metric on each split, and
# `higher_is_better` tells a consumer which direction a gap runs without having
# to guess it from the metric's name. Metrics files written in the old shape
# stay on disk and stay readable: nothing downstream reads either shape strictly.
METRICS_CONTRACT = """{
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
}"""

# One line of the learning-curve history every generated script writes while it
# trains. runner/checkups.py reads it LIVE and halts a run that is not learning;
# this literal is what the prompt shows, and the marker is gated in coder/pipeline.py.
PROGRESS_MARKER = "REPROBOT_PROGRESS"
PROGRESS_EXAMPLE = (
    PROGRESS_MARKER
    + """ {"kind": "epoch", "step": 7, "steps_total": 40, "train_loss": 0.2956, \
"eval_loss": 0.3106, "train_metric": 91.2, "eval_metric": 93.58, "metric": "test accuracy", \
"unit": "%", "higher_is_better": true, "chance_metric": 11.35, "target_value": 94.45, \
"loss_lower_bound": 0.0, "num_eval_samples": 10000, "elapsed_seconds": 713.2}"""
)

PROMPT = (
    """You are the Coder agent of an automated ML-paper replication \
pipeline. You are given TWO inputs describing ONE machine-learning paper, and \
you must write ONE self-contained training script that attempts to reproduce \
ONE specific numeric claim from that paper. Neural models are written in \
PyTorch as a plain, explicit training loop - NOT HuggingFace `Trainer` or any \
other training framework. Classical machine-learning estimators (SVMs, decision \
trees, random forests, k-NN, logistic regression, ...) are built with \
scikit-learn. Rule 5 says which applies.

The paper can be about any supervised learning task: image classification, \
classification of other inputs, regression on tabular data, and so on, with a \
neural network or a classical estimator. Nothing below assumes images unless it \
explicitly says so.

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
the model's input to its output. Then write one piece of code per `components` \
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
The same holds for a new OBJECTIVE: when the contribution is a loss function \
(a margin-based loss in place of softmax cross-entropy, a new penalty term), \
its own-method equation IS the loss your training loop must compute. Entries \
with `is_own_method: false` are the conventional formulation or a rival \
method's, printed only for contrast - implementing one of those silently \
builds the wrong method, and the result still trains and still reports a \
number. Check the flag on every equation before you write a line of it; read \
`defines` to see which is which. State in `architecture_used` which equation \
label(s) you implemented.

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
working code needs - layer widths, filter counts, kernel sizes, strides, \
padding, pooling windows, activation functions, initialization, whether a \
normalization layer exists. Go through it ENTRY BY ENTRY. For each one that \
your script has to decide in order to run: choose the standard, era-appropriate \
value, AND record it as an `assumptions` entry that names the gap it came from \
- phrase it like "unstated_details: <the gap, in the extraction's own words> - \
used <what you chose> because <why>". Entries that genuinely do not affect your \
script (a gap in a baseline architecture you are not building, a dataset you \
are not targeting) may be skipped, but say so rather than dropping them \
silently. This is the mechanism that keeps a paper's gaps VISIBLE instead of \
papered over: an invented-but-plausible layer width is indistinguishable from a \
stated one once it is in the code, unless it is written down here.

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

2. INFER THE TASK TYPE and record it in `task_type`. No upstream stage states \
it: work it out from the targeted claim's `metric` and `dataset`, the reader \
output's `method_summary`, and `data_pipeline`. Use `"classification"` when the \
model predicts one of a fixed set of discrete labels (metrics such as accuracy, \
error rate, top-k error) and `"regression"` when it predicts a continuous \
quantity (metrics such as RMSE, MSE, MAE, R^2). Those are the expected values, \
not a closed list: if the task is genuinely neither, use another short \
lowercase label and justify it in `claim_selection_reasoning`. Decide by what \
the model PREDICTS, not by how it is trained - a paper that swaps softmax \
cross-entropy for a different loss on a labelled benchmark is still \
`"classification"`; the loss is its contribution, the task is unchanged. The \
task type decides the output layer, the loss when the paper does not define \
its own, how `train_metric`/`eval_metric` are computed, and `higher_is_better`.

3. MATCH THE CLAIM'S EXPERIMENTAL REGIME. Papers routinely report several \
regimes side by side - different preprocessing (mean/std normalization vs. \
ZCA whitening, standardized vs. raw features, dimensionality reduction to a \
fixed size or none), different learning-rate schedules per dataset, \
with-dropout vs. without-dropout, different augmentation settings - and the \
reader output tags them via each entry's `model_variant`, `source`, or the \
wording of its value. Work out which regime your targeted claim actually came \
from (its `model_variant` and `source` table/section tell you), then take the \
hyperparameters and data-pipeline entries belonging to THAT regime. Picking \
the wrong regime silently reproduces a different number than the one you are \
targeting, which is the single worst failure mode of this stage. If the \
reader output gives a regime-specific and a regime-neutral value for the same \
setting, prefer the regime-specific one and say so in `assumptions`.

4. USE THE HYPERPARAMETERS VERBATIM. Learning rate, its full schedule \
(milestones and decay factor, not just the initial value), optimizer and every \
argument of it the paper gives (momentum, weight decay, nesterov, betas, \
epsilon), batch size, epoch count, dropout rate, loss-specific constants (a \
margin, a penalty weight), augmentation - all straight from the reader \
output's `hyperparameters` entries for the matched regime. Record each one you \
actually used in `hyperparameters_used` as `{name, value_used}`, where \
`value_used` is the concrete value the script encodes (e.g. "0.1, x0.2 at \
epochs 60/120/160").

5. CHOOSE THE LIBRARY BY MODEL FAMILY, THEN DISCLOSE WHAT THE TRANSLATION \
CHANGES. The deciding question is what kind of model the paper itself trained.
   - CLASSICAL ESTIMATORS -> scikit-learn. Support vector machines (kernel SVC, \
LinearSVC, LIBSVM/LIBLINEAR), decision trees, random forests, extra trees, \
scikit-learn's gradient boosting, k-nearest neighbours, logistic and linear \
regression, naive Bayes, perceptrons and SGD linear classifiers - and \
`MLPClassifier`/`MLPRegressor` when the paper itself used scikit-learn's. Build \
the scikit-learn estimator that implements the paper's algorithm, with EVERY \
hyperparameter passed explicitly by name, `random_state=args.seed` wherever the \
estimator accepts one, and `n_jobs` stated explicitly where it exists. Never \
re-implement a classical estimator in PyTorch: an SVM trained by SGD on a hinge \
loss is a different optimizer reaching a different solution, and scikit-learn's \
`SVC` wraps LIBSVM itself, so it can reproduce a LIBSVM paper exactly. Rules 6, \
7 and 12 below are for neural models and do not apply; instead fit the \
estimator on the training split and predict on both splits. Record \
`model_family` as "classical". The Runner's image \
has scikit-learn 1.5.2 and no xgboost, lightgbm or catboost.
   - NEURAL MODELS -> PyTorch, per rules 6 and 7. Anything the paper defines as \
a network of parameters trained by gradient descent is neural, INCLUDING \
tree-shaped models trained that way (a soft decision tree with learned \
filters is a PyTorch model, not a scikit-learn tree). Record `model_family` as \
"neural".
   - LIBRARY VERSIONS SHIP DIFFERENT DEFAULTS. A paper that says "default \
parameters", or leaves a setting unstated, inherited the defaults of the library \
VERSION it used, and scikit-learn has changed many since. For example: `SVC`'s \
`gamma` default was `'auto'` (1 / n_features) before 0.22 and is `'scale'` now; \
LIBSVM's `svm-train` defaults to an RBF kernel with C=1 and gamma = 1 / \
number_of_features, which is `SVC(kernel='rbf', C=1.0, gamma='auto')`; \
`RandomForestClassifier` defaulted to `n_estimators=10` before 0.22 and \
`max_features='auto'` (sqrt for classifiers), a value since removed; \
`LogisticRegression` defaulted to `solver='liblinear'` with one-vs-rest before \
0.22; `GradientBoostingClassifier`'s `loss='deviance'` is now \
`loss='log_loss'`. These are illustrations, not a checklist. Work out the \
paper's era from its date and set the values it actually ran with EXPLICITLY, \
translating removed parameter names to their current equivalents, and record \
each such difference in `assumptions` as "library version: <setting> defaulted \
to <then> in <library, version or era> vs <now> in scikit-learn 1.5.2 - used \
<what you chose> because <why>". LIBSVM's `svm-scale -l -1 -u 1` is \
`MinMaxScaler(feature_range=(-1, 1))` fit on the training split.
   - A NEURAL PAPER WRITTEN IN ANOTHER FRAMEWORK IS STILL REPRODUCED IN PYTORCH. \
The Runner's sandbox has no other deep-learning framework, so a paper built with \
Keras/TensorFlow, JAX, Theano, Caffe, MATLAB or anything else is still \
reimplemented in PyTorch. That translation is not neutral: frameworks ship \
different DEFAULTS, and a paper that says only "we used Adam" silently meant ITS \
framework's Adam. Whenever the paper names a framework other than PyTorch, or \
its text/code URLs make one evident:
   - Work out which of that framework's defaults the paper's unstated settings \
inherited. For example: Keras `Adam` defaults `epsilon` to 1e-7, while \
`torch.optim.Adam` defaults `eps` to 1e-8; Keras `Dense` initializes its kernel \
Glorot-uniform with zero biases, while `torch.nn.Linear` uses Kaiming-uniform \
weights and non-zero uniform biases; Keras `BatchNormalization` uses \
`momentum=0.99, epsilon=1e-3`, while `torch.nn.BatchNorm*` uses `momentum=0.1, \
eps=1e-5` and defines momentum as the complement (the Keras-equivalent PyTorch \
value is `momentum=0.01`); Keras `Model.fit` trains with `batch_size=32` when \
none is given. These are illustrations, not a checklist - apply the same \
reasoning to whatever framework and API the paper actually used.
   - Where matching the original framework's behaviour is cheap and exact, DO \
IT EXPLICITLY IN CODE - pass `eps=1e-7`, call `nn.init.xavier_uniform_` and \
`nn.init.zeros_` on each layer - rather than silently inheriting PyTorch's \
default.
   - Record EVERY such difference as its own `assumptions` entry, whether or \
not you matched it, phrased like "framework: paper used <framework>; <setting> \
defaults to <their value> there vs <PyTorch value> in PyTorch - used <what you \
chose> because <why>".
   - When the paper names no framework at all, do not guess one: make \
explicit, standard choices and disclose them under rule 9 as usual.

6. FOR A NEURAL MODEL, WRITE A HAND-ROLLED `nn.Module`, built from \
`architecture_notes` exactly as the section above prescribes, from `torch.nn` \
primitives. Do NOT load a pretrained or stock model from any library (HuggingFace `AutoModel*`, \
`torchvision.models`, `timm`, ...): stock models carry assumptions of their \
own - HuggingFace's built-in ResNets, for instance, assume ImageNet's 224x224 \
stem (7x7 stride-2 convolution followed by max-pooling), which destroys \
CIFAR's 32x32 inputs before the first block. Equally, do NOT emit a generic \
network that merely resembles the paper: if the paper's contribution is a \
custom layer or a custom loss, that custom piece is the thing being \
reproduced, and a plain layer stack or a stock `torch.nn` loss in its place is \
a failed replication no matter what number it reaches. Name your classes after \
the paper's own component names so the mapping is readable. \
`architecture_used` must describe what you built concretely (depth, widths, \
layer/block structure, ordering, output layer, and the loss function) and tie \
each part back to the `architecture_notes` component or equation label it came \
from.

7. FOR A NEURAL MODEL, WRITE A PLAIN, EXPLICIT PYTORCH TRAINING LOOP. Do not \
use HuggingFace `Trainer`, `accelerate`, PyTorch Lightning, Keras, skorch, or \
any other training framework or wrapper - write the loop yourself:
   - `torch.utils.data` datasets and `DataLoader`s for both splits.
   - The optimizer constructed with EVERY argument passed explicitly by name \
(learning rate, momentum, weight decay, betas, eps, ...), never leaning on a \
library default you have not written down (see rule 5).
   - The learning-rate scheduler, if the paper has one, stepped at the \
granularity the paper describes - per epoch or per iteration - and say which \
in `hyperparameters_used`.
   - `model.train()` while training; `model.eval()` inside `torch.no_grad()` \
while evaluating.
   - The loss this paper actually uses (rule C), computed from the model's \
outputs.
   - Nothing the paper does not state: no gradient clipping, warmup, label \
smoothing, weight averaging, mixed precision or early stopping unless the \
reader output states it - in which case implement exactly what it states.
   - The device chosen as `cuda` when available, else `cpu`, and logged.
   - The loss and the claim's metric logged every epoch (or at a stated \
logging interval, for runs of thousands of epochs).
   This is WHY no framework is used, and why these points matter: a framework's \
model contract (a dict of inputs and labels in, logits and loss out) fits \
regression and paper-defined losses badly, and a framework brings its own \
optimizer, schedule and loop defaults that the paper never stated - numbers the \
reproduction would use that nobody chose. A plain loop has no hidden defaults: \
every setting in it is one you wrote and can disclose.

8. LOAD DATA FROM A STABLE, DOCUMENTED SOURCE, AND CACHE IT UNDER `--data-dir`.
   - IMAGE DATASETS that `torchvision.datasets` provides (CIFAR-10/100, MNIST, \
SVHN, ...): load them with it, e.g. `torchvision.datasets.CIFAR10(\
root=args.data_dir, train=..., download=True, transform=...)`, and express \
augmentation with `torchvision.transforms` (reflection padding, random crops \
and horizontal flips map onto it one-to-one). Apply train-time augmentation to \
the training split only; the evaluation split gets deterministic preprocessing \
only. If the regime flattens images into vectors, do that in the transform or \
the model, and say which.
   - TABULAR AND OTHER DATASETS, in this order of preference - use the FIRST \
that has the dataset:
     1. OpenML through `sklearn.datasets.fetch_openml(..., \
data_home=args.data_dir, as_frame=True)`, pinned EITHER by `data_id` alone OR \
by `name` together with an explicit `version` (a bare name can resolve to a \
different upload later) - never `data_id` and `version` together, which \
`fetch_openml` rejects with a ValueError. OpenML is versioned, maintained, and \
hosts most classic tabular benchmarks (Boston Housing is `data_id=531`).
     2. A direct URL, ONLY if the dataset is on neither `torchvision.datasets` \
nor OpenML. Download it once into `--data-dir` and read that file on every \
later run.
   VERIFY THE DATA IS THE DATA YOU MEANT, IN CODE, BEFORE TRAINING ON IT. An \
OpenML `data_id` that is wrong by a digit does not fail - it downloads a \
different dataset that happens to hold that number, and the script then trains \
on it and reports a plausible metric for the wrong problem. So immediately after \
`fetch_openml` returns, assert that the fetched dataset is the intended one - \
check `bunch.details["name"]` against the expected name, and its row and column \
counts against what the paper states when it states them - and raise with a \
clear message naming the expected and actual values if they differ. Do the \
equivalent sanity check (row count, column names) after reading a URL \
download. This turns a silent wrong-dataset run into an immediate, explained \
failure.
   Do not reach for a URL or loader you remember rather than one you know is \
current. Both go dead: scikit-learn dropped `load_boston` in 1.2, and the \
dataset mirrors that old tutorials link to get taken offline - CMU StatLib's \
Boston Housing URL, the one most tutorials cite, now answers every client with \
HTTP 403. A remembered URL looks exactly as plausible as a working one, which \
is why OpenML comes first. Load the result with `pandas`/`numpy`, then convert \
it to tensors.
   - Files in LIBSVM/svmlight format (the LIBSVM datasets site): download \
them into `--data-dir` and read them with \
`sklearn.datasets.load_svmlight_file`, passing `n_features` explicitly so the \
training and test matrices have the same width.
   - DATA THE PAPER GENERATES from a stated procedure (a synthetic task) is \
generated in code, seeded from `--seed`, following the procedure exactly; \
record any size or distribution detail the paper leaves out in `assumptions`.
   - Never `datasets.load_dataset` or any HuggingFace Hub download: the \
`datasets` library is not in the Runner's image.
   - EVERY download lands under `--data-dir`. Pass it explicitly as the \
library's root/cache argument (`root=`, `data_home=`, the path you save a URL \
download to); never rely on a library's own default cache location.
   - Whenever the paper does not name where its data came from (it names only \
the dataset, or nothing), record the exact source you chose as an \
`assumptions` entry: "data source: <what the paper says> - used <torchvision \
class / OpenML data_id, or name and version / URL> because <why>".
   - SPLITS. Reproduce the paper's split as stated: the dataset's official \
train/test split when that is what the paper used, the stated sizes or ratio \
otherwise. When the paper states split SIZES or a RATIO but not HOW rows were \
assigned, the reproduced metric depends on a choice the paper never made for \
you: use a fixed, seeded random split (seeded from `--seed`, so one integer \
reproduces the whole run), log the resulting sizes, and record an \
`assumptions` entry saying the split method is unstated, what you used, and \
that the metric depends on it.
   - Fit every data-dependent preprocessing step - normalization statistics, \
feature scaling, PCA or other dimensionality reduction, target scaling - on \
the TRAINING split only, then apply the fitted transform to the evaluation \
split. Fitting on all rows leaks evaluation data into training and flatters \
the reproduced number. If the paper explicitly says it did otherwise, follow \
the paper and record that in `assumptions`.
   - `--max-train-samples` / `--max-eval-samples` take a SEEDED RANDOM subset \
of each split AFTER the split is made - stratified by label for a \
classification task - never simply the first N rows. Dataset files are often \
sorted by label (svmguide1's first 2,000 training rows are all one class), so \
the first N rows of a capped check run can hold a single class, and the check \
crashes or measures nothing. Seed the subset from `--seed` so it is reproducible.

9. NEVER INVENT AN UNSTATED DETAIL SILENTLY. If neither input states \
something the script cannot run without (weight initialization, the exact \
normalization constants, the data source, how a split was drawn, a framework \
default, `num_workers`, the eval batch size, ...), choose the canonical \
standard default AND record it as one entry in `assumptions`, phrased as \
"<what was missing> - used <what you chose> because <why>". `assumptions` is \
where EVERY kind of gap lands: the ones `architecture_notes.unstated_details` \
already named for you (walk that list, per rule E above), the framework \
differences from rule 5, the data-source and split choices from rule 8, and \
any further one you hit while writing the code. An empty `assumptions` array \
means you genuinely needed nothing beyond the two inputs; with a non-empty \
`unstated_details` that is impossible, so never empty it just to look tidy. \
This differs deliberately from the Reader's rule: the Reader is allowed to \
record "not stated" and stop, but generated code has to actually run, so the \
rule here is CHOOSE SENSIBLY, ALWAYS DISCLOSE.

10. ALWAYS DEFINE THESE ARGPARSE FLAGS, spelled exactly like this:
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
pre-populated dataset cache at that exact relative location, and every dataset \
download must be pointed at it. Getting this wrong does not fail: it silently \
re-downloads the dataset on every run - for CIFAR-10 that is ~170 MB, and it \
made a real run about twelve times slower.
   This is a load-bearing interface contract with the pipeline's Runner stage, \
which uses these flags to force fast capped smoke runs before a full one. \
EVERY flag must be defined and accepted even when it does not apply to this \
paper's method - no epochs, no learning rate, no mini-batches, as for most \
classical estimators. Such a flag stays in the parser as a documented no-op: \
its `help` text says it is ignored for this method, and the script logs that at \
startup. Never drop one. Map a flag onto an estimator parameter only when it is \
the same quantity (`--epochs` to `MLPClassifier`'s `max_iter`, `--lr` to its \
`learning_rate_init`, `--batch-size` to its `batch_size`); the number of trees \
in a forest is not an epoch count. `--max-train-samples` and \
`--max-eval-samples` always apply, because they are what make the Runner's \
cheap check runs cheap. List every \
flag your `argparse` actually defines in `cli_flags_included`.

11. EMIT THE METRICS JSON in exactly this shape - write it to the \
`--metrics-output` path AND print the identical JSON as the FINAL single line \
on stdout:

"""
    + METRICS_CONTRACT
    + """

   - `claim_id`, `metric` and `unit` must be copied VERBATIM from the targeted \
claim - do not normalize, rename, or unit-convert them (if the claim says \
"test error" and "%", the script emits "test error" and "%", never "accuracy" \
or "error_rate"; if a claim's `unit` is an empty string, emit the empty \
string).
   - `task_type` is the same label you record in the `task_type` tool field.
   - `higher_is_better` is a JSON boolean set from what the claim's metric \
MEANS: `false` for error rates, RMSE, MSE, MAE and other loss-like metrics; \
`true` for accuracy, R^2, F1, AUC and other score-like metrics. A consumer uses \
it to tell a shortfall from an improvement, so it is never omitted or guessed \
from habit.
   - `train_metric` and `eval_metric` are the claim's OWN metric, in the \
claim's unit, computed after training on the training split and on the \
evaluation split respectively, with the model in eval mode (no dropout, no \
augmentation). They are NOT accuracy unless the claim's metric is accuracy: \
for a claim of "test error" in "%", `train_metric` is the training error in % \
and `eval_metric` the evaluation error in %; for a claim of RMSE, both are \
RMSEs. When targets were scaled for training, invert that scaling on the \
predictions first - an RMSE on standardized targets is not the paper's RMSE.
   - `value` is the reproduced number a future Critic diffs against the \
claim's `reported_value` with no conversion at all: the claim's metric, in the \
claim's unit, on the split the claim reports - so it equals `eval_metric`.
   - `train_loss` and `eval_loss` are the paper's training objective averaged \
over the training and evaluation splits, computed in the same post-training, \
eval-mode passes as `train_metric` and `eval_metric` - so each loss and metric \
pair describes the same model on the same data.
   - For a classical estimator with no iterative training objective, \
`train_loss` and `eval_loss` are `null` unless the paper reports a loss for it \
(then compute that loss), and `epochs_completed` is `null` unless the estimator \
iterates (`MLPClassifier`'s `n_iter_`, for example).
   - REPRODUCE THE CLAIM'S EVALUATION PROTOCOL. When the claim is an average \
over repeated runs - several shuffles or seeds, k-fold cross-validation, a \
number of random initializations - the script performs every repetition, \
seeded deterministically from `--seed`, logs each run's result, and reports the \
MEAN as `value` and `eval_metric`, adding `"num_runs"` and \
`"run_values"` (the per-run list) to the JSON. A capped check run (either \
`--max-*-samples` flag set) performs a single repetition, and says so in the log.
   - `num_train_samples` / `num_eval_samples` are the sizes actually used, \
after any `--max-*-samples` cap. `wall_clock_seconds` is measured from the \
start of the run: take the start time as the very first statement of `main()`, \
before argument parsing and data loading - not at the start of the training loop.
   - Every numeric field is a plain JSON number - convert tensors and numpy \
scalars with `float()` / `int()` before serializing.

11b. WRITE A LEARNING-CURVE HISTORY WHILE TRAINING RUNS. The Runner reads it \
LIVE, while the script is still running, and stops a run that is not learning \
(a loss below its lower bound, a model worse than chance, no progress, a \
collapse after learning). It is a load-bearing contract like the metrics JSON.
   - THE FILE: derive it from `--metrics-output` by replacing a trailing \
`.json` with `.history.jsonl` (`metrics.full.json` -> \
`metrics.full.history.jsonl`; append `.history.jsonl` when the path does not \
end in `.json`). Truncate it once at startup, then APPEND one JSON object per \
line and flush after every record, so a reader sees each record immediately.
   - THE SAME LINE ON STDOUT: print every record as one line, prefixed with the \
marker and a space, with `flush=True`. One record looks like this:

"""
    + PROGRESS_EXAMPLE
    + """

   - WHEN: a neural model writes one record after every epoch, with \
`"kind": "epoch"` (for runs longer than 200 epochs, at least 100 evenly spaced \
records, always including the first and the last epoch). Write each record ONCE, \
after its step has completed, carrying that step's results - never a partial \
record before training or evaluation has produced them. A classical estimator \
writes one record after each repetition, fold or seed (`"kind": "repetition"` \
or `"fold"`), or one record after its single fit (`"kind": "fit"`).
   - FIELDS - write every key, using `null` only when a value genuinely does not \
exist: `step` (1-based) and `steps_total` (the planned total after any caps); \
`train_loss` / `eval_loss` (the training objective, averaged over the epoch and \
over an evaluation pass); `train_metric` / `eval_metric` (the claim's own metric \
in the claim's unit - evaluate the evaluation split for EVERY recorded epoch, in \
eval mode); `metric`, `unit` and `higher_is_better`, identical to the metrics \
JSON; `num_eval_samples`; `elapsed_seconds` since the start of the run; and \
three fields the Runner's checks depend on:
     * `chance_metric` - the claim's metric, in its unit, scored on the \
evaluation split by a trivial predictor fitted to the TRAINING split only: the \
training set's majority class for classification (as an accuracy or an error \
rate, whichever the claim uses), the training set's mean target for regression \
(0 for R^2). Compute it once, before training.
     * `target_value` - the targeted claim's `reported_value`.
     * `loss_lower_bound` - the smallest value the training objective can take: \
0.0 for cross-entropy, squared or absolute error, hinge and squared-hinge losses, \
and any sum of such terms with non-negative weights and penalties; `null` when \
the objective can legitimately be negative (a log-likelihood or evidence bound \
written without its constant) or when you are not certain.
   - Write non-finite values as they are - `json.dumps` emits `NaN` and \
`Infinity` - never replace them with `null` or clip them: a NaN is exactly what \
the Runner needs to see. Never skip a record to save time.

12. GUARD THE ZERO-BATCH EDGE CASE. A capped smoke run with \
`--max-train-samples` smaller than `--batch-size` combined with \
`drop_last=True` silently yields zero batches, "succeeds", and reports \
meaningless numbers. The script must detect this and either fail loudly with \
a clear message or clamp the batch size down (logging the clamp) - it must \
never silently train on nothing. Apply the same reasoning to the evaluation \
split.

13. USE STDLIB `logging` (or `print`) IN THE GENERATED SCRIPT - NOT `loguru`. \
The script runs standalone inside a Docker container and must not depend on \
this repository's tooling. Its third-party imports are limited to `torch` and \
`numpy`, plus `torchvision` (image datasets and transforms), `pandas`, \
`scipy` and `scikit-learn` (fetching, splitting and preprocessing data, and - \
under rule 5 - the classical estimator being reproduced). Those are what the \
Runner's image guarantees; any other import, including xgboost, lightgbm and \
catboost, is a failure waiting to happen there. A classical-estimator script \
need not use `torch` at all.

14. THE SCRIPT MUST BE COMPLETE AND RUNNABLE AS WRITTEN. No placeholders, no \
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
        "Record a complete, self-contained training script (a plain PyTorch "
        "loop for a neural model, a scikit-learn estimator for a classical one; "
        "no Trainer or other framework) targeting one specific "
        "claim from the paper, plus the bookkeeping explaining which claim it "
        "targets, which task type that claim measures, which architecture/"
        "dataset/hyperparameters it encodes, what had to be assumed, and which "
        "CLI flags it defines."
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
            "task_type": {
                "type": "string",
                "description": (
                    "The learning task the targeted claim measures, inferred from the "
                    "claim's metric and dataset plus method_summary and data_pipeline. "
                    "Expected values are 'classification' (predicts a discrete label) "
                    "and 'regression' (predicts a continuous quantity); use another "
                    "short lowercase label only if the task is genuinely neither. The "
                    "script's metrics JSON must carry this same string."
                ),
            },
            "model_family": {
                "type": "string",
                "enum": ["neural", "classical"],
                "description": (
                    "'neural' when the script trains a PyTorch model by gradient descent "
                    "(including tree-shaped models trained that way); 'classical' when it "
                    "builds a scikit-learn estimator such as an SVM, a tree ensemble or "
                    "k-NN. The Runner picks the cheap stages' sizes from this."
                ),
            },
            "architecture_used": {
                "type": "string",
                "description": (
                    "The architecture the script implements, described concretely "
                    "(depth, widths, layer/block structure, ordering, output layer "
                    "and loss function) and tied back to the architecture_notes "
                    "component or equation label each part came from."
                ),
            },
            "dataset_used": {
                "type": "string",
                "description": (
                    "The dataset the script trains and evaluates on: its exact source "
                    "(torchvision class, OpenML data_id or name+version, or URL), how it is "
                    "split, and how it is preprocessed, normalized and augmented."
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
            "task_type",
            "model_family",
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
    task_type: str
    model_family: str
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
    "task_type",
    "model_family",
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


def _normalise_model_family(raw: str) -> str:
    """'neural' or 'classical'; anything else falls back to 'neural', loudly.

    Neural is the safe fallback because it keeps today's reproduce.sh stage sizes.
    """
    value = raw.strip().lower()
    if value in ("neural", "classical"):
        return value
    logger.warning(
        f"  [training_script] model_family {raw!r} is not 'neural' or 'classical' - "
        f"treating it as 'neural' for the stage ladder"
    )
    return "neural"


def _log_task_type(task_type: str) -> None:
    """Log the inferred task type, flagging a label outside the expected set.

    Deliberately a warning and never a rejection: `task_type` is open-ended by
    design, so an unexpected label may be exactly right for an unusual paper. But
    every downstream choice in the script - output layer, loss, how the per-split
    metric is computed, `higher_is_better` - hangs off it, so an unexpected value
    has to be visible on the console before anyone reads the code.
    """
    logger.info(f"  [training_script] TASK TYPE: {task_type}")
    if task_type not in KNOWN_TASK_TYPES:
        logger.warning(
            f"  [training_script] task_type {task_type!r} is not one of the expected "
            f"{', '.join(KNOWN_TASK_TYPES)} - accepted as-is, but check "
            f"claim_selection_reasoning for the justification the prompt requires"
        )


class TrainingScriptWriter(CodeWriter[TrainingScript]):
    """Writes one self-contained training script (PyTorch or scikit-learn) targeting one claim."""

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
            task_type=_optional_str(payload, "task_type").strip(),
            model_family=_normalise_model_family(_optional_str(payload, "model_family")),
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
        _log_task_type(result.task_type)
        logger.info(f"  [training_script] MODEL FAMILY: {result.model_family}")
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
