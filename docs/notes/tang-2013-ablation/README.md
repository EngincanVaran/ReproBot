# Tang 2013 — why the `full` run collapsed, and the ablation that found it

Investigation from 2026-09-13. Paper: *Deep Learning using Linear Support Vector
Machines* (Tang, 2013), `extra-papers/2013-06 - ...pdf`. Targeted claim: **MNIST
test error 0.87%** (DLSVM), with the paper's softmax baseline at 0.99%.

This was ReproBot's **first ever `full`-stage run** — the first run long enough
to produce a number comparable to a paper's claim. It collapsed, and nothing in
the pipeline noticed.

## What the full run did

The generated script trained with the paper's stated setup (PCA 784→70, two
512-unit ReLU layers, minibatch 200, SGD with momentum, lr 0.1 linearly decayed
to 0 over 400 epochs, input Gaussian noise σ=1.0 linearly decayed, L2 cost 0.001
on the top layer, L2-SVM squared-hinge loss) plus four values the paper never
states, each disclosed in `coder_output.json`'s `assumptions`:
**momentum 0.9**, **SVM `C` = 1.0**, PyTorch-default init, unwhitened PCA.

Real epoch log, read live from the container (the Runner writes stage logs only
when a stage finishes, and the run was killed mid-way, so this record exists
nowhere else):

| Epoch | train_loss | train_err |
|---:|---:|---:|
| 1 | 1.888 | 27.1% |
| 20 | 3.554 | 83.2% |
| 40 | 3.781 | 85.5% |
| 100 | 3.611 | 89.5% |
| 140 | 3.613 | 89.7% |
| 200 | 3.607 | 89.5% |
| 240 | 3.606 | 89.4% |

Killed at ~epoch 250 of 400. It was unrecoverable (see below), so the remaining
CPU was better spent on the ablation.

## Diagnosis: a dead network

For one-vs-rest squared hinge with `C=1`, a network that **ignores its input**
can do no better than a constant score per class, `s_k = 2p_k − 1`, giving loss
`Σ_k 4p_k(1−p_k)`. With MNIST's real class frequencies:

| | Input-ignoring optimum | Observed |
|---|---:|---:|
| train_loss | **3.599** | **~3.607** |
| train_err | **88.8%** (always predicts "1", the most common digit) | **~89.5%** |

The small gap is the L2 term plus input noise. So: early oversized updates pushed
every hidden ReLU's pre-activation negative, every unit output 0 for every input,
gradients to the hidden layers became exactly zero, and only the output bias
could still learn — which it did, to the constant-guess optimum. **The decaying
learning rate cannot rescue this**: zero gradient times any learning rate is zero.

## The ablation

`ablate.py` imports the model, loss and data loaders **from the actual generated
`train.py`** (so it tests the real reproduction, not a re-implementation), holds
every paper-stated value fixed, and varies only the four unstated guesses, one at
a time. It uses the paper's 400-epoch lr/noise schedule but stops at 30 epochs —
the real run had collapsed by epoch 20. It also measures the fraction of hidden
units that output exactly 0 for every one of 2,000 probe samples. Ran in
`reprobot-runner:latest`, 6 configs in parallel, ~140 s each. Raw per-epoch data:
`ablation.json`.

| Config (one change) | test err @5 | test err @30 | dead fc2 @30 | Outcome |
|---|---:|---:|---:|---|
| **A — as generated** (mom 0.9, C 1.0) | 6.6% | **47.1%** (peak 62%) | 0.71 | unstable |
| **B — momentum 0.5** | 4.0% | **2.62%** | 0.38 | stable |
| **C — momentum 0.0** | 4.2% | **2.66%** | 0.14 | stable |
| **D — C = 0.1** (mom kept 0.9) | 4.4% | **2.80%** | **0.04** | stable, healthiest |
| **E — PCA whitened** | 79.8% | **90.3%** | **0.996** | total collapse |
| **F — init N(0, 0.01)** | 5.0% | 4.2% | 0.31 | stable, slower |

Measured PCA feature scale: unwhitened PC1 std 2.26 / PC70 std 0.32, mean ‖x‖² 46.1;
whitened all std 1.0, mean ‖x‖² 70.0.

## Findings

1. **Dead ReLUs are the mechanism — measured, not inferred.** Config E sits at loss
   3.627, on the input-ignoring optimum, with 99.6% of layer-2 units dead.
2. **Always layer 2.** `dead_fc1` was 0.000 in every config.
3. **It is a combination of two guesses.** Momentum 0.9 amplifies steps ~10×;
   `C=1.0` scales the squared-hinge gradient 10× over `C=0.1`. Reducing *either*
   alone stabilizes training. Each guess is a reasonable standard default on its
   own; together they are too aggressive for this loss at lr 0.1.
4. **Whitening is wrong, and the measured scale predicts why:** it enlarges the
   inputs (‖x‖² 46→70), so it collapses fastest. That supports the Coder's
   literal, unwhitened reading.
5. **The setup is on the edge of stability.** Config A degraded but did not fully
   collapse in 30 epochs; the real run did. They used different noise RNG streams,
   so collapse is seed-dependent — exactly what makes it hard to catch.

## Recommended fix, not yet applied

**`C = 0.1`**, changing nothing else. Healthiest network (4% dead), keeps momentum
0.9 consistent with "SGD *with* momentum", and the paper itself treats `C` as tuned
("selected using validation" in its CIFAR section). Every stated value stays.

Apply as a **controlled experiment**: edit one line of the existing script —
`coder/output/2013-06 - .../train.py` line 174, `svm_C = 1.0` → `0.1` — then run
`uv run python -m runner.pipeline --input "coder/output/2013-06 - Deep Learning
using Linear Support Vector Machines" --mode full` (~28 min CPU). Regenerating via
the Coder instead would risk the model changing other things at the same time.

## What this says about the pipeline

- **Nothing judges whether a number is sane.** Had the run finished, the Runner
  (success = exit code 0) and the Orchestrator would have reported `success` for
  ~89% error against a claimed 0.87%. The Critic is the missing piece.
- **The escalation ladder cannot catch at-scale instability.** `capped` trains ~15
  steps on 512 images and genuinely learned (train err 75%→57%→31% across
  probe→smoke→capped). The collapse needed thousands of steps.
- **The `capped` stage's docstring says "does it actually learn?" but nothing reads
  `train_metric` to check.** It passed on exit code alone.

## Result: the `C = 0.1` rerun replicates the claim (2026-09-13)

The one-line change above was applied to the existing script (edit saved 21:31:18,
container started 21:31:47; nothing else touched) and `full` was run in the Runner.
It **passed in 3223 s** (~54 min — slower than the ~28 min estimate because the
Wijaya run shared the CPU), with no collapse at any point:

| Epoch | lr | train_loss | train_err |
|---:|---:|---:|---:|
| 1 | 0.1000 | 0.1790 | 27.44% |
| 40 | 0.0903 | 0.0673 | 9.81% |
| 100 | 0.0753 | 0.0402 | 5.49% |
| 200 | 0.0503 | 0.0148 | 1.55% |
| 300 | 0.0253 | 0.0050 | 0.27% |
| 400 | 0.0003 | 0.0028 | 0.06% |

**Final: test error 0.82% against the claimed 0.87%** (82 vs 87 misclassified
digits out of 10,000; train error 0.062%). The test set is evaluated once, after
the last epoch — no checkpoint is selected on it.

**How to read it:**
- **Consistent with the claim, not better than it.** For a test error near 0.85%
  on 10,000 images, one standard error of binomial sampling noise is ≈ 0.09
  percentage points; the 0.05-point gap is about half of that. One seed (42).
- **This is not an autonomous result.** A human intervened between the collapse
  and the rerun: the ablation chose `C`. The selection criterion was network
  health (fraction of dead units, loss vs the input-ignoring floor), but the
  ablation *did* log short-horizon test error (epochs 5 and 30), so the choice was
  not made blind to the test set. The honest statement is "the generated
  implementation replicates the claim once one unstated hyperparameter is
  corrected", and the paper itself treats `C` as tuned.
- **What the pipeline would have needed to do this alone:** notice the collapse
  (a Critic, or a Runner that reads `train_metric`), attribute it to unstated
  hyperparameters rather than to code, and search over them — none of which exists.
