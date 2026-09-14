# orchestrator/ — the retry loop

Joins the two stages that could never close the circle on their own: `coder/`
writes a training script but never runs it, `runner/` runs one and classifies its
failure but never rewrites it. This stage owns the **shared-memory object**
(project plan §1.2), drives **generate → run → triage → regenerate**, and decides
when to stop.

**It makes no LLM call of its own.** Every token it spends belongs to a stage it
drives: one Sonnet generation per attempt (`coder/`), one Haiku triage per
genuine failure (`runner/`). §2.1 says the Orchestrator's transitions are
deterministic given the triage category and the retry count, and they are — the
routing table is plain Python.

**It calls the Critic, but is not the Critic.** After a successful run, `critic/`
compares the reproduced number with the paper's claim. The Critic is arithmetic,
with no LLM. Its verdict can trigger extra seeds or one guided fidelity retry. See
"The Critic phase" below and `critic/README.md`.

## The loop

```
                    ┌──────────────────────────────┐
                    │  CODER: generate (feedback)   │
                    └───────────────┬────────────────┘
                       ScriptSyntaxError │ script
                    ◄──────────────┘     ▼
                  (feedback = the         ┌────────────────────────┐
                   parser message,        │  PLATEAU GUARD          │
                   no Docker run)         │  similarity vs. previous │
                                          └──────┬──────────┬────────┘
                                       ≥ threshold│          │below
                                                  ▼          ▼
                                            plateaued   ┌──────────────┐
                                                        │  RUNNER: docker │
                                                        └───────┬──────────┘
                       ┌───────────────┬──────────────┬─────────┴──┐
                       ▼               ▼              ▼             ▼
                    success         timeout    environment_    recoverable_
                       │               │          error           error
                       ▼               ▼            ▼               │
                    SUCCESS         TIMEOUT   ENVIRONMENT_ERROR      │
                                                             retry_count
                                                             < budget?
                                                              │      │
                                                          yes │      │ no
                                                              │      ▼
                                                              │  RETRY_BUDGET_
                                                              │   EXHAUSTED
                                                              ▼
                                              feedback = triage.suggested_fix
                                                       (back to CODER)
```

### The six ways it stops, and why each is separate

| Verdict | Meaning | Why it is not just "failed" |
|---|---|---|
| `success` | a stage exited 0 | — |
| `environment_error` | triage blames the container or its inputs | **This distinction is the point of the triage call.** Regenerating the script cannot fix a broken image, so retrying would spend the whole budget learning nothing. |
| `timeout` | the run blew its wall-clock budget | `runner/` produces **no triage** for a timeout by design — its cause is mechanical. There is no `suggested_fix` to feed back, and telling the Coder to "fix" a script that may be correct but slow would make things worse, not better. |
| `retry_budget_exhausted` | budget spent, still failing | — |
| `plateaued` | the regenerated script is near-identical to the one that just failed | §2.1's explicit requirement. Stops **before** paying for a container that would reproduce the same failure. |
| `untriaged_error` | it failed, but no triage came back | No category means no routing decision. The loop stops rather than guessing which of the two it was. |
| `coder_failed` | the Coder stage itself raised | A stage failure, not a script bug. |

These are **execution** verdicts. They are deliberately not the Critic's
`pass` / `retry` / `fail` vocabulary from §2.5, which judges *numbers*. Keeping
the two apart is what stops a `success` here from ever being read as "the paper
was reproduced" — a `probe` run's numbers come from two optimizer steps.

### The syntax-error shortcut

A `ScriptSyntaxError` from `coder/`'s own `ast.parse` gate is **not** a verdict.
It is a code bug the Coder can fix, it is caught before any container starts, and
the fix costs **no Docker run** — so the parser's message is fed straight back as
feedback and the loop continues. It still consumes one unit of retry budget: a
model emitting invalid Python three times running is not converging either.

### The Critic phase (added 2026-09-14)

When `decide_after_run` says `done`, `_critic_phase` judges the result with
`critic.judge` and routes the verdict through a fourth pure function,
`decide_after_critic`:

| Critic verdict | Condition | Action |
|---|---|---|
| `pass`, `not_evaluated` | — | accept; the loop ends `success` |
| `inconclusive` (one run, no noise estimate) | the full run took ≤ `--seed-budget` (1800 s) and `reproduce.sh` has `seed2\|seed3` | run `seed2` and `seed3` in the same container setup, judge the mean of three, route again (once) |
| `inconclusive` | seeds already ran, or unaffordable | accept as inconclusive |
| `fail` | a fidelity retry is left (`--fidelity-retry-budget`, default 1), the retry budget has room, and the Coder recorded unstated choices | **retry**: regenerate with `guided_retry_feedback`, which covers the gap and the Coder's own `assumptions` as the only things it may change |
| `fail` | otherwise | accept as fail |

The loop's `verdict` stays an **execution** verdict (a fidelity `fail` still ends
`success` if the script ran). The fidelity verdict lives in `critic_output` and on
each attempt as `critic_verdict`. A fidelity retry charges one unit of the normal
retry budget and increments `fidelity_retry_count`. After the retry, the regenerated
script runs through the whole ladder again and is judged again. Its judgement is
appended to `critic_output.judgements`, so a report can show both.

`--no-critic` restores the execution-only loop.

### A halted run retries with its evidence (added 2026-09-14)

`runner/` now watches every stage live and can end one with status **`halted`**:
the script ran, but its learning-curve history showed it was not learning — a loss
below its lower bound, a model worse than chance, a collapse after learning (rules
in `runner/README.md`). A halt is not a verdict. `decide_after_run` routes it like a
fixable script error, except that **no triage call is made**: the check-up's own
evidence — which rule fired, at which epoch, with which numbers, plus advice on what
to check — goes to the Coder verbatim as feedback. It consumes one unit of retry
budget; when the budget is spent the verdict is `retry_budget_exhausted`. The rule
and its message are kept on the attempt as `checkup_rule` / `checkup_message`
(state files written before these fields existed still load).

Verified end to end: a soft decision tree with a deliberately sign-flipped loss was
halted 5 s into `probe` (`loss_below_lower_bound`); the regenerated script, written
with that feedback, trained healthily to `capped` — `success` after one retry, with
no human involved. `clear_stale_metrics` now removes stale history files too, so a
retry is never judged on the previous attempt's curve. A Coder gate rejection for a
missing progress history (`ScriptContractError`) takes the same no-Docker shortcut as
a syntax error.

## Plateau guard

Project plan §2.1, verbatim: *"if the Coder's last two attempts produced
near-identical scripts, escalate to `fail` early rather than exhausting the
budget on a plateaued fix, echoing PaperBench's finding that agents plateau
rather than making steady long-horizon progress."*

```python
difflib.SequenceMatcher(None, normalise_lines(prev), normalise_lines(cur), autojunk=False).ratio()
```

Two mechanics worth knowing, because both are traps:

1. **Lines, not characters.** A line is the unit a code change happens in, and
   it is the same unit `unified_diff` reports — so the ratio and the
   `diff_from_previous` stored beside it describe the same thing. `normalise_lines`
   drops blank lines and *trailing* whitespace; leading whitespace is kept,
   because indentation is semantic in Python.
2. **`autojunk=False`.** `SequenceMatcher`'s default heuristic treats any element
   appearing in more than 1% of a ≥200-element sequence as junk. On a long script
   that silently discards every repeated line and returns a ratio that means
   nothing. Measured: a 400-line script containing 58 identical `    return None`
   lines, with exactly one other line changed, scores **0.9975** with the
   heuristic off — and **0.7625** with it on.

### Reading the threshold

For whole-line replacements the ratio is exactly `1 - changed_lines /
total_lines` (verified across 0/1/4/5/40 changed lines out of 200). So the
default **0.98 means "2% or fewer of the lines changed"**, not "two lines
changed" — on a 200-line script that is 4 lines, on a 430-line one about 8.

That is calibrated for **this** Coder, which regenerates the whole file from
scratch every attempt: two independent Sonnet generations of a ~430-line script
differ far more than that even when converging, so scoring above 0.98 means the
model reproduced its previous output almost verbatim. **If the Coder ever becomes
patch-based this threshold is far too aggressive** — a correct one-line fix in a
430-line file scores ~0.998 and would be called a plateau — and it should move
toward 0.999.

0.98 is a starting point, not a measurement. That is why it is a constructor
parameter (`--plateau-threshold`) and why **every comparison logs its actual
ratio**, so it can be tuned from real runs rather than argued about forever.
The one real data point so far, from the verified repair run below — a genuine
regeneration that fixed the bug it was told about:

```
[plateau] v1 vs v2 line similarity 0.4079 (stop at >= 0.98)
[plateau] the regenerated script differs enough to be worth running (similarity 0.4079 < threshold 0.98)
```

0.4079 against a 0.98 threshold is an enormous margin, which is the useful
finding: a whole-file regeneration lands nowhere near the plateau line even when
it is making a small, targeted fix. The threshold has plenty of room to be
raised; it has no room to be lowered.

## `state.py` — the shared-memory object

Project plan §1.2 made concrete. One `ReproState` per paper, written to
`orchestrator/output/<paper>/state.json`.

```
ReproState
├── paper_id           "2016-05 - Wide Residual Networks"
├── source_pdf         dataset/<paper>.pdf, or null (provenance only)
├── reader_output      the whole reader/output/<paper>.json, embedded
├── coder_output       { script_path, script_version, diff_from_previous }
├── runner_output      { status, reproduced_metrics, logs_path, error_trace }
├── critic_output      latest Critic judgement + seed_run_values + all judgements, or null
├── retry_count        regenerations spent
├── retry_budget       regenerations allowed
├── fidelity_retry_count  guided retries spent after a Critic fail
├── history            append-only [{ timestamp, stage, event, detail }]
├── verdict            the loop's terminal answer
└── attempts           per-attempt record (see below)
```

§1.2's own words for why this is one object rather than scattered state: it is
what lets the Orchestrator answer *"have we already tried this fix"* and lets a
future Report Generator produce a claim-by-claim table without re-deriving
anything.

Four notes on fidelity to the plan:

- **`critic_output` is `null` until the Critic judges an attempt.** It was reserved
  from the start so the schema did not change shape when the Critic landed. It holds
  the latest judgement's fields (verdict, claimed, reproduced, tolerance, evidence,
  merged claim ids, …) plus `script_version`, `seed_run_values` and the list of every
  `judgements` entry. State files written before the Critic still load.
- **`history` entries are structured, not prose strings.** §1.2 sketches them as
  free text. Splitting each into `timestamp` / `stage` / `event` / `detail` is
  the same append-only audit log with its fields separated, which is what makes
  "have we already tried this" a query instead of a substring search.
- **`attempts` and `verdict` are additions.** `history` is the narrative;
  `attempts` is the record the loop actually routes on — which version ran, what
  feedback produced it, what the Runner said, how triage classified it, how
  similar it was to its predecessor.
- **`runner_output` carries only §1.2's four fields.** `runner/`'s own
  `RunnerOutput` has far more (per-stage timings, container names, an 8000-char
  excerpt). Copying it all in would make `state.json` unreadable and duplicate a
  file that already exists per attempt; the fields the loop routes on are lifted
  into `AttemptRecord` and `logs_path` points at the rest.

### `script_version` and `diff_from_previous` are live now

`coder/pipeline.py` has always emitted these two **inert** (`1` / `null`) —
a single-shot stage has no previous version to diff against. The loop populates
them for real: the version increments per attempt, and `diff_from_previous` is a
`difflib.unified_diff` against the previous attempt's script, capped at 8000
chars with an explicit `...[N chars of diff truncated]...` marker. Unlike the
similarity ratio, the diff is taken over the **raw** lines — it is an audit
artifact of what changed on disk, so a whitespace-only edit should show up in it.

### Every attempt's script is kept

```
orchestrator/output/<paper>/
├── state.json
├── attempts/
│   ├── v1_train.py            # exactly what ran on attempt 1
│   ├── v2_train.py
│   └── v3_train.py.invalid    # a syntax-gate rejection is archived too
└── logs/
    ├── attempt-1/probe.stdout.log   # full, untruncated
    ├── attempt-1/probe.stderr.log
    └── attempt-2/...
```

`coder/output/<paper>/train.py` remains the **current** script, overwritten each
attempt — that is the file `reproduce.sh` runs. The archive is what makes a
plateau claim checkable after the fact instead of something you have to take the
ratio's word for, and it is why per-attempt logs go in separate directories
rather than letting attempt 2 overwrite attempt 1's.

## Class architecture

```
orchestrator/state.py
┌──────────────────────────────────────────────┐
│  ReproState  (+ CoderOutputState,               │
│                RunnerOutputState,                │
│                AttemptRecord, HistoryEntry)       │
│    record(stage, event, detail)                    │
│    to_dict / from_dict / save / load                │
│  Pure data. Imports no stage, no SDK, no docker.     │
└──────────────────────────────────────────────┘

orchestrator/loop.py
┌──────────────────────────────────────────────┐
│  PURE:  normalise_lines / script_similarity      │
│         make_diff / feedback_from_triage          │
│         decide_after_run                           │
│         decide_after_syntax_error                   │
│         decide_after_regeneration   -> LoopDecision   │
│  Orchestrator(coder, runner, modes,                    │
│               retry_budget, plateau_threshold)          │
│    run(...) -> ReproState                                │
└──────────────────────────────────────────────┘

orchestrator/pipeline.py
┌──────────────────────────────────────────────┐
│  discover_reader_outputs / run_dataset / main   │
│  Argument parsing, skip-if-done, per-paper       │
│  try/except, state writing. The only entry point. │
└──────────────────────────────────────────────┘
```

Same split as `coder/` and `runner/`: the pipeline owns I/O and the CLI, the
worker module owns the mechanics, and neither reaches into the other's job. The
Orchestrator itself owns **no** mechanics — `CoderPipeline` writes the script and
runs its gates, `DockerRunner` executes it and classifies the failure, and this
class only decides what happens next and records it.

## Plain Python, not LangGraph — and why that is a decision, not an omission

Project plan §1.3 recommends **LangGraph** for exactly this loop, and that
recommendation is right *eventually*. It is deliberately not taken yet.

What §1.3 argues LangGraph buys: explicit, inspectable transitions for the
Critic→Coder retry edge, the budget-exhausted→fail edge and the
pass→Report-Generator edge, plus a retry loop testable without invoking an LLM.
At this slice the graph is **three nodes and one conditional edge**, and both of
those benefits are already in hand:

- The transitions are explicit — they are three pure functions returning a
  `LoopDecision`, which is more inspectable than a graph edge, not less.
- The loop is already testable without an LLM: the whole routing table is
  verified against synthetic input with no Docker and no API key (see Status).

Against that, `CLAUDE.md` is explicit that graph machinery must not be added
preemptively, and a framework here would mean a heavy dependency, a state-schema
translation layer between `ReproState` and a graph state, and indirection around
a loop a reader can currently follow top to bottom in one function.

**What makes the migration cheap when it is worth doing.** Two properties are
maintained on purpose: `ReproState` stays exactly §1.2-shaped and imports no
framework, and every transition decision lives in a pure function that takes
primitives and returns a verdict. A LangGraph node then wraps one stage call, and
a conditional edge calls `decide_after_run` and switches on `decision.action`.
Nothing about the routing rules would have to be rewritten.

**When to do it:** the Critic has now landed and added a second conditional edge
(inconclusive → seeds, fail → guided retry). It is still four pure functions and one
loop a reader can follow top to bottom, so the migration is still not worth its
cost. Revisit when the Report Generator and the Reader re-parse edge (§1.1) arrive,
or when a human-escalation edge needs to pause and resume a run.

## What this deliberately does not do

**No fidelity verdict on cheap stages.** The Critic judges only a `full` run. The
escalation modes `probe`, `smoke` and `capped` are execution gates, not
reproductions: a `probe` is two optimizer steps, and its `test error` of ~92%
against a claimed 4.00% says nothing except that the script ran. A run that stops
below `full` is `not_evaluated`. Judging it would give a confident but meaningless
verdict, which is the failure mode §2.5 cites Agent Laboratory for.

**No open-ended fidelity tuning.** One guided retry per paper, on unstated choices
only. More retries would slowly fit unstated hyperparameters to the test split.

Also not built: escalation to a human on `fail` (§2.1), the Reader re-parse edge
(§1.1), and the Report Generator (§2.6) that consumes this state object.

## Logging

Detailed by design, via `loguru` (not `print`) — the retry budget and threshold
at startup, each attempt's number and remaining budget, the exact feedback text
handed to the Coder, where each version was archived, the diff size recorded,
**the plateau ratio for every comparison**, every stale metrics file removed, the
Runner's verdict and triage category, and the terminal verdict with the reason it
was reached. A loop that ends in the wrong place should be diagnosable from
console output alone. The stages it drives keep their own instrumentation.

## Setup

```bash
uv sync --extra orchestrator --group dev
cp .env.example .env   # ANTHROPIC_API_KEY — required: the Coder runs every attempt
```

Docker Desktop (macOS) or the docker engine (Linux) must be running, and
`runner/`'s image must exist (`--build` builds it).

```bash
# one paper: generate, run, repair, up to 3 regenerations
uv run python -m orchestrator.pipeline \
    --input "reader/output/2016-05 - Wide Residual Networks.json"

# the cheap loop — prove the script starts at all, two retries, reuse the image
uv run python -m orchestrator.pipeline --input reader/output \
    --max-stage probe --retry-budget 2 --no-build

# re-drive the loop over a script you already have, without paying to regenerate it
uv run python -m orchestrator.pipeline --input reader/output/paper.json \
    --use-existing-script

# single-shot (generate + run, never retry), and a stricter plateau guard
uv run python -m orchestrator.pipeline --input reader/output \
    --retry-budget 0 --plateau-threshold 0.999

# a CPU-sized paper to its full setup, judged by the Critic (seeds if <= 30 min)
uv run python -m orchestrator.pipeline --input reader/output/paper.json \
    --max-stage full --seed-budget 1800 --fidelity-retry-budget 1

# execution only, no fidelity judgement
uv run python -m orchestrator.pipeline --input reader/output/paper.json --no-critic
```

`--retry-budget` counts **regenerations**, not attempts: 2 means at most attempt
1 plus two retries. `--use-existing-script` skips generation for attempt 1 only;
every retry regenerates normally. Papers with an existing `state.json` are
skipped unless `--force` is given.

Stage flags (`--max-stage` / `--mode` / `--image` / `--build` / `--cache-dir` /
`--timeout` / `--memory` / `--cpus` / `--network`) are passed straight through to
`runner/`'s `DockerRunner` and mean exactly what `runner/README.md` says they
mean. There is deliberately **no `--no-triage`**, unlike `runner/`'s CLI: the
triage category *is* this stage's routing decision, so disabling it would end
every failure as `untriaged_error` on the first attempt and the loop would never
retry anything.

## Dependency choice

`orchestrator` is its own `pyproject.toml` extra (`anthropic` +
`python-dotenv`), identical in shape to `reader`'s, `coder`'s and `runner`'s, and
torch-free on the host. It makes no API call itself — the SDK is there because it
constructs the client the two stages it drives use.

## Status

**Verified end to end against a real repair, in Docker, with real API calls.**
The test that matters is not "does the loop run" but "does it fix a broken
script", so a *runtime* bug that `ast.parse` cannot catch was deliberately
injected into the working Wide Residual Networks script — the historical
`self.optimizer`-never-assigned defect documented in `coder/README.md`:

```python
class _StepDecayScheduler:
    def __init__(self, optimizer, milestones, gamma):
        self._milestones = set(milestones)     # note: no self.optimizer
    def step(self):
        for group in self.optimizer.param_groups:   # AttributeError at run time
```

`self.optimizer` read twice, assigned zero times; `ast.parse` passes.

**It was repaired on the first retry.** The real run, end to end:

```
[attempt 1] retry_count=0/2, no feedback (first attempt)
  [coder] reusing the existing script ... (369 lines)
  [probe] FAILED in 169.5s (exit 1)
    AttributeError: '_StepDecayScheduler' object has no attribute 'optimizer'
  [triage] classifying failure of stage 'probe' (exit 1) with claude-haiku-4-5...
  [triage] category: recoverable_error
  [triage] suggested fix: In the `_StepDecayScheduler` class definition (around
           line 285), ensure that the `__init__` method stores the optimizer
           instance. ... assigns the passed optimizer to `self.optimizer`
           (e.g., `self.optimizer = optimizer`).
  [decide] RETRY - triage says the fault is in the generated script
[attempt 2] retry_count=1/2, feedback: In the `_StepDecayScheduler` class ...
[training_script] regenerating with feedback: In the `_StepDecayScheduler` ...
  [archive] v2 -> .../attempts/v2_train.py
  [diff] v1 -> v2: 193 diff line(s), 8038 chars recorded in state.coder_output
  [plateau] v1 vs v2 line similarity 0.4079 (stop at >= 0.98)
  [probe] metrics (file): claim_id=c34 test error=92.1875 %
  [probe] PASSED in 2000.8s (exit 0)
[loop] VERDICT: success
[loop] finished after 2 attempt(s), 1/2 retries used
```

The regenerated script did not merely happen to be correct — it addressed the
feedback by name, keeping the class the triage named and fixing exactly what it
was told to:

```python
class _StepDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.2, last_epoch=-1):
        ...
        # NOTE: torch.optim.lr_scheduler._LRScheduler.__init__ already assigns
        # `self.optimizer = optimizer`; we keep it explicit here for clarity and
        # to guarantee self.optimizer.param_groups is always accessible ...
        self.optimizer = optimizer
```

The resulting `state.json`: `verdict: "success"`, `retry_count: 1`,
`retry_budget: 2`, `critic_output: null`, `coder_output.script_version: 2` with a
real 8038-char `diff_from_previous`, two `attempts` (v1 `error`/`recoverable_error`/
`action: retry`, v2 `success`/`action: done`/`plateau_ratio: 0.4079`), and a
7-entry `history` running orchestrator → coder → runner → orchestrator → coder →
runner → orchestrator.

### The Critic phase, end to end (2026-09-14)

Wijaya 2023 (Boston Housing, claim RMSE 3.02) with `--max-stage full --force`:

1. One Sonnet generation passed all three Coder gates.
2. `probe`, `smoke` and `capped` passed healthy. `full` ran 1000 epochs in 182 s with
   101 live progress records and reported RMSE **3.75**.
3. The Critic merged claims c8 and c11 and judged the run **`inconclusive`**: one
   RMSE, no noise model.
4. `decide_after_critic` routed it to **`run_seeds`**, because 182 s is within the
   1800 s seed budget and the new `reproduce.sh` has seed modes.
5. `seed2` gave **2.84** and `seed3` gave **3.40**, 119 s and 118 s. Each seed also
   redraws the paper's unstated split.
6. The Critic re-judged the three runs: mean **3.33**, tolerance ±1.07 from their
   spread, **`pass`**.
7. The decision was `accept`, loop verdict `success`, 0 retries.

`critic_output` holds the judgement with `run_values`, `seed_run_values` and
`script_version: 1`, and attempt 1 carries `critic_verdict: "pass"`. The guided
fidelity retry (`fail` → retry) has not happened in a live run yet. It is covered by
`tests/test_loop_critic.py` only.

### Verified

- **The full repair chain, for real:** broken script → `probe` fails
  (`AttributeError`) → Haiku triage returns `recoverable_error` with a concrete
  `suggested_fix` → the Coder regenerates **with that text in its prompt** → the
  new script's `probe` passes. Verdict `success`, `retry_count: 1`, 2 attempts.
- `ruff check`, `ruff format --check`, `mypy --strict orchestrator/ coder/` and
  `pre-commit run --all-files` all pass.
- **118 assertions with no Docker and no API key**, by injecting fake
  Coder/Runner objects: every branch of the routing table (success,
  `environment_error` stopping immediately *with budget left*, `environment_error`
  beating budget exhaustion, budget exhaustion at and past the limit,
  `--retry-budget 0`, timeout stopping even when a triage exists, missing triage,
  unknown status raising, empty `suggested_fix` falling back to the reasoning);
  the plateau maths at, above and below the threshold, the exact
  `1 - changed/total` property, the `autojunk` trap, and the threshold being
  honoured rather than hard-coded; diff generation including truncation and the
  identical-scripts marker; `ReproState` round-tripping through JSON on disk with
  §1.2's field sets asserted; and the loop end to end for all seven outcomes —
  including that a plateau spends **no** second container, that a syntax error
  retries **without** a Docker run, that `environment_error` never calls the
  Coder again, and that attempt 1's archived script is not overwritten by
  attempt 2.

### Known limitations

- **One paper, one injected bug.** The loop is verified to repair *this* failure.
  Whether it converges on a bug the Coder itself produced repeatedly is unknown —
  and the plateau guard exists precisely because it might not.
- **The retry regenerates the whole script, it does not patch it.** The feedback
  goes into the prompt (`## Feedback on your previous attempt`), and the model
  writes a fresh file. A regeneration can therefore fix the named bug and
  introduce a different one; nothing here diffs semantics, only text.
- **The plateau threshold has one real data point** (0.4079 for a genuine
  regeneration). It has never fired on a real run.
- **`environment_error`, `timeout`, `untriaged_error` and `coder_failed` are
  verified against fakes only.** No real container has produced one.
- **A regeneration can silently lose the shared dataset cache, and did.**
  Attempt 1's `probe` took 169 s; attempt 2's took **2000 s** — and 1740 s of
  that was re-downloading CIFAR-10 at ~94 kB/s. The regenerated script defaulted
  its data directory to `./wrn28-10-cifar10-output/data` instead of `./data`, so
  the `runner/cache/datasets → /workspace/data` mount simply did not apply to it.

  `runner/README.md` calls this out as a known property (*"`./data` is a
  convention, not a contract… the mount is simply unused and that paper's
  download lands in its own mount — slower, never broken"*), and it behaved
  exactly as documented: slower, not broken. But it is materially worse in a
  **loop** than in a single-shot run, because every regeneration is a fresh roll
  of the dice on that path, and a 12× stage slowdown eats wall-clock budget that
  was calibrated on a warm cache. Two options if this recurs: add `--data-dir` to
  `REQUIRED_CLI_FLAGS` and pin it in `reproduce.sh`, or mount the cache at both
  paths. Neither was done here — `runner/` is out of scope for this stage, and
  one observation is not yet a pattern.
