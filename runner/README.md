# runner/ — sandboxed execution

Takes one paper's generated `coder/output/<paper>/` directory and **executes it
inside a Docker container**, escalating through cheap gates until something
fails or the requested depth is reached. Captures the logs, reads the metrics
back out, and — only when a run genuinely fails — spends one Claude Haiku call
classifying the failure.

**This stage never imports torch and never runs a training script on the host.**
Everything executes inside the image; on the host it shells out to the `docker`
CLI, so its own dependencies are just `anthropic` + `python-dotenv`.

## The interface is `reproduce.sh`, not CLI flags

This is the central design decision of the stage.

The only command ever sent into a container is:

```bash
bash reproduce.sh <mode>          # mode ∈ probe | smoke | capped | full
```

The Runner **never** constructs a `python` command, **never** passes a `--flag`,
and **never** needs to know what arguments a generated script accepts. That is
what keeps it paper-agnostic: a future paper whose script takes entirely
different arguments changes only its own `reproduce.sh`, and nothing in this
folder. The contract narrows from "eight CLI flags spelled exactly right" to
"four mode names".

Mode semantics come from the script's own header (see `coder/README.md`):

| Mode | What it costs | What it proves |
|---|---|---|
| `probe` | a couple of optimizer steps, seconds | data → model → loss → step works at all; catches shape/dtype errors |
| `smoke` | one full epoch over a small slice | execution reaches the eval path and the metrics write |
| `capped` | 5 epochs / 512 samples, minutes | training actually learns — train accuracy climbing above CIFAR-10's 10% chance |
| `full` | the paper's real setup, hours | the only numbers comparable to the paper's claim; needs a GPU |

Each mode writes its **own** `metrics.<mode>.json`, so a cheap stage's numbers
can never be mistaken for a real run's.

> One sharp edge worth knowing: `reproduce.sh` with **no** argument defaults to
> `full`. The Runner always passes an explicit mode, and the image's default
> `CMD` passes `probe`, so neither can start a multi-hour run by accident.

## Escalating gates

```
      probe ──ok──► smoke ──ok──► capped ──(only if asked)──► full
        │             │             │
      exit≠0        exit≠0        exit≠0
        ▼             ▼             ▼
             stop, triage, record
```

Stages run in order and stop at the **first** non-zero exit — a shape error
costs seconds instead of hours. `full` is never reached by escalation; the
default `--max-stage` is `capped`, and `full` has to be named explicitly.

## Class architecture

```
runner/triage.py
┌──────────────────────────────────────────────┐
│  triage_failure(exit_code, failed_stage,        │
│                 log_excerpt, client)              │
│    -> TriageResult(category, reasoning,            │
│                    suggested_fix)                   │
│  A plain function. One fixed step, one call site,    │
│  so an ABC would be abstraction nobody uses.          │
└──────────────────────────────────────────────┘

runner/docker_runner.py
┌──────────────────────────────────────────────┐
│  DockerRunner(image, cache_dir, stage_timeouts, │
│               memory, cpus, network, run_triage) │
│    check_daemon() / ensure_image(build)           │
│    run_stage(paper_dir, logs_dir, mode) -> StageRun │
│    run_paper(...) -> RunnerOutput                    │
│  Everything that touches the docker CLI lives here.   │
└──────────────────────────────────────────────┘

runner/pipeline.py
┌──────────────────────────────────────────────┐
│  discover_paper_dirs / run_dataset / main       │
│  Argument parsing, skip-if-done, per-paper       │
│  try/except, output writing. The only entry point. │
└──────────────────────────────────────────────┘
```

Same split as `coder/`: the pipeline owns I/O and the CLI, the worker module owns
the mechanics, and neither reaches into the other's job.

## The timeout trap

The single easiest thing to get wrong in this stage, so it is handled explicitly:

**`subprocess.run(timeout=...)` kills the `docker run` *client*, not the
container.** The container is a child of dockerd, not of the Python process. Kill
the client and the container keeps running — burning CPU, holding the bind
mounts, and writing into the paper directory — indefinitely.

So every run:

1. gets an explicit `--name`, chosen **before** launch (`container_name_for()`),
   because after the client dies the name is the only handle left on it;
2. on `TimeoutExpired`, issues `docker kill <name>`, treating "No such container"
   as success (with `--rm` the daemon may already have reaped it) and falling
   back to `docker rm --force`;
3. carries a **per-stage** budget, enforced independently, so a hung `probe`
   cannot eat the budget of the `capped` run behind it.

Defaults, overridable with a repeatable `--timeout MODE=SECONDS`:

| Stage | Budget | Why |
|---|---|---|
| `probe` | 1200 s | 2 optimizer steps, plus the cold-cache CIFAR-10 download (~170 MB) that this stage pays for everyone |
| `smoke` | 2400 s | 1 epoch over 512 samples plus an eval pass |
| `capped` | 7200 s | 5 epochs over 512 samples on CPU |
| `full` | 86400 s | the paper's real setup; only ever run deliberately |

A second, subtler trap is handled too: on the timeout path `TimeoutExpired`
carries whatever partial output it had collected as **raw bytes even in text
mode**. `decode_stream()` normalizes it, so a timeout cannot crash the handler
that is trying to report the timeout.

## The mounts — and how results get out

```
coder/output/<paper>/  ──►  /workspace        (rw)   ← metrics.<mode>.json written here
runner/cache/datasets  ──►  /workspace/data   (rw)   ← shared CIFAR-10, nested on purpose
runner/cache/home      ──►  /cache            (rw)   ← HOME/HF_HOME/TORCH_HOME
```

**There is no `docker cp`.** The paper directory mount *is* the host directory,
so when the script writes `metrics.<mode>.json` the file is already on the host.
Read-write is not optional for that mount — it is the entire results channel.

The CIFAR-10 cache is the interesting one. The generated scripts default
`--data-dir` to `./data` relative to the script, and `reproduce.sh` never
overrides it, so a naive setup downloads 170 MB **per paper**. Mounting a shared
host directory at `/workspace/data` — nested inside the `/workspace` mount, which
Docker allows because it orders mounts by path depth — turns that into one shared
copy across all eight papers.

That nesting depends on `./data`, which is a **convention, not a contract**: it
is the default `coder/`'s prompt happens to produce, and it is not in
`REQUIRED_CLI_FLAGS`. If a future paper's script uses a different directory name,
the mount is simply unused and that paper's download lands in its own
`/workspace` mount — slower, never broken. Nothing detects or depends on it.

`runner/cache/` and `runner/output/` are both gitignored.

## Resource limits — the decision, and why

`--memory` and `--cpus` are **exposed as flags but unset by default**, and that
is deliberate rather than an oversight.

A CPU-only CIFAR-10 run's memory is dominated by the dataset plus activations,
and it varies with the paper's batch size — a number this stage does not know,
because it deliberately never reads the script's arguments. An arbitrary cap
would produce an **exit code 137** (OOM-killed), which looks exactly like a
crash in the generated script and would be triaged as one, sending the Coder off
to fix a bug that does not exist. Costing a false `recoverable_error` is worse
than an uncapped container on a developer machine that is running one paper at a
time.

The flags exist for the cases where the trade-off flips — a shared CI box, or
several papers in parallel — and `triage.py`'s prompt names exit 137 explicitly
as an `environment_error` so a deliberate cap is still classified correctly.

`--network` stays on `bridge` by default because torchvision downloads CIFAR-10
on the first run. Once the cache is warm, `--network none` makes the sandbox
fully offline.

**Known limitation:** the container runs as root, so on a **Linux** host the
files it writes into the bind mounts (`metrics.*.json`, the CIFAR-10 cache) come
out root-owned. On macOS, Docker Desktop's filesystem mapping hides this and the
files are owned by the host user. Adding `--user $(id -u):$(id -g)` would fix it
but needs a writable `HOME` inside the container and cannot be verified here
without a daemon, so it was deliberately not added blind.

## Log capture and truncation

Both streams are captured separately and the **full, untruncated** logs always go
to disk:

```
runner/output/<paper>/logs/<mode>.stdout.log
runner/output/<paper>/logs/<mode>.stderr.log
```

Separately, a `log_excerpt` of roughly **8000 characters** is built for the
triage/Critic prompt — head-and-tail, with an explicit `...[N chars
truncated]...` marker carrying the exact count:

| Stream | Head | Tail |
|---|---|---|
| stdout | 800 | 2400 |
| stderr | 1200 | 3600 |

The 1:3 head:tail ratio is the point. The head catches an import or setup failure
that aborts before anything else prints; the tail catches the traceback; the
middle — epoch after epoch of progress lines — is the part nothing is learned
from.

**Each stream is truncated *before* combining**, which matters more than it
looks. Concatenate first and a chatty stdout (one progress line per epoch) can
push stderr's traceback entirely out of the window — exactly the thing the
excerpt exists to preserve. This is verified against a 150k-char stdout in the
checks below.

`error_trace` is the **last** traceback in stderr (a `raise ... from` chain
prints the original first; the final one is what actually ended the process). If
there is no traceback at all — an OOM kill, a signal, a shell error before Python
started — the tail of stderr is returned instead, since it is still the best
evidence available.

## Reading the metrics back

After each stage, `metrics.<mode>.json` is read out of the bind mount and parsed
against the contract documented in `coder/README.md` — `claim_id`, `metric`,
`unit`, `value` are copied verbatim by the generated script and are never
normalized here either.

If the file is missing, there is a fallback: the generated script also prints the
identical object as its final line of stdout, so `parse_metrics_from_stdout()`
scans upward for it. The scan requires a contract key (`claim_id` / `metric` /
`value`) before accepting a line, so HuggingFace `Trainer`'s own
`{'loss': 2.30, 'epoch': 1.0}` progress lines — Python reprs with single quotes,
not valid JSON — cannot be mistaken for the metrics.

A stage that exits 0 without producing metrics is **not** an error: `probe` may
exit before reaching the write. It is logged as a warning and recorded with
`metrics: null`.

## Triage — one Haiku call, only when it earns itself

```
stage exit 0        ─► success   ─► NO API call
stage timed out     ─► timeout   ─► NO API call
stage exit non-zero ─► error     ─► one claude-haiku-4-5 call
```

Success and timeout are decided **mechanically**. That is deliberate: a zero-exit
run needs no explanation, and a timeout's cause is already "it did not finish in
N seconds", which no model can improve on. The common path — everything passes —
costs nothing.

On a genuine failure, `triage_failure()` sends only the three things that carry
signal: the exit code, which stage failed, and the truncated log excerpt. The
generated script's source is **not** sent — the traceback already names the
failing line, and shipping several hundred lines of code would cost far more than
the classification is worth.

It returns one of two categories, which is the routing decision the future
Orchestrator needs:

- **`recoverable_error`** — the fault is in the generated script; a rewrite could
  fix it. `suggested_fix` is written to be handed verbatim to
  `CodeWriter.write(feedback=...)`.
- **`environment_error`** — the fault is in the container or its inputs;
  regenerating the script would change nothing.

The prompt calls out the genuinely ambiguous case explicitly: a missing import is
an `environment_error` if the image lacks the package, but a `recoverable_error`
if the script imports a name that does not exist inside an installed one.

The category is re-validated on parse even though the tool schema declares an
`enum` — the schema is a strong hint to the model, not an SDK-enforced guarantee,
and a silently mislabelled category would misroute the entire retry decision.

## The Dockerfile

`python:3.11-slim`, **not** `pytorch/pytorch`: the official PyTorch images are
CUDA-based and carry the CUDA runtime, cuDNN and NCCL layers — several GB that a
CPU-only CIFAR-10 gate run never touches. CPU-only wheels come from
`--index-url https://download.pytorch.org/whl/cpu` (`--index-url`, not
`--extra-index-url`, so a CUDA build from PyPI can never be resolved instead).

**This stage is the platform trap's solution, not another instance of it.**
`CLAUDE.md` and `ocr/README.md` document the host constraint: Intel Mac + Python
3.13, and no PyTorch release ships Intel-macOS wheels and 3.13 support at the same
time. None of that applies *inside* the container, which runs its own Linux
CPython 3.11 against Linux wheels. The host's OS, interpreter and lockfile are
irrelevant to it — which is exactly why the generated scripts can finally be run.

**Nothing generated is copied in.** There is no `COPY train.py`; the paper
directory is bind-mounted at `docker run` time. The image stays completely
static, so the Coder can regenerate every script in the repo without rebuilding
it or invalidating a single layer.

Versions are pinned so a rebuild months from now reproduces the sandbox a
recorded run used — an unpinned image would silently change what "the script
failed" means. `torch` is pinned as `2.5.1`, **not** `2.5.1+cpu`: the CPU index
tags the x86_64 wheel with the `+cpu` local version but the aarch64 one without
it, so the bare pin resolves on both the Intel dev machine and an Apple Silicon
collaborator's.

## Logging

Detailed by design, via `loguru` (not `print`) — the daemon version, the build,
the exact `docker run` argv for every stage, the container name, the budget, the
captured byte counts and where the full logs went, the parsed metrics field by
field, each stage's verdict and elapsed time, the escalation stop, and the
triage category, reasoning and suggested fix. A run that fails should be
diagnosable from console output alone.

## Setup

```bash
uv sync --extra runner --group dev
cp .env.example .env   # ANTHROPIC_API_KEY — only needed if a run fails
```

Docker Desktop (macOS) or the docker engine (Linux) must be running. There is no
Python docker SDK dependency — this stage shells out to the CLI.

```bash
# one paper, escalating probe -> smoke -> capped, building the image first
uv run python -m runner.pipeline --input "coder/output/2016-05 - Wide Residual Networks"

# every generated paper, reusing an already-built image
uv run python -m runner.pipeline --input coder/output --no-build

# just the cheapest gate
uv run python -m runner.pipeline --mode probe

# tighter budgets, capped resources, offline (warm cache), no API call at all
uv run python -m runner.pipeline --timeout probe=300 --timeout smoke=900 \
    --memory 8g --cpus 4 --network none --no-triage

# the paper's real setup — needs a GPU and hours; never reached by default
uv run python -m runner.pipeline --input coder/output/<paper> --max-stage full
```

`--max-stage` and `--mode` are mutually exclusive: the first escalates up to a
stage, the second runs exactly one. Papers with an existing `runner_output.json`
are skipped unless `--force` is given.

Output per paper (gitignored):

```
runner/output/<paper>/
├── runner_output.json         # status, stages, metrics, excerpt, triage
└── logs/
    ├── probe.stdout.log       # FULL, untruncated
    ├── probe.stderr.log
    └── ...
```

`runner_output.json` carries the top-level verdict (`status`, `stage_reached`,
`failed_stage`, `exit_code`, `wall_clock_seconds`, `reproduced_metrics`,
`logs_path`, `log_excerpt`, `error_trace`, `triage`) plus a `stages` list with
the same detail per stage — so the Critic can compare against the claim while the
Orchestrator sees exactly how far the escalation got.

## Dependency choice

`runner` is its own `pyproject.toml` extra (`anthropic` + `python-dotenv`),
identical in shape to `reader`'s and `coder`'s and, importantly, **torch-free**
on the host. torch exists only inside the image. That is what lets this stage
stay in the managed `uv.lock` on the Intel-macOS dev machine while being the
thing that finally executes torch code.

## Status

**Docker Desktop was blocked on a macOS privileged-access password dialog for
the entire session, so `docker build` and `docker run` could not be exercised.**
Following `ocr/README.md`'s precedent for Docling/MinerU, here is exactly what is
verified and what is written-but-unrun.

### Verified

- **The generated script genuinely runs, and the stage budgets below are
  calibrated against that run rather than estimated.** `reproduce.sh probe` was
  executed natively on the dev machine (outside Docker, in a throwaway Python
  3.11 venv — see CLAUDE.md's Tooling section for the narrow legacy pin-set that
  makes that possible on an Intel Mac). It completed with **exit 0** and wrote a
  `metrics.probe.json` matching the documented contract exactly, all twelve keys:

  ```
  train_runtime             48.7 s   (256 samples, 1 epoch, 2 optimizer steps)
  train_samples_per_second   5.256
  eval passes               ~6.4 s (128 eval) + ~12.2 s (256-sample train-acc)
  total compute             ~67 s
  total wall clock          1774 s   <- the other ~1707 s was the CIFAR-10 download
  ```

  Two things follow, and both changed the code. First, the cold dataset fetch
  dominates everything else, so `probe`'s budget has to tolerate it (the earlier
  1200 s guess would have timed out on the very first run, looking exactly like a
  hung script) and the shared cache mount is load-bearing, not an optimisation.
  Second, extrapolating 5.256 samples/s to `full` — 50,000 samples × 200 epochs —
  gives roughly **22 days** on this CPU, which settles any ambiguity about
  whether `full` is GPU-only.

  What it does *not* verify: any of this happening **inside a container**. The
  script ran on the host. Everything Docker-shaped below is still unproven.
- `ruff check`, `ruff format --check`, `mypy --strict runner/`, and
  `pre-commit run --all-files` all pass.
- **110 assertions** over every daemon-free function, against synthetic input:
  log truncation (boundary cases, exact dropped-char accounting, the
  chatty-stdout property proved against a 150k-char stdout), traceback
  extraction (last-of-a-chain, the no-traceback fallback), metrics parsing
  (contract object recovered; HF `Trainer`'s single-quoted repr and unrelated
  JSON both correctly rejected; invalid/array/missing files), stage escalation,
  `--timeout` parsing and all four of its rejection paths, container-name
  legality against Docker's name grammar, `docker run` argv assembly, byte/str
  stream decoding, paper discovery, and triage payload parsing.
- **`run_stage` and `run_paper` end-to-end with `subprocess.run` monkeypatched**,
  which needs no daemon and covers: exit-0 success with metrics read from the
  mount, full untruncated logs written to disk, the **timeout path issuing
  `docker kill` against the exact `--name` it launched with**, partial raw-bytes
  output surviving decode, escalation stopping before `capped` when `smoke`
  fails, triage firing on failure with `suggested_fix` captured, **no API call
  on either success or timeout**, `--no-triage` suppressing it, and
  `runner_output.json` round-tripping through `asdict`/`json`.
- **The `reproduce.sh` interface, against the real generated file** for Wide
  Residual Networks: `bash -n` syntax check passes; its four `case` arms are
  `probe/smoke/capped/full`, matching `STAGE_ORDER` exactly; an unknown mode
  exits 2 with a usage message without reaching `python`.
- **The Dockerfile structurally**: all 9 logical instructions parse and use valid
  keywords, base is `python:3.11-slim`, no `COPY`/`ADD` of generated code, torch
  from the CPU-only index, everything pinned.
- **Every pinned version exists as a real artifact** (checked over HTTP, since it
  could not be checked by building):
  `torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl` (HTTP 200, 174.7 MB),
  `torchvision-0.20.1+cpu-cp311-cp311-linux_x86_64.whl` (HTTP 200), both aarch64
  counterparts (HTTP 200), and `transformers==4.46.3`, `accelerate==1.1.1`,
  `numpy==2.1.3`, `pillow==11.0.0` on PyPI (HTTP 200).

### Written but never executed

- `docker build` of this image has **never been run**. The pins are confirmed to
  exist and the instructions parse, but no layer has actually been built, so
  apt/pip resolution inside the image is unproven.
- `docker run` has **never been run**, so nothing here has yet driven a real
  container: the mounts (including the nested `/workspace/data`), the working
  directory, and `bash reproduce.sh <mode>` reaching the generated script are all
  argv-verified and design-verified, not execution-verified.
- No real `metrics.<mode>.json` has been produced by a container. The parser is
  verified against the documented contract shape, not against a file some
  container actually wrote.
- The real Haiku triage call has never been made — `triage.py`'s parsing is
  verified against synthetic payloads, but no live API response has exercised it.
- Consequently the stage budgets in the table above are **estimates**, not
  measurements. The first real run should be `--mode probe` on a single paper,
  with the wall-clock time it reports used to re-tune them.

### Not built

The Coder↔Runner retry loop. `triage.py` produces a `recoverable_error` category
and a `suggested_fix` written specifically to be handed to
`CodeWriter.write(feedback=...)`, and `coder/base.py` already accepts that
parameter — but nothing wires the two together yet. That is the Orchestrator's
job, and is the increment this stage was built to make possible.

Worth knowing about the class of failure this stage exists to catch:
`coder/README.md` documents a real `AttributeError` (`_NoOpScheduler.get_last_lr()`
reading a `self.optimizer` the class never assigns) that `ast.parse` could not
catch, because a script can be perfectly valid Python and still die on its first
step. That bug came from a **discarded** generation and is *not* in the committed
`coder/output/` script — checked directly, which is the only way to be sure of a
claim like this — so it is not a prediction about the first `probe` run. It is
the reason `probe` exists at all: a syntax gate proves a script parses, never
that it runs.
