# coder/ — training script generation

Turns one paper's `reader/output/<paper>.json` into a self-contained,
runnable training script plus a pinned `requirements.txt` — the Coder
agent's job in the pipeline `ocr → reader → coder → runner (not yet built)
→ critic (not yet built)`.

## Pipeline shape — a chain, not a loop

`reader/`'s three extractors are genuinely interchangeable: same input
(`markdown_text`), same shape, so `reader/base.py`'s `Extractor[ResultT]`
ABC lets `ReaderPipeline` loop over them generically. `coder/`'s three
steps are different in kind — a fixed **sequential chain** where each
step's input is the *previous step's differently-shaped output*, and one
step isn't even an LLM call:

```
reader/output/<paper>.json
        │
        ▼
┌─────────────────────┐   CodePlan   ┌─────────────────────┐   script (str)   ┌───────────────────────┐
│   MethodTranslator    │ ───────────► │   CodeSynthesizer     │ ───────────────► │   DependencyResolver    │
│ (Claude call)           │              │ (Claude call)           │                 │ (deterministic, no LLM)  │
│ coder/method_translator.py│           │ coder/code_synthesizer.py│                │ coder/dependency_resolver.py│
└─────────────────────┘              └─────────────────────┘                 └───────────────────────┘
                                                                                          │
                                                                                          ▼
                                                                              Dependencies (pinned requirements)
```

There is no shared `Stage` ABC and no `CoderPipeline` class holding
config/state — `coder/pipeline.py`'s `run_pipeline()` just calls the three
steps in order, the same function-based shape `ocr/vlm_extract.py` uses
rather than `reader/pipeline.py`'s class-based one. Each step does still
accept the same `feedback: str | None = None` parameter reader/'s
extractors have, unused for now — that's where a future Critic verdict
plugs in once `runner/`/`critic/`/`orchestrator/` exist.

## The v1 fidelity trade-off

`reader/` doesn't extract a paper's architecture yet (`architecture_notes`
is still design-only — see CLAUDE.md), and precisely translating an
architecture description in prose into a correct novel `nn.Module` is
failure-prone. So `MethodTranslator` doesn't attempt it: it either names an
existing Hugging Face Hub model id as an architectural stand-in (e.g.
`microsoft/resnet-50` for a paper proposing a novel residual network), or,
if nothing fits, describes a simple custom architecture "in the spirit of"
the method. Either way `model_choice.is_exact_reproduction` is always
`false` in v1, and `model_choice.caveats` records exactly what was
substituted — a real fidelity trade-off, made visible in the output JSON
rather than silently assumed. A live test against *Wide Residual Networks*
did in fact write out a correct from-scratch `WideResNet(depth=28,
widen_factor=10)` (no suitable Hub stand-in exists), matching the paper's
actual construction closely — the substitution path is the fallback, not
the only path.

## The `REPROBOT_RESULT` convention

Every generated script must print exactly one line at the end:
```
REPROBOT_RESULT {"metric": "...", "value": ..., "unit": "..."}
```
matching `target_claim.metric`/`.unit` from the `CodePlan`. This is the
*only* channel the result is meant to leave the script by. It's being
established now, before `runner/` exists, specifically so `runner/` can
grep container stdout for one exact line instead of parsing
`transformers.Trainer`'s noisy training logs — and so every script
`coder/` has already generated stays compatible once `runner/` is built,
rather than needing to be regenerated.

## Dependency pinning

`dependency_resolver.py` regex-scans the generated script's own
`import`/`from ... import` lines and looks each top-level module up in a
small curated table — a module *not* in the table is left out, never
guessed, and Python stdlib modules are recognized via
`sys.stdlib_module_names` rather than a hand-maintained list. Versions are
chosen for the future `runner/` Docker image (a fresh Linux Python
environment) — **not** the Intel-macOS host's legacy `torch==2.2.2` pin set
CLAUDE.md documents for local smoke-testing, which is irrelevant once code
runs in a container. `transformers` is pinned to `4.46.3` specifically: a
live test's generated script used
`TrainingArguments(evaluation_strategy=...)`, a parameter renamed to
`eval_strategy` in later releases — `4.46.3` is the last version confirmed
to still accept the old name. `accelerate` is added automatically whenever
`transformers` is resolved, since `Trainer` requires it at runtime even
though no generated script ever `import`s it by name.

## Setup

```bash
uv sync --extra coder --group dev
```
No torch here — `coder/` only *writes* code, it never executes it, so this
extra stays exactly as lightweight as `reader/`'s (`anthropic` +
`python-dotenv`) and in the shared lock, unlike the generated script's own
torch/torchvision/transformers dependencies (which belong to `runner/`'s
future Docker image, not this project's environment).

## Usage

```bash
uv run python -m coder.pipeline --input "reader/output/2016-05 - Wide Residual Networks.json"
uv run python -m coder.pipeline --input reader/output --output coder/output
```
Same `--input`/`--output`/skip-if-already-done CLI shape as `ocr/`'s and
`reader/`'s scripts. Writes, per paper: `<paper>.py` (the script),
`<paper>_requirements.txt` (pinned pip requirements), and `<paper>.json`
(the `CodePlan`, `Dependencies`, and any synthesis notes — everything a
future `runner/`/`critic/` or a human needs, without re-reading the
script). Before writing anything, `run_pipeline()` runs `ast.parse()` on
the generated script and raises if it isn't valid Python, so a broken
generation is caught immediately rather than surfacing confusingly later
when someone tries to run the file.
