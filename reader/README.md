# reader/ — structured extraction pipeline

Turns one paper's already-OCR'd Markdown (`ocr/output/vlm/<paper>.md`) into
structured data — **claims** (the paper's own reported results),
**hyperparameters** (its training configuration), and **data_pipeline**
(dataset source, preprocessing, augmentation, reference URLs) — each
source-grounded to a table/page or prose section, cross-checked against
the paper by a validation step that can trigger a bounded retry loop. This
is the Reader agent's complete first phase: everything the eventual Coder
needs from a paper except architecture details (not yet built).

## Class architecture

```
reader/base.py
┌─────────────────────────────────────────┐
│  Extractor[ResultT]  (ABC)                 │
│    name: ClassVar[str]                      │
│    extract(markdown_text, client,            │
│             feedback: str | None) -> ResultT  │
└─────────────────────────────────────────┘
           ▲              ▲               ▲
           │              │               │
┌──────────┴───┐  ┌───────┴────────┐  ┌────┴─────────────┐
│ ClaimsExtractor │  │HyperparametersEx-│  │DataPipelineExtractor│
│ name="claims"    │  │tractor            │  │name="data_pipeline"  │
│                    │  │name="hyperparam- │  │                       │
│ reader/claims.py    │  │eters"             │  │reader/data_pipeline.py│
│                      │  │reader/hyperparam- │  │                        │
│                       │  │eters.py            │  │                        │
└─────────────────┘  └────────────────┘  └───────────────────┘

reader/validator.py
┌─────────────────────────────────────────┐
│  ExtractionValidator                       │
│    validate(markdown_text,                  │
│             results: dict[str, Any],         │
│             client) -> ValidationResult        │
│  (iterates `results` generically by stage        │
│   name — does NOT hardcode which stages exist,    │
│   so adding a 4th extractor needs zero changes here)│
└─────────────────────────────────────────┘

reader/pipeline.py
┌─────────────────────────────────────────┐
│  ReaderPipeline(stages, validator,          │
│                 max_retries=3)                │
│    run(markdown_path, client) -> ReaderOutput    │
│  Owns the retry loop (below). Adding a new         │
│  extractor = add one instance to `stages`,           │
│  nothing else in this file changes.                    │
└─────────────────────────────────────────┘
```

**Why classes, not functions:** the pipeline needs to treat every
extraction stage uniformly — call `.extract()`, look up results by
`.name`, and on a validation flag, route it back to whichever stage owns
that name and call `.extract()` again with `feedback` set. A shared base
class makes that loop's code stage-agnostic. Concretely: `data_pipeline.py`
was added after this design existed, and required **zero changes** to
`pipeline.py`'s loop logic, `validator.py`'s cross-checking, or the flag
routing — just one new class and one new line in the default stage list.
That's the whole point.

## The retry loop — now real, and proven to catch real bugs

```
ReaderPipeline.run(markdown_path, client)
        │
        ▼
  load markdown ONCE
        │
        ▼
  run every stage ONCE
  (claims.extract(), hyperparameters.extract(), data_pipeline.extract())
        │
        ▼
  ┌──────────────────────────────────────────┐
  │  validator.validate(markdown, results)       │◄─────────────┐
  └──────────────────┬───────────────────────────┘                │
                      ▼                                              │
              ┌───────────────┐                                       │
              │ any flags?      │                                       │
              ▼ no                ▼ yes                                 │
             done         split flags into:                              │
                            (a) ROUTABLE — relates_to prefix                │
                                exactly matches one stage's .name             │
                                (e.g. "claims: c3" -> claims)                   │
                            (b) UNROUTABLE — doesn't match any one              │
                                stage (e.g. "cross-check: claims vs.             │
                                hyperparameters") -> left as report-only,         │
                                never auto-retried                                  │
                                   │                                                 │
                          for each stage with routable flags:                        │
                            re-run stage.extract(markdown, client,                     │
                              feedback=<concatenated flag descriptions>)                 │
                            (a full re-extraction with the specific                       │
                             complaint folded into the prompt — not a                      │
                             patch to individual fields)                                     │
                                   │                                                          │
                          replace that stage's result ─────────────────────────────────────────┘
                          (capped at max_retries=3 total validation passes,
                           i.e. up to 2 retry rounds; if flags remain after
                           the cap, finish anyway with them still listed —
                           never hard-fails)
```

### Proof this actually works: Wide Residual Networks, real run

The very first real run against Wide Residual Networks caught a genuine
bug (see "Known issues, fixed" below) that made claims extraction return
**zero results** three times in a row despite the validator correctly and
specifically naming real missing numbers each time. Once fixed, a
subsequent real run demonstrated the loop doing exactly what it's for:

- Pass 1: 73 claims extracted, but the validator caught that claim `c42`
  labeled a value `4.00%` as a **CIFAR-100** result — it was actually the
  paper's CIFAR-10 value from the same table row, duplicated under the
  wrong dataset label.
- Retry: `claims` re-extracted with that specific complaint folded into
  the prompt. The duplicate/misattributed entry was removed.
- Final: 66 claims, `reference_urls` correctly captured all 4 URLs in the
  paper (including `https://github.com/szagoruyko/wide-residual-networks`),
  all 4 datasets covered in `data_pipeline`.

**Honest limitation, not swept under the rug:** validation did *not*
converge to zero flags for this paper — flag count went 5→6→8 across the 3
passes (increasing, not decreasing) and finished with 8 unresolved. WRN
has ~7x more claims than the papers this was tuned against (66 vs.
Network In Network's ~10), which gives a validator a lot more surface area
to find *something* to comment on, and not everything it flags is a real
error (some are nuance-of-phrasing observations, not mistakes). The loop
mechanically works — it retries the right stage with the right feedback,
and it did fix a genuine bug above — but "always converges to zero" is not
a guarantee for large/complex papers with the current 3-pass cap. Worth
revisiting (higher cap? tighter validator prompt?) once more papers have
been run through it, not something to solve speculatively now.

## `claims.py` — `ClaimsExtractor`

AutoP2C-style extraction:

1. Locate every results table, note table + transcribed page in
   `tables_examined`.
2. Classify each row as the paper's **own proposed method** or a
   **baseline/prior-work** row (baselines cite another paper, e.g. `[14]`;
   own-method rows don't and are usually bolded).
3. One claim per distinct own-method row — multiple headline configurations
   for the same metric+dataset (e.g. with/without augmentation) each get
   their own claim, distinguished by `model_variant`.
4. Baseline rows dropped entirely, even from a table that also has kept
   claims.

Each claim: `claim_id`, `metric`, `dataset`, `reported_value` (float),
`unit`, `source` (e.g. `"Table 1, transcribed page 5"`), optional
`model_variant`. Matches the project plan's `Claim` shape (§1.2) plus
`model_variant`.

## `hyperparameters.py` — `HyperparametersExtractor`

Looks in **both** a dedicated hyperparameters table **and**
"Implementation Details" prose — confirmed necessary, not optional: Network
In Network has *no* dedicated table at all, everything came from prose.
Baseline configurations excluded, same as claims.

Each hyperparameter: `name`, `value` (**string**, not float — schedules
like `"lowered by a scale of 10 at epochs 82 and 123"` must survive
verbatim, not collapse to a bare number), optional `unit`, `source` (table
or prose reference), optional `model_variant`.

**Known, not-yet-fixed:** the `sources_examined` bookkeeping field
consistently logs as empty even when hyperparameters are successfully
found with real sources — the model isn't reliably populating that one
field despite it being schema-required. Doesn't affect the extracted data
itself, just that one log line's accuracy.

## `data_pipeline.py` — `DataPipelineExtractor`

Per-dataset entries (`dataset`, `dataset_source`, `normalization`,
`augmentation`, `split_convention`, `source`) plus a paper-level
`reference_urls` list (GitHub/code-repo/dataset-download URLs found
anywhere in the paper text).

**Deliberately does not guess.** When a paper defers a detail to a cited
work without giving the actual numbers — e.g. Network In Network's *"global
contrast normalization and ZCA whitening (same as Goodfellow et al.)"*,
no formula given in-paper — the extractor records that fact ("not stated,
cites Goodfellow et al. [8]") rather than inventing plausible-sounding
numbers. This is a known, real gap, not a bug: resolving it would need a
paper-lineage mechanism (fetch and read the cited paper) that's explicitly
out of scope for this slice — see `docs/notes/coder-agent-precedents.md`
for why that's a real, evidenced idea worth building later, just not now.

`dataset_source` may be *inferred* rather than paper-stated (e.g. Wide
ResNet never names a PyTorch/HF loader, only Torch7 — the extractor notes
the inferred standard-library equivalent and flags it as inferred).

## `validator.py` — `ExtractionValidator`

One more Claude call, given the full paper Markdown plus every stage's
combined output (iterated generically from the `results` dict — doesn't
hardcode which stages exist), prompted to flag — not fix — two kinds of
problems: (a) inconsistencies between stages (a `model_variant` in claims
with no matching hyperparameter entry, a claim's dataset with no
corresponding `data_pipeline` entry), and (b) anything in the paper that
looks like it should have been captured but wasn't. Each flag:
`relates_to` (which stage/field, used for retry routing — see loop diagram
above), `description`.

## Logging

Detailed by design, via `loguru` (not `print`) — every table/prose/dataset
source examined, every item kept, every validation flag raised, and every
retry (which stage, which flags triggered it) printed as it happens, plus
final pass/fail summary. A bad extraction — or the loop not converging —
should be obvious from console output alone, which is exactly how the
Wide ResNet bug above was caught in the first place.

## Setup

```bash
uv sync --extra reader --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

```bash
# a single paper
uv run python -m reader.pipeline --input "ocr/output/vlm/2013-12 - Network In Network.md"

# every Markdown file in a directory, skipping papers already processed
uv run python -m reader.pipeline --input ocr/output/vlm

# custom output location
uv run python -m reader.pipeline --input ocr/output/vlm --output somewhere/else
```

Output: `reader/output/<paper>.json` (gitignored) —
`{ claims, hyperparameters, data_pipeline, validation: { flags, attempts, retried_stages } }`.

## Dependency choice

`reader` is its own `pyproject.toml` extra (`anthropic` + `python-dotenv`),
not a reuse of `ocr`'s `vlm` extra — this stage never touches a PDF or
renders a page image (`pypdfium2`, which `vlm` needs), it only reads
Markdown `ocr/vlm_extract.py` already produced.

## Known issues, fixed

**`max_tokens` too small for large papers (fixed).** First real run against
Wide Residual Networks (8 results tables — more than any paper tested so
far) returned **zero claims across all 3 retry attempts**, despite the
validator correctly naming specific missing numbers every time. Diagnosed
directly: `stop_reason: max_tokens` — the call hit its 4096-token cap after
generating `tables_examined` and `candidates_considered`, before ever
starting the `claims` array. Fixed by bumping `max_tokens` 4096→8192 across
all four Claude calls in `reader/` (not just the one that broke — all four
have the same unbounded-output risk for a large enough paper). Same class
of bug already hit once before in `ocr/vlm_extract.py`.

## Status

Verified end-to-end against two papers:
- **Network In Network**: claims/hyperparameters extraction stable across
  runs (counts shift slightly, expected LLM non-determinism), zero
  `reference_urls` correctly (paper has none).
- **Wide Residual Networks** (the actual first Coder validation target):
  66 claims, 4 datasets in `data_pipeline`, all 4 real reference URLs
  captured correctly. Validation loop fired for real, fixed a genuine
  duplicate-value bug, but did not fully converge for this larger/more
  complex paper (8 flags remained after the 3-pass cap) — see "Honest
  limitation" above.

Not yet built: `architecture_notes.py` (next slice — needed once Coder
moves past stock-architecture papers like Wide ResNet to something like
Network In Network's custom `mlpconv` layer), and any auto-resolution for
`cross-check: ...` style flags that don't route to a single stage.
