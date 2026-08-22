# reader/ — structured extraction pipeline

Turns one paper's already-OCR'd Markdown (`ocr/output/vlm/<paper>.md`) into
structured data — **method_summary** (what the paper proposes, in prose),
**architecture_notes** (the model to build, and what the paper fails to
specify about it), **claims** (the paper's own reported results),
**hyperparameters** (its training configuration), and **data_pipeline**
(dataset source, preprocessing, augmentation, reference URLs) — each
source-grounded to a table/page, figure block, or prose section, cross-checked
against the paper by a validation step that can trigger a bounded retry loop.
Those five stages are the four `reader_output` fields the project plan's §1.2
specifies (`architecture_notes` and `method_summary` completed the set), so
this is now the Reader agent's full scope: everything the Coder needs from a
paper.

## Class architecture

```
reader/base.py
┌─────────────────────────────────────────┐
│  Extractor[ResultT]  (ABC)                 │
│    name: ClassVar[str]                      │
│    extract(markdown_text, client,            │
│             feedback: str | None) -> ResultT  │
└─────────────────────────────────────────┘
           ▲
           │  implemented once per reader_output field —
           │  five stages, all reading the same Markdown,
           │  none consuming another's output
           │
           ├── MethodSummaryExtractor       name="method_summary"
           │     reader/method_summary.py
           ├── ArchitectureNotesExtractor   name="architecture_notes"
           │     reader/architecture_notes.py
           ├── ClaimsExtractor              name="claims"
           │     reader/claims.py
           ├── HyperparametersExtractor     name="hyperparameters"
           │     reader/hyperparameters.py
           └── DataPipelineExtractor        name="data_pipeline"
                 reader/data_pipeline.py

reader/validator.py
┌─────────────────────────────────────────┐
│  ExtractionValidator                       │
│    validate(markdown_text,                  │
│             results: dict[str, Any],         │
│             client) -> ValidationResult        │
│  (iterates `results` generically by stage        │
│   name — does NOT hardcode which stages exist,    │
│   so adding an Nth extractor needs no code change  │
│   here, only one more example parenthetical in the  │
│   prompt so its cross-checks are actually asked for)  │
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
class makes that loop's code stage-agnostic. Concretely: `data_pipeline.py`,
then `method_summary.py` and `architecture_notes.py`, were each added after
this design existed, and each required **zero changes** to `pipeline.py`'s
loop logic, `validator.py`'s cross-checking, or the flag routing — just one
new class and one new line in the default stage list. (The validator prompt
does gain one example parenthetical per new stage — not to make cross-checking
*possible*, which it already is, but to make it *asked for*.) That's the whole
point.

## The retry loop — now real, and proven to catch real bugs

```
ReaderPipeline.run(markdown_path, client)
        │
        ▼
  load markdown ONCE
        │
        ▼
  run every stage ONCE
  (method_summary, architecture_notes, claims,
   hyperparameters, data_pipeline — .extract() each;
   stages are independent, order is only reading order)
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

## `method_summary.py` — `MethodSummaryExtractor`

The project plan's §1.2 `method_summary` field. The only stage whose output
is prose rather than records: `problem` (what the paper attacks), `core_idea`
(the central contribution — the mechanism, not just its name), `novelty`
(what is new against the prior work the paper *itself* names), `summary` (one
coherent paragraph, the §1.2 field proper), plus `sources_examined`.

Consumers are the Coder — which needs to know what it is building before it
reads `architecture_notes`' component list — and the eventual Report
Generator.

**Deliberately narrow, and the prompt says so explicitly:** no result numbers,
no hyperparameters, no dataset preprocessing, no layer-level architecture
spec. Not tidiness — every stage's output goes to `validator.py` *together*,
so a summary that restates a number slightly differently from the stage that
owns it manufactures a cross-check flag about a disagreement that was never
real. Naming the architecture and its central mechanism in a sentence is
correct; enumerating its filter sizes is not.

## `architecture_notes.py` — `ArchitectureNotesExtractor`

The other half of §1.2, and the higher-stakes one: its consumer is the Coder,
which has to emit a real `nn.Module`, so this stage is graded on whether an
engineer could write the model from it.

`model_name`, `overall_structure` (prose: how the parts compose end to end),
`components` (each with `name` / `role` / `specification` / `source`),
`key_equations` (verbatim LaTeX, the paper's own numbering kept),
`depth_or_scale`, `unstated_details`, `sources_examined`.

**Figures are first-class source text.** `ocr/vlm_extract.py` renders every
page image and transcribes figures in-line as bracketed `[Figure N: ...]`
blocks — caption plus a description of what the figure depicts — and for
architecture that block is frequently where the structure actually lives.
Network In Network's *"three mlpconv layers and one global average pooling
layer"* is stated in the Figure 2 block; the prose around it only gestures at
the structure. The prompt directs the model at those blocks by name and
requires citing them as sources.

**`specification` is a string, never a number** — same reasoning as
`hyperparameters.value`, sharper here: "not stated" and prose formulas ("a
three-layer perceptron; the number of layers is flexible") have to survive
verbatim instead of collapsing into a plausible-looking integer.

**`key_equations` entries carry a role, and that is the point.** Each is an
object: `latex` (verbatim, the paper's own `\tag` numbering kept), `label`,
`defines` (one line on what it specifies), and `is_own_method`.

That last field is load-bearing rather than decorative. A paper introducing a
new layer almost always prints the conventional formulation — and often a
rival's — right beside its own. Network In Network prints all three within two
pages, and the first version of this extractor returned exactly that: three
bare LaTeX strings, indistinguishable. Telling the model in the prompt to
"skip baseline equations" did not work; it kept returning them. Making the
schema *force* a role declaration did:

```
eq 1  [contrast only]   Conventional convolution + ReLU, shown for contrast
eq 2  [IMPLEMENT THIS]  The mlpconv layer's per-patch MLP computation
eq 3  [contrast only]   Maxout, a prior-work baseline contrasted against mlpconv
```

Three plausible specifications with no way to choose between them is worse
for a code generator than no equations at all — it will confidently implement
one of them. A consumer now filters on `is_own_method` and gets eq. (2) alone.
A bare string from an older run is still parsed, but is marked
`is_own_method=false`, which is the conservative reading.

### `unstated_details` — required, not optional

This is the field the stage exists for. It lists, explicitly, every
architectural detail needed to write working code that the paper does **not**
state — per-layer filter counts, kernel sizes, strides, pooling windows, MLP
widths, initialization — including details deferred to supplementary material
or to a citation without numbers, quoted in the paper's own wording.

Same "deliberately does not guess" discipline `data_pipeline.py` established,
with a higher stake, and the prompt spells out why: **code has to actually
run.** A wrong-but-plausible channel count does not fail loudly — it produces
a model that trains happily and reports a number for the *wrong network*,
which is strictly worse than a gap labelled as a gap. So the gap gets
surfaced loudly (an empty `unstated_details` even logs a `warning`, because a
paper that fully specifies its architecture is rare) instead of being silently
filled in.

**Honest limitation:** this stage cannot make a paper say what it does not
say. For Network In Network it correctly reports that the main text never
gives filter counts or kernel sizes — the paper defers them to supplementary
material this pipeline never sees. That is the *right* output, but it is not a
buildable spec: the Coder still has to resolve those gaps some other way
(a `depth_or_scale` formula, a reference repo from `data_pipeline.reference_urls`,
a documented assumption, or a human). Closing them automatically would need
the same paper-lineage mechanism `data_pipeline.py`'s citation-deferral gap
needs — see `docs/notes/coder-agent-precedents.md` — which is out of scope
here. What this stage guarantees is that the gap is *visible* rather than
invented; it does not guarantee the gap is filled.

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
`{ method_summary, architecture_notes, claims, hyperparameters, data_pipeline,
validation: { flags, attempts, retried_stages } }`. Keys are one per stage
name, written from the results dict, so the JSON grows a key when a stage is
added and downstream consumers (`coder/`) read them by name.

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

Because that class of bug is silent by construction, `method_summary.py` and
`architecture_notes.py` additionally check `message.stop_reason` and
`logger.error` when it is `max_tokens`, so a truncated call announces itself
instead of being diagnosed after the fact. **Open follow-up:** the four older
calls (`claims`, `hyperparameters`, `data_pipeline`, `validator`) still don't
check it — same four-line guard, deliberately left out of the slice that
introduced it to keep the diff reviewable.

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

`method_summary` and `architecture_notes` were then added and run end to end
against the same two papers, which is where the two malformed-payload issues
in "Known issues, open" below surfaced. What the new stages produced:

- **Network In Network** (the hard case — a custom `mlpconv` layer whose
  dimensions the paper never gives). `architecture_notes` captured the
  three-mlpconv-layers-plus-global-average-pooling structure, eq. (2)
  verbatim with its `\tag{2}`, `depth_or_scale` correctly reporting that the
  paper gives no closed-form rule (quoting *"The number of layers in both NIN
  and the micro networks is flexible"*), and **11 `unstated_details`** — filter
  counts, receptive-field size, pooling window, initialization, plus the
  paper's own deferral, quoted: *"The detailed settings of the parameters are
  provided in the supplementary materials"*. No invented numbers anywhere in
  the output.
- **Wide Residual Networks** (the near-stock case). 9 components over 19
  sources: Table 1's per-group structure with exact widths (`[3×3, 16]` for
  conv1, `16×k`/`32×k`/`64×k` for conv2–4), BN-ReLU-conv ordering, dropout
  placement read off the Figure 1(d) block, avg-pool `[8×8]`, and the honest
  note that Table 1 declares the final classification layer *"omitted for
  clearance"*. `depth_or_scale` records the `WRN-n-k` naming convention **and
  states that the paper gives no closed-form `n = 6N+4` relation** — which is
  correct; the formula is not in this paper's text, and the validator
  independently re-checked and confirmed it absent.

Both papers ran the full 3 validation passes and finished with flags
remaining (NIN 4, WRN 6), consistent with the "Honest limitation" above.
Routing worked in both directions on the new stages: `architecture_notes` was
re-run from a routed flag on both papers, `method_summary` on Wide ResNet.

Still open: any auto-resolution for `cross-check: ...` style flags that don't
route to a single stage, and the `unstated_details` gap itself (this stage
reports the gap; nothing yet closes it — see `architecture_notes.py` above).

## Known issues, open

**Tool input arriving double-encoded (guarded in the two newest stages,
unguarded in the older four).** Caught while verifying `architecture_notes`:
`data_pipeline` returned *zero* datasets for Network In Network, three runs
out of three, on a call whose `stop_reason` was a clean `tool_use`. Dumping
the raw block showed the extraction had actually succeeded — the model had
just emitted the whole payload as a JSON *string* under a single key
(`{"datasets": "{\"datasets_examined\": [...], \"datasets\": [...]}"}`)
instead of as the schema's object. Every `payload.get(...)` then missed,
`_as_list()` turned each miss into `[]`, and the stage reported an empty
result that looked like a clean run.

`method_summary.py` and `architecture_notes.py` detect that shape, unwrap it,
and `logger.warning` about it. `claims.py`, `hyperparameters.py`,
`data_pipeline.py`, and `validator.py` do **not** yet — they will silently
drop a good extraction if the model does this to them.

**An all-empty tool payload, intermittently, on any stage.** A second, related
shape: the tool call returns `stop_reason: tool_use` but with every required
field missing, so the stage produces a structurally valid, completely empty
result. Seen across three different stages in three consecutive runs —
`data_pipeline` and then `claims` on Network In Network, `method_summary` on
Wide Residual Networks — and not reproducible on demand (probing the same
stage/paper pair directly returned good payloads both times). So: intermittent,
model-side, and *not* specific to any one stage or prompt.

The retry loop is what currently saves this, and it demonstrably does: in
every case the validator flagged the empty extraction specifically ("the
`method_summary` fields are all empty, despite the paper providing clear
content for these"), `pipeline.py` routed the flag to that stage by name, and
the re-run came back populated. That is the loop earning its keep on a failure
mode it was not designed for. But it costs a full extra validation pass out of
a budget of three, so it eats retry budget that should be going to real
extraction errors.

The proper fix for both shapes is one shared tool-use helper (unwrap +
`stop_reason` check + a loud "required field missing" path, and probably an
immediate single re-request on an empty payload rather than waiting for the
validator) used by all six stages, replacing the six copies of
`_tool_input`/`_as_list`. That is a refactor across every module in this
package and belongs in its own commit, not smuggled into the one that found
the bug. For now, the two newest stages report the diagnosis
(`logger.error` naming exactly which keys were missing and which arrived) so
the next occurrence is readable straight off the console instead of needing
another raw-payload dump.
