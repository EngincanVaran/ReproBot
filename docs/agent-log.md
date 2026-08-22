# Agent Log

A running record of delegated subagent work on ReproBot, so Engincan can
track what each "role" was asked to do and what it found/built without
re-reading full transcripts. Claude (in the main session) acts as the
orchestrator: it decides when a task warrants delegating to a specialized
subagent, dispatches it, and appends an entry here once it returns.

**Roles used so far:**
- **Research Agent** — surveys papers/notes/dataset/existing output, gives
  implementation guidance; doesn't write code.
- **Coding Agent** — implements a specific, scoped change.
- **Explainer Agent** — reads a diff/result and explains it in plain terms.

Newest entries at the bottom. Each entry: who, what they were asked, what
came back (summarized — full output lives in the actual commit/files, not
duplicated here).

---

### Research Agent — scope the first `reader/` increment

**Asked:** Read CLAUDE.md, the project plan's shared-memory schema,
`docs/notes/reader-agent-precedents.md`, the `ocr/` scripts, and the actual
sample output already on disk (pdfplumber + VLM extractions of *Network In
Network*), then recommend the smallest useful first slice for a new
`reader/` folder — not the whole Reader agent.

**Result:**
1. **Input source:** Claude-VLM output only, not pdfplumber — pdfplumber's
   sample has no word-spacing in prose and renders every figure as an empty
   table placeholder; VLM output is clean Markdown with correctly
   caption-grounded figure descriptions.
2. **First slice:** claims extraction only, not the full `reader_output` —
   validates the hardest part (precise source-grounding to a table/figure)
   in isolation, before hyperparameters/data-pipeline/architecture (each
   independently important per AutoP2C's own modality ablation).
3. **Prompt approach:** AutoP2C-style narrow scoping (locate results tables
   specifically, extract only rows for the paper's own proposed method vs.
   baselines, one claim per distinct reported variant, require a page/table
   source citation, no metric-name normalization yet) rather than
   AutoReproduce's single broad summarization pass.
4. **Schema:** reuse the project plan's `Claim` shape as-is, plus one
   addition — an optional `model_variant` field, since papers like NIN
   report multiple headline configurations (e.g. with/without data
   augmentation) for the same metric+dataset pair.
5. **Validation paper:** `Network In Network.pdf` — already OCR'd by both
   backends, single clean CIFAR-10 results table with an unambiguous
   headline claim (10.41% / 8.81% test error).

Decision: build `reader/extract_claims.py` next — Claude-VLM output in,
`Claim` list out (+ `model_variant`), validated against Network In Network.

---

### Coding Agent — extend VLM extraction to cover the whole paper

**Asked:** `ocr/vlm_extract.py`'s only prior run was capped at 2 pages
(smoke test). Strengthen table-transcription fidelity in the prompt, raise
`max_tokens` (4096→8192) to avoid mid-table truncation, add detailed
per-page progress logging, verify with ruff/mypy/pre-commit, then actually
run the full paper (no page cap) on *Network In Network* and confirm the
CIFAR-10 results table comes through with exact numbers.

**Result:** All lint/type/pre-commit checks pass. Real run, 10/10 pages,
~13s/page:
```
[page 5/10] 13.2s, 3696 chars, table=yes
[page 6/10] 12.7s, 3376 chars, table=yes
[page 7/10] 10.2s, 2822 chars, table=yes
```
Results table (page 5) confirmed exact: **NIN+Dropout 10.41%**,
**NIN+Dropout+DataAug 8.81%** — both target numbers present, no truncation.
Tables 2–5 (CIFAR-100, SVHN, MNIST, ablation) also captured cleanly.
`ocr/output/vlm/Network In Network.md` now holds the real full-paper output.

---

### Validator Agent — audit OCR coverage across dataset/

**Asked:** Read-only audit of what's actually been OCR'd across all 8
`dataset/` papers by both working backends (pdfplumber, VLM), and sanity-
check quality of whatever exists.

**Result:** Ran concurrently with the Coding Agent above and happened to
read the VLM file *before* that fix landed — its "VLM: only 2/10 pages"
finding is now stale (see previous entry; fixed and reconfirmed 10/10).
Still-valid findings:
- **pdfplumber** quality issue (unaddressed, separate from VLM): body text
  has no inter-word spacing (`"MinLin1,2,QiangChen2..."`) and includes
  garbled/reversed arXiv sidebar text interleaved into the main content
  (`"4102"`, `"raM"`, `"]EN.sc["`). Not blocking — Reader consumes VLM
  output per the earlier Research Agent recommendation — but worth knowing
  if pdfplumber is ever used as a fallback/cross-check source.
- **7 of 8 dataset papers** have zero OCR output from either backend yet —
  expected, we've deliberately only been validating against Network In
  Network so far, not a bug. Page counts confirmed via `pdfinfo`: AutoAugment
  14p, Stochastic Depth 16p, ResNet 12p, DenseNet 9p, Compact Transformers
  18p, NIN 10p, All-Conv-Net 14p, WideResNet 15p.

---

### Coding Agent — implement `reader/extract_claims.py`

**Asked:** Build the first `reader/` script per the earlier Research Agent
scoping: read a paper's VLM Markdown, use Claude tool-use (structured
output, not free-text parsing) to extract own-method claims (excluding
baseline rows) with `claim_id`/`metric`/`dataset`/`reported_value`/`unit`/
`source`/`model_variant`, matching `ocr/`'s exact code style (local
dataclasses, argparse CLI, no shared package). Detailed per-table/per-claim
logging required. Validate against the now-complete Network In Network VLM
output.

**Result:** `reader/__init__.py`, `reader/extract_claims.py`,
`reader/README.md` added; own `reader` extra in `pyproject.toml`
(anthropic + python-dotenv, deliberately not `pypdfium2` — reader never
touches a PDF, only ocr/'s Markdown output). ruff/mypy --strict/pre-commit
all pass. Real run against Network In Network: 5 tables examined, 24
candidate rows, 8 kept as own-method claims (baseline rows correctly
excluded — verified no baseline value from Table 1 leaked through). The
two target claims came through exact: **10.41%** (NIN + Dropout) and
**8.81%** (NIN + Dropout + Data Augmentation), CIFAR-10, Table 1.

**Open question flagged by the agent, resolved:** the run also kept 3
extra own-method claims from Table 5 (a GAP-vs-fully-connected ablation,
also CIFAR-10) — broader than the 2-claim example used when scoping this
task. Engincan decided: keep everything (no code change needed, this is
already current behavior) — capture every reported own-method variant
across every results table, not just the headline claim per dataset.

---

### Review Agent — critique the VLM extraction prompt before the big batch run

**Asked:** `dataset/` was just renamed and outputs cleared; before running
the full 8-paper batch (real API cost, not repeating soon), critically
review `ocr/vlm_extract.py`'s `PROMPT` and recommend concrete edits — not
vague advice. Grounded in `reader-agent-precedents.md`'s AutoP2C section,
`reader/extract_claims.py`'s actual consumption pattern, and a live 1-page
test run.

**Result:** 5 concrete edits recommended, ranked by the agent's own
judgment of value:
1. **Figure/diagram depth** — upgrade from "briefly" to AutoP2C's 3-part
   guidance (code-relevant detail only, capture every numerical element,
   cross-reference the caption) — this was already a documented TODO, now
   is the cheap moment to do it.
2. **Two-column reading order** — "preserve reading order" alone risks a
   VLM reading across columns instead of down them; needs an explicit
   left-column-fully-then-right-column instruction.
3. **Page furniture (headers/footers/arXiv sidebar)** — explicit exclusion
   instruction, mirroring the exact pdfplumber failure mode found in an
   earlier audit (sidebar text interleaved into body paragraphs) — with an
   explicit carve-out to still transcribe in-text code/data repo URLs.
5. **Bibliography** — compact to one line per entry (author/year/title
   only) rather than full transcription or a full skip — `extract_claims.py`
   never reads it, but a full skip forecloses AutoReproduce-style citation
   lineage work later.
6. **Table/figure caption preservation — highest-value catch.** The current
   table instruction covers the table body but never says to capture the
   caption line itself, and `extract_claims.py`'s `source` field
   (`"Table 1, transcribed page 5"`) depends on exactly that caption text —
   a silent, direct break in the one thing already consuming this output.

Decision: apply all 5 to `PROMPT` as part of the loguru refactor, before
the batch run.

---

### Coding Agent — loguru migration + apply prompt fixes

**Asked:** Apply the Review Agent's 5 prompt edits to `ocr/vlm_extract.py`,
and replace every `print()` with `loguru` across all 5 scripts (3 ocr/
backends + vlm_extract + reader/extract_claims), zero shared logging
config (each file imports `from loguru import logger` independently, per
the no-shared-package-between-stages convention).

**Result:** All 5 prompt edits applied (two-column order, page-furniture
exclusion + URL carve-out, table+figure caption capture, AutoP2C-depth
figure description, compacted bibliography). `loguru>=0.7` added as a base
dependency (used by every stage, unlike stage-specific extras). All
`print()` calls replaced (`logger.info` for progress, `logger.warning` for
early-exits, `logger.error` for caught exceptions). ruff/mypy/pre-commit
all pass (one pre-existing, unrelated mypy gap on `docling_extract.py` —
docling isn't installed locally, expected). Smoke test (1 page, real API
call) confirmed clean colorized loguru output and that the new prompt
didn't break single-column transcription. Also fixed while here: the
pre-commit large-file exclude pattern didn't cover `dataset/`, which would
have blocked committing the renamed papers — added it to the exclude list.

---

### Research Agent — scope the next Reader extraction slice (parallel with the batch run)

**Asked:** Of `hyperparameters`, `architecture_notes`, `data_pipeline`
(the three unbuilt fields in `reader_output`), which should be built next,
and how — same "smallest useful slice" discipline as claims extraction.

**Result:** Build **`hyperparameters` next**, not architecture or data
pipeline:
1. Most blocking for a Coder to produce *any* working script — a Coder can
   fake architecture via a stock `AutoModelForImageClassification` and
   refine later, but `TrainingArguments` structurally needs lr/batch
   size/epochs/optimizer or the Coder is just guessing defaults with no
   traceable feedback signal when the Critic later finds a metric gap.
   Data pipeline and architecture both need harder inputs (scattered prose,
   or VLM figure descriptions specifically) — hyperparameters mostly live
   in dedicated tables, same shape claims extraction already proved out.
2. **Approach:** same pattern as claims — one Claude tool-use call over the
   full VLM Markdown, `record_hyperparameters` tool, `sources_examined`
   bookkeeping, own-config-only (not baseline).
3. **Schema:** `Hyperparameter(name, value: str, unit: str | None, source,
   model_variant: str | None)` — value stays a **string**, not float,
   specifically so schedules/optimizer descriptions ("step decay at epochs
   82/123") survive rather than getting collapsed to a bare number.
4. **Failure mode to guard against:** the project plan's own worked
   Critic-feedback example — a paper stating a base LR in one place and a
   *schedule* elsewhere; extracting only the scalar produces a script that
   runs but silently diverges numerically. Mitigation is exactly why value
   is a string and why the prompt must cover both tables AND the
   "Implementation details" prose section, not tables alone.
5. **File:** separate `reader/extract_hyperparameters.py`, not a mode
   bolted onto `extract_claims.py` — matches the one-slice-per-script
   convention already established.

Decision: build `reader/extract_hyperparameters.py` next (not this session
— logged for the next increment).

---

### Full 8-paper VLM batch run — 2 papers hit an unfixable content-filter false-positive

**What happened:** ran `ocr.vlm_extract` over all 8 renamed `dataset/`
papers. 6 succeeded fully. 2 failed with `Output blocked by content
filtering policy` from the Anthropic API, both specifically on **page 2**:
*Deep Networks with Stochastic Depth* and *Densely Connected Convolutional
Networks*.

**Investigated directly (no subagent — quick, hands-on diagnosis):**
rendered both pages to PNG and visually confirmed each is completely benign
academic prose (screenshotted, no images/tables, nothing remotely
sensitive). Retried both papers fresh — same error, same page, both times
(deterministic, not a transient flake). Tested whether it was an encoding
artifact: re-rendered Stochastic Depth's page 2 at `scale=1.5` instead of
`2.0`, and swapped the detailed transcription prompt for a one-line
"Transcribe this page briefly" — **identical failure both times.** Ruled
out our prompt and our render scale as the cause; this is a genuine,
deterministic false-positive in Claude's vision content classifier tied to
how this specific page's image renders, not something fixable client-side.

**Real gap this exposed:** `run_vlm()` only writes `markdown_path` after
its full per-page loop completes, and a page-level exception propagates
all the way up through `extract_dataset`'s per-*paper* try/except — so a
single bad page (page 2 of 16, or 2 of 9) discards the *entire* paper,
including every page that already transcribed successfully. Proposed fix:
catch failures per-page, insert a placeholder marker, and continue —
not implemented this session.

**Decision (Engincan):** leave these 2 papers as-is for now (0 pages
extracted each), don't build per-page resilience yet — revisit once
`reader/` actually needs to consume them specifically.

---

### Research Agent — Coder-loop precedents in AutoReproduce and AutoP2C

**Asked:** Same method as `reader-agent-precedents.md` (read the actual
PDFs, not summaries) but scoped to the next stage: the Coder. Deep-dive
code-generation granularity, the debug/repair loop mechanics, pre-full-run
validation, and ablation evidence for whether any of it matters. Write-up
saved to `docs/notes/coder-agent-precedents.md`.

**Result — most load-bearing finding:** AutoReproduce's ablation shows
removing its debug/refine loop barely moves the LLM-judged *alignment*
score (69.97→68.32) but explodes the *Performance Gap* metric from 31.62%
to 88.78% — the single largest degradation in their whole ablation table,
worse than removing MinerU OCR or Paper Lineage entirely. Code can look
like a faithful implementation and still be numerically wrong if it's
never actually executed and fixed. AutoP2C's loop is even more load-bearing
— removing it produces 0% executable code across every test paper.

Other confirmed specifics: AutoReproduce's exact edit mechanism is a
structured `EDIT\nN M\n<new code>` command replacing only lines N–M (never
a full rewrite), with **two independent retry triggers** — execution error,
and separately a semantic-alignment check that can fire even when code runs
fine (a proto-Critic check baked into the Coder stage). AutoP2C uses
whole-file localize-then-correct instead (`LLM_LocalizeError` →
`LLM_CorrectError`) since it targets a multi-file repo, not a single
script, and wires `ray.tune` HPO in only after first clean execution, with
search space bounded by the paper's own reported hyperparameter ranges.

Decision: use this to scope the Coder+Runner architecture — see below.

---

### Coding Agent — consolidate reader/ into pipeline.py + build hyperparameters + validation report

**Asked:** Restructure `reader/` per Engincan's request: stop accumulating
standalone scripts per extraction type, refactor `extract_claims.py`'s
logic into an importable `claims.py` module, add a new `hyperparameters.py`
module (already-approved design from the earlier scoping entry), and build
`pipeline.py` as the one real entry point — loads a paper's Markdown once,
runs claims then hyperparameters, then one more Claude call that flags
(doesn't fix) inconsistencies between the two extractions or gaps vs. the
paper text. Auto-fix-on-flag explicitly deferred to a future increment,
build the report version first and see what it actually catches.

**Result:** `claims.py`, `hyperparameters.py`, `pipeline.py` added;
`extract_claims.py` removed (superseded). ruff/mypy --strict/pre-commit all
pass. Real run against Network In Network:
- Claims: same 8/8 as before, unchanged (22 vs. 24 candidates considered —
  model-reported bookkeeping count, varies run-to-run, not a regression).
- Hyperparameters: 16 found, correctly pulled from prose sections (this
  paper has no dedicated hyperparameters table — confirms the prose-fallback
  design was necessary, not optional) — LR schedule kept verbatim as a
  string ("...lowered by a scale of 10... repeated once..."), not collapsed
  to a number.
- Validation: 5 flags, genuinely useful — missing CIFAR-10 feature-map
  count (that CIFAR-100/MNIST entries reference but never state), missing
  dropout rate (0.5), missing 3-layer-MLP-per-mlpconv-layer detail, and a
  missing Section 4.6 CNN+GAP comparison table from claims. One false-alarm
  flag that second-guesses itself mid-description — prompt could be
  tightened later, not blocking.

Decision: next Reader increment (`data_pipeline.py` + dataset/reference
URLs) gets its own scoping pass when we get to it; auto-fix loop on
validation flags stays deferred until we've seen this report run against
more papers.

---

### Coding Agent — refactor reader/ to classes, implement the validation retry loop

**Asked:** Refactor the function-based `claims.py`/`hyperparameters.py`
into `Extractor` subclasses (a shared ABC: `name`, `extract(markdown,
client, feedback=None)`), pull `validate_extraction` out of `pipeline.py`
into its own `validator.py` (`ExtractionValidator`, generic over any number
of named stages, not hardcoded to claims+hyperparameters), and turn
`pipeline.py` into a `ReaderPipeline` class implementing the previously
report-only validation into a real retry loop: route each flag to the one
stage its `relates_to` prefix names, re-run only that stage with the flag
folded into its prompt as feedback, re-validate, repeat up to
`max_retries` (3) total validation passes. Flags that don't map to exactly
one known stage (e.g. `"cross-check: claims vs. hyperparameters"`) are left
as report-only, not auto-retried — routing a genuine cross-stage
disagreement to "the one wrong side" isn't a solved problem yet.

**Result:** `reader/base.py` (new, `Extractor[ResultT]` ABC using PEP 695
generic syntax), `reader/validator.py` (new), `reader/claims.py` and
`reader/hyperparameters.py` rewritten as `Extractor` subclasses (same
prompts/schemas/logic, unchanged behavior), `reader/pipeline.py` rewritten
around `ReaderPipeline`. ruff/mypy --strict/pre-commit all pass.

**Real run — the loop genuinely fired and resolved real issues**, deleted
the stale straight-line output and reran fresh against Network In Network:
- Pass 1: 8 claims, 13 hyperparameters → 5 flags (4 routed, 1 unroutable
  cross-check flag correctly left as report-only).
- Retry round 1: `claims` re-extracted, found a whole **missing claim** the
  original run silently dropped (14.51% CIFAR-10, NIN without dropout) plus
  3 more from a missed Section 4.6 comparison table (8→12 claims).
  `hyperparameters` re-extracted, added the missing baseline-CNN
  architecture note and CIFAR-10 feature-map count (13→18).
- Pass 2: 3 new flags (mostly precision/duplication checks, not gaps).
- Retry round 2: both stages re-extracted again, tightened/consolidated
  (claims stayed at 12, hyperparameters settled at 15, explicitly noting
  some values are deferred to unreproduced supplementary material rather
  than guessed).
- Pass 3: **0 flags** — clean.

Final: 3 validation passes, both stages retried in both rounds, ended with
zero remaining flags (vs. the straight-line version's 5, including a
completely missing 14.51% claim). Verified directly against
`reader/output/2013-12 - Network In Network.json`: 12 claims, 15
hyperparameters, `validation.flags: []`, `attempts: 3`.

---

### Research Agent — scope reader/data_pipeline.py

**Asked:** Read NIN's and Wide Residual Networks' actual paper text (not
just hyperparameters.json's incidental capture) and scope a new
`data_pipeline.py` `Extractor`: exact field schema, the "cites another
paper, no exact numbers" problem, validation-loop compatibility, and which
paper to validate against — Wide ResNet is the actual Coder target, unlike
NIN which was just convenient.

**Result:**
1. **Per-dataset schema**, same pattern as hyperparameters: `dataset`,
   `dataset_source` (confirmed-or-inferred; WRN never names a PyTorch/HF
   loader, only Torch7), `normalization`, `augmentation`, `split_convention`
   (all strings, "not stated" is a valid value), `source`, plus a
   paper-level (not per-dataset) `reference_urls` list.
2. **Confirmed by direct text search, not assumption**: NIN has zero URLs
   anywhere in the paper. Wide ResNet has three, including
   `https://github.com/szagoruyko/wide-residual-networks` stated twice
   ("Our code and models are available at..."). **The citation-without-
   numbers gap is real and worse on NIN**: NIN defers ALL 4 datasets'
   preprocessing to "Goodfellow et al. [8]" with zero numbers in-paper, and
   MNIST gets no preprocessing sentence at all. Wide ResNet is more
   self-contained (exact augmentation numbers given directly) but still
   defers ZCA whitening's exact parameters to the same citation. Decision:
   record "not stated, cites Goodfellow et al. [8]" rather than inventing
   plausible numbers — correctly out of scope for this slice, exactly the
   gap a future paper-lineage mechanism would close.
3. **No code change needed in `validator.py`** — it already iterates the
   results dict generically — but its two example parentheticals should
   mention `data_pipeline` explicitly so cross-checks against it are
   prompted for, not just implicitly possible.
4. **Validate against Wide Residual Networks**, not just NIN — it's the
   actual first Coder target and the harder case for this slice (real URLs
   to test against, a citation gap that still applies despite being more
   self-contained overall). NIN kept as a regression check.

Decision: build `data_pipeline.py` per this schema, add its example to
`validator.py`'s prompt, verify against both papers (WRN primary, NIN
regression).

---

### Coding Agent — build reader/data_pipeline.py, wire into the loop

**Asked:** Build `DataPipelineExtractor` per the approved schema
(per-dataset entries + paper-level `reference_urls`), add it to
`ReaderPipeline`'s default stage list, add a `data_pipeline` example to
`validator.py`'s prompt, verify against Wide Residual Networks (primary)
and Network In Network (regression).

**Result:** File created, wired in correctly (confirmed independently:
zero changes needed to `ReaderPipeline`/`_route_flags`/`ExtractionValidator`
to support the third stage — exactly the payoff the class design was for).
The delegated agent's own verification run got cut off mid-task (a
background command it didn't wait on) before reporting real results, so
the two verification runs below were completed directly instead.

**Bug found and fixed during verification — the delegated agent never got
this far, so it's not in its report.** First real run against Wide
Residual Networks: **claims came back completely empty (0/0) across all 3
retry attempts**, despite the validator correctly and repeatedly naming
specific real numbers that were missing (WRN-28-10 CIFAR-10 4.00%, etc.).
Diagnosed directly: `stop_reason: max_tokens` — the claims call hit its
4096-token cap after generating `tables_examined` (8 tables, more than any
paper tested so far) and `candidates_considered`, before ever starting the
`claims` array. Same class of bug already fixed once for
`ocr/vlm_extract.py`. Fix: bumped `max_tokens` 4096→8192 across all four
reader/ Claude calls (`claims.py`, `hyperparameters.py`,
`data_pipeline.py`, `validator.py`), not just the one that broke — all four
have the same unbounded-output-size risk.

**Re-verified after the fix:**
- **Wide Residual Networks** (primary target): claims 0→66 (peaked at 73,
  the loop's own retries removed some duplicate/misattributed entries —
  e.g. caught and fixed a CIFAR-100 claim that had actually duplicated a
  CIFAR-10 value from the same table row). `data_pipeline.reference_urls`
  correctly captured all 4 URLs in the paper, including
  `https://github.com/szagoruyko/wide-residual-networks` (confirmed present
  by the earlier research pass). All 4 datasets covered.
  **Honest finding: validation did NOT converge to zero for this paper** —
  flags went 5→6→8 over 3 passes (increasing, not decreasing) and finished
  with 8 unresolved. WRN is a much larger surface area (66 claims vs. NIN's
  ~10) for a validator to find *something* to comment on; some flags were
  genuine (the duplicate value above), others were fairly nitpicky
  (nuance-of-phrasing observations). Not a correctness blocker, but a real
  limit of the current loop worth knowing: more complex papers may need a
  higher retry cap, or the validator prompt may need tightening to stop
  finding new things to say each round rather than converging.
- **Network In Network** (regression): `data_pipeline.reference_urls`
  correctly empty (paper has zero URLs, confirmed by the research pass).
  Claims/hyperparameters counts shift slightly run-to-run (non-deterministic
  LLM calls, expected) — finished at 4 unresolved flags this run vs. 0 in
  the previous run, same non-determinism, not a regression from the fix.

**Minor, non-blocking finding:** `hyperparameters.py`'s `sources_examined`
bookkeeping field logs `(0)` every run even when hyperparameters are found
with real `source` values — the model isn't reliably populating that one
field despite it being schema-required. Cosmetic, not fixed this session.

---

### Validator Agent — audit ocr/ and reader/ output quality before continuing

**Asked:** Read-only audit of the actual generated content (not just line
counts) of all 6 `ocr/output/vlm/*.md` files and both `reader/output/*.json`
files, looking for defects distinct from ones already documented (WRN's
non-converging validation, `sources_examined` bug, pdfplumber garbling, the
2-paper content-filter failure, deliberate no-guessing on cited-but-unstated
data_pipeline details).

**Result:**
1. **Duplicated figure captions** — `Striving for Simplicity (All
   Convolutional Net).md`, Figures 5/6 only: caption restated as a bare
   paragraph right after the bracketed `[Figure N: ...]` description block.
   Figures 1-4 in the same file don't do this; a page-specific VLM slip, not
   systemic.
2. **Hallucinated stray heading** — `AutoAugment.md`, page 5: a spurious
   `# Page Content` heading inserted before the real continuation sentence,
   not present in the source PDF.
3. **Real extractor gap, confirmed** — `Wide Residual Networks.json`:
   ImageNet has 14 reported claims (c52-c65) but zero hyperparameter entries
   at all (not even a "not stated" placeholder, unlike every other gap in
   the file) — this is exactly what validation flag #8 already points at,
   now confirmed as a genuine coverage miss rather than a nitpick.

**Confirms working correctly:** page counts match source PDFs exactly across
all 6 VLM outputs (no silent page drops beyond the known 2-paper failure);
spot-checked numeric tables (WRN, ResNet, NIN) all match the source PDFs
exactly, including a genuine WRN-internal inconsistency (Table 8 vs. Table 9
disagree on WRN-50-2-bottleneck top-5 error, 6.03% vs. 5.79%) that both OCR
and claims extraction correctly preserved as two distinct claims rather than
silently merging/picking one; CCT.md correctly stitches a bibliography entry
split across a page boundary; NIN.json and WRN.json both have well-grounded
sources and correct `reference_urls`. Of WRN's 8 unresolved validation
flags, content-checked directly: 5 are the validator second-guessing itself
and finding no issue, 1 is a defensible interpretive edge case, only 2 are
substantive (the ImageNet gap above, and narrative comparison deltas
correctly excluded as derived-not-reported values).

Decision: fix the two OCR prompt issues (duplicate caption restatement,
stray page-break heading) and the ImageNet hyperparameters gap before
starting `architecture_notes.py` — not blocking, but cheap and now
concretely diagnosed rather than theoretical.

**Superseded by Engincan's redirect:** he chose to move on to the Coder and
Runner stages instead of fixing these first. The three findings stay logged
here as known, diagnosed, unfixed work — not silently dropped.

---

### Research Agents (x2, parallel) — scope `coder/` and `runner/` first slices

**Asked:** Two independent scoping passes, run concurrently since neither
depends on the other's output. (a) `coder/`: smallest useful slice that
turns `reader/output/<paper>.json` into one self-contained HuggingFace
`Trainer` script. (b) `runner/`: Docker sandbox design, AutoReproduce's
two-stage dry-run mechanic, CPU-feasible capped run, log truncation, Haiku
triage, timeout enforcement. Both grounded in
`docs/notes/coder-agent-precedents.md` §4 (treated as settled research, not
re-derived) and project plan §2.3/§2.4.

**Result (coder):** one Sonnet 5 tool-use call; hand-rolled `WideResNet`
`nn.Module` rather than `AutoModelForImageClassification` (HF's built-in
ResNets assume ImageNet's 224x224 7x7-conv+maxpool stem — wrong for CIFAR's
32x32); `torchvision.datasets.CIFAR10` rather than `datasets.load_dataset`
(the paper's 4px-reflection-pad augmentation maps directly onto
`torchvision.transforms`, and it keeps `pyarrow` out of the future Runner
image); `ast.parse()` as a free pre-write syntax gate; `max_tokens` starting
at 8192+ given this repo's two prior `stop_reason: max_tokens` incidents.

**Result (runner):** `python:3.11-slim` + CPU torch wheel (NOT
`pytorch/pytorch`, which drags in CUDA layers) — worth being precise that
the container runs its own **Linux** Python, so the host's Intel-macOS +
3.13 trap simply doesn't apply inside it; this stage is that trap's
*solution*, not another instance of it. Script bind-mounted in (not `COPY`d,
since the Coder regenerates it every retry — a bind mount keeps the image
static and cacheable); results bind-mounted out (survives a timeout-kill,
unlike `docker cp`). Three escalating stages, each just a differently-flagged
invocation of the same script: shape probe → structural dry run → capped
run. Log truncation head-and-tail (~8k chars: first 2k catches import/setup
failures, last 6k catches the traceback), full log always kept on disk.
Haiku called **only** to disambiguate a genuine non-timeout failure
(`recoverable_error` vs `environment_error`) — success and timeout are both
determined mechanically, saving an API call on the common path. Timeout via
host-side `subprocess.run(timeout=...)` plus an explicit `docker kill`,
since `TimeoutExpired` only kills the `docker run` *client* while the
container keeps running inside `dockerd` (and no native `docker run
--timeout` flag exists).

**Reconciliation needed (the two agents disagreed):** each independently
proposed a different `metrics.json` schema — Coder's claim-linked
(`claim_id`/`metric`/`unit`/`value`), Runner's training-diagnostics-focused
(`train_loss`/`epochs_completed`/...). Merged into one 12-key shape carrying
both, with `metric`/`unit` copied verbatim from the targeted claim so a
future Critic can diff against `reported_value` with no unit conversion.

**Blocking prerequisite found:** the Docker daemon is not running on this
machine (`docker --version` works, v29.4.1; `docker info` and `docker run
hello-world` both fail to connect to the socket). Docker Desktop must be
started before `runner/` can be built or tested.

---

### Coding Agent — build `coder/`

**Asked:** Implement the scoped design as a production-ready stage, with one
change Engincan requested directly: the Coder should read **the paper itself**
alongside `reader/`'s JSON. Rationale — `reader/` has no `architecture_notes`
stage yet, so the architecture description exists only in the OCR Markdown;
passing both grounds the architecture in the paper's real text instead of
pretrained knowledge, and closes exactly the gap the earlier scoping pass had
flagged as its open question #2. Both inputs are only ~18k tokens combined,
so no retrieval machinery was needed. Verify with ruff/mypy --strict/
pre-commit, then actually run against Wide Residual Networks.

**Result:** `coder/{__init__,base,script_writer,pipeline}.py` + `README.md`;
`pyproject.toml` (`coder` extra + packages), `.gitignore`, and
`.pre-commit-config.yaml` (mypy `files:` widened to `^(ocr|reader|coder)/`)
updated. ruff/mypy --strict/pre-commit all pass.

**Real run, verified independently rather than taken on the agent's word:**
346-line script, `stop_reason=tool_use` (not `max_tokens`), both gates
passed, 0 missing CLI flags, 9 hyperparameters, 7 assumptions recorded.
Direct inspection of the generated `train.py` confirms: correct WRN
(`n=(depth-4)//6`, widths `16/16k/32k/64k`, pre-activation BN-ReLU-Conv
B(3,3), Kaiming init); all 8 required flags with the paper's real values as
defaults; `torchvision` + `Pad(4, padding_mode="reflect")`; the exact 12-key
metrics contract with `"test error"`/`"%"` verbatim from c34. **Regime
matching — the thing most likely to silently reproduce the wrong number —
is correct**: CIFAR milestones `[60,120,160]` γ=0.2, not SVHN's
`[80,120]` γ=0.1; no dropout; mean/std normalization, not ZCA. Unprompted
nice touch: it rescales the LR milestones proportionally when `--epochs` is
reduced, so a Runner smoke test still gets a sensible schedule.

**Two findings worth keeping:**
1. **Intermittent tool-field leak (worked around).** In roughly half the
   observed calls the model serialized later tool fields as literal
   `<parameter name="...">` text *inside* an earlier field, twice swallowing
   `script_content` entirely. `_recover_leaked_fields()` splits them back
   out deterministically, restricted to the 8 known field names so `<`/`>`
   inside generated code can't false-trigger. Free, but a workaround for a
   model-behavior quirk, not a fix — worth revisiting if it changes.
2. **A real bug in a generated script that the gates cannot catch.** One run
   produced a `_NoOpScheduler` whose `get_last_lr()` reads
   `self.optimizer.param_groups`, but the class never sets `self.optimizer`
   — an `AttributeError` the moment `Trainer` logs the LR. `ast.parse` proves
   a script *parses*, not that it *runs*. Deliberately not fixed with a third
   static-analysis gate: this is precisely the failure class the Runner
   exists to catch, and it's concrete local evidence for what
   `coder-agent-precedents.md` already argues from AutoReproduce's ablation
   (dropping the debug loop barely moved their alignment score but nearly
   tripled their performance gap). Not present in the final committed script.

---

### Coding Agent — build `runner/` (Docker path unproven)

**Asked:** Implement the scoped Runner design, with one correction Engincan
made directly: **`reproduce.sh <mode>` is the entire interface.** Runner never
builds a `python` command and never passes a `--flag`, so it stays
paper-agnostic — a future paper needing different arguments changes only its
own `reproduce.sh`. The earlier flag-passing design is superseded. Told
explicitly that the Docker daemon was blocked behind a macOS password prompt,
and to verify everything daemon-independent rather than fake it or block.

**Result:** `runner/{__init__,docker_runner,triage,pipeline}.py`, `Dockerfile`,
`README.md`; `pyproject.toml`/`.gitignore`/`.pre-commit-config.yaml` updated.
ruff/mypy --strict/pre-commit all pass (re-run independently, not taken on
trust). 110 assertions over every daemon-free function, including
`subprocess.run` monkeypatched end-to-end. The genuinely valuable ones: the
timeout path issues `docker kill` against the exact `--name` it launched with;
partial `TimeoutExpired` output arrives as raw **bytes** even in text mode and
is decoded rather than crashing the handler reporting the timeout; a 150k-char
stdout does not push stderr's traceback out of the truncated excerpt (with a
control case showing it *does* if you combine streams before truncating); no
API call is spent on success or on timeout. Image pins were verified over HTTP
since they could not be verified by building.

**Verification path taken: (b), the honest one.** No image was ever built and
no container ever ran. `runner/README.md`'s Status section separates verified
from unrun, following `ocr/README.md`'s docling/mineru precedent.

**Caught in review — the agent got one thing wrong.** It read
`coder/README.md`'s "Known issues" entry about a `_NoOpScheduler` bug and
concluded that bug is in the committed script, so its report predicted the
first `probe` run would fail. Checked directly: `grep -c "_NoOpScheduler"`
returns **0**. That bug came from a *discarded* generation. The claim had
propagated into `runner/README.md` and was corrected there. Worth noting as a
delegation lesson: an agent reading a "Known issues" section can mistake
historical findings for current state, and a claim about what's in a file is
cheap to verify and should be.

**Open, deliberately unfixed:** stage timeout budgets (probe 1200s / smoke
2400s / capped 7200s) are estimates never checked against a real run; the
container runs as root, which macOS Docker Desktop hides but a Linux host will
not; `--memory`/`--cpus` are exposed but unset, because an arbitrary cap
produces exit 137 that looks identical to a script crash and would be triaged
as `recoverable_error`, sending the Coder to fix a bug that does not exist.

---

### Direct investigation (no subagent) — the Intel-mac torch trap is worse than documented

**Why:** with Docker blocked, the fallback plan was to run the generated script
natively in a Python 3.11 venv to get real timing numbers for Runner's budgets.

**Found:** Python 3.11 does install torch on this host — but only `2.2.2`, the
last Intel-mac wheel. Current `transformers` (5.x) requires torch >= 2.5 and,
on finding 2.2.2, **does not error**: it silently disables the PyTorch backend
and logs "Models won't be available", so an HF `Trainer` script fails
confusingly rather than at import. torch 2.2.2 is also compiled against NumPy
1.x and warns loudly under NumPy 2. One narrow legacy combination does work:
`torch==2.2.2` + `transformers==4.46.3` + `accelerate<1.2` + `numpy<2`.

**Why it matters:** it upgrades Docker from convenience to genuinely
load-bearing for `runner/`, and it means the container's `transformers==4.46.3`
pin is not arbitrary. Recorded in CLAUDE.md's Tooling section so nobody
re-derives it.

---

### End-to-end verification — the full chain runs, in and out of Docker

**Native run first (Docker still blocked):** `reproduce.sh probe` executed in a
throwaway 3.11 venv using the legacy pin-set above. **Exit 0**, and it wrote a
`metrics.probe.json` matching `coder/`'s documented contract exactly — all
twelve keys, `metric`/`unit` carried verbatim from claim c34. Measured: 48.7 s
train (256 samples, 5.256 samples/s), ~19 s across two eval passes, ~67 s
compute total — inside **1774 s** of wall clock, because the cold CIFAR-10
download took the other ~1707 s.

**That measurement caught a real bug in `runner/`.** Stage budgets had been
estimated, with the cold download folded into `probe` at 1200 s. The real run
took 1774 s, so the very first Docker run would have timed out — and a timeout
looks exactly like a hung script, so it would have been triaged
`recoverable_error`, sending the Coder to fix a bug that does not exist.
Budgets recalibrated from the measurement (probe 2700 / smoke 900 / capped
1800), and `runner/cache/datasets/` seeded with the already-downloaded CIFAR-10
so the first container run starts warm.

**Then Docker came up and the container path was verified for real:**
```
[docker] built reprobot-runner:latest in 224.2s
  [probe] PASSED in 166.4s (exit 0)
  [probe] metrics (file): claim_id=c34 test error=92.1875 %
[done] 1 paper(s) passed, 0 failed, 0 could not be run at all
```
`runner_output.json`: `status: success`, `triage: null` — confirming no API call
is spent on the success path. Logs captured to separate stdout/stderr files.

**The cache design paid for itself immediately:** the same `probe` took 1774 s
natively with a cold cache and **166 s** in the container with a warm one. That
~1600 s delta is the one-time download all 8 papers would otherwise repeat.

**Status after this:** the full chain — PDF → `ocr/` → `reader/` → `coder/` →
`runner/` — is proven end to end. What does *not* exist yet is the loop:
`coder/base.py` accepts `feedback`, `runner/triage.py` emits `suggested_fix`,
and nothing connects them. That is the next increment and the project's actual
differentiator.

---

### Coding Agent — build `orchestrator/`, the Coder↔Runner retry loop

**Asked:** Build the fifth stage: the project plan's §1.2 shared-memory state
object plus the retry loop that routes a Runner failure back into the Coder as
feedback. Four design calls handed down rather than left open — plain Python
over LangGraph (§1.3 recommends it, but at this slice the graph is three nodes
and one edge, and CLAUDE.md says not to add graph machinery preemptively);
`environment_error` stops immediately instead of consuming retries; the §2.1
plateau guard; and `script_version`/`diff_from_previous` populated for real.
Explicitly told: do NOT build the Critic, and if the loop fails to converge,
report that honestly rather than tuning the test until it passes.

**Result:** `orchestrator/{__init__,state,loop,pipeline}.py` + `README.md`, plus
one backwards-compatible change to `coder/pipeline.py` (`run()` gained
`feedback: str | None = None`, threaded into `writer.write()`; default `None`,
so a direct `coder.pipeline` run is unaffected). Seven terminal verdicts.
ruff/mypy --strict/pre-commit all pass; 118 assertions with no Docker and no API
key, covering every verdict path, the plateau maths at and around the threshold,
diff truncation, and state round-tripping.

**The key test — the loop repaired a real runtime bug, first retry.** The
historical `self.optimizer`-never-assigned bug was reintroduced into a *scratch
copy* of the WRN script (valid Python, so `ast.parse` passes — precisely the
failure class the loop exists for):
```
[probe] FAILED in 169.5s (exit 1)
  AttributeError: '_StepDecayScheduler' object has no attribute 'optimizer'
[triage] category: recoverable_error
[decide] RETRY - triage says the fault is in the generated script
[plateau] v1 vs v2 line similarity 0.4079 (stop at >= 0.98)
[probe] PASSED in 2000.8s (exit 0)
[loop] VERDICT: success - finished after 2 attempt(s), 1/2 retries used
```
Verified directly in `state.json`: `verdict: success`, `retry_count: 1`,
`script_version: 2`, `critic_output: null`, 7 history entries. The fix wasn't
luck — v2 kept the class triage named and did exactly what it was told.

**Agent's own finding, worth more than the feature:** `autojunk=True` (the
`SequenceMatcher` default) discards repeated lines and measured **0.9975 vs.
0.7625** on the same pair — with the 0.98 threshold that is the difference
between "plateaued, stop" and "keep going". Left off deliberately.

**A real defect the loop surfaced, now fixed.** Attempt 2's probe took **2000 s
vs. 169 s** — the regenerated script defaulted its data dir to
`./<output-dir>/data` instead of `./data`, missed `runner/`'s shared cache mount,
and re-downloaded 170 MB. Nothing failed, which is what makes it nasty: it
silently costs twelve minutes, and a retry loop re-rolls that dice every
regeneration while the stage budgets assume a warm cache. `runner/README.md` had
called this exact risk "a convention, not a contract" — so the convention was
made a contract: `--data-dir` added to `coder/`'s `REQUIRED_CLI_FLAGS` with the
prompt requiring exactly `"./data"`, enforced by the same literal check as every
other flag. (The committed script already complied; only the regeneration drifted.)

**Honest limits:** convergence is proven for one injected bug on one paper. The
retry regenerates the whole file rather than patching, so a fix can introduce a
new defect, and nothing diffs semantics. `environment_error`/`timeout`/
`untriaged_error`/`coder_failed` are verified against fakes only, never observed
in the wild.

---

### Coding Agent — complete the Reader: `method_summary` + `architecture_notes`

**Asked:** Build the two §1.2 fields never implemented. Schema for
`architecture_notes` specified up front (`model_name`, `overall_structure`,
`components[]`, `key_equations`, `depth_or_scale`, `unstated_details`), with
`data_pipeline`'s "deliberately does not guess" discipline carried over and
sharpened: code has to run, so an invented channel count trains the *wrong
network* and reports a number for it. Told to verify against Network In Network
specifically — the hard case, custom `mlpconv`, no dimensions in-paper — and to
report honestly if it invented anything.

**Result:** both extractors built and wired in (5 stages now). NIN came out
clean: the three-mlpconv + global-average-pooling structure read off the
Figure 2 block, equation (2) verbatim, 12 `unstated_details`, **zero invented
numbers**, quoting the paper's own deferral ("the detailed settings of the
parameters are provided in the supplementary materials"). Final NIN output: 12
claims, 17 hyperparameters, 4 datasets, 7 architecture components, 279-word
method summary, 4 flags after 3 passes.

**The agent corrected the orchestrator, and was right.** The task brief asserted
Wide Residual Networks states `depth = 6N + 4`. It does not — verified
independently by grepping the Markdown: the paper gives only Table 1's symbolic
`N` and the `WRN-n-k` naming convention. That formula is community knowledge.
The extractor refused to supply it and said so, and on a later pass the
validator independently re-checked and confirmed it absent. Worth recording
because the generated `train.py` from the earlier Coder run *asserts* that
formula — it came from the model's priors, not the paper. That silent
substitution is precisely what this stage exists to surface.

**Fix applied on review — `key_equations` was actively misleading.** It returned
three bare LaTeX strings for NIN: conventional convolution, mlpconv, and maxout.
Indistinguishable, so a Coder had three plausible specifications and would have
confidently implemented one. Instructing the model in the prompt to "skip
baseline equations" did not work — it kept returning them. Forcing the *schema*
to carry a role did: each entry is now an object with `latex`/`label`/`defines`/
`is_own_method`, and the run reports "3 (1 defining this paper's own method, 2
shown for contrast)". Three plausible specs is worse for a code generator than
none at all.

**The agent's own report was inaccurate about what it changed** — it said it
left `claims.py`/`hyperparameters.py`/`data_pipeline.py`/`validator.py`
untouched; git showed all four modified. It had refactored every call site onto
a new shared `reader/tooluse.py`. The refactor is *better* than what it
described (hardening lands once rather than six times), but the mismatch meant
its verification claims could not be trusted, so the full pipeline was re-run
directly. Delegation lesson, same shape as the earlier `_NoOpScheduler` miss: an
agent's description of its own diff is a claim, and `git diff` is cheap.

**That re-run proved the hardening's worth immediately.** `reader/tooluse.py`
handles three silent failure modes it found — `max_tokens` truncation, an
all-empty payload, and a *double-encoded* tool input where the model returns the
whole payload as a JSON string under one key on a clean `stop_reason: tool_use`.
The last one fired live during verification:
```
[data_pipeline] tool input arrived double-encoded (whole payload as a JSON
string under 'datasets') - unwrapped it; the extraction below is real, not empty
[data_pipeline] datasets extracted: 4
```
Before the fix that was a silent `0 datasets` on a clean run — indistinguishable
from a paper that genuinely had none.

**Open:** `hyperparameters.sources_examined` is still empty and is now costing
retry budget, since the validator flags it every run. Cost has risen materially
— five extractors plus a validator inside a retry loop, ~3 passes per paper.

---
