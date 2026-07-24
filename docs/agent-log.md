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
