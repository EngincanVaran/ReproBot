# viewer/ — pipeline dashboard

A Streamlit app that shows what `ocr/` and `reader/` have already produced:
a per-paper pipeline status table (OCR done? Reader done? how many
validation flags?), then, per selected paper, its extracted claims,
hyperparameters, data-pipeline info, validation flags, raw JSON, and the raw
OCR Markdown.

Mostly **display only** — it loads `ocr/output/vlm/*.md` and
`reader/output/*.json` from disk and renders them, without altering how
either stage extracts or operates. The one exception is the
**"Import a paper"** section, placed in the main page just above the paper
selector:

1. Upload a PDF, optionally edit its filename stem, optionally cap the page
   count (cheap testing before committing to the whole paper).
2. Click "Save + run OCR extraction". This saves the PDF to
   `viewer/uploads/` (**not** `dataset/` — `dataset/` is curated by someone
   else in parallel, see the repo's `CLAUDE.md`) and calls
   `ocr/vlm_extract.py`'s own `run_vlm()` function directly, unmodified —
   the same backend already used for the rest of the papers, writing to the
   same `ocr/output/vlm/` directory.
3. Once OCR finishes, the paper shows up in the status table next to the
   curated `dataset/` papers (`viewer/uploads/*.pdf` is merged into the
   paper list).

Any paper with OCR done but no Reader output yet also gets a
**"Run Reader extraction"** button on its Overview tab, which calls
`reader/pipeline.py`'s own `run_pipeline()` directly, unmodified — the same
claims/hyperparameters/data_pipeline extractors plus the validation retry
loop already used for the rest of the papers.

Both actions call the Anthropic API — OCR makes one paid request per page;
Reader extraction makes several (one per extractor, plus validation, with
up to `max_retries` retry passes) — same cost as running either script from
the CLI, and both require an `ANTHROPIC_API_KEY` in `.env` at the repo root
(copy `.env.example`).

## Setup

```bash
uv sync --extra viewer
```

## Run

```bash
uv run streamlit run viewer/app.py
```

Opens in a browser tab (default `http://localhost:8501`). Papers are listed
from `dataset/*.pdf`; a paper with no OCR or Reader output yet still shows
up in the status table, just marked as not run.
