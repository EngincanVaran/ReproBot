# viewer/ — pipeline dashboard

A Streamlit app that shows what `ocr/`, `reader/`, and `coder/` have already
produced: a per-paper pipeline status table (OCR done? Reader done? Coder
done? how many validation flags?), then, per selected paper, its method
summary, architecture notes, extracted claims, hyperparameters, data-pipeline
info, validation flags, raw JSON, the generated training script (+
`reproduce.sh`), and the raw OCR Markdown.

Mostly **display only** — it loads `ocr/output/vlm/*.md`,
`reader/output/*.json`, and `coder/output/<paper>/coder_output.json` from disk
and renders them, without altering how any stage extracts or operates. The
exceptions are the **"Import a paper"** section, placed in the main page just
above the paper selector, and the per-paper **"Run Reader extraction"** /
**"Generate code"** buttons:

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
method_summary/architecture_notes/claims/hyperparameters/data_pipeline
extractors plus the validation retry loop already used for the rest of the
papers.

Any paper with Reader output but no Coder output yet gets a
**"Generate code"** button, which calls `coder/pipeline.py`'s own
`run_pipeline()` directly, unmodified — it resolves the paper's
`ocr/output/vlm/<paper>.md` as its required second input, makes one Claude
tool-use call to write the training script, then runs the deterministic
`ast.parse` syntax gate and CLI-flag check. A run that fails the syntax gate
still surfaces its `train.py.invalid` + `coder_output.failed.json` in the
Code tab.

All three actions call the Anthropic API — OCR makes one paid request per
page; Reader extraction makes several (one per extractor, plus validation,
with up to `max_retries` retry passes); Coder makes one — same cost as
running each script from the CLI, and all require an `ANTHROPIC_API_KEY` in
`.env` at the repo root (copy `.env.example`).

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
