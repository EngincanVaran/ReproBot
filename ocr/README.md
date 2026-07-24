# ocr/ — PDF extraction backends

Four independent ways to turn a paper PDF into Markdown, so we can compare
extraction quality before committing to one for the Reader agent. Each
backend is its own script with the same CLI shape and writes to its own
subfolder under `ocr/output/` (gitignored — regenerate anytime).

| Backend | Script | Install | How it parses | Verified |
|---|---|---|---|---|
| **pdfplumber** | `pdfplumber_extract.py` | `uv sync --extra pdfplumber` | Rule-based, no ML models | ✅ ran end-to-end |
| **Claude (VLM)** | `vlm_extract.py` | `uv sync --extra vlm` + API key | Claude reads rendered page images directly | ✅ ran end-to-end |
| **Docling** | `docling_extract.py` | separate env — see below | Learned layout/table models (torch) | not runnable on this machine, see below |
| **MinerU** | `mineru_extract.py` | separate env — see below | Learned layout/OCR models (torch) | not runnable on this machine, see below |

See `docs/notes/reader-agent-precedents.md` for why learned-layout parsing
(MinerU/Docling-style) was the starting recommendation — both AutoReproduce
and AutoP2C use it, and AutoReproduce's own ablation shows it measurably
improves downstream code fidelity vs. naive text extraction. pdfplumber and
the Claude VLM backend are the two we can run and compare locally right now;
Docling and MinerU fill out the same comparison once run elsewhere.

## Setup (pdfplumber + Claude VLM — this repo's managed environment)

```bash
uv sync --extra pdfplumber --extra vlm --group dev
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

```bash
# whole dataset
uv run python -m ocr.pdfplumber_extract
uv run python -m ocr.vlm_extract

# a single paper
uv run python -m ocr.pdfplumber_extract --input "dataset/Deep Residual Learning for Image Recognition.pdf"

# cheap test run: only the first 2 pages of every paper (VLM backend costs real API tokens)
uv run python -m ocr.vlm_extract --max-pages 2

# custom output location
uv run python -m ocr.pdfplumber_extract --output somewhere/else
```

Both scripts default to reading from `dataset/` (the CIFAR-10 paper set) and
skip papers they've already extracted, so reruns are cheap.

## How the Claude (VLM) backend handles figures

`vlm_extract.py` renders each PDF page to a **full-page PNG** via pypdfium2
(`page.render(scale=...)`) and sends that whole image to Claude in one call —
it does not extract or strip figures separately first. Whatever's visually on
the page (body text, equations, diagrams, plots, tables) is in the pixels
Claude receives, so it genuinely looks at figures rather than working from
text alone. `ocr/output/vlm/Network In Network.md` is a good example: Claude
correctly described Figure 1's two-panel diagram, including that panel (b)
shows "a small multilayer perceptron (represented by two columns of circular
nodes)" — that detail only comes from actually reading the image.

That said, the current prompt (`PROMPT` in `vlm_extract.py`) only asks for a
*brief* bracketed description of each figure — it's not instructed to
transcribe every number on a plot axis or describe a diagram's full
structure. AutoP2C's own VLM-parsing prompt (see
`docs/notes/reader-agent-precedents.md`) goes further: capture *all*
numerical elements, ignore purely decorative detail, and cross-reference the
figure's caption to ground the description. Worth adopting that level of
detail once figure interpretation actually feeds the Reader's claim/
hyperparameter extraction — not needed just to compare raw backend output.

## Docling and MinerU — separate environment required

Both are **not** in this project's `pyproject.toml` / `uv.lock`, on purpose.
Both depend on `torch`, and there is no PyTorch release that supports Python
3.13 **and** ships Intel-macOS (x86_64) wheels — PyTorch added 3.13 support
at `2.5.0`, the same release range where it dropped Intel-macOS wheels
entirely (last Intel-mac wheel: `2.2.2`). Declaring either as an extra here
would make the *whole* project un-syncable on this machine, since `uv.lock`
resolves the union of all extras. If you're running these on
different hardware (Apple Silicon, Linux, CI) or in Docker, this constraint
doesn't apply to you.

The scripts themselves (`docling_extract.py`, `mineru_extract.py`) are
already written against the same CLI shape as the other two — just install
their dependencies into whatever environment you're using and run them the
same way:

```bash
# Docling — needs Python <=3.12 on Intel macOS (torch>=2.2.2 has no cp313 wheel there);
# fine as-is on Apple Silicon, Linux, or Python 3.13 on those platforms.
pip install "docling>=2.0"
python -m ocr.docling_extract
python -m ocr.docling_extract --input "dataset/some-paper.pdf"

# MinerU — needs a platform where torch>=2.6.0 has wheels (Apple Silicon, Linux,
# Windows, or Docker). Intel macOS cannot run this locally at any Python version.
pip install "mineru[pipeline]>=2.0"
python -m ocr.mineru_extract
```

Output layout matches the other two backends: `ocr/output/docling/<paper>.md`
and `ocr/output/mineru/<paper>/<mineru's own internal layout>/<paper>.md` +
`images/`.

## Status

- pdfplumber and Claude VLM: both ran end-to-end locally (`uv sync`, ruff,
  mypy --strict, and pre-commit all pass); sample output under
  `ocr/output/`.
- Docling and MinerU: code written, not yet run (need a non-Intel-macOS /
  Python-≤3.12 environment — someone else is running these).
- Extraction only so far — no claims/hyperparameter extraction is wired up
  yet, beyond the VLM backend's brief inline figure descriptions (see above).
  That's the next step, once we've eyeballed and compared all four backends'
  raw output on a few papers.
