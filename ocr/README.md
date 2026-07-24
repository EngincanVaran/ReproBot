# ocr/ — PDF extraction backends

Four independent ways to turn a paper PDF into Markdown, so we can compare
extraction quality before committing to one for the Reader agent. Each
backend is its own script with the same CLI shape and writes to its own
subfolder under `ocr/output/` (gitignored — regenerate anytime).

| Backend | Script | Install | How it parses |
|---|---|---|---|
| **pdfplumber** | `pdfplumber_extract.py` | `uv sync --extra pdfplumber` | Rule-based, no ML models |
| **Claude (VLM)** | `vlm_extract.py` | `uv sync --extra vlm` + API key | Claude reads rendered page images directly |
| **Docling** | `docling_extract.py` | separate env — see below | Learned layout/table models (torch) |
| **MinerU** | `mineru_extract.py` | separate env — see below | Learned layout/OCR models (torch) |

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

Extraction only, for now — no LLM-based parsing (claims/hyperparameters/
figures) is wired up yet beyond the VLM backend's raw page transcription.
That's the next step, once we've eyeballed and compared all four backends'
raw output on a few papers.
