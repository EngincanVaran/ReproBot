"""Streamlit viewer for the ocr/ and reader/ pipeline outputs.

Mostly read-only: it loads files that `ocr/` and `reader/` have already
written to `ocr/output/vlm/*.md` and `reader/output/*.json` and displays
them, without altering how either stage extracts or operates. Two
exceptions, both calling each stage's own entry-point function directly,
unmodified, exactly as their CLIs do:

- The "Import a paper" section lets a user upload a new PDF and run OCR on
  it, via `ocr/vlm_extract.py`'s `run_vlm()`.
- A paper with OCR done but no Reader output yet gets a "Run Reader
  extraction" button on its Overview tab, via `reader/pipeline.py`'s
  `run_pipeline()`.

Neither reimplements or changes the extraction logic itself.

Uploaded PDFs are saved to `viewer/uploads/`, not `dataset/` - `dataset/` is
curated by someone else in parallel (see CLAUDE.md), so this app never
writes into it. The paper list below merges `dataset/*.pdf` and
`viewer/uploads/*.pdf`, so an imported paper shows up next to the rest.

Requires the `viewer` extra: `uv sync --extra viewer`. Running OCR via the
import section additionally requires an ANTHROPIC_API_KEY in `.env` (same
requirement as running `ocr/vlm_extract.py` directly) and makes one paid
Claude API call per page.

Usage:
    uv run streamlit run viewer/app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from ocr.vlm_extract import run_vlm
from reader.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
UPLOADS_DIR = REPO_ROOT / "viewer" / "uploads"
OCR_DIR = REPO_ROOT / "ocr" / "output" / "vlm"
READER_DIR = REPO_ROOT / "reader" / "output"

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9 ._-]")


def list_papers() -> list[str]:
    """Paper names (no extension), sourced from dataset/ (the curated
    replication targets) plus viewer/uploads/ (papers imported through this
    app), so every paper any stage could act on shows up, even ones no stage
    has processed yet."""
    papers: list[str] = []
    if DATASET_DIR.is_dir():
        papers.extend(p.stem for p in DATASET_DIR.glob("*.pdf"))
    if UPLOADS_DIR.is_dir():
        papers.extend(p.stem for p in UPLOADS_DIR.glob("*.pdf") if p.stem not in papers)
    return sorted(papers)


def sanitize_paper_name(raw_name: str) -> str:
    """Reduce a user-supplied name to a safe filename stem: drop any
    directory components and extension, then strip characters outside a
    conservative allow-list so it can't escape `viewer/uploads/`."""
    stem = Path(raw_name).name
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    stem = _UNSAFE_NAME_CHARS.sub("", stem).strip()
    return stem or "untitled-paper"


def ocr_markdown_path(paper: str) -> Path:
    return OCR_DIR / f"{paper}.md"


def reader_json_path(paper: str) -> Path:
    return READER_DIR / f"{paper}.json"


@st.cache_data
def load_reader_output(path_str: str, mtime: float) -> dict[str, Any]:
    """`mtime` busts the cache when pipeline.py re-writes the file, e.g.
    after a retry loop finishes with new content."""
    path = Path(path_str)
    logger.info("Loading reader output: {}", path)
    return cast("dict[str, Any]", json.loads(path.read_text()))


@st.cache_data
def load_markdown(path_str: str, mtime: float) -> str:
    path = Path(path_str)
    logger.info("Loading OCR markdown: {}", path)
    return path.read_text()


def run_ocr_extraction(pdf_path: Path, paper_name: str, *, max_pages: int | None) -> None:
    """Invoke ocr/vlm_extract.py's own run_vlm() unmodified - same backend,
    same output path, same behavior as running the CLI by hand."""
    load_dotenv(REPO_ROOT / ".env")
    try:
        client = Anthropic()
    except Exception as exc:  # noqa: BLE001 - surface any client-setup error in the UI
        st.error(f"Could not create Anthropic client (check ANTHROPIC_API_KEY): {exc}")
        return

    with st.status(f"Running OCR on '{paper_name}'...", expanded=False) as status:
        try:
            result = run_vlm(pdf_path, OCR_DIR, client, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001 - surface any extraction error in the UI
            status.update(label=f"OCR failed: {exc}", state="error")
            logger.error("OCR extraction failed for {}: {}", paper_name, exc)
            return
        status.update(label=f"OCR done -> {result.markdown_path.name}", state="complete")
    logger.info("OCR extraction finished: {}", result.markdown_path)
    st.rerun()


def run_reader_extraction(markdown_path: Path, paper_name: str) -> None:
    """Invoke reader/pipeline.py's own run_pipeline() unmodified - same
    extractors, validator, and retry loop as running the CLI by hand."""
    load_dotenv(REPO_ROOT / ".env")
    try:
        client = Anthropic()
    except Exception as exc:  # noqa: BLE001 - surface any client-setup error in the UI
        st.error(f"Could not create Anthropic client (check ANTHROPIC_API_KEY): {exc}")
        return

    with st.status(f"Running Reader extraction on '{paper_name}'...", expanded=False) as status:
        try:
            output = run_pipeline(markdown_path, READER_DIR, client)
        except Exception as exc:  # noqa: BLE001 - surface any extraction error in the UI
            status.update(label=f"Reader extraction failed: {exc}", state="error")
            logger.error("Reader extraction failed for {}: {}", paper_name, exc)
            return
        status.update(
            label=f"Reader extraction done - {len(output.validation.flags)} flag(s)",
            state="complete",
        )
    logger.info("Reader extraction finished for {}", paper_name)
    st.rerun()


def render_import_section() -> None:
    st.subheader("Import a paper")
    uploaded_file = st.file_uploader("PDF file", type=["pdf"])
    if uploaded_file is None:
        return

    paper_name = st.text_input("Paper name", value=sanitize_paper_name(uploaded_file.name))
    limit_pages = st.number_input(
        "Limit pages (0 = whole paper)",
        min_value=0,
        value=0,
        help=(
            "Each page is one paid Claude API call - set a small limit to "
            "test cheaply before running the whole paper."
        ),
    )

    if st.button("Save + run OCR extraction"):
        clean_name = sanitize_paper_name(paper_name)
        if clean_name in list_papers():
            st.error(f"A paper named '{clean_name}' already exists. Choose another name.")
            return
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        target_pdf = UPLOADS_DIR / f"{clean_name}.pdf"
        target_pdf.write_bytes(uploaded_file.getvalue())
        logger.info("Saved uploaded paper: {}", target_pdf)
        run_ocr_extraction(target_pdf, clean_name, max_pages=limit_pages or None)


def render_pipeline_status(papers: list[str]) -> None:
    st.subheader("Pipeline status")
    rows = []
    for paper in papers:
        has_ocr = ocr_markdown_path(paper).exists()
        reader_path = reader_json_path(paper)
        has_reader = reader_path.exists()
        flags = 0
        if has_reader:
            data = load_reader_output(str(reader_path), reader_path.stat().st_mtime)
            flags = len(data.get("validation", {}).get("flags", []))
        rows.append(
            {
                "Paper": paper,
                "OCR": "done" if has_ocr else "-",
                "Reader": "done" if has_reader else "-",
                "Validation flags": flags if has_reader else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_claims(data: dict[str, Any]) -> None:
    claims = data.get("claims", {}).get("claims", [])
    if not claims:
        st.info("No claims extracted for this paper.")
        return
    st.dataframe(pd.DataFrame(claims), use_container_width=True, hide_index=True)


def render_hyperparameters(data: dict[str, Any]) -> None:
    hyperparameters = data.get("hyperparameters", {}).get("hyperparameters", [])
    if not hyperparameters:
        st.info("No hyperparameters extracted for this paper.")
        return
    st.dataframe(pd.DataFrame(hyperparameters), use_container_width=True, hide_index=True)


def render_data_pipeline(data: dict[str, Any]) -> None:
    pipeline = data.get("data_pipeline", {})
    reference_urls = pipeline.get("reference_urls", [])
    if reference_urls:
        st.markdown("**Reference URLs**")
        for url in reference_urls:
            st.markdown(f"- {url}")
    datasets = pipeline.get("datasets", [])
    if not datasets:
        st.info("No dataset preprocessing info extracted for this paper.")
        return
    for dataset in datasets:
        with st.expander(dataset.get("dataset", "(unnamed dataset)")):
            for key, value in dataset.items():
                if key == "dataset":
                    continue
                st.markdown(f"**{key}**: {value}")


def render_validation(data: dict[str, Any]) -> None:
    validation = data.get("validation", {})
    st.markdown(f"**Validation passes (attempts):** {validation.get('attempts', 0)}")
    retried = validation.get("retried_stages", [])
    st.markdown(f"**Retried stages:** {', '.join(retried) if retried else 'none'}")
    flags = validation.get("flags", [])
    if not flags:
        st.info("No validation flags.")
        return
    for flag in flags:
        st.markdown(f"**{flag.get('relates_to', '(unspecified)')}**")
        st.markdown(flag.get("description", ""))
        st.divider()


def render_paper(paper: str) -> None:
    reader_path = reader_json_path(paper)
    ocr_path = ocr_markdown_path(paper)

    if not reader_path.exists() and not ocr_path.exists():
        st.warning(f"No pipeline output yet for '{paper}'.")
        return

    data = (
        load_reader_output(str(reader_path), reader_path.stat().st_mtime)
        if reader_path.exists()
        else None
    )

    tab_names = ["Overview"]
    if data is not None:
        tab_names += ["Claims", "Hyperparameters", "Data Pipeline", "Validation", "Raw JSON"]
    if ocr_path.exists():
        tab_names.append("Raw OCR Markdown")

    tabs = st.tabs(tab_names)
    tab_by_name = dict(zip(tab_names, tabs, strict=True))

    with tab_by_name["Overview"]:
        st.markdown(f"### {paper}")
        st.markdown(f"- OCR Markdown: {'available' if ocr_path.exists() else 'not yet run'}")
        st.markdown(f"- Reader output: {'available' if data is not None else 'not yet run'}")
        if data is not None:
            st.markdown(f"- Claims: {len(data.get('claims', {}).get('claims', []))}")
            st.markdown(
                f"- Hyperparameters: "
                f"{len(data.get('hyperparameters', {}).get('hyperparameters', []))}"
            )
            st.markdown(f"- Datasets: {len(data.get('data_pipeline', {}).get('datasets', []))}")
        elif ocr_path.exists():
            st.caption(
                "Calls the Anthropic API (claims + hyperparameters + data_pipeline "
                "extractors, plus validation with retries)."
            )
            if st.button("Run Reader extraction", key=f"run_reader_{paper}"):
                run_reader_extraction(ocr_path, paper)

    if data is not None:
        with tab_by_name["Claims"]:
            render_claims(data)
        with tab_by_name["Hyperparameters"]:
            render_hyperparameters(data)
        with tab_by_name["Data Pipeline"]:
            render_data_pipeline(data)
        with tab_by_name["Validation"]:
            render_validation(data)
        with tab_by_name["Raw JSON"]:
            st.json(data)

    if ocr_path.exists():
        with tab_by_name["Raw OCR Markdown"]:
            st.markdown(load_markdown(str(ocr_path), ocr_path.stat().st_mtime))


def main() -> None:
    st.set_page_config(page_title="ReproBot Pipeline Viewer", layout="wide")
    st.title("ReproBot Pipeline Viewer")
    st.caption(
        "Displays ocr/output/ and reader/output/ as-is; the import section "
        "below can trigger ocr/vlm_extract.py's existing OCR backend on a "
        "newly uploaded paper, unmodified."
    )

    papers = list_papers()
    if not papers:
        st.error(f"No papers found in {DATASET_DIR} or {UPLOADS_DIR}.")
        render_import_section()
        return

    render_pipeline_status(papers)

    st.divider()
    render_import_section()
    st.divider()
    selected = st.selectbox("Select a paper", papers)
    render_paper(selected)


if __name__ == "__main__":
    main()
