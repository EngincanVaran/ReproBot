"""Streamlit viewer for the ocr/, reader/, coder/, and runner/ pipeline outputs.

Mostly read-only: it loads files that `ocr/`, `reader/`, `coder/`, and
`runner/` have already written to `ocr/output/vlm/*.md`,
`reader/output/*.json`, `coder/output/<paper>/coder_output.json`, and
`runner/output/<paper>/runner_output.json`, and displays them, without
altering how any stage extracts, generates, or executes. Three exceptions,
each calling a stage's own entry-point function directly, unmodified, exactly
as its CLI does:

- The "Import a paper" section lets a user upload a new PDF and run OCR on
  it, via `ocr/vlm_extract.py`'s `run_vlm()`.
- A paper with OCR done but no Reader output yet gets a "Run Reader
  extraction" button on its Overview tab, via `reader/pipeline.py`'s
  `run_pipeline()`.
- A paper with Reader output but no Coder output yet gets a "Generate
  code" button on its Overview tab, via `coder/pipeline.py`'s
  `run_pipeline()`. That call resolves the paper's OCR Markdown as its
  required second input, makes one Claude tool-use call, and applies the
  syntax + CLI-flag gates - it writes `coder/output/<paper>/train.py`,
  `coder_output.json`, and `reproduce.sh`.

None of these reimplement or change the extraction/generation logic itself.
`runner/` output has no trigger button here, deliberately: a real run can take
up to 45 minutes on a cold cache (see `runner/docker_runner.py`'s stage
timeouts) and needs a reachable Docker daemon, so it stays a CLI-only action
(`uv run python -m runner.pipeline`) and this app only displays what it wrote.

`critic/` is the opposite case: its verdict is arithmetic (no API key, no Docker), so the
Runner tab has a "Judge this run with the Critic" button that calls `critic.judge.judge()`
directly. The model review (one paid Claude Opus call) has its own button on the Critic tab,
which calls `critic.pipeline.run_review()`; re-judging keeps a stored review, and only the
review button replaces it.

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

import ast
import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from loguru import logger

from coder.pipeline import MissingPaperMarkdownError, ScriptSyntaxError
from coder.pipeline import run_pipeline as run_coder_pipeline
from critic.blocking import pass_blocking_reasons
from critic.checks import run_checks
from critic.claims import group_claims
from critic.history import write_history
from critic.judge import judge
from critic.pipeline import run_review
from critic.pipeline import write_output as write_critic_output
from ocr.vlm_extract import run_vlm
from reader.pipeline import run_pipeline as run_reader_pipeline
from report.pipeline import REPORT_FILENAME, SUMMARY_FILENAME, build_report, state_files
from report.render import render_index
from report.summary import render_summary
from viewer.full_run import (
    JOBS_DIR,
    STAGES,
    job_alive,
    job_dir,
    read_status,
    start_job,
    stop_job,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
EXTRA_DIR = REPO_ROOT / "extra-papers"
UPLOADS_DIR = REPO_ROOT / "viewer" / "uploads"
OCR_DIR = REPO_ROOT / "ocr" / "output" / "vlm"
READER_DIR = REPO_ROOT / "reader" / "output"
CODER_DIR = REPO_ROOT / "coder" / "output"
RUNNER_DIR = REPO_ROOT / "runner" / "output"
CRITIC_DIR = REPO_ROOT / "critic" / "output"
REPORT_DIR = REPO_ROOT / "report" / "output"
ORCHESTRATOR_DIR = REPO_ROOT / "orchestrator" / "output"

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9 ._-]")


def list_papers() -> list[str]:
    """Paper names (no extension), sourced from dataset/ (the curated
    replication targets) plus viewer/uploads/ (papers imported through this
    app), so every paper any stage could act on shows up, even ones no stage
    has processed yet."""
    papers: list[str] = []
    for source in (DATASET_DIR, EXTRA_DIR):
        if source.is_dir():
            papers.extend(p.stem for p in source.glob("*.pdf") if p.stem not in papers)
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


def coder_json_path(paper: str) -> Path:
    """coder/pipeline.py writes `coder/output/<paper>/coder_output.json` on a
    successful run, or `coder_output.failed.json` when the syntax gate fails."""
    return CODER_DIR / paper / "coder_output.json"


def coder_failed_json_path(paper: str) -> Path:
    return CODER_DIR / paper / "coder_output.failed.json"


def runner_json_path(paper: str) -> Path:
    """runner/pipeline.py writes `runner/output/<paper>/runner_output.json`
    after every escalation run, win or lose - a script failure is a recorded
    `status: "error"`, not a missing file."""
    return RUNNER_DIR / paper / "runner_output.json"


def orchestrator_state_path(paper: str) -> Path:
    """`orchestrator.pipeline` keeps a paper's whole run in one `state.json`: its Runner
    result under `runner_output` and its Critic result under `critic_output`. The Runner
    and Critic CLIs write their own files instead, so the viewer reads those first and
    falls back to this."""
    return ORCHESTRATOR_DIR / paper / "state.json"


def report_path(paper: str) -> Path:
    """`report.pipeline` writes `report/output/<paper>/report.md` (and `curve.svg` beside it)."""
    return REPORT_DIR / paper / REPORT_FILENAME


def critic_json_path(paper: str) -> Path:
    """`critic.pipeline` writes `critic/output/<paper>.json`: the verdict, the claim
    groups, and - only when it was run with `--review` - the model review."""
    return CRITIC_DIR / f"{paper}.json"


@st.cache_data
def load_json_output(path_str: str, mtime: float) -> dict[str, Any]:
    """Shared loader for reader/'s and coder/'s output JSON. `mtime` busts the
    cache when a pipeline re-writes the file, e.g. after a retry loop finishes
    with new content."""
    path = Path(path_str)
    logger.info("Loading JSON output: {}", path)
    return cast("dict[str, Any]", json.loads(path.read_text()))


def load_run_outputs(paper: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A paper's Runner and Critic results, from whichever pipeline produced them.

    `runner.pipeline` and `critic.pipeline` write their own files. `orchestrator.pipeline`
    keeps both inside `state.json`, where `runner_output` is only a summary (no image,
    stages or metrics mode), so those come from the last attempt. Own files win."""
    runner_path, critic_path = runner_json_path(paper), critic_json_path(paper)
    runner = (
        load_json_output(str(runner_path), runner_path.stat().st_mtime)
        if runner_path.exists()
        else None
    )
    critic = (
        load_json_output(str(critic_path), critic_path.stat().st_mtime)
        if critic_path.exists()
        else None
    )
    state_path = orchestrator_state_path(paper)
    if state_path.exists():
        state = load_json_output(str(state_path), state_path.stat().st_mtime)
        if runner is None and state.get("runner_output"):
            attempt = (state.get("attempts") or [{}])[-1]
            runner = {
                **state["runner_output"],
                "stage_reached": attempt.get("stage_reached"),
                "metrics_mode": attempt.get("stage_reached"),
                "wall_clock_seconds": attempt.get("wall_clock_seconds", 0),
                "failed_stage": attempt.get("failed_stage"),
                "exit_code": attempt.get("exit_code"),
            }
        loop_critic = state.get("critic_output") or {}
        if critic is not None and not critic.get("review") and loop_critic.get("review"):
            # A verdict re-run from this page carries no review; the run's review still applies.
            critic = {**critic, "review": loop_critic["review"]}
        critic = critic or loop_critic or None
    return runner, critic


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
            output = run_reader_pipeline(markdown_path, READER_DIR, client)
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


def run_coder_extraction(reader_output_path: Path, paper_name: str) -> None:
    """Invoke coder/pipeline.py's own run_pipeline() unmodified - it resolves
    the paper's OCR Markdown (`ocr/output/vlm/<paper>.md`) as its required
    second input, makes one Claude tool-use call to write the training script,
    then runs the deterministic syntax + CLI-flag gates."""
    load_dotenv(REPO_ROOT / ".env")
    try:
        client = Anthropic()
    except Exception as exc:  # noqa: BLE001 - surface any client-setup error in the UI
        st.error(f"Could not create Anthropic client (check ANTHROPIC_API_KEY): {exc}")
        return

    with st.status(f"Generating code for '{paper_name}'...", expanded=False) as status:
        try:
            output = run_coder_pipeline(reader_output_path, OCR_DIR, CODER_DIR, client)
        except MissingPaperMarkdownError as exc:
            status.update(label=f"Needs OCR Markdown first: {exc}", state="error")
            logger.error("Code generation blocked for {}: {}", paper_name, exc)
            return
        except ScriptSyntaxError as exc:
            # The invalid script + coder_output.failed.json are still on disk;
            # rerun so the Code tab can surface them.
            status.update(label=f"Generated script failed the syntax gate: {exc}", state="error")
            logger.error("Code generation syntax gate failed for {}: {}", paper_name, exc)
        except Exception as exc:  # noqa: BLE001 - surface any generation error in the UI
            status.update(label=f"Code generation failed: {exc}", state="error")
            logger.error("Code generation failed for {}: {}", paper_name, exc)
            return
        else:
            status.update(
                label=f"Code generated -> {Path(output.script_path).name}", state="complete"
            )
    logger.info("Code generation finished for {}", paper_name)
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
        has_coder = coder_json_path(paper).exists()
        coder_failed = coder_failed_json_path(paper).exists()
        flags = 0
        if has_reader:
            data = load_json_output(str(reader_path), reader_path.stat().st_mtime)
            flags = len(data.get("validation", {}).get("flags", []))
        runner_data, critic_data = load_run_outputs(paper)
        runner_status = runner_data.get("status", "unknown") if runner_data else "-"
        critic_verdict = critic_data.get("verdict", "unknown") if critic_data else "-"
        rows.append(
            {
                "Paper": paper,
                "OCR": "done" if has_ocr else "-",
                "Reader": "done" if has_reader else "-",
                "Coder": "done" if has_coder else ("failed" if coder_failed else "-"),
                "Validation flags": flags if has_reader else "-",
                "Runner": runner_status,
                "Critic": critic_verdict,
                "Report": "done" if report_path(paper).exists() else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_method_summary(data: dict[str, Any]) -> None:
    method_summary = data.get("method_summary", {})
    for label, key in (("Problem", "problem"), ("Core idea", "core_idea"), ("Novelty", "novelty")):
        if method_summary.get(key):
            st.markdown(f"**{label}**")
            st.markdown(method_summary[key])
    if method_summary.get("summary"):
        st.markdown("**Summary**")
        st.markdown(method_summary["summary"])
    sources = method_summary.get("sources_examined", [])
    if sources:
        with st.expander("Sources examined"):
            for source in sources:
                st.markdown(f"- {source}")


def render_architecture(data: dict[str, Any]) -> None:
    notes = data.get("architecture_notes", {})
    if notes.get("model_name"):
        st.markdown(f"**Model:** {notes['model_name']}")
    if notes.get("overall_structure"):
        st.markdown(notes["overall_structure"])
    if notes.get("depth_or_scale"):
        st.caption(f"Depth / scale: {notes['depth_or_scale']}")

    components = notes.get("components", [])
    if components:
        st.markdown("**Components**")
        st.dataframe(pd.DataFrame(components), use_container_width=True, hide_index=True)

    equations = notes.get("key_equations", [])
    if equations:
        st.markdown("**Key equations**")
        for equation in equations:
            if equation.get("latex"):
                try:
                    st.latex(equation["latex"])
                except Exception:  # noqa: BLE001 - fall back to raw text on any render error
                    st.code(equation["latex"], language="latex")
            meta = " · ".join(
                part
                for part in (
                    equation.get("label"),
                    equation.get("defines"),
                    "own method" if equation.get("is_own_method") else "prior work",
                )
                if part
            )
            if meta:
                st.caption(meta)

    unstated = notes.get("unstated_details", [])
    if unstated:
        st.markdown("**Unstated details (the Coder has to assume these)**")
        for detail in unstated:
            st.markdown(f"- {detail}")


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


def render_code(coder_data: dict[str, Any]) -> None:
    if not coder_data.get("syntax_ok", True):
        st.error(
            "The generated script failed the `ast.parse` syntax gate and was written "
            f"as `train.py.invalid`: {coder_data.get('syntax_error')}"
        )
    missing_flags = coder_data.get("missing_cli_flags", [])
    if missing_flags:
        st.warning(
            f"Script is missing {len(missing_flags)} CLI flag(s) the Runner needs: "
            f"{', '.join(missing_flags)}"
        )

    if coder_data.get("claim_targeted"):
        st.markdown(f"**Target claim:** `{coder_data['claim_targeted']}`")
    if coder_data.get("claim_selection_reasoning"):
        st.caption(coder_data["claim_selection_reasoning"])

    script_path = Path(coder_data["script_path"])
    st.markdown(f"**Script:** `{script_path.name}`")
    if script_path.exists():
        st.code(script_path.read_text(), language="python")
    else:
        st.warning(f"Script file not found on disk: {script_path}")

    reproduce_path = script_path.parent / "reproduce.sh"
    if reproduce_path.exists():
        with st.expander("reproduce.sh (the Runner's whole interface)"):
            st.code(reproduce_path.read_text(), language="bash")

    hyperparameters_used = coder_data.get("hyperparameters_used", [])
    if hyperparameters_used:
        st.markdown("**Hyperparameters used**")
        st.dataframe(pd.DataFrame(hyperparameters_used), use_container_width=True, hide_index=True)

    if coder_data.get("architecture_used"):
        st.markdown("**Architecture used**")
        st.markdown(coder_data["architecture_used"])
    if coder_data.get("dataset_used"):
        st.markdown("**Dataset used**")
        st.markdown(coder_data["dataset_used"])

    assumptions = coder_data.get("assumptions", [])
    if assumptions:
        st.markdown("**Assumptions the Coder had to make (not stated in the paper)**")
        for assumption in assumptions:
            st.markdown(f"- {assumption}")


def parse_training_curves(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover per-epoch test error and the learning-rate trace from a stage's stdout.

    The generated scripts use HF `Trainer`, which prints one Python-dict line per
    evaluation (`eval_accuracy`, `epoch`) and per logging step (`learning_rate`,
    `epoch`). Lines that do not parse are skipped: the log also carries progress
    text, and the final `train_runtime` summary line is not a curve point.
    """
    evals: list[dict[str, float]] = []
    rates: list[dict[str, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(row, dict) or "epoch" not in row or "train_runtime" in row:
            continue
        if "eval_accuracy" in row:
            evals.append(
                {"epoch": float(row["epoch"]), "Test error (%)": 100 - row["eval_accuracy"]}
            )
        elif "learning_rate" in row:
            rates.append(
                {"epoch": float(row["epoch"]), "Learning rate": float(row["learning_rate"])}
            )
    # The run ends with one extra evaluation at the final epoch; keep the last per epoch.
    return pd.DataFrame(evals).drop_duplicates("epoch", keep="last"), pd.DataFrame(rates)


def reader_claims(reader_data: dict[str, Any]) -> list[dict[str, Any]]:
    claims = reader_data.get("claims", [])
    if isinstance(claims, dict):
        claims = claims.get("claims", [])
    return list(claims)


def find_claim(reader_data: dict[str, Any] | None, claim_id: str | None) -> dict[str, Any] | None:
    """The reader claim a run targeted, so the run can be shown next to the paper's number."""
    if not reader_data or not claim_id:
        return None
    return next((c for c in reader_claims(reader_data) if c.get("claim_id") == claim_id), None)


def render_runner_context(runner_data: dict[str, Any], reader_data: dict[str, Any] | None) -> None:
    """Hardware, and reproduced-vs-reported. Informational only: no Critic exists yet,
    so this shows the difference and deliberately gives no pass/fail verdict."""
    image = str(runner_data.get("image") or "")
    if not image:
        st.markdown("**Hardware:** not recorded")
    else:
        kind = "GPU (CUDA image)" if image.endswith(":cuda") else "CPU"
        st.markdown(f"**Hardware:** {kind} - `{image}`")

    metrics = runner_data.get("reproduced_metrics")
    if not metrics:
        return
    mode = runner_data.get("metrics_mode")
    claim = find_claim(reader_data, metrics.get("claim_id"))
    if mode != "full":
        st.warning(
            f"These numbers come from the `{mode}` stage, a fraction of the paper's training "
            "budget. They show the script runs and learns - they are **not** comparable to the "
            "paper's reported value."
        )
        return
    if claim is None:
        return
    reproduced, reported = float(metrics["value"]), float(claim["reported_value"])
    left, middle, right = st.columns(3)
    label = f"Reproduced ({metrics.get('metric')}, {metrics.get('unit')})"
    left.metric(label, f"{reproduced:.2f}")
    middle.metric("Paper reports", f"{reported:.2f}")
    right.metric("Difference", f"{reproduced - reported:+.2f}")
    st.caption(
        f"Claim `{claim.get('claim_id')}`: {claim.get('model_variant') or '-'} on "
        f"{claim.get('dataset')}, {claim.get('source')}. Metrics are from "
        f"{metrics.get('num_eval_samples')} eval samples over "
        f"{metrics.get('epochs_completed')} epochs."
    )


def render_training_curves(stages: list[dict[str, Any]]) -> None:
    for stage in stages:
        path_str = stage.get("stdout_path")
        if not path_str:
            continue
        path = Path(path_str) if Path(path_str).is_absolute() else REPO_ROOT / path_str
        if not path.exists():
            continue
        evals, rates = parse_training_curves(path)
        if len(evals) < 3:
            continue
        st.markdown(f"**Training curves** (`{stage.get('mode')}` stage, {len(evals)} evaluations)")
        st.line_chart(evals.set_index("epoch"))
        if not rates.empty:
            st.line_chart(rates.set_index("epoch"))


def render_runner(runner_data: dict[str, Any], reader_data: dict[str, Any] | None = None) -> None:
    status = runner_data.get("status", "unknown")
    status_banner = {"success": st.success, "error": st.error, "timeout": st.warning}.get(
        status, st.info
    )
    status_banner(f"Status: **{status}**")

    st.markdown(f"**Stage reached:** {runner_data.get('stage_reached') or '-'}")
    if runner_data.get("failed_stage"):
        st.markdown(
            f"**Failed at:** `{runner_data['failed_stage']}` "
            f"(exit code {runner_data.get('exit_code')})"
        )
    st.markdown(f"**Total wall clock:** {runner_data.get('wall_clock_seconds', 0):.1f}s")

    render_runner_context(runner_data, reader_data)

    metrics = runner_data.get("reproduced_metrics")
    if metrics:
        st.markdown(f"**Reproduced metrics** (from the `{runner_data.get('metrics_mode')}` stage)")
        st.json(metrics)
    else:
        st.info("No metrics were produced by this run.")

    triage = runner_data.get("triage")
    if triage:
        st.markdown(f"**Triage:** `{triage.get('category')}`")
        if triage.get("reasoning"):
            st.markdown(triage["reasoning"])
        if triage.get("suggested_fix"):
            st.markdown(f"**Suggested fix:** {triage['suggested_fix']}")

    if runner_data.get("error_trace"):
        with st.expander("Error trace"):
            st.code(runner_data["error_trace"], language="python")

    stages = runner_data.get("stages", [])
    if not stages:
        return
    render_training_curves(stages)
    st.markdown("**Stages**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Mode": stage.get("mode"),
                    "Status": stage.get("status"),
                    "Exit code": stage.get("exit_code"),
                    "Wall clock (s)": stage.get("wall_clock_seconds"),
                    "Timeout (s)": stage.get("timeout_seconds"),
                }
                for stage in stages
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    for stage in stages:
        with st.expander(f"`{stage.get('mode')}` logs ({stage.get('status')})"):
            st.code(stage.get("log_excerpt", "") or "(empty)", language="text")
            for label, path_key in (("stdout", "stdout_path"), ("stderr", "stderr_path")):
                log_path = Path(stage[path_key]) if stage.get(path_key) else None
                if log_path and log_path.exists():
                    st.caption(f"Full {label}: `{log_path}`")


def run_critic_verdict(
    paper: str, reader_data: dict[str, Any], runner_data: dict[str, Any]
) -> None:
    """Judge a finished run with `critic.judge.judge()`, unmodified. The verdict is pure
    arithmetic - no API key, no Docker - so unlike the model review it is safe to run here.
    An existing model review is kept: it is a separate, paid artefact this button must not
    delete."""
    claims = reader_claims(reader_data)
    judgement = judge(
        claims,
        runner_data.get("reproduced_metrics"),
        runner_status=str(runner_data.get("status") or "missing"),
        metrics_mode=runner_data.get("metrics_mode"),
    )
    payload: dict[str, Any] = {
        **judgement.to_dict(),
        "claim_groups": [group.to_dict() for group in group_claims(claims)],
        "checks": run_checks(runner_data.get("reproduced_metrics")),
    }
    existing = critic_json_path(paper)
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if "review" in previous:
            payload["review"] = previous["review"]
    logger.info("Critic verdict for {}: {}", paper, payload["verdict"])
    write_critic_output(CRITIC_DIR, paper, payload)
    st.rerun()


def render_critic_button(
    paper: str,
    reader_data: dict[str, Any],
    runner_data: dict[str, Any],
    *,
    key: str,
    label: str,
) -> None:
    if st.button(label, key=key, help="Arithmetic only: no API key, no Docker, no cost."):
        run_critic_verdict(paper, reader_data, runner_data)


def run_critic_review(
    paper: str,
    reader_data: dict[str, Any],
    runner_data: dict[str, Any],
    *,
    chance: float | None,
) -> None:
    """Judge the run, then have the Critic's model review read the script against the paper.

    Same steps as running it by hand: rebuild the learning-curve history from the run's log
    if it is missing, judge, call `critic.pipeline.run_review()` unmodified with the RUNNER's
    log directory (the CLI would point it at the paper folder, where there is no log), and
    write the result. One paid Claude Opus call."""
    load_dotenv(REPO_ROOT / ".env")
    paper_dir = CODER_DIR / paper
    metrics = runner_data.get("reproduced_metrics")
    logs_path = runner_data.get("logs_path")
    logs_dir = (REPO_ROOT / logs_path) if logs_path else None
    claims = reader_claims(reader_data)

    with st.status("Running the Critic model review...", expanded=True) as status:
        try:
            history = paper_dir / "metrics.full.history.jsonl"
            if history.exists():
                status.write(f"Learning-curve history already present ({history.name}).")
            else:
                status.write("Building the learning-curve history from the run's log...")
                write_history(
                    paper,
                    chance=chance,
                    coder_output=CODER_DIR,
                    runner_output=RUNNER_DIR,
                    reader_output=READER_DIR,
                )
            status.write("Judging the run...")
            judgement = judge(
                claims,
                metrics,
                runner_status=str(runner_data.get("status") or "missing"),
                metrics_mode=runner_data.get("metrics_mode"),
            )
            status.write("Calling the model review (one Claude Opus call, usually 1-3 minutes)...")
            review = run_review(judgement, metrics, reader_data, paper_dir, logs_dir)
        except Exception as exc:  # noqa: BLE001 - surface any review error in the UI
            status.update(label=f"Critic review failed: {exc}", state="error")
            logger.error("Critic review failed for {}: {}", paper, exc)
            return
        payload: dict[str, Any] = {
            **judgement.to_dict(),
            "claim_groups": [group.to_dict() for group in group_claims(claims)],
            "checks": run_checks(metrics),
            "review": review.to_dict(),
        }
        write_critic_output(CRITIC_DIR, paper, payload)
        status.update(
            label=f"Review done - {review.method_fidelity}, {len(review.findings)} finding(s)",
            state="complete",
        )
    logger.info("Critic review finished for {}: {}", paper, review.method_fidelity)
    st.rerun()


def render_critic_review_button(
    paper: str,
    reader_data: dict[str, Any],
    runner_data: dict[str, Any],
    *,
    has_review: bool,
) -> None:
    """Only a full run is worth reviewing: the Critic itself refuses to judge a gate stage."""
    if runner_data.get("metrics_mode") != "full":
        return
    chance = st.number_input(
        "Chance level of the metric (optional)",
        min_value=0.0,
        value=None,
        placeholder="e.g. 90 for 10-class error",
        help="Lets the review judge how far above chance the learning curve starts.",
        key="critic_chance",
    )
    label = (
        "Re-run the model review (replaces the current one)"
        if has_review
        else ("Run the model review")
    )
    if st.button(
        label,
        key="critic_review",
        help="One paid Claude Opus call, usually 1-3 minutes. Needs ANTHROPIC_API_KEY and credit.",
    ):
        run_critic_review(paper, reader_data, runner_data, chance=chance)


def generate_reports() -> None:
    """Rebuild every orchestrated paper's report with `report.pipeline.build_report()`,
    plus the index and cross-run summary its CLI writes. Free: the generator reads
    finished-run files and makes no model call."""
    written: list[tuple[str, dict[str, Any], str]] = []
    for path in state_files(ORCHESTRATOR_DIR):
        try:
            name, href = build_report(path, CODER_DIR, REPORT_DIR)
        except Exception as exc:  # noqa: BLE001 - one bad state must not hide the rest
            logger.error(f"[viewer] report for {path.parent.name} failed: {exc}")
            continue
        written.append((name, json.loads(path.read_text(encoding="utf-8")), href))
    if written:
        (REPORT_DIR / "README.md").write_text(render_index(written), encoding="utf-8")
        (REPORT_DIR / SUMMARY_FILENAME).write_text(render_summary(written), encoding="utf-8")
    st.rerun()


_REPORT_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+\.svg)\)")


def render_report(paper: str) -> None:
    """Show the paper's generated Markdown report, drawing its SVG curve with `st.image`
    (a relative image link would not resolve inside `st.markdown`)."""
    if st.button(
        "Generate / refresh all reports", key="report_generate", help="Free: no API call."
    ):
        generate_reports()
    path = report_path(paper)
    if not path.exists():
        st.info(
            "No report yet - reports are built from orchestrator/output/<paper>/state.json, "
            "so run the paper through the Orchestrator, then press the button."
        )
        return
    text = path.read_text(encoding="utf-8")
    st.download_button(
        "Download the report (.md)", text, file_name=path.name, key="report_download"
    )
    match = _REPORT_IMAGE.search(text)
    if match and (path.parent / match.group(1)).exists():
        st.markdown(text[: match.start()], unsafe_allow_html=True)
        st.image((path.parent / match.group(1)).read_text(encoding="utf-8"))
        st.markdown(text[match.end() :], unsafe_allow_html=True)
    else:
        st.markdown(text, unsafe_allow_html=True)


_VERDICT_BANNERS = {"pass": st.success, "fail": st.error, "inconclusive": st.warning}


def _fmt(value: object, unit: str = "") -> str:
    return "-" if not isinstance(value, int | float) else f"{value:.4g}{unit}"


def render_critic_review(review: dict[str, Any]) -> None:
    """The Critic v2 model review of the script against the paper. Display only: it costs
    an API call, so it stays a CLI action (`critic.pipeline --review`)."""
    st.markdown("#### Model review")
    st.markdown(f"**Method fidelity:** {review.get('method_fidelity', '-')}")
    if review.get("summary"):
        st.markdown(review["summary"])
    if review.get("curve_assessment"):
        with st.expander("Learning-curve assessment"):
            st.markdown(review["curve_assessment"])
    findings = review.get("findings", [])
    if findings:
        st.markdown(f"**Findings** ({len(findings)})")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Kind": f.get("kind"),
                        "Aspect": f.get("aspect"),
                        "Severity": f.get("severity"),
                        "Verified": f.get("verified"),
                        "Explanation": f.get("explanation"),
                    }
                    for f in findings
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    hypotheses = review.get("hypotheses", [])
    if hypotheses:
        st.markdown("**Hypotheses for a gap**")
        for h in hypotheses:
            st.markdown(f"{h.get('rank')}. {h.get('hypothesis')}  \n*Change:* {h.get('change')}")
    st.caption(f"Reviewed by `{review.get('model', '-')}`.")


def render_critic(
    critic_data: dict[str, Any],
    paper: str,
    reader_data: dict[str, Any] | None,
    runner_data: dict[str, Any] | None,
) -> None:
    verdict = str(critic_data.get("verdict", "unknown"))
    blocked = pass_blocking_reasons(critic_data.get("review")) if verdict == "pass" else []
    if blocked:
        st.warning(
            "Verdict: **pass, blocked by the review** - the number matches, but the script is "
            "not the paper's method or data, so this is not a reproduction:\n"
            + "\n".join(f"- {reason}" for reason in blocked)
        )
    else:
        _VERDICT_BANNERS.get(verdict, st.info)(
            f"Verdict: **{verdict}** - {critic_data.get('reason')}"
        )

    unit = str(critic_data.get("unit") or "")
    st.caption(
        f"Claim `{critic_data.get('claim_id') or '-'}`: {critic_data.get('metric') or '-'} on "
        f"{critic_data.get('dataset') or '-'}"
    )
    if critic_data.get("claimed") is not None:
        first, second, third, fourth = st.columns(4)
        first.metric("Reproduced", _fmt(critic_data.get("reproduced"), unit))
        second.metric("Paper reports", _fmt(critic_data.get("claimed"), unit))
        third.metric("Gap", _fmt(critic_data.get("gap"), unit))
        fourth.metric("Tolerance (+/-)", _fmt(critic_data.get("tolerance"), unit))
        if critic_data.get("exceeds_claim"):
            st.info("The reproduced value is better than the paper's.")

    if critic_data.get("tolerance") is not None:
        st.markdown("**Where the tolerance comes from**")
        sources = (
            ("Reporting precision", _fmt(critic_data.get("reporting_precision"))),
            ("Test-set (binomial) noise", _fmt(critic_data.get("test_noise"))),
            ("Run-to-run spread", _fmt(critic_data.get("run_spread"))),
            ("Used", str(critic_data.get("evidence") or "-")),
        )
        st.dataframe(
            pd.DataFrame(sources, columns=["Source", "Value"]),
            use_container_width=True,
            hide_index=True,
        )
        if len(critic_data.get("run_values", [])) <= 1:
            st.warning(
                "This is a single run, so run-to-run spread was not measured. The tolerance "
                "comes from test-set noise alone."
            )
    for check in critic_data.get("checks", []):
        st.warning(f"**Check `{check.get('kind')}`:** {check.get('message')}")
    if critic_data.get("recommendation") not in (None, "none"):
        st.markdown(f"**Recommendation:** `{critic_data['recommendation']}`")
    merged = critic_data.get("merged_claim_ids", [])
    if len(merged) > 1:
        st.caption(f"Duplicate claims merged into one: {', '.join(merged)}")

    if reader_data is not None and runner_data is not None:
        render_critic_button(
            paper,
            reader_data,
            runner_data,
            key="critic_rerun",
            label="Re-run the Critic verdict",
        )
        render_critic_review_button(
            paper, reader_data, runner_data, has_review=bool(critic_data.get("review"))
        )

    groups = critic_data.get("claim_groups", [])
    if groups:
        with st.expander(f"All claim groups ({len(groups)})"):
            st.dataframe(pd.DataFrame(groups), use_container_width=True, hide_index=True)

    if critic_data.get("review"):
        render_critic_review(critic_data["review"])
    else:
        st.caption("No model review yet - use the button above (one paid Claude Opus call).")


def render_paper(paper: str) -> None:
    reader_path = reader_json_path(paper)
    ocr_path = ocr_markdown_path(paper)
    coder_path = coder_json_path(paper)

    if not reader_path.exists() and not ocr_path.exists():
        st.warning(f"No pipeline output yet for '{paper}'.")
        pdf_path = next(
            (
                source / f"{paper}.pdf"
                for source in (DATASET_DIR, EXTRA_DIR, UPLOADS_DIR)
                if (source / f"{paper}.pdf").exists()
            ),
            None,
        )
        if pdf_path is not None:
            st.caption("Calls the Anthropic API once per page (Claude reads each page image).")
            if st.button("Run OCR extraction", key=f"run_ocr_{paper}"):
                run_ocr_extraction(pdf_path, paper, max_pages=None)
        return

    coder_failed_path = coder_failed_json_path(paper)

    data = (
        load_json_output(str(reader_path), reader_path.stat().st_mtime)
        if reader_path.exists()
        else None
    )
    if coder_path.exists():
        coder_data = load_json_output(str(coder_path), coder_path.stat().st_mtime)
    elif coder_failed_path.exists():
        coder_data = load_json_output(str(coder_failed_path), coder_failed_path.stat().st_mtime)
    else:
        coder_data = None
    runner_data, critic_data = load_run_outputs(paper)

    tab_names = ["Overview"]
    if data is not None:
        if data.get("method_summary"):
            tab_names.append("Method Summary")
        if data.get("architecture_notes"):
            tab_names.append("Architecture")
        tab_names += ["Claims", "Hyperparameters", "Data Pipeline", "Validation", "Raw JSON"]
    if coder_data is not None:
        tab_names.append("Code")
    if runner_data is not None:
        tab_names.append("Runner")
    if critic_data is not None:
        tab_names.append("Critic")
    if runner_data is not None:
        tab_names.append("Report")
    if ocr_path.exists():
        tab_names.append("Raw OCR Markdown")

    tabs = st.tabs(tab_names)
    tab_by_name = dict(zip(tab_names, tabs, strict=True))

    with tab_by_name["Overview"]:
        st.markdown(f"### {paper}")
        st.markdown(f"- OCR Markdown: {'available' if ocr_path.exists() else 'not yet run'}")
        st.markdown(f"- Reader output: {'available' if data is not None else 'not yet run'}")
        st.markdown(f"- Coder output: {'available' if coder_data is not None else 'not yet run'}")
        if data is not None:
            st.markdown(f"- Claims: {len(data.get('claims', {}).get('claims', []))}")
            st.markdown(
                f"- Hyperparameters: "
                f"{len(data.get('hyperparameters', {}).get('hyperparameters', []))}"
            )
            st.markdown(f"- Datasets: {len(data.get('data_pipeline', {}).get('datasets', []))}")
        elif ocr_path.exists():
            st.caption(
                "Calls the Anthropic API (method_summary + architecture_notes + claims "
                "+ hyperparameters + data_pipeline extractors, plus validation with retries)."
            )
            if st.button("Run Reader extraction", key=f"run_reader_{paper}"):
                run_reader_extraction(ocr_path, paper)

        if coder_data is not None:
            st.markdown(f"- Generated script: {Path(coder_data['script_path']).name}")
            if coder_data.get("claim_targeted"):
                st.markdown(f"- Target claim: {coder_data['claim_targeted']}")
            if not coder_data.get("syntax_ok", True):
                st.markdown("- ⚠ Script failed the syntax gate (see Code tab)")
            if coder_data.get("missing_cli_flags"):
                st.markdown(f"- ⚠ Missing CLI flags: {', '.join(coder_data['missing_cli_flags'])}")
        elif data is not None:
            st.caption(
                "Needs the paper's OCR Markdown too. Makes one Claude tool-use call to "
                "write the training script, then runs deterministic syntax + CLI-flag "
                "gates (no extra call)."
            )
            if st.button("Generate code", key=f"run_coder_{paper}"):
                run_coder_extraction(reader_path, paper)

        if coder_data is not None and coder_data.get("syntax_ok", True):
            render_run_training_button(paper)

        if runner_data is not None:
            st.markdown(
                f"- Runner status: **{runner_data.get('status')}** "
                f"(reached `{runner_data.get('stage_reached') or '-'}`)"
            )
        elif coder_data is not None:
            st.caption(
                "Run via the CLI: `uv run python -m runner.pipeline --input "
                f'"coder/output/{paper}"`.'
            )

    if data is not None:
        if "Method Summary" in tab_by_name:
            with tab_by_name["Method Summary"]:
                render_method_summary(data)
        if "Architecture" in tab_by_name:
            with tab_by_name["Architecture"]:
                render_architecture(data)
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

    if coder_data is not None:
        with tab_by_name["Code"]:
            render_code(coder_data)

    if runner_data is not None:
        with tab_by_name["Runner"]:
            render_runner(runner_data, data)
            if data is not None and runner_data.get("reproduced_metrics"):
                st.divider()
                render_critic_button(
                    paper,
                    data,
                    runner_data,
                    key="critic_from_runner",
                    label="Judge this run with the Critic",
                )

    if critic_data is not None:
        with tab_by_name["Critic"]:
            render_critic(critic_data, paper, data, runner_data)

    if runner_data is not None:
        with tab_by_name["Report"]:
            render_report(paper)

    if ocr_path.exists():
        with tab_by_name["Raw OCR Markdown"]:
            st.markdown(load_markdown(str(ocr_path), ocr_path.stat().st_mtime))


SSH_FILE = JOBS_DIR / "gpu_ssh.txt"


def paper_pdf(paper: str) -> Path | None:
    for source in (DATASET_DIR, EXTRA_DIR, UPLOADS_DIR):
        if (source / f"{paper}.pdf").exists():
            return source / f"{paper}.pdf"
    return None


def render_compute_sidebar() -> None:
    """Where the Orchestrator runs. The SSH command is kept on disk so it survives reloads."""
    st.sidebar.header("Compute")
    target = st.sidebar.radio(
        "Run training on",
        ["gpu", "local"],
        format_func=lambda t: "GPU box (SSH)" if t == "gpu" else "Local Docker (CPU)",
        key="compute_target",
    )
    if target == "gpu":
        saved = SSH_FILE.read_text().strip() if SSH_FILE.exists() else ""
        ssh = st.sidebar.text_input(
            "SSH command (from Vast)", value=saved, placeholder="ssh -p 44683 root@1.2.3.4"
        )
        if ssh != saved:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            SSH_FILE.write_text(ssh)
        st.sidebar.caption(
            "Code, this paper's inputs and `.env` (the API key, for retries and the Critic) "
            "are copied to the box. It bills while idle - stop it on Vast when done."
        )
    st.sidebar.selectbox(
        "Highest stage",
        ["full", "capped", "smoke", "probe"],
        index=0 if target == "gpu" else 1,
        key="max_stage",
        help="probe -> smoke -> capped -> full. 'full' is the paper's real setup.",
    )


def current_ssh() -> str | None:
    if st.session_state.get("compute_target") != "gpu" or not SSH_FILE.exists():
        return None
    return SSH_FILE.read_text().strip() or None


STATE_ICONS = {"done": "✅", "skipped": "⏭️", "running": "⏳", "failed": "❌"}


def render_full_run(paper: str) -> None:
    # Collapsed unless this paper has a job, so it stays out of the way when browsing.
    with st.expander("Run full pipeline", expanded=read_status(paper) is not None):
        render_full_run_body(paper)


def render_full_run_body(paper: str) -> None:
    st.caption(
        "OCR → Reader → Coder → Orchestrator (Runner + Critic + retries) → Report. "
        "Runs in the background: refresh or close the tab freely. Stages already on "
        "disk are skipped."
    )
    pdf = paper_pdf(paper)
    alive = job_alive(paper)
    target = st.session_state.get("compute_target", "gpu")
    ssh = current_ssh()
    cols = st.columns([1, 1, 4])
    if pdf is None:
        cols[2].warning("No PDF found for this paper.")
    elif cols[0].button("▶ Run full pipeline", disabled=alive, key=f"full_{paper}"):
        if target == "gpu" and not ssh:
            st.error("Paste the GPU box's SSH command in the sidebar first.")
        else:
            start_job(paper, pdf, target, ssh, st.session_state.get("max_stage", "full"))
            st.rerun()
    if cols[1].button("■ Stop", disabled=not alive, key=f"stop_{paper}"):
        stop_job(paper, ssh)
        st.rerun()
    render_job_progress(paper)


def render_run_training_button(paper: str) -> None:
    """Start the same background job as 'Run full pipeline'; with code already on disk,
    OCR/Reader/Coder are skipped and it goes straight to the Orchestrator."""
    target = st.session_state.get("compute_target", "gpu")
    stage = st.session_state.get("max_stage", "full")
    where = "the GPU box" if target == "gpu" else "local Docker"
    st.caption(
        f"Runs the script on {where} up to `{stage}` (sidebar), with the Critic and retries, "
        "then writes the report. In the background - progress shows under "
        "'Run full pipeline' above."
    )
    pdf = paper_pdf(paper)
    alive = job_alive(paper)
    if st.button(
        "Training running…" if alive else "Run training",
        key=f"run_training_{paper}",
        disabled=alive or pdf is None,
    ):
        ssh = current_ssh()
        if target == "gpu" and not ssh:
            st.error("Paste the GPU box's SSH command in the sidebar first.")
        elif pdf is not None:
            start_job(paper, pdf, target, ssh, stage)
            st.rerun()


def render_critic_line(critic: dict[str, Any]) -> None:
    """'done' only means the run finished; this says whether the paper's number held."""
    verdict = critic.get("verdict")
    if not verdict:
        return
    numbers = ""
    if critic.get("reproduced") is not None and critic.get("claimed") is not None:
        unit = critic.get("unit", "")
        numbers = f": {critic['reproduced']:.4g}{unit} vs the paper's {critic['claimed']:.4g}{unit}"
    if verdict == "pass" and critic.get("blocked"):
        st.warning(
            f"**Critic: pass blocked{numbers}.** The number matches, but the review found "
            "the script is not the paper's method or data, so this is not a reproduction."
        )
    elif verdict == "pass":
        st.success(f"**Critic: pass{numbers}**")
    elif verdict == "fail":
        st.error(f"**Critic: fail{numbers}**")
    else:
        st.info(f"**Critic: {verdict}{numbers}**")


@st.fragment(run_every=5)
def render_job_progress(paper: str) -> None:
    status = read_status(paper)
    if status is None:
        return
    alive = job_alive(paper)
    state = status.get("state", "?")
    if state == "running" and not alive:
        state = "died"
    where = f"{status.get('target', '?')}, up to {status.get('max_stage', '?')}"
    label = {
        "running": "⏳ running",
        "done": "✅ done",
        "failed": "❌ failed",
        "stopped": "■ stopped",
        "died": "❌ job process ended unexpectedly",
    }.get(state, state)
    st.markdown(f"**Job:** {label} ({where}) · last update {status.get('updated', '?')}")
    if status.get("error"):
        st.error(status["error"])
    render_critic_line(status.get("critic") or {})
    stages = status.get("stages", {})
    cols = st.columns(len(STAGES))
    for col, name in zip(cols, STAGES, strict=True):
        info = stages.get(name)
        icon = STATE_ICONS.get(info["state"], "") if info else "·"
        col.markdown(f"{icon} **{name}**")
        if info and info.get("detail"):
            col.caption(info["detail"])
    log_path = job_dir(paper) / "job.log"
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        st.code("\n".join(lines[-40:]) or "(no output yet)", language="text")
    if state in ("done", "failed", "stopped", "died") and st.button(
        "Refresh results", key=f"refresh_{paper}"
    ):
        st.rerun(scope="app")


def main() -> None:
    st.set_page_config(page_title="ReproBot Pipeline Viewer", layout="wide")
    st.title("ReproBot Pipeline Viewer")
    st.caption(
        "Displays ocr/output/, reader/output/, coder/output/, and runner/output/ "
        "as-is; the import section below and the per-paper Overview buttons can "
        "trigger the OCR/Reader/Coder stages, unmodified. 'Run full pipeline' takes a "
        "paper through every stage as a background job, locally or on a GPU box."
    )
    render_compute_sidebar()

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
    render_full_run(selected)
    render_paper(selected)


if __name__ == "__main__":
    main()
