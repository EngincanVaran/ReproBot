"""Execute a paper's generated training script inside a Docker sandbox.

`DockerRunner` owns everything that touches the `docker` CLI: building the
image, running one escalation stage, enforcing a host-side timeout (and cleaning
up the container the timeout does *not* kill), capturing logs, and reading the
metrics file back out of the bind mount. `runner/pipeline.py` is the entry
point; this module does no argument parsing and writes only the files that
belong to a run's own output directory.

THE INTERFACE IS `reproduce.sh`, NOT CLI FLAGS
----------------------------------------------
The only command ever sent into a container is `bash reproduce.sh <mode>`, with
`mode` one of probe/smoke/capped/full. This module never constructs a `python`
command, never passes a `--flag`, and never needs to know what arguments a
generated script accepts. That is what keeps the Runner paper-agnostic: a future
paper whose script takes entirely different arguments changes only its own
`reproduce.sh`, and nothing here.

ESCALATING GATES
----------------
Stages run in order and stop at the first non-zero exit, cheapest first, so a
shape error costs seconds rather than hours:

    probe  -> a couple of optimizer steps      (is it runnable at all?)
    smoke  -> one full epoch over a slice      (does it reach eval + write metrics?)
    capped -> 5 epochs / 512 samples           (does it actually learn?)
    full   -> the paper's real setup           (GPU, hours - opt-in only)

`full` is never reached by escalation; it has to be asked for explicitly.

THE TIMEOUT TRAP
----------------
`subprocess.run(timeout=...)` kills the `docker run` *client*. The container
keeps running inside dockerd, holding CPU and its bind mounts. Every run
therefore gets an explicit `--name` chosen before launch, and the timeout path
issues a `docker kill` against that name (idempotent - "no such container" is
success, since `--rm` may already have reaped it).

Usage:
    from runner.docker_runner import DockerRunner
    runner = DockerRunner()
    runner.ensure_image(build=True)
    output = runner.run_paper(paper_dir, logs_dir, modes=("probe", "smoke"), client=None)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from anthropic import Anthropic
from loguru import logger

from runner.triage import TriageResult, triage_failure

RUNNER_DIR: Final[Path] = Path(__file__).resolve().parent
DOCKERFILE_PATH: Final[Path] = RUNNER_DIR / "Dockerfile"
BUILD_CONTEXT: Final[Path] = RUNNER_DIR
DEFAULT_CACHE_DIR: Final[Path] = RUNNER_DIR / "cache"

DEFAULT_IMAGE: Final[str] = "reprobot-runner:latest"

CONTAINER_WORKDIR: Final[str] = "/workspace"
CONTAINER_DATA_DIR: Final[str] = "/workspace/data"
CONTAINER_CACHE_DIR: Final[str] = "/cache"

STAGE_ORDER: Final[tuple[str, ...]] = ("probe", "smoke", "capped", "full")

# Per-stage wall-clock budgets, enforced independently so a hung cheap stage can
# never eat the budget of the expensive one behind it.
#
# CALIBRATED AGAINST A REAL RUN, not estimated. `reproduce.sh probe` was executed
# natively on the dev machine (8-core Intel Mac, WRN-28-10, ~36M params) and
# reported, from the run's own metrics and Trainer output:
#
#     train_runtime            48.7 s   (256 samples, 1 epoch, 2 optimizer steps)
#     train_samples_per_second  5.256
#     eval passes              ~6.4 s (128 eval) + ~12.2 s (256-sample train-acc)
#     total compute            ~67 s
#     total wall clock         1774 s   <-- the other ~1707 s was the CIFAR-10
#                                           download, on a slow connection
#
# That download is the whole story here. The previous budgets folded it into
# `probe` at 1200 s, which the measured 1774 s would have BLOWN on the very first
# run - a timeout that looks exactly like a hung script. `probe` therefore gets a
# budget that tolerates a slow cold fetch; every later stage assumes `probe`
# already paid it and the shared cache mount is warm.
#
# Extrapolating from 5.256 samples/s, warm: smoke ~134 s, capped ~525 s. Each
# budget below is roughly 3-5x its measured cost, which is headroom for a slower
# host, not an invitation to run something enormous.
#
# `full` is 50,000 samples x 200 epochs = ~1.9M seconds at this throughput, i.e.
# about 22 DAYS on this CPU. Its budget is not a real ceiling - it is a marker
# that this mode is GPU-only, and the 24 h figure only means anything on one.
DEFAULT_STAGE_TIMEOUTS: Final[dict[str, int]] = {
    "probe": 2700,  # 45 min: ~67 s compute + a possibly slow cold CIFAR-10 fetch
    "smoke": 900,  # 15 min: ~134 s measured-equivalent, warm cache
    "capped": 1800,  # 30 min: ~525 s measured-equivalent, warm cache
    "full": 86400,  # 24 h: GPU-only; ~22 days on this CPU, so never run it here
}

# Budget for the short-lived control commands (`docker info`, `image inspect`,
# `kill`). These talk to the daemon and return immediately; a slow one means the
# daemon is wedged, which is worth failing fast on rather than waiting out.
DOCKER_CONTROL_TIMEOUT: Final[int] = 120
BUILD_TIMEOUT: Final[int] = 3600

# Log excerpt budget for the triage/Critic prompt: ~8000 chars total, split
# across the two streams and head-and-tail within each. The 1:3 head:tail ratio
# is the point - the head catches an import or setup failure that aborts before
# anything else prints, the tail catches the traceback, and the middle (epoch
# after epoch of progress lines) is the part nothing is learned from.
STDOUT_HEAD_CHARS: Final[int] = 800
STDOUT_TAIL_CHARS: Final[int] = 2400
STDERR_HEAD_CHARS: Final[int] = 1200
STDERR_TAIL_CHARS: Final[int] = 3600

# The tail of stderr kept as `error_trace` when the process died without ever
# printing a Python traceback (an OOM kill, a signal, a shell-level failure).
ERROR_TRACE_FALLBACK_CHARS: Final[int] = 2000

type RunStatus = Literal["success", "error", "timeout"]


class DockerUnavailableError(RuntimeError):
    """The `docker` CLI is missing, or its daemon is not reachable."""


class ImageMissingError(RuntimeError):
    """The sandbox image does not exist locally and building was not requested."""


@dataclass
class StageRun:
    """One `bash reproduce.sh <mode>` execution and everything learned from it."""

    mode: str
    status: RunStatus
    exit_code: int | None
    wall_clock_seconds: float
    timeout_seconds: int
    container_name: str
    stdout_path: str
    stderr_path: str
    metrics_path: str | None
    metrics: dict[str, Any] | None
    log_excerpt: str
    error_trace: str | None


@dataclass
class RunnerOutput:
    """The whole run over one paper - what the Critic and Orchestrator consume."""

    paper: str
    image: str
    paper_dir: str
    status: RunStatus
    stage_reached: str | None
    failed_stage: str | None
    exit_code: int | None
    wall_clock_seconds: float
    reproduced_metrics: dict[str, Any] | None
    metrics_mode: str | None
    logs_path: str
    log_excerpt: str
    error_trace: str | None
    triage: TriageResult | None
    stages: list[StageRun] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pure helpers. Everything below this line is deterministic and daemon-free, so
# it is verifiable against synthetic input without Docker running at all.
# --------------------------------------------------------------------------- #


def stages_up_to(max_stage: str) -> tuple[str, ...]:
    """The escalation sequence ending at `max_stage`, cheapest stage first."""
    if max_stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {max_stage!r}; expected one of {', '.join(STAGE_ORDER)}")
    return STAGE_ORDER[: STAGE_ORDER.index(max_stage) + 1]


def parse_timeout_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse repeated `--timeout mode=seconds` arguments into a stage->seconds map."""
    overrides: dict[str, int] = {}
    for raw in values or []:
        mode, sep, seconds = raw.partition("=")
        if not sep:
            raise ValueError(f"expected --timeout MODE=SECONDS, got {raw!r}")
        mode = mode.strip()
        if mode not in STAGE_ORDER:
            raise ValueError(f"unknown stage {mode!r} in --timeout; expected one of {STAGE_ORDER}")
        try:
            parsed = int(seconds)
        except ValueError as exc:
            raise ValueError(f"timeout for {mode!r} must be an integer, got {seconds!r}") from exc
        if parsed <= 0:
            raise ValueError(f"timeout for {mode!r} must be positive, got {parsed}")
        overrides[mode] = parsed
    return overrides


def slugify(text: str) -> str:
    """Reduce a paper name to something Docker will accept in a container name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "paper"


def container_name_for(paper: str, mode: str) -> str:
    """A container name known BEFORE launch, and unique across concurrent runs.

    Knowing it in advance is what makes the timeout path possible at all: when
    `subprocess` gives up on the `docker run` client, the container it started is
    still alive and can only be reached by name. `time.time_ns()` keeps two runs
    of the same paper and mode - in parallel, or one retried after a kill - from
    colliding on a name the daemon still holds.
    """
    return f"reprobot-{slugify(paper)[:48]}-{mode}-{time.time_ns()}"


def truncate_head_tail(text: str, head_chars: int, tail_chars: int) -> str:
    """Keep the first `head_chars` and last `tail_chars`, marking what was dropped.

    The marker carries the exact number of characters removed so a reader (or the
    triage model) can tell a mildly clipped log from one where almost everything
    is gone.
    """
    if head_chars < 0 or tail_chars < 0:
        raise ValueError("head_chars and tail_chars must be non-negative")
    if len(text) <= head_chars + tail_chars:
        return text
    dropped = len(text) - head_chars - tail_chars
    head = text[:head_chars]
    tail = text[len(text) - tail_chars :] if tail_chars else ""
    return f"{head}\n...[{dropped} chars truncated]...\n{tail}"


def build_log_excerpt(stdout: str, stderr: str) -> str:
    """Combine the two streams into one ~8000-char excerpt for the triage prompt.

    Each stream is truncated BEFORE combining, on purpose. Concatenating first
    and truncating once would let a chatty stdout (an epoch-per-line progress
    log) push stderr's traceback entirely out of the window - which is precisely
    the part the excerpt exists to preserve.
    """
    parts: list[str] = []
    for label, text, head, tail in (
        ("stdout", stdout, STDOUT_HEAD_CHARS, STDOUT_TAIL_CHARS),
        ("stderr", stderr, STDERR_HEAD_CHARS, STDERR_TAIL_CHARS),
    ):
        excerpt = truncate_head_tail(text, head, tail)
        suffix = " (truncated)" if len(excerpt) != len(text) else ""
        body = excerpt if excerpt.strip() else "(empty)"
        parts.append(f"--- {label}{suffix} ---\n{body}")
    return "\n\n".join(parts)


def extract_error_trace(stderr: str) -> str | None:
    """Pull the Python traceback out of stderr, or fall back to its tail.

    Takes the LAST traceback: a `raise ... from exc` chain prints the original
    first, and the final one is the exception that actually ended the process.
    When there is no traceback at all - an OOM kill, a signal, a shell error
    before Python even started - the tail of stderr is still the best evidence
    available, so it is returned rather than reporting nothing.
    """
    if not stderr.strip():
        return None
    marker = "Traceback (most recent call last):"
    index = stderr.rfind(marker)
    if index == -1:
        return truncate_head_tail(stderr, 0, ERROR_TRACE_FALLBACK_CHARS)
    return stderr[index:]


def parse_metrics_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Recover the metrics object from the script's final stdout line.

    A fallback only. The generated script both writes `metrics.<mode>.json` and
    prints the identical object as its last line (see coder/README.md's metrics
    contract); the file in the bind mount is the primary source, and this covers
    the case where the process printed its result and then died before or during
    the file write.

    Scans upward from the end and requires a contract key, so an unrelated line
    of JSON-looking output cannot be mistaken for the metrics.
    """
    contract_keys = {"claim_id", "metric", "value"}
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and contract_keys & parsed.keys():
            return parsed
    return None


def metrics_path_for(paper_dir: Path, mode: str) -> Path:
    """Where `reproduce.sh <mode>` writes its metrics, on the host side of the mount."""
    return paper_dir / f"metrics.{mode}.json"


def read_metrics_file(paper_dir: Path, mode: str) -> dict[str, Any] | None:
    """Read `metrics.<mode>.json` back out of the bind-mounted paper directory.

    This is how results leave the container: the script writes into the mount,
    which IS the host directory, so there is no `docker cp` step and nothing to
    copy out after the container is gone.
    """
    path = metrics_path_for(paper_dir, mode)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"  [metrics] {path.name} exists but is not valid JSON: {exc}")
        return None
    if not isinstance(parsed, dict):
        logger.warning(f"  [metrics] {path.name} is valid JSON but not an object")
        return None
    return parsed


def build_run_command(
    *,
    image: str,
    paper_dir: Path,
    cache_dir: Path,
    mode: str,
    container_name: str,
    memory: str | None = None,
    cpus: str | None = None,
    network: str = "bridge",
) -> list[str]:
    """Assemble the `docker run` argv for one stage.

    Three mounts, each for a distinct reason:

    * `<paper_dir>` -> /workspace, read-write. Read-write is not optional: the
      generated script writes `metrics.<mode>.json` there, and that mount is the
      entire results channel.
    * `<cache>/datasets` -> /workspace/data, read-write. The generated scripts
      default `--data-dir` to `./data` relative to the script, and `reproduce.sh`
      never overrides it, so this nested mount turns a per-paper CIFAR-10
      download into ONE shared copy across all eight papers. Docker orders mounts
      by path depth, so /workspace lands first and this shadows it. If some
      future paper's script uses a different directory name the mount is simply
      unused and its download lands in the paper's own mount - slower, never
      broken. `./data` is a convention here, not a contract.
    * `<cache>/home` -> /cache, read-write. HOME/HF_HOME/TORCH_HOME point here in
      the image, so any HuggingFace or torch.hub fetch also survives `--rm`.

    `--rm` because the results are already on the host through the mount, so a
    stopped container carries nothing worth keeping.
    """
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--workdir",
        CONTAINER_WORKDIR,
        "--volume",
        f"{paper_dir.resolve()}:{CONTAINER_WORKDIR}",
        "--volume",
        f"{(cache_dir / 'datasets').resolve()}:{CONTAINER_DATA_DIR}",
        "--volume",
        f"{(cache_dir / 'home').resolve()}:{CONTAINER_CACHE_DIR}",
        "--network",
        network,
    ]
    if memory:
        command += ["--memory", memory]
    if cpus:
        command += ["--cpus", cpus]
    command += [image, "bash", "reproduce.sh", mode]
    return command


def decode_stream(value: str | bytes | None) -> str:
    """Normalize a captured stream to `str`.

    `subprocess.run(text=True)` decodes on the success path, but the
    `TimeoutExpired` path attaches whatever partial output it had joined as
    RAW BYTES even in text mode. Every timeout would otherwise crash the handler
    that is trying to report the timeout.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# --------------------------------------------------------------------------- #
# The daemon-facing runner.
# --------------------------------------------------------------------------- #


class DockerRunner:
    """Builds the sandbox image and runs a paper's `reproduce.sh` stages in it."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        stage_timeouts: dict[str, int] | None = None,
        memory: str | None = None,
        cpus: str | None = None,
        network: str = "bridge",
        run_triage: bool = True,
    ) -> None:
        self.image = image
        self.cache_dir = cache_dir
        self.stage_timeouts = {**DEFAULT_STAGE_TIMEOUTS, **(stage_timeouts or {})}
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.run_triage = run_triage

    # -- daemon plumbing ---------------------------------------------------- #

    def check_daemon(self) -> None:
        """Fail early, and with an actionable message, if Docker is not usable.

        Worth its own step: without it the first failure would be a `docker run`
        exiting non-zero, which is indistinguishable from the generated script
        failing - and would be triaged as such, spending an API call to learn
        that Docker was not running.
        """
        if shutil.which("docker") is None:
            raise DockerUnavailableError(
                "the `docker` CLI is not on PATH. Install Docker Desktop (macOS) or "
                "the docker engine (Linux); runner/ shells out to the CLI and needs "
                "no Python docker SDK."
            )
        probe = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_CONTROL_TIMEOUT,
            check=False,
        )
        if probe.returncode != 0:
            raise DockerUnavailableError(
                "the docker daemon is not reachable (`docker info` failed). Start "
                "Docker Desktop and let it finish initializing, then retry. Client "
                f"said: {probe.stderr.strip() or probe.stdout.strip()}"
            )
        logger.info(f"[docker] daemon reachable, server version {probe.stdout.strip()}")

    def image_exists(self) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
            text=True,
            timeout=DOCKER_CONTROL_TIMEOUT,
            check=False,
        )
        return result.returncode == 0

    def build_image(self) -> None:
        """Build the sandbox image from `runner/Dockerfile`.

        The build context is `runner/` itself - a handful of small files. No
        generated script is in it, by design: the image stays static across every
        Coder regeneration and its layer cache is never invalidated by one.
        """
        logger.info(f"[docker] building {self.image} from {DOCKERFILE_PATH}")
        logger.info(f"[docker] build context: {BUILD_CONTEXT} (no generated code is copied in)")
        started = time.monotonic()
        result = subprocess.run(
            [
                "docker",
                "build",
                "--tag",
                self.image,
                "--file",
                str(DOCKERFILE_PATH),
                str(BUILD_CONTEXT),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=BUILD_TIMEOUT,
            check=False,
        )
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            logger.error(f"[docker] build FAILED after {elapsed:.1f}s (exit {result.returncode})")
            for line in (result.stderr or result.stdout).splitlines()[-30:]:
                logger.error(f"  [build] {line}")
            raise RuntimeError(f"docker build failed with exit code {result.returncode}")
        logger.info(f"[docker] built {self.image} in {elapsed:.1f}s")

    def ensure_image(self, build: bool) -> None:
        """Make sure the sandbox image is present, building it only if asked."""
        exists = self.image_exists()
        if exists and not build:
            logger.info(f"[docker] image {self.image} already present, skipping build")
            return
        if not exists and not build:
            raise ImageMissingError(
                f"image {self.image} does not exist locally and --no-build was given. "
                f"Run once with --build, or `docker build -t {self.image} "
                f"-f {DOCKERFILE_PATH} {BUILD_CONTEXT}`."
            )
        self.build_image()

    def kill_container(self, name: str) -> None:
        """Kill a container by name, tolerating one that is already gone.

        THE reason `--name` exists. A `subprocess` timeout kills only the
        `docker run` client process; the container it asked for is a child of
        dockerd and keeps running - burning CPU and holding the bind mounts -
        until something explicitly stops it. "No such container" is a success
        here: with `--rm` the daemon may already have reaped it.
        """
        logger.warning(f"[docker] killing container {name} (the timeout did not stop it)")
        result = subprocess.run(
            ["docker", "kill", name],
            capture_output=True,
            text=True,
            timeout=DOCKER_CONTROL_TIMEOUT,
            check=False,
        )
        if result.returncode == 0:
            logger.info(f"[docker] container {name} killed")
            return
        stderr = result.stderr.strip()
        if "No such container" in stderr or "is not running" in stderr:
            logger.info(f"[docker] container {name} was already gone ({stderr})")
            return
        logger.error(f"[docker] `docker kill {name}` failed: {stderr}; trying `docker rm -f`")
        forced = subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            text=True,
            timeout=DOCKER_CONTROL_TIMEOUT,
            check=False,
        )
        if forced.returncode == 0:
            logger.info(f"[docker] container {name} force-removed")
        else:
            logger.error(
                f"[docker] could not remove container {name}: {forced.stderr.strip()}. "
                f"It may still be running - check `docker ps` and remove it by hand."
            )

    # -- execution ---------------------------------------------------------- #

    def ensure_cache_dirs(self) -> None:
        """Create the shared cache directories before any mount uses them.

        A bind-mount source that does not exist is created by the daemon instead,
        which on Linux means a root-owned directory the host user then cannot
        clean up. Creating them here keeps them owned by whoever ran the pipeline.
        """
        for subdirectory in ("datasets", "home"):
            (self.cache_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    def run_stage(self, paper_dir: Path, logs_dir: Path, mode: str) -> StageRun:
        """Run one `bash reproduce.sh <mode>` in the sandbox and report on it."""
        timeout_seconds = self.stage_timeouts[mode]
        container_name = container_name_for(paper_dir.name, mode)
        command = build_run_command(
            image=self.image,
            paper_dir=paper_dir,
            cache_dir=self.cache_dir,
            mode=mode,
            container_name=container_name,
            memory=self.memory,
            cpus=self.cpus,
            network=self.network,
        )
        logger.info(f"  [{mode}] container: {container_name}")
        logger.info(f"  [{mode}] budget: {timeout_seconds}s")
        logger.info(f"  [{mode}] command: {' '.join(command)}")

        started = time.monotonic()
        status: RunStatus
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            stdout = decode_stream(completed.stdout)
            stderr = decode_stream(completed.stderr)
            exit_code: int | None = completed.returncode
            status = "success" if completed.returncode == 0 else "error"
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout = decode_stream(exc.stdout)
            stderr = decode_stream(exc.stderr)
            exit_code = None
            status = "timeout"
            logger.error(f"  [{mode}] TIMEOUT after {elapsed:.1f}s (budget {timeout_seconds}s)")
            self.kill_container(container_name)

        stdout_path = logs_dir / f"{mode}.stdout.log"
        stderr_path = logs_dir / f"{mode}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        logger.info(
            f"  [{mode}] captured {len(stdout)} chars stdout -> {stdout_path.name}, "
            f"{len(stderr)} chars stderr -> {stderr_path.name} (full, untruncated)"
        )

        metrics = read_metrics_file(paper_dir, mode)
        metrics_source = "file"
        if metrics is None:
            metrics = parse_metrics_from_stdout(stdout)
            metrics_source = "stdout fallback"
        if metrics is None:
            if status == "success":
                logger.warning(
                    f"  [{mode}] exited 0 but wrote no metrics.{mode}.json and printed no "
                    f"metrics line - the stage ran, but produced no numbers to compare"
                )
        else:
            logger.info(
                f"  [{mode}] metrics ({metrics_source}): claim_id={metrics.get('claim_id')} "
                f"{metrics.get('metric')}={metrics.get('value')} {metrics.get('unit', '')} "
                f"(eval_accuracy={metrics.get('eval_accuracy')}, "
                f"train_accuracy={metrics.get('train_accuracy')}, "
                f"epochs_completed={metrics.get('epochs_completed')})"
            )

        if status == "success":
            logger.info(f"  [{mode}] PASSED in {elapsed:.1f}s (exit 0)")
        elif status == "error":
            logger.error(f"  [{mode}] FAILED in {elapsed:.1f}s (exit {exit_code})")

        metrics_file = metrics_path_for(paper_dir, mode)
        return StageRun(
            mode=mode,
            status=status,
            exit_code=exit_code,
            wall_clock_seconds=round(elapsed, 2),
            timeout_seconds=timeout_seconds,
            container_name=container_name,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            metrics_path=str(metrics_file) if metrics_file.exists() else None,
            metrics=metrics,
            log_excerpt=build_log_excerpt(stdout, stderr),
            error_trace=extract_error_trace(stderr) if status != "success" else None,
        )

    def run_paper(
        self,
        paper_dir: Path,
        logs_dir: Path,
        modes: tuple[str, ...],
        client: Anthropic | None = None,
    ) -> RunnerOutput:
        """Run a paper's stages in order, stopping at the first non-zero exit.

        Triage runs only on a genuine failure. Success needs no explanation, and
        a timeout's cause is already known mechanically - spending an API call on
        either would be paying to be told what the exit status already said.
        """
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_cache_dirs()
        reproduce_sh = paper_dir / "reproduce.sh"
        if not reproduce_sh.exists():
            raise FileNotFoundError(
                f"no reproduce.sh in {paper_dir} - runner/ drives generated scripts only "
                f"through that wrapper. Generate it with `uv run python -m coder.pipeline`."
            )

        logger.info(f"[run] {paper_dir.name}")
        logger.info(f"[run] stages: {' -> '.join(modes)}")
        stages: list[StageRun] = []
        total_elapsed = 0.0
        overall: RunStatus = "success"
        failed_stage: str | None = None
        exit_code: int | None = 0

        for mode in modes:
            stage = self.run_stage(paper_dir, logs_dir, mode)
            stages.append(stage)
            total_elapsed += stage.wall_clock_seconds
            if stage.status != "success":
                overall = stage.status
                failed_stage = mode
                exit_code = stage.exit_code
                logger.error(f"[run] stopping escalation at '{mode}' ({stage.status})")
                break

        last = stages[-1] if stages else None
        metrics_stage = next(
            (stage for stage in reversed(stages) if stage.metrics is not None), None
        )

        triage: TriageResult | None = None
        if overall == "error" and last is not None:
            if not self.run_triage:
                logger.info("[run] triage disabled, skipping the classification call")
            elif client is None:
                logger.warning(
                    "[run] failure needs triage but no Anthropic client was supplied "
                    "(missing ANTHROPIC_API_KEY?) - recording the failure untriaged"
                )
            else:
                triage = triage_failure(
                    exit_code=last.exit_code,
                    failed_stage=last.mode,
                    log_excerpt=last.log_excerpt,
                    client=client,
                )
        elif overall == "timeout":
            logger.info("[run] timeout is mechanical, no triage call made")
        elif overall == "success":
            logger.info(f"[run] all {len(stages)} stage(s) passed in {total_elapsed:.1f}s")

        return RunnerOutput(
            paper=paper_dir.name,
            image=self.image,
            paper_dir=str(paper_dir),
            status=overall,
            stage_reached=last.mode if last else None,
            failed_stage=failed_stage,
            exit_code=exit_code,
            wall_clock_seconds=round(total_elapsed, 2),
            reproduced_metrics=metrics_stage.metrics if metrics_stage else None,
            metrics_mode=metrics_stage.mode if metrics_stage else None,
            logs_path=str(logs_dir),
            log_excerpt=last.log_excerpt if last else "",
            error_trace=last.error_trace if last else None,
            triage=triage,
            stages=stages,
        )
