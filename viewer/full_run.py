"""One paper through every stage, as a background job the viewer starts and watches.

OCR -> Reader -> Coder -> Orchestrator (Runner + Critic + retries) -> Report. Each stage is
the stage's own CLI, unmodified, and each one skips work already on disk, so a stopped job
restarts where it left off. The Orchestrator runs either here in local Docker (CPU) or on a
rented GPU box over SSH: code and this paper's inputs are rsynced over, the run is started
under `nohup` there so a dropped SSH connection does not kill it, and its outputs are
rsynced back every poll so the viewer's Runner/Critic tabs and learning curve fill in live.

The viewer launches this with `start_job()`; status lives in `viewer/jobs/<paper>/`:
`status.json` (stage + state, rewritten at every transition), `job.log` (every stage's
output) and `pid`.

    uv run --extra viewer python -m viewer.full_run --paper "<paper>" --pdf <path> \
        --target gpu --ssh "ssh -p 44683 root@1.2.3.4"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from critic.blocking import pass_blocking_reasons
from runner.docker_runner import slugify

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_ROOT / "viewer" / "jobs"
REMOTE_ROOT = "/root/ReproBot"
STAGES = ["ocr", "reader", "coder", "remote_setup", "orchestrator", "report"]
POLL_SECONDS = 30
GPU_DOCKER_SETUP = """set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
if ! docker run --rm --gpus all python:3.11-slim test -e /dev/nvidia0; then
  echo "GPU not reachable from Docker - installing the NVIDIA Container Toolkit"
  export DEBIAN_FRONTEND=noninteractive
  KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o $KEYRING
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  docker run --rm --gpus all python:3.11-slim test -e /dev/nvidia0
fi
echo "GPU reachable from Docker"
"""
# Everything the box needs to run the Orchestrator: code and lock files, never outputs,
# caches or PDFs (one paper's inputs are pushed separately).
RSYNC_EXCLUDES = [
    ".git",
    ".venv",
    "__pycache__",
    "*.pdf",
    "ocr/output*",
    "reader/output*",
    "coder/output*",
    "runner/output*",
    "runner/cache",
    "orchestrator/output*",
    "critic/output*",
    "report/output*",
    "viewer/jobs",
    "viewer/uploads",
    "docs",
    "papers",
    ".agents",
    ".claude",
]


def job_dir(paper: str) -> Path:
    return JOBS_DIR / paper


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    user: str

    @classmethod
    def parse(cls, command: str) -> SshTarget:
        """Read `ssh -p PORT user@host [...]` as Vast prints it; extra flags are ignored."""
        tokens = shlex.split(command)
        port, dest = 22, None
        index = 1 if tokens and tokens[0] == "ssh" else 0
        while index < len(tokens):
            token = tokens[index]
            if token == "-p":
                port = int(tokens[index + 1])
                index += 2
                continue
            if token in ("-L", "-R", "-i", "-o"):
                index += 2
                continue
            if not token.startswith("-") and "@" in token:
                dest = token
            index += 1
        if dest is None:
            raise ValueError(f"no user@host in SSH command: {command!r}")
        user, host = dest.split("@", 1)
        return cls(host=host, port=port, user=user)

    def ssh_args(self) -> list[str]:
        return [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ConnectTimeout=20",
            f"{self.user}@{self.host}",
        ]

    def rsync_shell(self) -> str:
        return f"ssh -p {self.port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

    def remote(self, path: str) -> str:
        return f"{self.user}@{self.host}:{path}"


class Job:
    def __init__(self, paper: str) -> None:
        self.paper = paper
        self.dir = job_dir(paper)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "job.log"
        self.status: dict[str, Any] = {"paper": paper, "stages": {}, "state": "running"}

    def write_status(self, **fields: Any) -> None:
        self.status.update(fields, updated=datetime.now(UTC).isoformat(timespec="seconds"))
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=2))
        tmp.replace(self.dir / "status.json")

    def stage(self, name: str, state: str, detail: str = "") -> None:
        self.status["stages"][name] = {"state": state, "detail": detail}
        logger.info(f"[{name}] {state}{': ' + detail if detail else ''}")
        self.write_status(current_stage=name)

    def run(self, command: list[str], *, check: bool = True) -> int:
        """Run one command, appending its combined output to job.log as it streams."""
        logger.info(f"$ {shlex.join(command)}")
        with self.log_path.open("a") as log:
            log.flush()
            proc = subprocess.run(
                command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if check and proc.returncode != 0:
            raise RuntimeError(f"exit {proc.returncode}: {shlex.join(command[:4])} ...")
        return proc.returncode


def py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def run_local_stages(job: Job, pdf: Path) -> Path:
    paper = job.paper
    md = REPO_ROOT / "ocr" / "output" / "vlm" / f"{paper}.md"
    reader_json = REPO_ROOT / "reader" / "output" / f"{paper}.json"
    coder_json = REPO_ROOT / "coder" / "output" / paper / "coder_output.json"

    for name, done, command in (
        ("ocr", md, py("ocr.vlm_extract", "--input", str(pdf))),
        ("reader", reader_json, py("reader.pipeline", "--input", str(md))),
        ("coder", coder_json, py("coder.pipeline", "--input", str(reader_json))),
    ):
        if done.exists():
            job.stage(name, "skipped", "output already on disk")
            continue
        job.stage(name, "running")
        job.run(command)
        if not done.exists():
            raise RuntimeError(f"{name} finished without writing {done.relative_to(REPO_ROOT)}")
        job.stage(name, "done")
    return reader_json


def orchestrator_args(paper: str, pdf: Path, max_stage: str, retry_budget: int) -> list[str]:
    return [
        "--input",
        f"reader/output/{paper}.json",
        "--dataset-dir",
        str(pdf.parent.relative_to(REPO_ROOT)),
        "--use-existing-script",
        "--max-stage",
        max_stage,
        "--retry-budget",
        str(retry_budget),
    ]


def run_orchestrator_local(job: Job, pdf: Path, max_stage: str, retry_budget: int) -> None:
    job.stage("remote_setup", "skipped", "local Docker target")
    job.stage("orchestrator", "running", f"local Docker, up to {max_stage}")
    job.run(
        py("orchestrator.pipeline", *orchestrator_args(job.paper, pdf, max_stage, retry_budget))
    )
    job.stage("orchestrator", "done")


def tar_over_ssh(job: Job, target: SshTarget, rels: list[str], *, push: bool) -> int:
    """Copy repo-relative paths between the two repo roots with tar over SSH.

    Not rsync: macOS ships openrsync, which hands remote paths to the remote shell
    unprotected, and every paper name contains spaces."""
    names = " ".join(shlex.quote(rel) for rel in rels)
    if push:
        local = f"tar -C {shlex.quote(str(REPO_ROOT))} -cf - {names}"
        remote = f"mkdir -p {REMOTE_ROOT} && tar -C {REMOTE_ROOT} -xf -"
        command = f"{local} | {shlex.join(target.ssh_args())} {shlex.quote(remote)}"
    else:
        # Missing paths (nothing produced yet) are not an error on a pull.
        remote = (
            f"cd {REMOTE_ROOT} && set -- && for p in {names}; do "
            '[ -e "$p" ] && set -- "$@" "$p"; done; [ $# -gt 0 ] && tar -cf - "$@"'
        )
        local = f"tar -C {shlex.quote(str(REPO_ROOT))} -xf -"
        command = f"{shlex.join(target.ssh_args())} {shlex.quote(remote)} | {local}"
    return job.run(["bash", "-o", "pipefail", "-c", command], check=push)


def push_paper(job: Job, target: SshTarget) -> None:
    paper = job.paper
    rels = [
        f"ocr/output/vlm/{paper}.md",
        f"reader/output/{paper}.json",
        f"coder/output/{paper}",
        ".env",
    ]
    tar_over_ssh(job, target, rels, push=True)


def pull_paper(job: Job, target: SshTarget) -> None:
    """Bring this paper's run back so the viewer shows it; never fails the job."""
    paper = job.paper
    tar_over_ssh(job, target, [f"orchestrator/output/{paper}", f"coder/output/{paper}"], push=False)


def remote_setup(job: Job, target: SshTarget) -> None:
    """Idempotent: uv + Python 3.13 venv on the box, current code synced, GPU visible."""
    job.stage("remote_setup", "running", f"{target.host}:{target.port}")
    # Listing an `nvidia` runtime proves nothing (a Vast VM shipped one pointing at a
    # binary that was not installed); only a real `--gpus all` container does.
    job.run([*target.ssh_args(), GPU_DOCKER_SETUP])
    job.run(
        [
            *target.ssh_args(),
            "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)",
        ]
    )
    job.run([*target.ssh_args(), f"mkdir -p {REMOTE_ROOT}"])
    excludes = [arg for pattern in RSYNC_EXCLUDES for arg in ("--exclude", pattern)]
    # Excluded paths are never deleted on the receiver, so outputs and caches survive.
    job.run(
        [
            "rsync",
            "-a",
            "--delete",
            *excludes,
            "--exclude",
            ".env",
            "-e",
            target.rsync_shell(),
            f"{REPO_ROOT}/",
            target.remote(f"{REMOTE_ROOT}/"),
        ]
    )
    job.run(
        [
            *target.ssh_args(),
            "export PATH=$HOME/.local/bin:$PATH; command -v uv >/dev/null || "
            "curl -LsSf https://astral.sh/uv/install.sh | sh; "
            f"cd {REMOTE_ROOT} && $HOME/.local/bin/uv sync -q "
            "--extra orchestrator --extra critic --extra runner",
        ]
    )
    push_paper(job, target)
    job.stage("remote_setup", "done")


def remote_run_active(target: SshTarget, paper: str) -> bool:
    """True when this paper's Orchestrator is already running on the box, so a restarted
    job attaches to it rather than launching a second run (or re-pushing its files)."""
    # `[o]` keeps the pattern from matching the remote shell that runs pgrep itself.
    literal = re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", f"reader/output/{paper}.json")
    pattern = shlex.quote(f"[o]rchestrator\\.pipeline --input {literal}")
    probe = subprocess.run(
        [*target.ssh_args(), f"pgrep -f -- {pattern} >/dev/null"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


def run_orchestrator_remote(
    job: Job, pdf: Path, target: SshTarget, max_stage: str, retry_budget: int
) -> None:
    paper = job.paper
    remote_job = f"{REMOTE_ROOT}/viewer/jobs/{shlex.quote(paper)}"
    args = shlex.join(orchestrator_args(paper, pdf, max_stage, retry_budget))
    inner = (
        f"cd {REMOTE_ROOT} && $HOME/.local/bin/uv run -q --extra orchestrator "
        f"--extra critic --extra runner python -m orchestrator.pipeline {args} "
        f"--gpu --as-host-user --force; echo $? > {remote_job}/exit_code"
    )
    if remote_run_active(target, paper):
        job.stage("orchestrator", "running", f"GPU box, up to {max_stage} (attached)")
    else:
        job.stage("orchestrator", "running", f"GPU box, up to {max_stage}")
        # Only the nohup'd command may go to the background: with `a && b && c & d` the
        # whole chain is backgrounded and ssh waits on it for the entire run.
        job.run(
            [
                *target.ssh_args(),
                f"mkdir -p {remote_job} && rm -f {remote_job}/exit_code || exit 1; "
                f"setsid nohup bash -c {shlex.quote(inner)} > {remote_job}/run.log 2>&1 "
                f"< /dev/null & echo $! > {remote_job}/pid",
            ]
        )

    offset = 0
    while True:
        time.sleep(POLL_SECONDS)
        # Append only the new part of the remote log to job.log.
        tail = subprocess.run(
            [*target.ssh_args(), f"tail -c +{offset + 1} {remote_job}/run.log"],
            capture_output=True,
            check=False,
        )
        if tail.returncode == 0 and tail.stdout:
            offset += len(tail.stdout)
            with job.log_path.open("ab") as log:
                log.write(tail.stdout)
        pull_paper(job, target)
        code = subprocess.run(
            [*target.ssh_args(), f"cat {remote_job}/exit_code 2>/dev/null"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if code:
            # The run may have written its final state after this poll's pull; pull again.
            pull_paper(job, target)
            if code != "0":
                raise RuntimeError(f"orchestrator on the GPU box exited {code}")
            break
        job.write_status()  # heartbeat, so the viewer can tell a live job from a dead one
    job.stage("orchestrator", "done")


def run_verdict(paper: str) -> str:
    """The Orchestrator exits 0 whatever happened to the paper; its verdict is in state.json."""
    state_path = REPO_ROOT / "orchestrator" / "output" / paper / "state.json"
    if not state_path.exists():
        return "no state.json"
    return str(json.loads(state_path.read_text()).get("verdict"))


def critic_summary(paper: str) -> dict[str, Any]:
    """The fidelity side, which `run_verdict` cannot see: did the number match, and may the
    match count. A run can succeed and still fail - or pass on a method the review blocks."""
    state_path = REPO_ROOT / "orchestrator" / "output" / paper / "state.json"
    if not state_path.exists():
        return {}
    critic = json.loads(state_path.read_text()).get("critic_output") or {}
    return {
        "verdict": critic.get("verdict"),
        "reproduced": critic.get("reproduced"),
        "claimed": critic.get("claimed"),
        "unit": critic.get("unit") or "",
        "blocked": pass_blocking_reasons(critic.get("review"))
        if critic.get("verdict") == "pass"
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--target", choices=["local", "gpu"], default="local")
    parser.add_argument("--ssh", default=None, help="Vast's SSH command (gpu target)")
    parser.add_argument("--max-stage", default=None, help="Default: full on gpu, capped local")
    parser.add_argument("--retry-budget", type=int, default=3)
    args = parser.parse_args()

    job = Job(args.paper)
    logger.add(job.log_path, level="INFO")
    max_stage = args.max_stage or ("full" if args.target == "gpu" else "capped")
    job.write_status(
        target=args.target,
        max_stage=max_stage,
        pid=os.getpid(),
        started=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        run_local_stages(job, args.pdf.resolve())
        if args.target == "gpu":
            if not args.ssh:
                raise ValueError("--target gpu needs --ssh")
            target = SshTarget.parse(args.ssh)
            if remote_run_active(target, args.paper):
                job.stage("remote_setup", "skipped", "a run is already active on the box")
            else:
                remote_setup(job, target)
            run_orchestrator_remote(job, args.pdf.resolve(), target, max_stage, args.retry_budget)
        else:
            run_orchestrator_local(job, args.pdf.resolve(), max_stage, args.retry_budget)
        verdict = run_verdict(args.paper)
        job.write_status(verdict=verdict, critic=critic_summary(args.paper))
        job.stage("report", "running")
        job.run(py("report.pipeline", "--input", f"orchestrator/output/{args.paper}", "--force"))
        job.stage("report", "done")
        if verdict != "success":
            # The report is written either way; the failure belongs to the run itself.
            job.status["current_stage"] = "orchestrator"
            raise RuntimeError(f"the run ended with verdict '{verdict}' - see the log above")
        job.write_status(state="done")
    except Exception as exc:  # noqa: BLE001 - every failure must land in status.json
        logger.exception(f"[job] failed: {exc}")
        current = job.status.get("current_stage", "")
        if current:
            job.status["stages"][current] = {"state": "failed", "detail": str(exc)}
        job.write_status(state="failed", error=str(exc))
        raise SystemExit(1) from exc


def start_job(paper: str, pdf: Path, target: str, ssh: str | None, max_stage: str) -> int:
    """Launch main() detached from the Streamlit process; returns its pid."""
    directory = job_dir(paper)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.log").write_text("")
    (directory / "status.json").unlink(missing_ok=True)
    command = py(
        "viewer.full_run",
        "--paper",
        paper,
        "--pdf",
        str(pdf),
        "--target",
        target,
        "--max-stage",
        max_stage,
    )
    if ssh:
        command += ["--ssh", ssh]
    with (directory / "launcher.log").open("w") as out:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (directory / "pid").write_text(str(proc.pid))
    return proc.pid


def job_alive(paper: str) -> bool:
    pid_file = job_dir(paper) / "pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    # A finished job stays a zombie until Streamlit (its parent) reaps it; that is not alive.
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return not stat.stdout.strip().startswith("Z")


def stop_job(paper: str, ssh: str | None) -> None:
    """Kill the local job's process group, and the remote orchestrator if there is one."""
    pid_file = job_dir(paper) / "pid"
    if pid_file.exists():
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
    if ssh:
        target = SshTarget.parse(ssh)
        remote_job = f"{REMOTE_ROOT}/viewer/jobs/{shlex.quote(paper)}"
        # The orchestrator's own containers get --name reprobot-*, so kill those too.
        subprocess.run(
            [
                *target.ssh_args(),
                f"pkill -TERM -s $(cat {remote_job}/pid 2>/dev/null || echo 0) 2>/dev/null; "
                # Only this paper's containers: other papers may be training beside it.
                f"docker ps -q --filter name=^reprobot-{slugify(paper)[:48]}- "
                "| xargs -r docker kill",
            ],
            capture_output=True,
            check=False,
            timeout=60,
        )
    status_path = job_dir(paper) / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["state"] = "stopped"
        status_path.write_text(json.dumps(status, indent=2))


def read_status(paper: str) -> dict[str, Any] | None:
    path = job_dir(paper) / "status.json"
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data


if __name__ == "__main__":
    main()
