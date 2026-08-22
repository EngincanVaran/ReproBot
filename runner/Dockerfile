# ReproBot runner sandbox - the image coder/'s generated training scripts execute in.
#
# ---------------------------------------------------------------------------
# WHY THIS IMAGE IS THE ANSWER TO THE REPO'S PLATFORM TRAP, NOT ANOTHER INSTANCE OF IT
# ---------------------------------------------------------------------------
# CLAUDE.md and ocr/README.md document a hard constraint on the *host*: the dev
# machine is an Intel Mac (x86_64) on Python 3.13, and no PyTorch release ships
# Intel-macOS wheels and Python 3.13 support at the same time (3.13 support
# landed in 2.5.0; the last Intel-mac wheel was 2.2.2). That is why torch is
# deliberately absent from this repo's pyproject.toml / uv.lock.
#
# None of that applies INSIDE this container. The container runs its own Linux
# CPython 3.11 against Linux wheels; the host's OS, Python version and lockfile
# are irrelevant to it. Running the generated scripts here is precisely how this
# stage sidesteps the trap - runner/ itself stays torch-free on the host (its
# pyproject extra is just anthropic + python-dotenv) and shells out to the
# `docker` CLI, so nothing torch-shaped ever has to resolve in uv.lock.
#
# ---------------------------------------------------------------------------
# WHY python:3.11-slim AND NOT pytorch/pytorch
# ---------------------------------------------------------------------------
# The pytorch/pytorch images are CUDA-based: they carry the CUDA runtime, cuDNN
# and NCCL layers (several GB) that a CPU-only CIFAR-10 gate run never touches.
# Starting from python:3.11-slim and installing the CPU-only wheels from
# https://download.pytorch.org/whl/cpu gives an image a fraction of the size,
# faster to pull and rebuild, with nothing in it we do not use.
#
# 3.11 rather than 3.13: it is the version with the broadest, longest-settled
# wheel coverage across torch/torchvision/transformers, and this image has no
# reason to track the host's interpreter - it never imports this repo's code.
#
# ---------------------------------------------------------------------------
# WHY NOTHING GENERATED IS COPIED IN
# ---------------------------------------------------------------------------
# There is deliberately no `COPY train.py` / `COPY reproduce.sh` here. The
# paper's coder/output/<paper>/ directory is BIND-MOUNTED at `docker run` time
# (see runner/docker_runner.py). That keeps the image completely static: the
# Coder can regenerate every script in the repo and this image neither rebuilds
# nor invalidates its layer cache. It is also how results get back out - the
# generated script writes metrics.<mode>.json into the mounted directory, which
# is the host directory, so no `docker cp` is needed.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# libgomp1: OpenMP runtime for torch's threaded CPU kernels. The manylinux torch
#   wheel bundles its own copy, but having the system one present avoids the
#   loader edge cases that show up with mixed OpenMP runtimes.
# ca-certificates: torchvision downloads CIFAR-10 over HTTPS on the first run.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PYTHONUNBUFFERED matters more than it looks: on a timeout runner/ kills the
# container, and only output already flushed through the pipe survives. Buffered
# stdout would lose exactly the progress lines that explain where it hung.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false

# Caches point at /cache, which runner/ bind-mounts to a host directory so a
# torch.hub / HuggingFace download survives container teardown (--rm).
ENV HOME=/cache \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

# Versions are pinned so a rebuild months from now produces the same sandbox a
# recorded run used - an unpinned image would silently change what "the script
# failed" means. torch/torchvision must move together (2.5.1 <-> 0.20.1);
# transformers/accelerate are the Trainer API the generated scripts target, and
# accelerate is a hard requirement of transformers.Trainer, not an optional add.
#
# Two separate pip steps on purpose: --index-url (not --extra-index-url) pins
# the torch step to the CPU-only index, so a CUDA build from PyPI can never be
# resolved instead. The PyPI-only packages need their own step because that
# index does not serve them.
#
# The torch pin is deliberately `2.5.1` and NOT `2.5.1+cpu`: the CPU index tags
# the x86_64 wheel with the +cpu local version but the aarch64 one without it,
# so pinning the bare version resolves on both - the Intel dev machine and an
# Apple Silicon collaborator's. Both were confirmed to exist:
#   torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl                        (174.7 MB)
#   torch-2.5.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
#   torchvision-0.20.1+cpu-cp311-cp311-linux_x86_64.whl
#   torchvision-0.20.1-cp311-cp311-linux_aarch64.whl
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchvision==0.20.1

RUN pip install --no-cache-dir \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        numpy==2.1.3 \
        pillow==11.0.0

# Mount points, created up front so their ownership/permissions are known rather
# than being invented by the daemon at mount time.
#   /workspace      <- coder/output/<paper>/   (rw; metrics.<mode>.json is written here)
#   /workspace/data <- shared CIFAR-10 cache   (rw; nested inside /workspace on purpose)
#   /cache          <- shared HF/torch cache   (rw)
RUN mkdir -p /workspace /workspace/data /cache

WORKDIR /workspace

# runner/ always overrides this with an explicit `bash reproduce.sh <mode>`.
# The default is the cheapest gate, so an image run by hand cannot accidentally
# start a full multi-hour training run.
CMD ["bash", "reproduce.sh", "probe"]
