"""Derive pinned pip requirements for a generated training script.

Deterministic - no Claude call, unlike Method Translator and Code
Synthesizer. Regex-scans the script's own `import`/`from ... import` lines
and looks each top-level module up in a small curated version table; a
module not in the table is simply left out (not guessed), and Python
stdlib modules are recognized and skipped from that "unrecognized" set
entirely via `sys.stdlib_module_names` rather than a hand-maintained list.

Versions are chosen for the future runner/ Docker image (a fresh Linux
Python environment), NOT the Intel-mac host's legacy torch==2.2.2 pin set
documented in CLAUDE.md - that pin set exists only to work around this
host's platform trap for local smoke-testing and is irrelevant once code
actually runs in a container. `transformers` is pinned to 4.46.3
specifically: a live test of `code_synthesizer.py`'s output generated a
script using `TrainingArguments(evaluation_strategy=...)`, a parameter
renamed to `eval_strategy` in later releases - 4.46.3 is the last version
confirmed to still accept the old name (see docs/agent-log.md).

This module is an importable step, not a standalone script - `coder/
pipeline.py` calls `resolve_dependencies()` after Code Synthesizer.
"""

from __future__ import annotations

import re
import sys

from loguru import logger

from coder.models import Dependencies, Dependency

# Import name -> actual pip package name, for the common cases where they differ.
_IMPORT_TO_PIP_NAME: dict[str, str] = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
}

_KNOWN_PACKAGES: dict[str, str] = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.46.3",
    "datasets": "3.1.0",
    "numpy": "2.1.3",
    "evaluate": "0.4.3",
    "accelerate": "1.1.1",
    "scikit-learn": "1.5.2",
    "pillow": "11.0.0",
    "opencv-python": "4.10.0.84",
    "pyyaml": "6.0.2",
}

_STDLIB_MODULES = set(sys.stdlib_module_names)

_IMPORT_LINE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _extract_imported_names(script: str) -> set[str]:
    names: set[str] = set()
    for line in script.splitlines():
        match = _IMPORT_LINE.match(line)
        if match:
            names.add(match.group(1))
    return names


def resolve_dependencies(script: str) -> Dependencies:
    """No file I/O here - the pipeline owns writing the requirements out."""
    imported_names = _extract_imported_names(script)
    pip_names = {_IMPORT_TO_PIP_NAME.get(name, name) for name in imported_names}

    resolved: set[str] = set()
    unrecognized: set[str] = set()
    for name in pip_names:
        if name in _KNOWN_PACKAGES:
            resolved.add(name)
        elif name not in _STDLIB_MODULES:
            unrecognized.add(name)

    if "transformers" in resolved:
        # transformers.Trainer requires accelerate at runtime, but nothing in
        # a generated script ever `import`s it by name, so it's never picked
        # up by the scan above - add it explicitly whenever Trainer is used.
        resolved.add("accelerate")

    for name in sorted(unrecognized):
        logger.info(f"  [dependency_resolver] skipped unrecognized import: {name}")
    for name in sorted(resolved):
        logger.info(f"  [dependency_resolver] pinned {name}=={_KNOWN_PACKAGES[name]}")

    requirements = [
        Dependency(package=name, version=_KNOWN_PACKAGES[name]) for name in sorted(resolved)
    ]
    return Dependencies(requirements=requirements)
