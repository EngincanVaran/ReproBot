"""Coder agent: turns a paper's structured Reader output into a runnable training script.

`pipeline.py` is the entry point (input resolution, the deterministic syntax
and CLI-flag gates, output writing); `script_writer.py` is an importable
`CodeWriter` subclass (see `base.py`), not a standalone script. This stage
only calls the Claude API and writes files - it never executes the generated
script and never imports torch."""
