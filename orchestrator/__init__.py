"""Orchestrator: the shared-memory state object and the Coder<->Runner retry loop.

`state.py` implements the project plan's §1.2 shared-memory schema, `loop.py` the
bounded retry loop over it, and `pipeline.py` is the only entry point.
"""
