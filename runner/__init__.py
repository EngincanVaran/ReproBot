"""Runner agent: executes coder/'s generated training scripts in a Docker sandbox.

`pipeline.py` is the entry point (paper discovery, escalating stage gates,
output writing); `docker_runner.py` owns the `docker build`/`docker run`
mechanics, host-side timeouts and log capture; `triage.py` is a single
Claude-Haiku call that classifies a genuine failure as a script bug vs. an
environment problem.

This stage never imports torch and never runs a training script on the host -
everything executes inside the container, whose own Linux CPython is what makes
the repo's Intel-macOS/Python-3.13 torch trap (see CLAUDE.md) inapplicable
there. The only interface to a generated script is `bash reproduce.sh <mode>`;
this stage never builds a `python` command and never passes a `--flag`."""
