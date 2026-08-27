"""Coder agent: turns reader/'s structured extraction into a runnable script.

`pipeline.py` is the entry point and runs a fixed sequential chain -
`method_translator.py` -> `code_synthesizer.py` -> `dependency_resolver.py` -
not a homogeneous loop like reader/'s Extractor stages, since each step's
input is the previous step's differently-shaped output. `models.py` holds
the data shapes threaded between them."""
