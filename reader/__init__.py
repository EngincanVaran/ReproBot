"""Reader agent: turns VLM-transcribed paper Markdown into structured data.

`pipeline.py` is the entry point and owns the validation retry loop;
`claims.py`/`hyperparameters.py`/`data_pipeline.py` are importable
`Extractor` subclasses (see `base.py`), not standalone scripts."""
