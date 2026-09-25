"""Report generator: turn a finished run into a Markdown replication report.

Deterministic and free: it reads what the earlier stages already wrote and makes no model
call, so a report can be regenerated at any time and always agrees with the run's files.
"""
