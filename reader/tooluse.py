"""Shared tool-use plumbing for every Claude call in `reader/`.

Every stage in this package makes the same shape of call - one forced
tool-use request over a paper's Markdown, one structured payload back - and
every stage therefore has the same three ways of getting a *structurally
valid but wrong* answer back. All three are silent by construction: they do
not raise, they produce an empty-looking extraction that is indistinguishable
from a clean run over a paper that had nothing to say.

1. **`max_tokens` truncation.** The tool-use JSON is cut off mid-generation,
   so whatever parses out of it is incomplete. Hit twice in this repo already
   (`ocr/vlm_extract.py`, then `reader/claims.py` returning zero claims for
   Wide Residual Networks across all three retries).
2. **A double-encoded tool input.** `stop_reason` is a clean `tool_use`, but
   instead of the schema's object the model emits a single key whose value is
   the entire payload re-encoded as a JSON *string*. The extraction succeeded;
   the field-by-field `.get()` calls then all miss and the result is thrown
   away. Reproduced 3/3 against `claude-sonnet-5` on `data_pipeline` +
   Network In Network.
3. **An all-empty payload.** `stop_reason: tool_use`, well-formed object, every
   required field missing. Intermittent and model-side: seen on three different
   stages across three consecutive runs, never reproducible on demand.

This module owns the response side of all three, so a stage module holds only
its prompt, its tool schema, and its own parsing - and so a fix lands once
rather than six times. It owns the *request* as well, not just the parse,
because handling (3) properly means being able to re-ask immediately; see the
long comment in `request_tool_use` for that tradeoff.

`log_prefix` is the stage's own console tag (`"claims"`, `"hparams"`,
`"data_pipeline"`, `"validate"`, ...) passed in rather than inferred, so the
warnings raised here land in the same `  [stage] ...` column as the stage's
own logging and a reader of the console can tell instantly which call
misbehaved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, ToolChoiceToolParam, ToolParam
from loguru import logger


def as_list(value: object) -> list[object]:
    """Coerce a tool-input field to a list, tolerating an absent or malformed one.

    Every list-valued field in every `reader/` tool schema is read through this,
    so a field that arrives as `None` (absent) or as some other type degrades to
    `[]` instead of raising mid-parse. Deliberately silent: deciding whether an
    empty list is *news* belongs to the missing-key report below, which can see
    the whole payload, not to a per-field coercion that cannot.
    """
    return value if isinstance(value, list) else []


def as_int(value: object, log_prefix: str, field: str) -> int:
    """Coerce a tool-input field the schema types as `integer` to an int.

    `as_list`'s counterpart, and it exists because the failure it absorbs was
    caught live rather than imagined: on Wide Residual Networks the model
    answered the `integer`-typed `candidates_considered` with the string
    *"Let me count carefully."* - narrating instead of filling the schema. A bare
    `int()` raises `ValueError` on that, which does not merely lose the field, it
    aborts the whole paper (`pipeline.py` gives up on the file; `run_dataset`
    moves to the next one). That is strictly worse than the silent-empty
    extractions this module exists to catch: an empty stage still reaches the
    validator, which can flag it and trigger a retry, whereas a crash reaches
    nothing.

    So a malformed value degrades to 0 and says so loudly, exactly as a missing
    one would, and the run continues to the validator that can do something
    about it.
    """
    if isinstance(value, bool):  # bool is an int subclass; a flag here is malformed
        pass
    elif isinstance(value, int):
        return value
    elif isinstance(value, float) and value.is_integer():
        return int(value)
    elif isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    elif value is None:
        return 0
    logger.error(
        f"  [{log_prefix}] '{field}' is schema-typed as an integer but arrived as "
        f"{type(value).__name__} {value!r} - recording 0. The rest of this payload may "
        f"be unreliable too; check what the model actually returned."
    )
    return 0


def _check_stop_reason(message: Message, log_prefix: str) -> None:
    """A `max_tokens` stop means the tool-use JSON was cut off mid-generation, so
    whatever parses out of it is silently incomplete. Make it loud; this repo has
    already been bitten twice by exactly this."""
    if message.stop_reason == "max_tokens":
        logger.error(
            f"  [{log_prefix}] response hit max_tokens - the tool-use JSON is TRUNCATED "
            f"and this extraction is incomplete. Raise max_tokens for this call."
        )


def _unwrap_double_encoded(payload: dict[str, object], log_prefix: str) -> dict[str, object]:
    """Recover a tool input the model serialized *twice*.

    Observed live against `claude-sonnet-5` (reproducibly, on
    `reader/data_pipeline.py` + Network In Network): instead of the tool
    schema's object, the model emitted a single key whose value is the whole
    payload re-encoded as a JSON *string* -
    `{"datasets": "{\\"datasets_examined\\": [...], \\"datasets\\": [...]}"}`.
    The extraction had actually succeeded; every field-by-field `.get()`
    downstream then missed, and the stage returned empty while looking like a
    clean run. Unwrap it and say so loudly rather than discarding a good
    extraction.

    The shape test is narrow on purpose - exactly one key, whose value is a
    string that parses as a JSON object - so it cannot fire on a legitimate
    single-key payload holding real prose.
    """
    if len(payload) != 1:
        return payload
    ((key, value),) = payload.items()
    if not isinstance(value, str):
        return payload
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return payload
    if not isinstance(decoded, dict):
        return payload
    logger.warning(
        f"  [{log_prefix}] tool input arrived double-encoded (whole payload as a JSON "
        f"string under '{key}') - unwrapped it; the extraction below is real, not empty"
    )
    return cast(dict[str, object], decoded)


def _tool_input(message: Message, tool_name: str, log_prefix: str) -> dict[str, object]:
    """Pull the forced tool call's input out of the response, unwrapping (2).

    `tool_choice` forces the call, so a response with no `tool_use` block is a
    protocol-level failure rather than a bad extraction - it raises, naming the
    tool that was asked for, exactly as each stage's own copy of this did.
    """
    for block in message.content:
        if block.type == "tool_use":
            return _unwrap_double_encoded(block.input, log_prefix)
    raise RuntimeError(f"Claude did not call the {tool_name} tool")


def _report_payload_problems(
    payload: dict[str, object],
    log_prefix: str,
    required_keys: Sequence[str],
    may_be_empty_keys: Sequence[str],
) -> bool:
    """Say which schema-required keys did not arrive usable, and what did.

    Every key checked here is `required` in the caller's tool schema, so a miss
    is a malformed response rather than a paper with nothing to say. Without
    this, `as_list()` and `str(... or "")` turn each miss into `[]`/`""` and the
    stage reports an all-empty extraction indistinguishable from a clean run.

    The two lists exist because "required" means two different things in these
    schemas. `required_keys` must arrive with a *truthy* value - no real paper
    has zero claims or an empty `summary`. `may_be_empty_keys` are required to
    be present but legitimately empty: `validator.flags` is empty when nothing
    is wrong, `architecture_notes.unstated_details` when a paper fully specifies
    itself. For those, only absence is reported.

    Returns True when *every* `required_keys` entry is missing or empty - i.e.
    failure shape (3) - which is what `request_tool_use` uses to decide whether
    to re-ask.
    """
    missing = [key for key in required_keys if not payload.get(key)]
    absent = [key for key in may_be_empty_keys if key not in payload]
    if missing:
        logger.error(
            f"  [{log_prefix}] tool payload missing/empty for {missing} - keys actually "
            f"received: {sorted(payload)}. This extraction is incomplete; the validator "
            f"should flag it and the pipeline retry this stage."
        )
    if absent:
        logger.error(
            f"  [{log_prefix}] tool payload omitted required key(s) {absent} entirely - "
            f"keys actually received: {sorted(payload)}. An empty value for these is a "
            f"legitimate answer; their absence is a malformed response."
        )
    return bool(required_keys) and len(missing) == len(required_keys)


def request_tool_use(
    client: Anthropic,
    *,
    log_prefix: str,
    model: str,
    max_tokens: int,
    tool: dict[str, Any],
    user_content: str,
    required_keys: Sequence[str] = (),
    may_be_empty_keys: Sequence[str] = (),
) -> dict[str, object]:
    """Make one forced tool-use call and return its (guarded) input payload.

    Guards truncation, double-encoding, and missing keys on the way out, and
    re-asks exactly once if the payload comes back entirely empty. The caller
    still owns its own prompt, tool schema, and parsing; what it gets back here
    is a plain dict that has already been checked and complained about.
    """
    tool_name = str(tool["name"])

    def send() -> dict[str, object]:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=[cast(ToolParam, tool)],
            tool_choice=cast(ToolChoiceToolParam, {"type": "tool", "name": tool_name}),
            messages=[{"role": "user", "content": user_content}],
        )
        _check_stop_reason(message, log_prefix)
        return _tool_input(message, tool_name, log_prefix)

    payload = send()
    all_empty = _report_payload_problems(payload, log_prefix, required_keys, may_be_empty_keys)
    if not all_empty:
        return payload

    # --- Why re-ask here rather than leaving it to the validator ---
    #
    # The validator loop does already catch an all-empty payload, and
    # demonstrably so: it flags "the fields are all empty", `pipeline.py` routes
    # that flag back to the owning stage by name, and the re-run comes back
    # populated. The objection is the price, not the outcome. That path spends a
    # whole validation pass out of a budget of three, and a validation pass is
    # the most expensive call in this package - the full paper Markdown *plus*
    # every stage's combined output - before the stage re-run it triggers is even
    # paid for. Worse, it spends retry budget on a malformed response, when the
    # budget exists for genuine extraction *errors*: the Wide ResNet run that
    # proved the loop works finished with 8 unresolved flags against a cap of 3
    # passes, so a pass burned here is a real extraction problem left unexamined.
    # Re-asking costs one more call with the same prompt and leaves that intact.
    #
    # Against that, a retry that cannot help is pure waste, so the trigger is
    # deliberately narrow and the cost of being wrong is bounded:
    #   - Only when EVERY `required_keys` entry is missing or empty. A partly
    #     populated payload is a real extraction with a gap in it - whether that
    #     gap is wrong is a judgement about the paper, which is precisely the
    #     validator's job - and re-asking would throw away good data.
    #   - Only once. If the emptiness is deterministic (a prompt or schema bug
    #     rather than the intermittent model-side shape documented above) a retry
    #     cannot fix it, and looping would double the bill on every run forever.
    #     One re-ask, then hand the empty payload back and let the validator have
    #     its turn: the old safety net stays exactly where it was, it just stops
    #     being the first line of defence.
    #   - Stages passing no `required_keys` never reach here at all. That is what
    #     `validator.py` does, because an empty `flags` list is its *success*
    #     case; re-asking there would double the cost of every clean validation.
    #
    # The re-ask resends the identical request rather than adding a "you returned
    # nothing, try again" nudge. The failure is intermittent - re-probing the
    # same stage/paper pair by hand returned good payloads both times - so a
    # plain re-ask is enough, and a nudged prompt would quietly make the retry a
    # different question from the one whose answer was lost.
    logger.warning(
        f"  [{log_prefix}] tool payload came back entirely empty on a clean 'tool_use' "
        f"stop - re-requesting once immediately rather than spending a validation pass "
        f"on it (this failure shape is intermittent and model-side; see reader/tooluse.py)"
    )
    payload = send()
    if _report_payload_problems(payload, log_prefix, required_keys, may_be_empty_keys):
        logger.error(
            f"  [{log_prefix}] the immediate re-request ALSO came back entirely empty - "
            f"this looks deterministic rather than intermittent, so it is left to the "
            f"validator retry loop. Suspect the prompt or the tool schema, not the model."
        )
    else:
        logger.info(
            f"  [{log_prefix}] re-request returned a populated payload - validation "
            f"budget preserved"
        )
    return payload
