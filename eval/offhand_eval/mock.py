"""Build a mock runner that emits the reference-correct call for each item.

Used by the test suite and by `run_eval.py --dry-run` to exercise the full
pipeline without a model. A "reference-correct" call carries the gold value for
every scored slot **and** a placeholder for any other required param a real
model would also have to supply (e.g. an SMS `body` or an event
`start_datetime`), so that a perfect model scores 100% on `schema_valid` too.
"""

from __future__ import annotations

import json

from .runners import MockRunner


def _gold_value(accepted):
    if isinstance(accepted, list):
        return accepted[0]
    if isinstance(accepted, dict) and "contains" in accepted:
        return accepted["contains"]
    return accepted


def reference_output(item: dict, tools_by_name: dict[str, dict]) -> str:
    gold = item["gold"]
    if gold.get("no_call"):
        return "Here's a plain-text answer with no tool call."

    args = {k: _gold_value(v) for k, v in gold.get("arguments", {}).items()}
    required = tools_by_name[gold["name"]]["function"]["parameters"].get("required", [])
    for req in required:
        args.setdefault(req, "placeholder")

    call = {"name": gold["name"], "arguments": args}
    return f"<tool_call>{json.dumps(call)}</tool_call>"


def build_mock_runner(items: list[dict], tools_by_name: dict[str, dict]) -> MockRunner:
    mapping = {item["query"]: reference_output(item, tools_by_name) for item in items}
    return MockRunner(mapping=mapping)
