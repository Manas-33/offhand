"""Score a parsed tool call against a gold answer.

Four signals, deliberately separated so the M0c writeup can show *where* a
precision drop bites:

* ``schema_valid``  - names a real tool, arguments are an object, required
  params present, no unknown params. (Grammar-constrained decoding in M2 should
  drive this to ~1.0 regardless of precision.)
* ``right_tool``    - correct function name.
* ``right_args``    - every gold-specified argument matches an accepted value.
* ``no_call``       - for irrelevance items: the model correctly stayed quiet.

Argument matching supports three gold forms:
  * list            -> predicted value must equal one of the members
  * {"contains": s} -> normalized substring match (for free-text slots)
  * scalar          -> exact normalized equality
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _arg_matches(predicted: Any, accepted: Any) -> bool:
    if isinstance(accepted, dict) and "contains" in accepted:
        return _norm(accepted["contains"]) in _norm(predicted)
    if isinstance(accepted, list):
        return _norm(predicted) in {_norm(a) for a in accepted}
    return _norm(predicted) == _norm(accepted)


def is_schema_valid(call: dict, tools_by_name: dict[str, dict]) -> bool:
    name = call.get("name")
    if name not in tools_by_name:
        return False
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False
    params = tools_by_name[name]["function"]["parameters"]
    properties = params.get("properties", {})
    required = params.get("required", [])
    if any(req not in args for req in required):
        return False
    if any(key not in properties for key in args):  # strict: no unknown params
        return False
    return True


def args_match(predicted_args: dict, gold_args: dict) -> bool:
    for key, accepted in gold_args.items():
        if key not in predicted_args:
            return False
        if not _arg_matches(predicted_args[key], accepted):
            return False
    return True


@dataclass
class ItemScore:
    id: str
    category: str
    expected_call: bool
    made_call: bool
    schema_valid: bool = False
    right_tool: bool = False
    right_args: bool = False
    no_call_correct: bool | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def score_item(item: dict, calls: list[dict], tools_by_name: dict[str, dict]) -> ItemScore:
    gold = item["gold"]
    expects_call = not gold.get("no_call", False)
    score = ItemScore(
        id=item["id"],
        category=item.get("category", "single"),
        expected_call=expects_call,
        made_call=bool(calls),
    )

    if not expects_call:
        score.no_call_correct = len(calls) == 0
        return score

    if not calls:
        return score

    call = calls[0]
    score.schema_valid = is_schema_valid(call, tools_by_name)
    score.right_tool = call.get("name") == gold["name"]
    if score.right_tool and score.schema_valid:
        score.right_args = args_match(call.get("arguments", {}), gold.get("arguments", {}))
    return score


def aggregate(scores: list[ItemScore]) -> dict[str, Any]:
    call_items = [s for s in scores if s.expected_call]
    irr_items = [s for s in scores if not s.expected_call]

    def rate(items: list[ItemScore], attr: str) -> float | None:
        if not items:
            return None
        return round(sum(bool(getattr(s, attr)) for s in items) / len(items), 4)

    return {
        "n_total": len(scores),
        "n_call_items": len(call_items),
        "n_irrelevance_items": len(irr_items),
        "schema_valid_rate": rate(call_items, "schema_valid"),
        "right_tool_rate": rate(call_items, "right_tool"),
        "call_accuracy": rate(call_items, "right_args"),
        "irrelevance_accuracy": rate(irr_items, "no_call_correct"),
    }
