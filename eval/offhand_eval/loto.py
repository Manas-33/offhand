"""Leave-one-tool-out (LOTO) reporting on top of the eval results.

The specialist is trained on 7 tools and three are held out of training entirely
(but still shown in the prompt, the "a new tool was added to the app" condition).
This groups the per-item scores the harness already produces into:

* SEEN   - the trained tools, per-tool and pooled
* UNSEEN - the three held-out tools, per-tool and pooled
* NO-CALL - the irrelevance items

The headline for the kill-test (Gate K) is the gap between seen and unseen
call-accuracy: small gap => the specialist generalizes its tool-calling to tools
it never trained on; large gap => the capability floor is about generality, not
just capacity.

No new scoring lives here. The held-out tools are already scored on their
discriminating slot because ``mini_eval.jsonl`` gold only specifies that slot
(e.g. a calendar event's ``title``, not its hallucinated ``start_datetime``), and
``scoring.args_match`` only checks gold-specified args.
"""

from __future__ import annotations

from typing import Any

# Must match train/gen_data.HELD_OUT (the tools filtered out of training).
HELD_OUT = ("toggle_flashlight", "create_calendar_event", "draft_sms")

_RATE_KEYS = ("right_tool", "call_accuracy", "schema_valid")


def _rates(scores: list[dict]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {"n": 0, "right_tool": None, "call_accuracy": None, "schema_valid": None}
    return {
        "n": n,
        "right_tool": round(sum(bool(s["right_tool"]) for s in scores) / n, 4),
        "call_accuracy": round(sum(bool(s["right_args"]) for s in scores) / n, 4),
        "schema_valid": round(sum(bool(s["schema_valid"]) for s in scores) / n, 4),
    }


def loto_breakdown(items: list[dict], results: list[dict], held_out=HELD_OUT, which: str = "strict") -> dict[str, Any]:
    """Group ``results`` (from harness.run_eval) into seen / unseen / no-call.

    ``items`` supplies each id's gold tool name; ``which`` selects the strict or
    lenient score for each item.
    """
    held_out = set(held_out)
    score_key = f"{which}_score"
    gold_name = {
        it["id"]: (None if it["gold"].get("no_call") else it["gold"].get("name"))
        for it in items
    }

    per_tool: dict[str, list[dict]] = {}
    no_call: list[dict] = []
    for r in results:
        name = gold_name.get(r["id"])
        if name is None:
            no_call.append(r[score_key])
        else:
            per_tool.setdefault(name, []).append(r[score_key])

    seen_tools = sorted(t for t in per_tool if t not in held_out)
    unseen_tools = sorted(t for t in per_tool if t in held_out)

    def pooled(tools: list[str]) -> dict[str, Any]:
        rates = _rates([s for t in tools for s in per_tool[t]])
        return {"tools": tools, **rates}

    nc_n = len(no_call)
    return {
        "which": which,
        "held_out": sorted(held_out),
        "per_tool": {t: _rates(per_tool[t]) for t in seen_tools + unseen_tools},
        "seen": pooled(seen_tools),
        "unseen": pooled(unseen_tools),
        "no_call": {
            "n": nc_n,
            "accuracy": round(sum(bool(s["no_call_correct"]) for s in no_call) / nc_n, 4) if nc_n else None,
        },
    }


def format_loto(bd: dict) -> str:
    def fmt(x: Any) -> str:
        return f"{x:.2f}" if isinstance(x, (int, float)) else "-"

    def row(name: str, m: dict) -> str:
        return (f"  {name:<24}{fmt(m.get('right_tool')):>11}{fmt(m.get('call_accuracy')):>10}"
                f"{fmt(m.get('schema_valid')):>9}{m.get('n', 0):>5}")

    header = f"  {'tool':<24}{'right_tool':>11}{'call_acc':>10}{'schema':>9}{'n':>5}"
    out = [f"LOTO breakdown [{bd['which']}]  (held-out: {', '.join(bd['held_out'])})", ""]

    out.append("  SEEN (trained tools)")
    out.append(header)
    for t in bd["seen"]["tools"]:
        out.append(row(t, bd["per_tool"][t]))
    out.append(row("-- seen overall --", bd["seen"]))

    out += ["", "  UNSEEN (held-out trio)", header]
    for t in bd["unseen"]["tools"]:
        out.append(row(t, bd["per_tool"][t]))
    out.append(row("-- unseen overall --", bd["unseen"]))

    nc = bd["no_call"]
    out += ["", f"  NO-CALL   accuracy={fmt(nc['accuracy'])}  n={nc['n']}"]

    seen_acc, unseen_acc = bd["seen"]["call_accuracy"], bd["unseen"]["call_accuracy"]
    if seen_acc is not None and unseen_acc is not None:
        gap = round(seen_acc - unseen_acc, 4)
        out += ["", f"  Gate signal: unseen call_acc {unseen_acc:.2f} vs seen {seen_acc:.2f}  (generalization gap {gap:+.2f})"]
    return "\n".join(out)
