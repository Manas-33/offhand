#!/usr/bin/env python3
"""Generate the teacher-labeled training set for the 0.6B tool-call specialist.

Query source is hybrid: template-seeded skeletons give controlled argument and
tool coverage, and (optionally, on Colab) a teacher paraphrase pass adds phrasing
diversity. The teacher (Qwen3-4B) then LABELS each query with a tool call through
the tool-calling path, and the eval harness scorer keeps only the items the
teacher got right (correct tool + schema-valid).

Eval hygiene baked in:
 - Leave-one-tool-out: the held-out tools are filtered out of training but still
   listed in the prompt tool set (the deployment condition, a new tool added to
   the app), and their items go to a separate held-out eval file.
 - The listed-tool set varies per example, so the model cannot learn a closed
   set ("the answer is always one of these N").

Runs locally against the mock teacher; on Colab pass --teacher hf for the real 4B.
Note: the template set below covers a representative subset of tools to prove the
pipeline; adding the remaining tools is just more TEMPLATES/SLOTS entries.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Protocol

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))

from offhand_eval.dataset import load_tools, tools_by_name  # noqa: E402
from offhand_eval.parse import parse_tool_calls  # noqa: E402
from offhand_eval.scoring import is_schema_valid  # noqa: E402

# One trivial (flashlight), one argument-heavy (calendar), one entity-referencing (sms).
HELD_OUT = ("toggle_flashlight", "create_calendar_event", "draft_sms")

SLOTS = {
    "time": ["7am", "6:30", "5 PM", "quarter past 8", "noon", "9:15", "8pm"],
    "duration": ["10 minute", "half an hour", "45 minute", "2 hour", "20 minute"],
    "person": ["Mom", "John", "Sarah", "Dad", "Alex", "my boss"],
    "title": ["dentist appointment", "team meeting", "lunch with Sam", "project review"],
    "note": ["buy milk", "call the plumber", "water the plants", "book flights"],
}

TEMPLATES = {
    "set_alarm": ["set an alarm for {time}", "wake me up at {time}", "can you set an alarm for {time}"],
    "set_timer": ["start a {duration} timer", "set a timer for {duration}", "give me a {duration} timer"],
    "create_note": ["make a note to {note}", "note down: {note}", "jot down {note}"],
    "create_calendar_event": ["add a {title} to my calendar", "schedule a {title} for tomorrow", "put {title} on my calendar"],
    "draft_sms": ["text {person} that I'll be late", "send a message to {person}", "draft a text to {person}"],
    "toggle_flashlight": ["turn on the flashlight", "turn off the torch", "switch on the flashlight", "kill the flashlight"],
}

NO_CALL_QUERIES = [
    "what's the capital of France?", "tell me a joke", "how tall is Mount Everest?",
    "what is 15 percent of 240?", "who won the world cup in 2018?", "explain photosynthesis",
]


class Teacher(Protocol):
    def generate(self, query: str, tools: list[dict]) -> str: ...


class MockTeacher:
    """Deterministic, mostly-correct labels for local testing (no model needed)."""

    _MAP = {
        "alarm": ("set_alarm", {"time": "07:00"}),
        "wake me": ("set_alarm", {"time": "07:00"}),
        "timer": ("set_timer", {"duration_minutes": 10}),
        "note": ("create_note", {"content": "buy milk"}),
        "jot": ("create_note", {"content": "buy milk"}),
        "calendar": ("create_calendar_event", {"title": "event", "start_datetime": "2026-09-12T10:00"}),
        "schedule": ("create_calendar_event", {"title": "event", "start_datetime": "2026-09-12T10:00"}),
        "text ": ("draft_sms", {"recipient": "Mom", "body": "running late"}),
        "message": ("draft_sms", {"recipient": "Mom", "body": "running late"}),
        "flashlight": ("toggle_flashlight", {"state": "on"}),
        "torch": ("toggle_flashlight", {"state": "on"}),
    }

    def generate(self, query: str, tools: list[dict]) -> str:
        q = query.lower()
        for key, (name, args) in self._MAP.items():
            if key in q:
                return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'
        return "I can answer that directly."  # no tool call


def build_seeds(n_per_tool: int, rng: random.Random) -> list[dict]:
    """Template-seeded (query, intended_tool) pairs, plus the no-call queries."""
    seeds: list[dict] = []
    for tool, templates in TEMPLATES.items():
        made: set[str] = set()
        attempts = 0
        while len(made) < n_per_tool and attempts < n_per_tool * 20:
            attempts += 1
            template = rng.choice(templates)
            fills = {s: rng.choice(SLOTS[s]) for s in SLOTS if "{" + s + "}" in template}
            query = template.format(**fills)
            if query in made:
                continue
            made.add(query)
            seeds.append({"intended_tool": tool, "query": query})
    for query in NO_CALL_QUERIES:
        seeds.append({"intended_tool": None, "query": query})
    return seeds


def listed_tools(all_names: list[str], intended, rng: random.Random, k_min=3, k_max=6) -> list[str]:
    """A varied subset of tools to show in this example's prompt (blocks closed-set leak)."""
    others = [n for n in all_names if n != intended]
    k = rng.randint(k_min, min(k_max, len(others)))
    names = rng.sample(others, k) + ([intended] if intended else [])
    rng.shuffle(names)
    return names


def label_and_validate(seed: dict, by_name: dict, teacher: Teacher, rng: random.Random) -> dict | None:
    """Have the teacher label the query, keep it only if the label is clean."""
    names = listed_tools(list(by_name), seed["intended_tool"], rng)
    tools = [by_name[n] for n in names]
    raw = teacher.generate(seed["query"], tools)
    calls = parse_tool_calls(raw)

    if seed["intended_tool"] is None:
        if calls:  # a no-call item where the teacher wrongly called a tool
            return None
        target = raw.strip()
    else:
        if not calls:
            return None
        call = calls[0]
        if call.get("name") != seed["intended_tool"] or not is_schema_valid(call, by_name):
            return None
        target = f'<tool_call>{json.dumps(call)}</tool_call>'

    return {
        "intended_tool": seed["intended_tool"],
        "query": seed["query"],
        "tools_listed": names,
        "target": target,
    }


def _write(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the specialist training set")
    ap.add_argument("--tools", default=str(HERE.parent / "eval" / "tools.json"))
    ap.add_argument("--n-per-tool", type=int, default=200)
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--teacher", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_name = tools_by_name(load_tools(args.tools))
    rng = random.Random(args.seed)

    if args.teacher == "mock":
        teacher: Teacher = MockTeacher()
    else:
        from offhand_eval.runners import HFRunner
        teacher = HFRunner(model_id=args.model)

    seeds = build_seeds(args.n_per_tool, rng)
    train, heldout = [], []
    dropped = 0
    for i, seed in enumerate(seeds):
        item = label_and_validate(seed, by_name, teacher, rng)
        if item is None:
            dropped += 1
            continue
        item["id"] = f"gen_{i}"
        (heldout if seed["intended_tool"] in HELD_OUT else train).append(item)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "train.jsonl", train)
    _write(out / "heldout_eval.jsonl", heldout)
    print(f"kept {len(train) + len(heldout)}, dropped {dropped} -> "
          f"train={len(train)}, heldout={len(heldout)} in {out}")


if __name__ == "__main__":
    main()
