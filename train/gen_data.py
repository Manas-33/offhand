#!/usr/bin/env python3
"""Generate the teacher-labeled training set for the 0.6B tool-call specialist.

Query source is hybrid:
 - Tool calls: template-seeded skeletons give controlled argument/tool coverage,
   then a teacher paraphrase pass (--teacher hf, on Colab) rewrites each seed into
   many phrasings, scaling a handful of templates into a few thousand varied items.
 - No calls: short, tool-agnostic general questions sampled from Dolly on the real
   run (the mock run uses a few built-in queries so it needs no download).

The teacher (Qwen3-4B) then LABELS each query through the tool-calling path, and we
keep only the items it got right: for a tool query, the intended tool with a
schema-valid call; for a no-call query, no tool call at all (its answer trimmed to
one short sentence, since a 0.6B should abstain briefly, not recite an essay).

Eval hygiene baked in:
 - Leave-one-tool-out: the held-out tools are filtered out of training but still
   listed in the prompt tool set (the deployment condition, a new tool added to
   the app), and their items go to a separate held-out eval file.
 - The listed-tool set varies per example, so the model cannot learn a closed
   set ("the answer is always one of these N").

Runs locally against the mock teacher; on Colab pass --teacher hf for the real 4B.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Protocol

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))

from offhand_eval.dataset import load_tools, tools_by_name  # noqa: E402
from offhand_eval.parse import parse_tool_calls  # noqa: E402
from offhand_eval.scoring import is_schema_valid  # noqa: E402

try:
    from tqdm.auto import tqdm  # progress + ETA for the long teacher run
except ImportError:  # optional; degrade to a no-op so the mock run needs nothing
    def tqdm(iterable, **kwargs):
        return iterable

# One trivial (flashlight), one argument-heavy (calendar), one entity-referencing (sms).
HELD_OUT = ("toggle_flashlight", "create_calendar_event", "draft_sms")

SLOTS = {
    "time": ["7am", "6:30", "5 PM", "quarter past 8", "noon", "9:15", "8pm",
             "midnight", "half past 6", "7:45", "10 in the morning"],
    "duration": ["10 minute", "half an hour", "45 minute", "2 hour", "20 minute",
                 "90 second", "5 minute", "three hour"],
    "person": ["Mom", "John", "Sarah", "Dad", "Alex", "my boss", "Priya",
               "Marcus", "Grandma", "the landlord"],
    "title": ["dentist appointment", "team meeting", "lunch with Sam", "project review",
              "1:1 with Dana", "gym session", "parent teacher meeting"],
    "note": ["buy milk", "call the plumber", "water the plants", "book flights",
             "renew the passport", "pick up the dry cleaning", "email the accountant"],
    "panel": ["wifi", "bluetooth", "display", "sound", "battery", "location"],
    "music": ["some jazz", "the Beatles", "some lo-fi beats", "some classical music",
              "my workout playlist", "some Taylor Swift", "a relaxing playlist"],
}

TEMPLATES = {
    "set_alarm": ["set an alarm for {time}", "wake me up at {time}", "can you set an alarm for {time}"],
    "set_timer": ["start a {duration} timer", "set a timer for {duration}", "give me a {duration} timer"],
    "create_note": ["make a note to {note}", "note down: {note}", "jot down {note}"],
    "set_reminder": ["remind me to {note}", "set a reminder to {note}", "remind me to {note} at {time}", "can you remind me to {note}"],
    "open_settings": ["open {panel} settings", "go to the {panel} settings", "take me to {panel} settings", "open the {panel} settings panel"],
    "draft_email": ["email {person} about the {title}", "draft an email to {person}", "write an email to {person}", "send an email to {person} about the {title}"],
    "play_music": ["play {music}", "put on {music}", "pause the music", "skip this song", "play the next track", "resume the music"],
    "create_calendar_event": ["add a {title} to my calendar", "schedule a {title} for tomorrow", "put {title} on my calendar"],
    "draft_sms": ["text {person} that I'll be late", "send a message to {person}", "draft a text to {person}"],
    "toggle_flashlight": ["turn on the flashlight", "turn off the torch", "switch on the flashlight", "kill the flashlight"],
}

NO_CALL_QUERIES = [
    "what's the capital of France?", "tell me a joke", "how tall is Mount Everest?",
    "what is 15 percent of 240?", "who won the world cup in 2018?", "explain photosynthesis",
]

NO_CALL_MAX_CHARS = 120


def shorten_no_call(text: str, max_chars: int = NO_CALL_MAX_CHARS) -> str:
    """Trim a teacher no-call answer to one short sentence.

    A 0.6B should learn to abstain with a brief reply, not reproduce a
    multi-paragraph essay (the real 4B answered "explain photosynthesis" with
    ~850 characters). Short targets also keep the training signal on the
    decision to abstain rather than on reciting facts. Take the first non-empty
    line, then its first sentence, then a word-boundary cap.
    """
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    match = re.search(r"[.!?](?:\s|$)", line)
    if match:
        line = line[: match.end()].strip()
    if len(line) > max_chars:
        line = line[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return line


class Teacher(Protocol):
    def generate(self, query: str, tools: list[dict]) -> str: ...


class MockTeacher:
    """Deterministic, mostly-correct labels for local testing (no model needed).

    ``paraphrase`` here is a stand-in that just prepends canned lead-ins, enough
    to exercise the scale-up path offline; the real phrasing diversity comes from
    the 4B teacher on Colab. Key order matters: ``settings`` is matched before
    ``play`` (so "display settings" is not routed to music), and ``remind``
    before ``email`` (so "remind me to email X" stays a reminder).
    """

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
        "settings": ("open_settings", {"panel": "wifi"}),
        "remind": ("set_reminder", {"text": "call the plumber"}),
        "pause the music": ("play_music", {"action": "pause"}),
        "skip this song": ("play_music", {"action": "next"}),
        "next track": ("play_music", {"action": "next"}),
        "resume the music": ("play_music", {"action": "play"}),
        "put on": ("play_music", {"action": "play"}),
        "play": ("play_music", {"action": "play"}),
        "email": ("draft_email", {"to": "Mom", "body": "running late"}),
    }

    _PARA_PREFIXES = (
        "", "hey, ", "could you ", "please ", "i need you to ",
        "would you mind, ", "quick, ", "when you get a sec, ", "ok so ",
    )

    def generate(self, query: str, tools: list[dict]) -> str:
        q = query.lower()
        for key, (name, args) in self._MAP.items():
            if key in q:
                return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'
        return "I can answer that directly."  # no tool call

    def paraphrase(self, query: str, n: int) -> list[str]:
        return [(pre + query) if pre else query for pre in self._PARA_PREFIXES][:n]


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
        target = shorten_no_call(raw)
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


def expand_with_paraphrases(seeds: list[dict], teacher: Teacher, k: int, rng: random.Random) -> list[dict]:
    """Multiply each tool seed into up to ``k`` paraphrases (keeping the original).

    The teacher rewrites the query; the intended tool carries over unchanged. A
    paraphrase that drifts to a different intent is not a problem here: the
    labeling pass re-runs the teacher and drops any item whose label no longer
    matches ``intended_tool``.
    """
    if k <= 0 or not hasattr(teacher, "paraphrase"):
        return seeds
    out: list[dict] = []
    for seed in tqdm(seeds, desc="paraphrasing"):
        variants = {seed["query"]}
        try:
            for para in teacher.paraphrase(seed["query"], k):
                para = para.strip()
                if para:
                    variants.add(para)
        except Exception as exc:  # a teacher hiccup should not lose the seed
            print(f"  paraphrase failed for {seed['query']!r}: {exc}", file=sys.stderr)
        for query in variants:
            out.append({"intended_tool": seed["intended_tool"], "query": query})
    rng.shuffle(out)
    return out


def load_dolly_no_call(n: int, rng: random.Random) -> list[str]:
    """Pull short, tool-agnostic general questions to use as no-call seeds.

    Human-written and single-turn; we keep the open/general/brainstorm/classify
    categories that have no supplied context and are phone-assistant length.
    Whether each one is really a no-call is decided later by the teacher: if it
    fires a tool, ``label_and_validate`` drops it, so this only supplies variety.
    """
    from datasets import load_dataset  # heavy dep, only needed for the real run

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    keep = {"open_qa", "general_qa", "brainstorming", "classification"}
    cands = [
        r["instruction"].strip()
        for r in ds
        if r["category"] in keep
        and not r["context"].strip()
        and 8 <= len(r["instruction"].strip()) <= 120
    ]
    rng.shuffle(cands)
    return cands[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the specialist training set")
    ap.add_argument("--tools", default=str(HERE.parent / "eval" / "tools.json"))
    ap.add_argument("--n-per-tool", type=int, default=40,
                    help="template seeds per tool, before paraphrasing")
    ap.add_argument("--paraphrases", type=int, default=8,
                    help="teacher paraphrases per tool seed (0 disables the scale-up)")
    ap.add_argument("--no-call", type=int, default=500,
                    help="no-call seeds to pull from Dolly (only on --teacher hf)")
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
    tool_seeds = [s for s in seeds if s["intended_tool"] is not None]
    no_call_seeds = [s for s in seeds if s["intended_tool"] is None]

    # Phrasing diversity: the teacher paraphrases each template seed.
    tool_seeds = expand_with_paraphrases(tool_seeds, teacher, args.paraphrases, rng)

    # No-call variety: pull real general questions on the real run; the mock run
    # keeps the built-in queries so the smoke test needs no network download.
    if args.teacher == "hf" and args.no_call > 0:
        no_call_seeds = [
            {"intended_tool": None, "query": q}
            for q in load_dolly_no_call(args.no_call, rng)
        ]

    all_seeds = tool_seeds + no_call_seeds
    rng.shuffle(all_seeds)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stream each validated item straight to disk so a long teacher run is
    # crash-safe: a Colab disconnect leaves the items generated so far, and
    # writing to a mounted Drive path (--out-dir) makes them survive the VM.
    n_train = n_heldout = n_no_call = dropped = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as ftrain, \
         open(out / "heldout_eval.jsonl", "w", encoding="utf-8") as fheld:
        for i, seed in enumerate(tqdm(all_seeds, desc="labeling")):
            item = label_and_validate(seed, by_name, teacher, rng)
            if item is None:
                dropped += 1
                continue
            item["id"] = f"gen_{i}"
            if seed["intended_tool"] in HELD_OUT:
                fheld.write(json.dumps(item) + "\n")
                n_heldout += 1
            else:
                ftrain.write(json.dumps(item) + "\n")
                n_train += 1
                if item["intended_tool"] is None:
                    n_no_call += 1
            if (i + 1) % 25 == 0:  # bound how much a crash can lose
                ftrain.flush()
                fheld.flush()

    print(f"kept {n_train + n_heldout} (dropped {dropped}) -> "
          f"train={n_train} (no_call={n_no_call}), heldout={n_heldout} in {out}")


if __name__ == "__main__":
    main()
