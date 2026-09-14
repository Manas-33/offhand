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

Hard call/no-call families (--hard). The v1 set made the call/no-call decision
trivially separable: every no-call item was a general question with only unrelated
phone tools listed, and every call item was a phone action. The specialist learned
that shortcut (a question means answer in text, a tool on the topic means call it)
and paid for it on BFCL. These families make the decision rest on whether a listed
tool can actually serve the request:
 - info_call: question-style requests a listed informational tool answers
   (general_tools.json, training-only), so a question can need a call.
 - info_unlisted: some of the same questions with that tool absent, so they
   need a text answer instead.
 - near_miss_phone / near_miss_general: a related tool is listed but cannot do
   the request (reading alarms with only set_alarm, a product price with only an
   exchange-rate tool), so a topical match alone is no reason to call.
 - unserviceable: action requests no listed tool can perform.
The labels still go through the teacher: a near miss the teacher answers with a
tool call is dropped, so an ambiguous seed never becomes a training target. None
of this comes from BFCL, and train/check_contamination.py verifies that.

Eval hygiene baked in:
 - Leave-one-tool-out: the held-out tools are filtered out of training but still
   listed in the prompt tool set (the deployment condition, a new tool added to
   the app), and their items go to a separate held-out eval file. The hard
   families never list a held-out tool at all, so they cannot teach its scope.
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
from collections import Counter
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

# --- Hard call/no-call families (--hard) ------------------------------------
# Informational tools live in general_tools.json and are used for training only.
# The two city slots draw from disjoint lists, so a pair never repeats a city.

INFO_SLOTS = {
    "city": ["Paris", "Tokyo", "Chicago", "Mumbai", "Sydney", "Toronto", "Berlin", "Cairo", "Seoul", "Denver"],
    "city2": ["London", "Osaka", "Boston", "Delhi", "Melbourne", "Vancouver", "Munich", "Nairobi", "Busan", "Austin"],
    "days": ["3", "5", "7", "two", "four"],
    "fx_pair": ["dollars to euros", "euros to yen", "pounds to dollars", "rupees to dollars", "Canadian dollars to euros"],
    "unit_q": ["5 miles in kilometers", "70 degrees Fahrenheit in Celsius", "3 cups in milliliters",
               "180 pounds in kilograms", "2 liters in ounces", "6 feet in meters"],
    "pct": ["15", "18", "7.5", "40", "12"],
    "num": ["240", "86", "1,200", "59.99", "3,450"],
    "bill": ["84", "120", "56.40", "230", "97.50"],
    "tip": ["15", "18", "20"],
    "people": ["2", "3", "4", "6"],
    "word": ["ephemeral", "ubiquitous", "serendipity", "laconic", "quixotic", "petrichor"],
    "phrase": ["good morning", "where is the train station", "thank you very much", "I'm allergic to peanuts"],
    "lang": ["Spanish", "Japanese", "French", "Hindi", "German"],
    "ticker": ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"],
    "company": ["Apple", "Nvidia", "Microsoft", "Amazon", "Tesla"],
    "team": ["Lakers", "Yankees", "Arsenal", "Maple Leafs", "Chiefs", "Warriors"],
    "flight": ["UA 837", "DL 405", "AA 100", "BA 283", "LH 454"],
    "carrier": ["UPS", "FedEx", "USPS", "DHL"],
    "tracking": ["1Z999AA10123456784", "9400111899223856", "7489 3345 1120", "JD014600006281"],
    "ingredients": ["chicken and rice", "eggs, spinach and feta", "chickpeas and tomatoes", "leftover salmon", "tofu and broccoli"],
    "food": ["oatmeal", "chicken breast", "avocado", "white rice", "almonds", "greek yogurt"],
    "grams": ["100", "150", "50", "200"],
    "country": ["Japan", "Germany", "India", "Brazil", "Canada", "Mexico"],
    "year": ["2026", "2027"],
    "stop": ["Union Station", "Central Square", "Shinjuku station", "Kings Cross", "Main Street"],
    "line": ["the Red Line", "the 22 bus", "the N train", "Line 4"],
    "date": ["June 21", "December 1", "March 15", "next Friday"],
    "movie": ["Dune Part Two", "Inside Out 2", "Oppenheimer", "The Batman", "Wicked"],
    "book": ["The Hobbit", "Beloved", "Dune", "Sapiens", "Pride and Prejudice"],
    "coin": ["Bitcoin", "Ethereum", "Solana", "Dogecoin"],
    "fiat": ["dollars", "euros", "yen", "pounds"],
    "principal": ["250,000", "18,000", "35,000", "400,000"],
    "rate": ["6.5", "4.2", "7.9", "5.1"],
    "years": ["30", "5", "15", "20"],
}

INFO_TEMPLATES = {
    "weather_now": ["what's the weather like in {city} right now?", "is it raining in {city}?", "how hot is it in {city} at the moment?"],
    "forecast_outlook": ["what's the forecast for {city} for the next {days} days?", "will it rain in {city} this week?", "what will the weather be in {city} tomorrow?"],
    "fx_rate": ["what's the exchange rate from {fx_pair}?", "how much is it to convert {fx_pair} today?"],
    "convert_units": ["what is {unit_q}?", "convert {unit_q}", "how much is {unit_q}?"],
    "percent_of": ["what's {pct} percent of {num}?", "what is {pct}% of {num}?"],
    "split_bill": ["our bill is ${bill}, how much does each of the {people} of us pay with a {tip} percent tip?",
                   "split a ${bill} bill {people} ways with a {tip}% tip"],
    "local_time": ["what time is it in {city} right now?", "is it the middle of the night in {city}?", "what's the local time in {city}?"],
    "define_word": ["what does {word} mean?", "define {word}", "what's the meaning of the word {word}?"],
    "translate_phrase": ["how do you say {phrase} in {lang}?", "translate {phrase} into {lang}"],
    "stock_quote": ["what's {ticker} trading at?", "how much is a share of {company} right now?", "what's a share of {company} going for today?"],
    "game_score": ["what was the score of the {team} game?", "did the {team} win last night?", "how are the {team} doing in their game?"],
    "flight_status": ["is flight {flight} on time?", "when does {flight} land?", "has {flight} taken off yet?"],
    "track_parcel": ["where is my {carrier} package {tracking}?", "track {carrier} shipment {tracking}"],
    "dish_ideas": ["what can I cook with {ingredients}?", "find me a recipe using {ingredients}"],
    "nutrition_facts": ["how many calories are in {grams} grams of {food}?", "how much protein is in {grams}g of {food}?"],
    "public_holidays": ["what are the public holidays in {country} in {year}?", "which days are national holidays in {country} in {year}?"],
    "next_departures": ["when's the next train from {stop}?", "when does {line} next leave {stop}?"],
    "city_distance": ["how far is {city} from {city2}?", "what's the distance between {city} and {city2}?"],
    "aqi_reading": ["how's the air quality in {city} today?", "is the air bad in {city} right now?"],
    "sun_times": ["what time is sunset in {city} on {date}?", "when does the sun rise in {city} on {date}?"],
    "cinema_listings": ["when is {movie} playing in {city}?", "what are the showtimes for {movie} in {city}?"],
    "book_details": ["who wrote {book}?", "how many pages is {book}?", "when was {book} published?"],
    "crypto_price": ["what's the price of {coin} in {fiat}?", "how much is one {coin} worth in {fiat}?"],
    "loan_payment": ["what's the monthly payment on a ${principal} loan at {rate} percent for {years} years?",
                     "if I borrow ${principal} at {rate}% over {years} years, what do I pay each month?"],
}

# A related informational tool is listed but cannot serve the request.
GENERAL_NEAR_MISSES = {
    "weather_now": ["what was the weather in {city} last July?", "will it snow in {city} next month?"],
    "forecast_outlook": ["what was the temperature in {city} yesterday?", "what will the weather be like in {city} in three weeks?"],
    "fx_rate": ["how much does an iPhone cost in Japan?", "is it a good time to exchange dollars for euros?"],
    "convert_units": ["how far is {city} from {city2}?", "how heavy is a blue whale?"],
    "percent_of": ["is 97 a prime number?", "what's the square root of 144?"],
    "split_bill": ["how much should I tip a hairdresser?", "is tipping expected in Japan?"],
    "local_time": ["when does daylight saving time end?", "what time zone is {city} in?"],
    "define_word": ["translate 'good night' into Italian", "give me a synonym for happy"],
    "translate_phrase": ["what does the word {word} mean?", "how do you pronounce quinoa?"],
    "stock_quote": ["should I buy {company} stock?", "what was {company}'s revenue last year?"],
    "game_score": ["when is the next {team} game?", "who is the best player on the {team}?"],
    "flight_status": ["book me a flight to Rome", "how much is a ticket from New York to London?"],
    "track_parcel": ["where's the nearest post office?", "how much does it cost to ship a box to Canada?"],
    "dish_ideas": ["how much sugar is in a mango?", "is it safe to eat raw cookie dough?"],
    "nutrition_facts": ["how do I bake banana bread?", "what's the healthiest breakfast?"],
    "public_holidays": ["when is my mom's birthday?", "how many vacation days do people get in {country}?"],
    "next_departures": ["how much is a monthly transit pass?", "is the subway safe at night?"],
    "city_distance": ["how long is the flight from {city} to {city2}?", "which is bigger, {city} or {city2}?"],
    "aqi_reading": ["what's the pollen count in {city}?", "is the tap water safe to drink in {city}?"],
    "sun_times": ["when is the next full moon?", "what's the weather at sunset in {city}?"],
    "cinema_listings": ["is {movie} worth watching?", "who directed {movie}?"],
    "book_details": ["summarize the plot of {book}", "recommend a book like {book}"],
    "crypto_price": ["should I invest in {coin}?", "how do I buy {coin}?"],
    "loan_payment": ["what's my credit score?", "which bank has the best mortgage rates?"],
}

# A related phone tool is listed but cannot serve the request (read, change or
# delete what exists). Only trained tools appear here; open_settings is left out
# because "open the right panel" is a fair answer to most settings questions.
PHONE_NEAR_MISSES = {
    "set_alarm": ["what alarms do I have set?", "delete my 7am alarm", "turn off all my alarms",
                  "did my alarm go off this morning?", "snooze the alarm that's ringing"],
    "set_timer": ["how much time is left on my timer?", "stop the timer", "cancel my pasta timer",
                  "pause the timer for a sec"],
    "create_note": ["read me my notes", "delete the note about milk", "what did I write in my last note?",
                    "search my notes for the wifi password"],
    "set_reminder": ["what are my reminders for today?", "cancel the reminder about the dentist",
                     "did I set a reminder for mom's birthday?", "mark the laundry reminder as done"],
    "draft_email": ["read my latest email", "did Sarah reply to my email?", "how many unread emails do I have?",
                    "delete all the emails from my boss"],
    "play_music": ["what song is playing right now?", "who sings this song?", "add this song to my favorites",
                   "what are the lyrics to Yesterday?"],
}

# Action requests that none of the listed tools can perform.
UNSERVICEABLE = [
    "order me a large pepperoni pizza", "call an Uber to the airport", "book a table for two at 8 tonight",
    "send $50 to Sam on Venmo", "turn off the living room lights", "lock the front door",
    "buy more paper towels", "post this photo on Instagram", "check me in for my flight",
    "start my car", "set the thermostat to 70", "take a selfie", "call Mom",
    "order a coffee for pickup", "cancel my Netflix subscription", "pay my electricity bill",
    "reserve a hotel room in Chicago for Friday", "find me a dentist nearby", "unlock my laptop",
]


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

    Informational tools (general_tools.json) are routed when a word of the tool's
    name appears in the query, with every required argument filled with a
    placeholder: enough to exercise the --hard path offline.
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
        phone = {name for name, _ in self._MAP.values()}
        for tool in tools or []:
            fn = tool["function"]
            if fn["name"] in phone:
                continue
            if any(word in q for word in fn["name"].split("_") if len(word) >= 4):
                call = {"name": fn["name"], "arguments": self.fill_required(fn["parameters"])}
                return f"<tool_call>{json.dumps(call)}</tool_call>"
        return "I can answer that directly."  # no tool call

    @staticmethod
    def fill_required(params: dict) -> dict:
        """Placeholder values for every required argument, typed per the schema."""
        args = {}
        for key in params.get("required", []):
            spec = params["properties"][key]
            if "enum" in spec:
                args[key] = spec["enum"][0]
            elif spec.get("type") in ("number", "integer"):
                args[key] = 1
            elif spec.get("type") == "boolean":
                args[key] = True
            else:
                args[key] = "x"
        return args

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


def build_hard_seeds(info_per_tool: int, unlisted_frac: float, rng: random.Random) -> list[dict]:
    """Seeds for the hard call/no-call families (see the module docstring).

    Each seed names its ``pool``, the tools its prompt may list, so the listing
    is drawn per item after paraphrasing. A near miss carries the
    ``related_tool`` that must be listed without being able to serve it.
    """
    def fill(template: str) -> str:
        return template.format(**{s: rng.choice(v) for s, v in INFO_SLOTS.items() if "{" + s + "}" in template})

    seeds: list[dict] = []
    for tool, templates in INFO_TEMPLATES.items():
        made: set[str] = set()
        attempts = 0
        while len(made) < info_per_tool and attempts < info_per_tool * 20:
            attempts += 1
            query = fill(rng.choice(templates))
            if query in made:
                continue
            made.add(query)
            seeds.append({"kind": "info_call", "intended_tool": tool, "query": query, "pool": "info"})
            if rng.random() < unlisted_frac:
                seeds.append({"kind": "info_unlisted", "intended_tool": None, "query": query, "pool": "seen_phone"})
    for tool, queries in PHONE_NEAR_MISSES.items():
        for query in queries:
            seeds.append({"kind": "near_miss_phone", "intended_tool": None, "related_tool": tool,
                          "query": query, "pool": "seen_phone"})
    for tool, templates in GENERAL_NEAR_MISSES.items():
        for template in templates:
            seeds.append({"kind": "near_miss_general", "intended_tool": None, "related_tool": tool,
                          "query": fill(template), "pool": "seen_phone"})
    for query in UNSERVICEABLE:
        seeds.append({"kind": "unserviceable", "intended_tool": None, "query": query, "pool": "seen_phone"})
    return seeds


def hard_pools(phone_names: list[str], general_names: list[str]) -> dict[str, list[str]]:
    """Which tools each hard family may list. Held-out tools are never listed."""
    seen = [n for n in phone_names if n not in HELD_OUT]
    return {"seen_phone": seen, "info": seen + list(general_names)}


def listed_tools(all_names: list[str], intended, rng: random.Random, k_min=3, k_max=6) -> list[str]:
    """A varied subset of tools to show in this example's prompt (blocks closed-set leak)."""
    others = [n for n in all_names if n != intended]
    k = rng.randint(k_min, min(k_max, len(others)))
    names = rng.sample(others, k) + ([intended] if intended else [])
    rng.shuffle(names)
    return names


def _label(seed: dict, names: list[str], by_name: dict, teacher: Teacher) -> dict | None:
    """Have the teacher answer with ``names`` listed; keep the item only if it agrees."""
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


def label_and_validate(seed: dict, by_name: dict, teacher: Teacher, rng: random.Random) -> dict | None:
    """Have the teacher label the query, keep it only if the label is clean."""
    names = listed_tools(list(by_name), seed["intended_tool"], rng)
    return _label(seed, names, by_name, teacher)


def label_hard(seed: dict, by_name: dict, teacher: Teacher, rng: random.Random, pools: dict) -> dict | None:
    """Label a hard-family seed with a tool listing drawn from its pool.

    A near miss always lists its related tool; otherwise the same agreement rule
    as the v1 families applies. Empty no-call answers are dropped.
    """
    must = seed["intended_tool"] or seed.get("related_tool")
    names = listed_tools(pools[seed["pool"]], must, rng)
    item = _label(seed, names, by_name, teacher)
    if item is None or not item["target"]:
        return None
    item["kind"] = seed["kind"]
    if seed.get("related_tool"):
        item["related_tool"] = seed["related_tool"]
    return item


def _write(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")


def expand_with_paraphrases(seeds: list[dict], teacher: Teacher, k: int, rng: random.Random) -> list[dict]:
    """Multiply each tool seed into up to ``k`` paraphrases (keeping the original).

    The teacher rewrites the query; the intended tool carries over unchanged, as
    do any other seed fields (a hard seed's kind, pool and related tool). A
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
            out.append({**seed, "query": query})
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
    ap.add_argument("--hard", action="store_true",
                    help="add the hard call/no-call families")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the v1 families, e.g. to add hard items to an existing train.jsonl")
    ap.add_argument("--extra-tools", default=str(HERE / "general_tools.json"),
                    help="training-only informational tools used by the hard families")
    ap.add_argument("--info-per-tool", type=int, default=6,
                    help="info_call seeds per informational tool, before paraphrasing")
    ap.add_argument("--info-unlisted-frac", type=float, default=0.3,
                    help="share of info seeds that also get a tool-absent (no-call) twin")
    ap.add_argument("--hard-paraphrases", type=int, default=4,
                    help="teacher paraphrases per hard seed")
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

    all_seeds: list[dict] = []
    if not args.skip_base:
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
        all_seeds += tool_seeds + no_call_seeds

    hard_by_name, pools = by_name, {}
    if args.hard:
        general = tools_by_name(load_tools(args.extra_tools))
        hard_by_name = {**by_name, **general}
        pools = hard_pools(list(by_name), list(general))
        hard_seeds = build_hard_seeds(args.info_per_tool, args.info_unlisted_frac, rng)
        all_seeds += expand_with_paraphrases(hard_seeds, teacher, args.hard_paraphrases, rng)

    rng.shuffle(all_seeds)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stream each validated item straight to disk so a long teacher run is
    # crash-safe: a Colab disconnect leaves the items generated so far, and
    # writing to a mounted Drive path (--out-dir) makes them survive the VM.
    n_train = n_heldout = n_no_call = dropped = 0
    kinds: Counter = Counter()
    with open(out / "train.jsonl", "w", encoding="utf-8") as ftrain, \
         open(out / "heldout_eval.jsonl", "w", encoding="utf-8") as fheld:
        for i, seed in enumerate(tqdm(all_seeds, desc="labeling")):
            if seed.get("kind"):
                item = label_hard(seed, hard_by_name, teacher, rng, pools)
            else:
                item = label_and_validate(seed, by_name, teacher, rng)
            if item is None:
                dropped += 1
                continue
            item.setdefault("kind", "no_call" if item["intended_tool"] is None else "phone_call")
            item["id"] = f"gen_{i}"
            if seed["intended_tool"] in HELD_OUT:
                fheld.write(json.dumps(item) + "\n")
                n_heldout += 1
            else:
                ftrain.write(json.dumps(item) + "\n")
                n_train += 1
                kinds[item["kind"]] += 1
                if item["intended_tool"] is None:
                    n_no_call += 1
            if (i + 1) % 25 == 0:  # bound how much a crash can lose
                ftrain.flush()
                fheld.flush()

    print(f"kept {n_train + n_heldout} (dropped {dropped}) -> "
          f"train={n_train} (no_call={n_no_call}), heldout={n_heldout} in {out}")
    print(f"train kinds: {dict(kinds)}")


if __name__ == "__main__":
    main()
