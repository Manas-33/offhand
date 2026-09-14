"""Quality-filter tests, no model or data needed: `python train/tests/test_filter_hard.py`.

Most cases are replies the real 4B teacher produced in the first hard-set run,
including the ones an earlier version of the rules dropped by mistake.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import filter_hard as fh


def _no_call(query, target, kind="near_miss_phone"):
    return {"kind": kind, "intended_tool": None, "query": query, "tools_listed": [], "target": target}


def test_drops_false_action_claims():
    assert fh.reasons(_no_call("Cancel my 7 a.m. alarm.", "I will cancel your 7 a.m."))
    assert fh.reasons(_no_call("snooze the alarm that's ringing", "Sure, I can help you snooze the alarm."))
    assert fh.reasons(_no_call("find the wifi password in my notes", "I can help you find the wifi password in your notes."))
    assert fh.reasons(_no_call("check off the laundry reminder", "I'll create a reminder for you to check off the laundry."))
    assert fh.reasons(_no_call("read me my notes", "I can help you read your notes."))
    # honest declines are kept, including the offer after one that points at what a listed tool does
    assert not fh.reasons(_no_call("delete my 7am alarm", "I cannot directly remove an alarm."))
    assert not fh.reasons(_no_call("what's the pollen count in Toronto?",
                                   "I cannot provide the pollen count directly, but I can check the air quality index (AQI) for Toronto.",
                                   kind="near_miss_general"))
    assert not fh.reasons(_no_call("what's the weather at sunset in Denver?",
                                   "I cannot check the weather directly, but I can help you find the sunset time in Denver.",
                                   kind="near_miss_general"))


def test_drops_made_up_tags():
    assert fh.reasons(_no_call("delete my 7am alarm", "<delete_alarm>"))


def test_drops_live_facts_but_keeps_computed_answers():
    assert fh.reasons(_no_call("is it the middle of the night in Berlin?", "It is currently 2:30 AM in Berlin.", kind="info_unlisted"))
    assert fh.reasons(_no_call("how hot is it in Cairo?", "It's currently 34°C in Cairo.", kind="info_unlisted"))
    assert not fh.reasons(_no_call("what is 40% of 240?", "40% of 240 is 96.", kind="info_unlisted"))
    assert not fh.reasons(_no_call("what's 70°F in Celsius?", "The temperature of 70°F is equivalent to 21.1°C.", kind="info_unlisted"))
    assert not fh.reasons(_no_call("exchange rate from dollars to euros?", "I cannot provide real-time exchange rates.", kind="info_unlisted"))


def test_drops_paraphraser_commentary():
    item = _no_call('First, I\'ll think of different ways to phrase "start my car"', "It seems you're exploring phrasings.", kind="unserviceable")
    assert fh.reasons(item)
    assert not fh.reasons(_no_call("start my car", "I cannot start your car.", kind="unserviceable"))


def test_tool_call_targets_are_never_read_as_text():
    call = {"kind": "info_call", "intended_tool": "fx_rate", "query": "dollars to euros rate?", "tools_listed": ["fx_rate"],
            "target": '<tool_call>{"name": "fx_rate", "arguments": {"base": "USD", "quote": "EUR"}}</tool_call>'}
    assert fh.reasons(call) == []


def test_duplicates_drop_within_a_kind_only():
    a = _no_call("what does serendipity mean?", "Serendipity means a happy accident.", kind="info_unlisted")
    b = dict(a)
    c = {**a, "kind": "info_call", "intended_tool": "define_word",
         "target": '<tool_call>{"name": "define_word", "arguments": {"word": "serendipity"}}</tool_call>'}
    kept, dropped = fh.filter_items([a, b, c])
    assert kept == [a, c] and dropped == [(b, ["duplicate question"])]


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("all filter_hard tests passed.")


if __name__ == "__main__":
    _run_all()
