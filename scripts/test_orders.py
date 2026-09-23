#!/usr/bin/env python3
"""test_orders.py — what may act on what she heard, and what may not.

    ./venv/bin/python scripts/test_orders.py

WHY THIS FILE EXISTS. On 2026-09-19 a video playing in the room woke Friday
337 times in two hours, and the reflex path acted on what it heard: "F right
A, speak Hindi." pivoted her ninety degrees, "See you next time." got the
clock read out, "Now go back to it." drove her backwards. None of those is a
wake-word problem that a threshold can fix -- they are the reflex table
matching a keyword inside somebody else's sentence.

The rule the fixes put back is offline._looks_like_an_order: in STRICT mode
(the path taken when a brain IS available) an order has to look like one --
the verb in front, or nothing in the sentence but the order itself. Refusing
costs a round trip to something that can read the sentence properly. Acting
costs a pivot nobody asked for.

Every KEEP line below is a real order Harsh gives. Every REFUSE line is
something a television actually said in this room, or a sentence built the
same way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain                                                # noqa: E402
import offline                                              # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"

KEEP = [
    ("forward.", "go_forward"), ("back", "go_backward"),
    ("left", "left"), ("right", "right"),
    ("Now move forward", "go_forward"), ("Go backwards.", "go_backward"),
    ("turn left", "left"), ("turn around", "around"),
    ("go left two meters", "go_left"), ("go forward two seconds", "go_forward"),
    ("a bit to the right", "right"), ("move back a little", "go_backward"),
    ("hey friday, turn right", "right"),
    ("chalo aage", "go_forward"), ("peeche chalo", "go_backward"),
    ("thoda sa aage chalo", "go_forward"),
    ("what time is it", "time"), ("what's the time", "time"),
    ("tell me the time", "time"), ("time?", "time"), ("Friday, time?", "time"),
    ("what's the weather", "weather"), ("how's the weather", "weather"),
    ("weather?", "weather"), ("is it raining", "weather"),
    ("mausam kaisa hai", "weather"),
    # Every one of these reached the BRAIN before, which answered from the
    # situation block and put no card on the screen -- 2026-09-20 00:47.
    ("I have a headache, tell me the weather.", "weather"),
    ("tell me the weather", "weather"), ("show me the weather", "weather"),
    ("friday whats the weather like", "weather"),
    ("weather animation", "weather"), ("how hot is it", "weather"),
    ("mausam batao", "weather"),
    ("explore", "explore"), ("start exploring", "explore"),
    ("explore the room", "explore"),
    ("Friday, explore for twenty minutes", "explore"),
    # File 14 navigation, 2026-09-23: never a brain turn.
    ("Friday, explore for 2 minutes and come back", "explore_return"),
    ("explore and come back", "explore_return"),
    ("come back", "nav_come_back"), ("wapas aao", "nav_come_back"),
    ("go home", "nav_home"), ("ghar chalo", "nav_home"),
    ("try again", "try_again"), ("phir se try karo", "try_again"),
    ("remember this as home", "remember_place"),
    ("Friday, remember this place as the kitchen", "remember_place"),
    ("save this spot as my desk", "remember_place"),
    ("this place is the kitchen", "remember_place"),
    ("go to the garage", "nav_go"),     # unknown: the TOOL refuses plainly
]

OFFLINE_NAV = [
    ("Friday, go to the kitchen", "nav_go"),
    ("go to the desk and come back", "nav_go"),
    ("go home", "nav_home"), ("ghar chalo", "nav_home"),
    ("come back", "nav_come_back"), ("wapas aao", "nav_come_back"),
    ("try again", "try_again"), ("phir se try karo", "try_again"),
    ("explore for 2 minutes and come back", "explore_return"),
    ("remember this as home", "remember_place"),
    ("yeh jagah kitchen hai", "remember_place"),
    ("stop", "stop"), ("go to sleep", "sleep"),
]

REFUSE = [
    "F right A, speak Hindi.",          # pivoted her 90 degrees, 22:27
    "Now go back to it.",               # drove her backwards, 22:56
    "See you next time.",               # read her the clock, 22:50 and 22:55
    "See you next time!",
    "come on.",
    "Let's go!",
    "Thank you.",
    "You ready?",
    "subscribe, can see that they are both also.",
    "I'm not going to do this. I'm going to take it.",
    "We are going to talk about the story, but we are going",
    "I can't wait to see what happens.",
    "You are not in the same place.",
    "one more time",
    "it's time to go",
    "that's the one on the left",
    "I think we should go right now",
    "we explore the caves later",
    "the weather is nice in Spain",
    "we talked about the weather for hours",
    "whether or not you come",
    "they will explore the island in the next episode",
    # Navigation starts journeys that last minutes, so these must not.
    "you should come back tomorrow",
    "don't come back",
    "I will try again later if it doesn't work out today",
    "remember that my exam is on Monday",
    "remember this, the exam is on Monday",
    "I have to go to the library later",
]

# Stop is never refused, on any path, at any length -- see _ORDERS_ONLY.
STOPS = ["stop", "Friday stop!", "stop it", "ruko", "stop please"]

# THE OTHER END OF THE SAME JUDGEMENT: once the brain has said "not said to
# me", brain.asked_of_her decides whether the conversation window overrides
# it. Every line here was really said in this room on 2026-09-19/20 -- his in
# ASKED, the television in NOT_ASKED -- and the transcriber mangled his as
# badly as it mangled the video's, which is the whole difficulty.
ASKED = [
    "describe about more about the thing you saw, like the people you saw.",
    "Tell me, see and tell me what the people are doing.",
    "So, how do you know me like what do you like what's it?",
    "what do you see", "show me the map", "friday, look at this",
    "can you describe the room", "batao kya dikh raha hai",
    "see what's in front of you",
]
NOT_ASKED = [
    "I'm gonna take the second one, we're gonna... I'm going to take the first one.",
    "I don't care about these guys, I can't get them.",
    "We must go to the history of the society.",
    "you'll be listening fast as the screen will be",
    "Thank you.", "See you next time.", "See ya.", "Come on.", "Let's go!",
    "Take care.",
]

# HOW HE CHANGES THE WAY SHE BEHAVES — never dropped, window or no window,
# because File 09's learned preferences have no other way in. Measured
# 2026-09-20: all three of the first three came back IGNORE and the live facts
# table had not one preference row in it.
INSTRUCTIONS = [
    "talk shorter", "speak in hindi from now on",
    "don't describe the room every time", "be more brief",
    "stop calling me that", "answer in one line", "use simple words",
    "call me Harsh", "never say that again", "from now on speak in hindi",
    "friday, talk less", "don't repeat yourself",
    # The long way round -- how he actually said it, 2026-09-20 00:49, and it
    # was IGNORED as television.
    "When I ask you to tell me the weather you should always show the "
    "animation for a minimum.",
    "you should always show the card", "I want you to speak slower",
    "make sure you look before you move", "remember to call me Harsh",
    "you must not say that", "you need to be quicker",
]
NOT_INSTRUCTIONS = [
    "come on.", "let's go!", "take care.", "see you next time.",
    "go and get it", "call the police", "thank you.", "stop",
    "We must go to the history of the society.", "what do you see",
    "you should have seen his face", "we should go now",
]

# HER OWN NOTES, READ OUT AS IF THEY WERE THE ANSWER. Measured live
# 2026-09-20 04:24 on "tell me a bit about yourself" — the situation block
# and the reflection note narrated back at him, sentence by sentence.
HER_NOTES = [
    "This is a straightforward question about myself, asked at 4:24 am on a "
    "Sunday when I last spoke 5 minutes ago.",
    "The context shows I'm sleepy, my last exploration failed, and I've been "
    "reflecting on being more honest about uncertainty.",
    "This is genuine small talk directed at me — he's asking me to tell him "
    "about myself.",
    "The notes say he wants shorter replies.",
]
REAL_ANSWERS = [
    "I'm Friday, a wheeled robot built by Harsh.",
    "It's 4:24 in the morning.",
    "Tokyo.",
    "I have cameras, LiDAR and depth sensors.",
    "We last spoke about the weather.",
    "I'm not sure — I'd have to ask Harsh.",
    "That's a good question about my sensors.",
]

# What the brain returns when it means "this was not for me". The written-out
# one was SPOKEN out loud on 2026-09-19 23:39 before is_ignore knew it.
IGNORES = [
    "IGNORE", "IGNORE.", "ignore",
    "This sounds like a sentence from a video or conversation I overheard, "
    "not something said to me — there's no question in it.",
    "I don't think that was meant for me.",
    "That was not addressed to me, so I stayed quiet.",
]
NOT_IGNORES = [
    "Tokyo.", "you can ignore that one", "I'm doing well, thanks for asking.",
    "He said to me that the rain would stop.",
    "I overheard the news earlier and it said rain.",
]


def main() -> int:
    bad = 0
    print("orders that must still skip the brain")
    for text, name in KEEP:
        intent = offline.understand(text, strict=True)
        ok = intent is not None and intent.name == name
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text!r} -> "
              f"{intent.name if intent else None} (wanted {name})")

    # WITH NO INTERNET main.py takes the loose branch (strict=False). File 14
    # navigation has to work there exactly as it does online -- Harsh,
    # 2026-09-23: "it will be all offline".
    print("\nnavigation with no internet at all (the loose branch)")
    for text, name in OFFLINE_NAV:
        intent = offline.understand(text)
        ok = intent is not None and intent.name == name
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text!r} -> "
              f"{intent.name if intent else None} (wanted {name})")

    print("\nsentences that must reach the brain instead")
    for text in REFUSE:
        intent = offline.understand(text, strict=True)
        bad += intent is not None
        print(f"  {PASS if intent is None else FAIL}  {text!r} -> "
              f"{intent.name if intent else None}")

    print("\nstop is never refused")
    for text in STOPS:
        ok = offline.is_stop(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text!r}")

    print("\nwhat the conversation window rescues from silence")
    for text in ASKED:
        ok = brain.asked_of_her(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text[:62]!r}")
    for text in NOT_ASKED:
        ok = not brain.asked_of_her(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  (silent) {text[:52]!r}")

    print("\ninstructions about her behaviour are never dropped")
    for text in INSTRUCTIONS:
        ok = brain.is_instruction(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text!r}")
    for text in NOT_INSTRUCTIONS:
        ok = not brain.is_instruction(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  (not one) {text!r}")

    print("\nher own notes are never said out loud")
    for text in HER_NOTES:
        ok = brain._is_meta(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  (silent) {text[:56]!r}")
    for text in REAL_ANSWERS:
        ok = not brain._is_meta(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text[:56]!r}")

    print("\n\"not said to me\", however the brain phrases it")
    for text in IGNORES:
        ok = brain.is_ignore(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  {text[:62]!r}")
    for text in NOT_IGNORES:
        ok = not brain.is_ignore(text)
        bad += not ok
        print(f"  {PASS if ok else FAIL}  (spoken) {text[:52]!r}")

    print(f"\n{'all good' if not bad else str(bad) + ' WRONG'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
