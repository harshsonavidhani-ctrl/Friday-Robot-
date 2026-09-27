#!/usr/bin/env python3
"""File 15: describe_scene and continuous narration. No camera, no money.

    ./venv/bin/python scripts/test_narration.py

Every paid call is replaced by a fake that records what it was asked, so the
cases that matter can be counted: how many Tier 2 calls a still room costs
(none), whether a change is narrated once and not twice, whether anything is
said over him. All state files go to a temp dir -- the real switch, status
and scene files are never touched.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-narration-"))
config.NARRATION_FILE = TMP / "narration.json"
config.NARRATION_STATUS_FILE = TMP / "narration_status.json"
config.SCENE_JSON = TMP / "scene.json"
config.SCENE_IMAGE = TMP / "scene_latest.jpg"
config.RATE_LIMIT_FILE = TMP / "rate_limits.json"
config.LISTENING_FILE = TMP / "listening.json"
config.SPEAKING_FILE = TMP / "speaking.json"
config.ANTHROPIC_API_KEY = ""          # belt and braces: nothing may reach the API
config.DESCRIBE_KEY_FILE = TMP / "no.env"   # ...including the .env fallback

import narration                                        # noqa: E402
import offline                                          # noqa: E402
import scene_describe                                   # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


# --- a fake camera: scene.json + a JPEG, as scene.py writes them ------------

def _jpeg(block: tuple[int, int, int, int] | None = None, shade: int = 128) -> bytes:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (640, 480), (shade, shade, shade))
    if block:
        ImageDraw.Draw(image).rectangle(block, fill=(20, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


def write_scene(objects=(), block=None, shade=128, age=0.0) -> float:
    taken = time.time() - age
    config.SCENE_IMAGE.write_bytes(_jpeg(block, shade))
    config.SCENE_JSON.write_text(json.dumps({"taken_at": taken,
                                             "objects": list(objects)}))
    return taken


class FakeNarrate:
    """scene_describe.narrate, recorded. Answers from a script."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.answers: list[str] = []

    def __call__(self, jpeg, previous=None, hint=""):
        self.calls.append({"previous": previous, "hint": hint})
        answer = self.answers.pop(0) if self.answers else "I can see a wall."
        if answer == "FAIL":
            return None, "ConnectionError: no internet"
        scene_describe.last_cost_usd = 0.0015
        scene_describe.last_model = "fake-haiku"
        return answer, ""


fake = FakeNarrate()
scene_describe.narrate = fake


def fresh_watch() -> narration.NarrationWatch:
    watch = narration.NarrationWatch()
    watch._why_not_now_real = watch._why_not_now
    return watch


# ---------------------------------------------------------------------------
print("\nthe switch (step 5)")

config.NARRATION_FILE.unlink(missing_ok=True)
check("missing file reads OFF", narration.mode()["on"] is False)
config.NARRATION_FILE.write_text("{not json")
check("corrupt file reads OFF", narration.mode()["on"] is False)
m = narration.set_mode(True, by="voice")
check("on by voice, with a timeout",
      m["on"] and abs(m["until"] - time.time() - config.NARRATION_TIMEOUT_S) < 5)
since = m["since"]
time.sleep(0.01)
m2 = narration.set_mode(True, by="dashboard")
check("on again keeps the session and restarts the clock",
      m2["since"] == since and m2["until"] >= m["until"])
narration.boot_reset()
check("boot_reset turns it off", narration.mode()["on"] is False
      and narration.mode()["by"] == "boot")
narration.set_mode(True, by="voice")
later = time.time() + config.NARRATION_TIMEOUT_S + 1
check("past its timeout it reads OFF, and says it expired",
      narration.mode(later)["on"] is False and narration.mode(later)["expired"])
narration.set_mode(False, by="test")

# ---------------------------------------------------------------------------
print("\nTier 1 -- free, and quiet in a still room (step 3)")

base = narration.Snapshot.of({"taken_at": 1, "objects": []},
                             narration.thumb(_jpeg()))
same = narration.Snapshot.of({"taken_at": 2, "objects": []},
                             narration.thumb(_jpeg(shade=134)))
check("a lighting shift is not a change",
      narration.what_changed(base, same) is None,
      str(narration.what_changed(base, same)))
big = narration.Snapshot.of({"taken_at": 3, "objects": []},
                            narration.thumb(_jpeg(block=(0, 0, 320, 480))))
check("half the picture changing is a change",
      "picture changed" in (narration.what_changed(base, big) or ""))
chair = {"label": "chair", "score": 0.8, "side": "left", "bearing_deg": -20}
with_chair = narration.Snapshot.of({"taken_at": 4, "objects": [chair]},
                                   base.thumb)
check("a new label is a change",
      narration.what_changed(base, with_chair) == "new: chair (left)")
check("a label leaving is a change",
      narration.what_changed(with_chair, base) == "gone: chair")
weak = dict(chair, score=0.4)
check("a label under the score floor is not",
      narration.what_changed(base, narration.Snapshot.of(
          {"taken_at": 5, "objects": [weak]}, base.thumb)) is None)
moved = dict(chair, bearing_deg=15, side="right")
check("a label that crossed the view is a change",
      "moved: chair" in (narration.what_changed(with_chair, narration.Snapshot.of(
          {"taken_at": 6, "objects": [moved]}, base.thumb)) or ""))
nudged = dict(chair, bearing_deg=-12)
check("a label that wobbled 8 deg is not",
      narration.what_changed(with_chair, narration.Snapshot.of(
          {"taken_at": 7, "objects": [nudged]}, base.thumb)) is None)

real = sorted(Path("/tmp/claude-1000").glob("**/scratchpad/frames/*.jpg"))
if len(real) >= 3:
    thumbs = [narration.thumb(p) for p in real]
    worst = max(narration.pixel_change(thumbs[0], t) for t in thumbs[1:])
    check(f"real still-room frames from this robot ({len(real)}) never trip it",
          worst < config.NARRATION_PIXEL_CHANGE, f"worst {worst:.3f}")

# ---------------------------------------------------------------------------
print("\nTier 2 -- once per change, and only then (step 4)")

def ready_watch() -> narration.NarrationWatch:
    watch = narration.NarrationWatch()
    watch._why_not_now = lambda now: None        # the gates are tested below
    return watch

fake.calls.clear()
watch = ready_watch()
write_scene()
check("off: no call, nothing said", watch.step() is None and not fake.calls)

narration.set_mode(True, by="voice")
fake.answers = ["I can see a white wall and a suitcase on a cupboard."]
item = watch.step()
check("first look after he turns it on is spoken",
      item is not None and item["kind"] == "first"
      and fake.calls[-1]["previous"] is None, str(item))

calls = len(fake.calls)
for _ in range(5):
    write_scene(shade=130)
    watch.step()
check("a still room costs nothing: 5 polls, 0 calls",
      len(fake.calls) == calls, f"{len(fake.calls) - calls} calls")

write_scene(block=(0, 0, 320, 480))
fake.answers = ["NO_CHANGE"]
check("a change the model calls NO_CHANGE says nothing",
      watch.step() is None and len(fake.calls) == calls + 1)
check("...and passes the last thing said as context",
      fake.calls[-1]["previous"].startswith("I can see a white wall"))
write_scene(block=(0, 0, 320, 480))
watch.step()
check("...and the same view is not paid for twice",
      len(fake.calls) == calls + 1)

write_scene(objects=[chair], block=(0, 0, 320, 480))
fake.answers = ["Someone just put a chair on my left."]
item = watch.step()
check("a real change is narrated, with the Tier 1 hint",
      item is not None and item["kind"] == "change"
      and fake.calls[-1]["hint"] == "new: chair (left)", str(item))

# step 6: the line waited too long in the queue -> missed -> looked at again
watch.missed()
write_scene(objects=[chair], block=(0, 0, 320, 480))
fake.answers = ["There's a chair on my left now."]
again = watch.step()
check("a line that went stale is looked at again, not lost",
      again is not None and "chair" in again["say"]
      and fake.calls[-1]["previous"].startswith("I can see a white wall"))
watch.delivered()

# failures back off; a stale frame is not what is in view
write_scene(block=(320, 0, 640, 480))
fake.answers = ["FAIL"]
n = len(fake.calls)
check("a failed call says nothing", watch.step() is None)
watch._why_not_now = narration.NarrationWatch._why_not_now.__get__(watch)
write_scene(block=(320, 0, 640, 480))
check("...and backs off instead of retrying every poll",
      watch.step() is None and len(fake.calls) == n + 1
      and "retrying" in narration.status().get("status", ""),
      narration.status().get("status", ""))

# ---------------------------------------------------------------------------
print("\nthe executive layer (step 6)")

watch = narration.NarrationWatch()
narration.set_mode(False, by="test")
narration.set_mode(True, by="voice")
write_scene()
n = len(fake.calls)
watch.pause()
check("mid-turn: no call", watch.step() is None and len(fake.calls) == n)
watch.resume()
config.LISTENING_FILE.write_text(json.dumps({"listening": True,
                                             "at": time.time()}))
check("he is talking: no call", watch.step() is None and len(fake.calls) == n,
      narration.status().get("status", ""))
config.LISTENING_FILE.write_text(json.dumps({"listening": False,
                                             "at": time.time()}))
config.SPEAKING_FILE.write_text(json.dumps({"state": "speaking",
                                            "at": time.time()}))
check("she is speaking: no call", watch.step() is None and len(fake.calls) == n,
      narration.status().get("status", ""))
config.SPEAKING_FILE.unlink()
watch.heard_harsh()
check("he just finished a turn: a moment's grace",
      watch.step() is None and len(fake.calls) == n)
watch._last_heard = 0.0
fake.answers = ["I can see a grey wall."]
check("...then the pending first look goes ahead",
      (watch.step() or {}).get("kind") == "first" and len(fake.calls) == n + 1)

write_scene(age=config.NARRATION_FRAME_MAX_AGE_S + 5, block=(0, 0, 640, 240))
check("a stale frame (dashboard holding the camera) is never described",
      watch.step() is None and len(fake.calls) == n + 1
      and "no fresh picture" in narration.status().get("status", ""))

# ---------------------------------------------------------------------------
print("\nthe timeout (step 5)")

raw = json.loads(config.NARRATION_FILE.read_text())
raw["until"] = time.time() - 1
config.NARRATION_FILE.write_text(json.dumps(raw))
item = watch.step()
check("past the limit she says she is stopping, and it is OFF",
      item is not None and item["kind"] == "timeout"
      and narration.mode()["on"] is False
      and narration.mode()["by"] == "timeout")
check("...once", watch.step() is None)
check("the dashboard says why",
      "time limit" in narration.panel()["status"])

# ---------------------------------------------------------------------------
print("\nhis words are the only voice switch (step 5)")

for said, want in (("Friday, keep describing what you see.", True),
                   ("keep describing", True),
                   ("start narrating", True),
                   ("narration mode on", True),
                   ("Hey Friday, stop describing.", False),
                   ("stop narrating please", False),
                   ("turn off the narration", False)):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} -> narration {'on' if want else 'off'}",
          intent is not None and intent.name == "narration"
          and intent.args.get("on") is want, str(intent))
for said in ("stop describing", "stop narrating"):
    check(f"{said!r} is not a motor stop", not offline.is_stop(said))
check("'stop' is still a motor stop", offline.is_stop("stop"))
for said in ("describe what you see", "what do you see",
             "I was describing my day to her"):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} does not switch narration",
          intent is None or intent.name != "narration", str(intent))

# ---------------------------------------------------------------------------
print("\nscene_describe (no network)")

real_narrate = scene_describe._call
seen: dict = {}


def fake_call(limit, jpeg, system, text, max_tokens):
    seen.update(limit=limit, system=system, text=text)
    return seen.get("answer", "NO_CHANGE."), ""


scene_describe._call = fake_call
import importlib                                        # noqa: E402
narrate_fn = importlib.reload(scene_describe).narrate   # the real one back
scene_describe._call = fake_call
seen["answer"] = "NO_CHANGE."
check("'NO_CHANGE.' with a previous is NO_CHANGE",
      scene_describe.narrate(b"x", previous="a wall") == (scene_describe.NO_CHANGE, ""))
seen["answer"] = "No change — the room is the same."
check("'No change — ...' too",
      scene_describe.narrate(b"x", previous="a wall")[0] == scene_describe.NO_CHANGE)
seen["answer"] = "NO_CHANGE"
check("NO_CHANGE on a FIRST look is never spoken",
      scene_describe.narrate(b"x", previous=None)[0] is None)
check("narration pays from its own rate limit", seen["limit"] == "vision_narrate")
seen["answer"] = "I can read 'EXIT' on the sign."
text, _ = scene_describe.describe(b"x", focus="what's written on that")
check("describe(focus) puts his question in the prompt",
      "what's written on that" in seen["text"] and seen["limit"] == "describe_scene")
scene_describe.describe(b"x")
check("describe() without focus asks for a description",
      "Describe the view." in seen["text"])
scene_describe._call = real_narrate
check("with no API key the real call refuses instead of raising",
      importlib.reload(scene_describe)._call(
          "describe_scene", b"x", "s", "t", 10) == (None, "no API key"))
for name in ("describe_scene", "vision_narrate"):
    check(f"RATE_LIMITS has {name}", name in config.RATE_LIMITS)
check("describe_scene cooldown is the spec's 2-3 s",
      2.0 <= config.RATE_LIMITS["describe_scene"]["min_interval_s"] <= 3.0)

# ---------------------------------------------------------------------------
print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
