"""She says what she sees while she explores.

Harsh, 2026-10-30, after the hackathon's first round: the judge said she could
not name the object in front of her. "if it detects or the camera sees the thing
it should speak aloud person detected or any thing, it say its name and then
continue exploring" and "anything it should detect, not particular thing".

Why she could not, measured on the run the judge watched:
  * the run has its cameras off (MODE_CAMERAS=0: they cost 36% of a tick and
    steered nothing), so its own detector named 0 frames;
  * scene.py pauses while a run holds the robot, so nothing else looked either;
  * the up camera is tilted at the wall and ceiling: a shoe on the floor is never
    in its picture. Only the down camera sees the floor;
  * and COCO has no "shoe": SSD-MobileNet knows 80 things.

So this is a thread in friday-mcp, beside the scene watcher, that watches the
camera's live stream: all the time with EYES_ALWAYS (Harsh, later that day: "it
should be on, not take photos"), else only while a run holds the robot. NOTHING
HERE TOUCHES THE RUN: it is another process,
it reads no sensor but the cameras, it cannot steer, and the run does not wait
for it. She keeps driving while she talks. Two models, because neither is enough:
SSD finds whole objects anywhere in the picture (people above all), and a 1000
class ImageNet classifier names what fills the view (a shoe, a backpack, a box).

Frames come from the dashboard (GET /api/cam/<which>/latest.jpg), never from a
second handle on a camera: its ffmpeg owns the device, so his phone can watch
while she looks, and the down camera is not opened and closed again and again,
which is what knocks it off the USB bus.

Offline by design: no model call, no API credit, no internet.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import config

log = logging.getLogger("friday.eyes")

_thread: threading.Thread | None = None
_stop = threading.Event()

# Every class under this is an animal. ImageNet has 120 breeds of dog, and a
# robot that says "Chihuahua" at a slipper loses the room.
FIRST_THING_CLASS = 398

# What the detector is confidently wrong about indoors unless it is very sure:
# "the detector has said stop sign and aeroplane in this room" (2026-09-25).
UNLIKELY_INDOORS = {
    "aeroplane", "airplane", "train", "boat", "bus", "truck", "stop sign",
    "fire hydrant", "parking meter", "giraffe", "zebra", "elephant", "bear",
    "horse", "sheep", "cow", "kite", "surfboard", "skis", "snowboard",
    "frisbee", "baseball bat", "baseball glove", "skateboard", "tennis racket",
    "traffic light", "bird", "television", "tv", "dog", "cat",
}

# Places and textures, not things: what a classifier says about a wall, a floor
# or a ceiling when there is nothing to name. First word of each ImageNet class.
NOT_A_THING = {
    "prison", "monastery", "church", "castle", "library", "restaurant",
    "barbershop", "bookshop", "grocery store", "shoe shop", "toyshop",
    "tobacco shop", "confectionery", "butcher shop", "bakery", "cinema",
    "home theater", "dock", "patio", "palace", "mosque", "planetarium",
    "greenhouse", "boathouse", "barn", "bell cote", "yurt", "cliff dwelling",
    "stupa", "dome", "vault", "altar", "pier", "breakwater", "dam", "lakeside",
    "seashore", "valley", "volcano", "cliff", "alp", "promontory", "sandbar",
    "coral reef", "geyser", "web site", "comic book", "book jacket", "menu",
    "crossword puzzle", "jigsaw puzzle", "window shade", "window screen",
    "shower curtain", "theater curtain", "sliding door", "doormat",
    "picket fence", "chainlink fence", "worm fence", "stone wall", "prayer rug",
    "manhole cover", "maze", "tile roof", "honeycomb", "shoji", "spotlight",
    "street sign", "traffic light", "lampshade", "wall clock", "bubble",
    "steel arch bridge", "suspension bridge", "viaduct", "triumphal arch",
    "water tower", "beacon", "mobile home", "fountain", "pedestal", "plate rack",
    "shopping cart", "stage", "scoreboard", "rule", "paper towel", "tray",
    "swimming trunks", "bath towel", "dishrag", "handkerchief", "wool", "velvet",
    "jean", "mosquito net", "quilt", "pajama", "apron", "bib", "poncho",
    "fur coat", "sarong", "kimono", "abaya", "academic gown", "vestment",
    "cloak", "stole", "feather boa", "brassiere", "maillot", "bikini",
    "miniskirt", "hoopskirt", "overskirt", "gown", "military uniform",
}

# ImageNet is exact where a person is plain. Names she should say instead.
PLAIN_NAMES = {
    "running shoe": "shoe", "Loafer": "shoe", "clog": "shoe", "sandal": "sandal",
    "cowboy boot": "boot", "sock": "sock", "cellular telephone": "phone",
    "dial telephone": "phone", "pay-phone": "phone", "notebook": "laptop",
    "laptop": "laptop", "desktop computer": "computer", "monitor": "screen",
    "screen": "screen", "television": "screen", "water bottle": "bottle",
    "pop bottle": "bottle", "beer bottle": "bottle", "wine bottle": "bottle",
    "pill bottle": "bottle", "water jug": "jug", "coffee mug": "cup", "cup": "cup",
    "beer glass": "glass", "goblet": "glass", "computer keyboard": "keyboard",
    "typewriter keyboard": "keyboard",
    "studio couch": "sofa", "folding chair": "chair", "rocking chair": "chair",
    "barber chair": "chair", "throne": "chair", "dining table": "table",
    "desk": "desk", "plastic bag": "plastic bag", "purse": "bag", "mailbag": "bag",
    "backpack": "backpack", "carton": "box", "crate": "box", "packet": "packet",
    "ballpoint": "pen", "fountain pen": "pen", "sunglass": "sunglasses",
    "sunglasses": "sunglasses", "ashcan": "dustbin", "remote control": "remote",
    "wardrobe": "cupboard", "file": "cupboard", "bookcase": "bookshelf",
    "crash helmet": "helmet", "cab": "car", "iron": "iron", "wallet": "wallet",
    "digital watch": "watch", "analog clock": "clock", "ping-pong ball": "ball",
    "soccer ball": "ball", "tennis ball": "ball", "basketball": "ball",
    "volleyball": "ball", "hand-held computer": "phone", "iPod": "phone",
    "Band Aid": "bandage", "hard disc": "hard disk",
    "modem": "router", "power drill": "drill", "stretcher": "stretcher",
}


# The detector's COCO words, said the way a person says them.
DETECTOR_NAMES = {
    "cell phone": "phone", "mobile phone": "phone", "wine glass": "glass",
    "wineglass": "glass", "sports ball": "ball",
    "potted plant": "plant", "dining table": "table", "tv": "screen",
    "television": "screen", "mouse": "computer mouse", "handbag": "bag",
}
# Round 2's objects spread their probability over neighbouring classes (a paper
# cup read 0.15 "coffee mug" + 0.13 "beer glass" live), so each is scored on the
# SUM of its family and can win although no one class did.
FAMILIES = {
    "cup": ("coffee mug", "cup", "beer glass", "measuring cup"),
    "phone": ("cellular telephone", "iPod", "hand-held computer", "remote control"),
    "bottle": ("water bottle", "pop bottle", "beer bottle", "wine bottle",
               "pill bottle", "water jug"),
    "pen": ("ballpoint", "fountain pen", "quill"),
}

# Everyday things beyond PLAIN_NAMES that she may name as they are.
EVERYDAY = {
    "umbrella", "pillow", "bucket", "broom", "plunger", "teapot", "coffeepot",
    "toaster", "microwave", "refrigerator", "washer", "vacuum", "fan", "lamp",
    "candle", "hammer", "screwdriver", "scissors", "ruler", "stapler", "lipstick",
    "hair spray", "lotion", "soap dispenser", "toilet seat", "bath towel",
    "suitcase", "joystick", "printer", "projector", "radio", "speaker", "guitar",
    "violin", "drum", "balloon", "banana", "orange", "lemon", "strawberry",
    "apple", "pineapple", "bicycle-built-for-two", "tricycle", "mask", "whistle",
}


def spoken_name(raw: str) -> str:
    """The one word she says for an ImageNet class ("" = do not say it).

    A KNOWN LIST, not the 1000. Live on 2026-10-30 the classifier called the top
    of Harsh's head "pretzel" at 0.76: a thousand classes means a confident
    nonsense word for anything it has not been shown, and a robot saying "pretzel"
    at a judge is worse than one saying nothing.
    """
    first = raw.split(",")[0].strip()
    if first in NOT_A_THING:
        return ""
    if first in PLAIN_NAMES:
        return PLAIN_NAMES[first]
    return first.lower() if first.lower() in EVERYDAY else ""


def sentence(labels: list[str]) -> str:
    """["person"] -> "Person detected." ; two -> "Person and chair detected."."""
    if not labels:
        return ""
    joined = " and ".join(labels[:2])
    return f"{joined[0].upper()}{joined[1:]} detected."


def run_sentence(labels: list[str]) -> str:
    """What she says once at the end of a run."""
    if not labels:
        return ""
    named = [f"{'an' if w[0] in 'aeiou' else 'a'} {w}" for w in labels]
    if len(named) == 1:
        return f"On this run I saw {named[0]}."
    return f"On this run I saw {', '.join(named[:-1])} and {named[-1]}."


class Sightings:
    """What is real, and what she has said. Pure logic, no I/O.

    A label counts when it is in EYES_CONFIRM of the last EYES_CONFIRM_OF looks
    through that camera, or at once at EYES_SURE_SCORE: a moving robot sees the
    same thing twice only if it is there. She says one name once per
    EYES_SAY_AGAIN_S, or the wake word is deaf all run (friday-deaf-while-she-
    narrates) and a chair in a hall of chairs becomes a chant.
    """

    def __init__(self) -> None:
        self.looks: dict[str, deque] = {}
        self.said: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, which: str, seen: dict[str, float], now: float) -> list[str]:
        looks = self.looks.setdefault(
            which, deque(maxlen=int(config.EYES_CONFIRM_OF)))
        looks.append(dict(seen))
        evidence: dict[str, float] = {}
        for look in looks:
            for label, score in look.items():
                evidence[label] = evidence.get(label, 0.0) + score
        # The judges' objects compete: live, a black paper cup also read "mobile
        # phone" 0.48-0.63. Only the one with the most evidence in these looks
        # may be named, so one object cannot be called two things in turn.
        rivals = [k for k in evidence if k in config.EYES_FOCUS]
        leader = max(rivals, key=evidence.get) if rivals else None
        # ...and it has to win clearly: a blurry phone that flips cup 0.73, phone
        # 0.55 is a thing she cannot read, so she says nothing rather than guess.
        if leader is not None and any(
                k != leader and evidence[k] * config.EYES_LEAD > evidence[leader]
                for k in rivals):
            leader = None
        fresh = []
        for label, score in sorted(seen.items(), key=lambda kv: -kv[1]):
            if label in config.EYES_FOCUS and label != leader:
                continue
            sure = (config.EYES_FOCUS_SURE if label in config.EYES_FOCUS
                    else config.EYES_SURE_SCORE)
            if (score < sure
                    and sum(label in look for look in looks) < config.EYES_CONFIRM):
                continue
            again = (config.EYES_SAY_AGAIN_PERSON_S if label == "person"
                     else config.EYES_SAY_AGAIN_S)
            if now - self.said.get(label, -1e12) < again:
                continue
            fresh.append(label)
        if any(label != "person" for label in fresh):
            fresh = [label for label in fresh if label != "person"]
        return fresh

    def spoken(self, labels: list[str], now: float) -> None:
        for label in labels:
            self.said[label] = now
            self.counts[label] = self.counts.get(label, 0) + 1

    def everything(self) -> list[str]:
        """Most often seen first."""
        return sorted(self.counts, key=lambda k: -self.counts[k])


class Classifier:
    """MobileNetV2, 1000 ImageNet classes, ~60 ms a look on one thread."""

    SIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, session, names: list[str]) -> None:
        self.session, self.names = session, names
        self.input = session.get_inputs()[0].name
        first = {n.split(",")[0].strip(): i for i, n in enumerate(names)}
        self.family_ids = {name: [first[c] for c in classes if c in first]
                           for name, classes in FAMILIES.items()}

    @classmethod
    def load(cls) -> "Classifier | None":
        model, labels = Path(config.EYES_CLASSIFIER), Path(config.EYES_CLASSIFIER_LABELS)
        if not model.exists() or not labels.exists():
            log.warning("eyes: no classifier at %s: naming with the detector only", model)
            return None
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(config.EYES_THREADS)
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), options,
                                       providers=["CPUExecutionProvider"])
        names = [line.split(" ", 1)[1].strip()
                 for line in labels.read_text().splitlines() if " " in line]
        return cls(session, names)

    def best(self, frame) -> tuple[str, float]:
        """The top class of a BGR frame: (spoken name or "", confidence)."""
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        side = min(h, w)
        x, y = (w - side) // 2, (h - side) // 2
        square = cv2.resize(frame[y:y + side, x:x + side], (self.SIZE, self.SIZE))
        rgb = square[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.MEAN, np.float32)) / np.array(self.STD, np.float32)
        logits = self.session.run(None, {self.input: rgb.transpose(2, 0, 1)[None]})[0][0]
        logits = logits - logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        family, weight = "", 0.0
        for name, classes in FAMILIES.items():
            total = sum(float(probs[i]) for i in self.family_ids.get(name, ()))
            if total > weight:
                family, weight = name, total
        top = int(probs.argmax())
        if weight >= float(probs[top]) * config.EYES_FAMILY_EDGE:
            return family, weight
        if top < FIRST_THING_CLASS or top >= len(self.names):
            return "", float(probs[top])
        return spoken_name(self.names[top]), float(probs[top])


def centre_view(frame):
    """What is right in front of her, close up. A cup at 80 cm is 30 pixels of a
    640 frame and 15 of the detector's 320: too small to name. Cut from the full
    picture it is twice that, for the price of one more look. The middle of the
    picture, not the bottom: live, what he held up was never in the bottom."""
    h, w = frame.shape[:2]
    side = int(min(h, w) * config.EYES_CLOSE_CROP)
    x, y = (w - side) // 2, (h - side) // 2
    return frame[y:y + side, x:x + side]


def looked_at(which: str, frame, detector, classifier,
              turn: int | None = None) -> dict[str, float]:
    """Everything worth naming in one frame: {label: score}.

    EYES_FOCUS is what the judges will put in front of her (Harsh, 2026-10-30:
    "cup phone bottle"): those pass at a lower score, because a small object on
    the floor seen on the move never scores like a person does. Being seen in 2
    of 3 looks still has to hold, which is what keeps that honest.
    `turn`: 0 = the whole picture only, 1 = the close-up only, None = both. The
    loop takes turns, so one detector pass (~0.4 s) is one look and a thing is
    named sooner than by doing both every time.
    """
    seen: dict[str, float] = {}
    close = centre_view(frame)

    def keep(label: str, score: float) -> None:
        seen[label] = max(seen.get(label, 0.0), float(score))

    if detector is not None:
        views = (frame, close) if turn is None else ((frame, close)[turn],)
        for view in views:
            for d in detector._infer(view, min_score=config.EYES_FOCUS_SCORE):
                if d.label in config.EYES_NEVER_SAY:
                    continue
                name = DETECTOR_NAMES.get(d.label, d.label)
                if name in config.EYES_FOCUS:
                    keep(name, d.score)
                elif d.score < config.EYES_MIN_SCORE:
                    continue
                elif d.label in UNLIKELY_INDOORS and d.score < config.EYES_SURE_SCORE:
                    continue
                else:
                    keep(name, d.score)
    if classifier is not None:
        name, score = classifier.best(close)
        floor = (config.EYES_CLASS_MIN_DOWN if which == "down"
                 else config.EYES_CLASS_MIN_UP)
        if name in config.EYES_FOCUS:
            floor = config.EYES_FOCUS_CLASS_SCORE
        if name and score >= floor:
            # Never EYES_SURE_SCORE: its guess must be seen twice, however keen.
            keep(name, min(score, config.EYES_FOCUS_SURE - 0.01))
    if config.EYES_ONLY_FOCUS:
        # A wrong name in front of a judge costs more than a missed one: live it
        # said "Refrigerator" at a glass door and "Vase" at nothing.
        seen = {k: v for k, v in seen.items()
                if k in config.EYES_FOCUS or k == "person"}
    return seen


def _frame(which: str):
    """The newest frame from the dashboard's stream, or None."""
    import cv2
    import numpy as np
    url = f"{config.DASHBOARD_LOCAL_URL}/api/cam/{which}/latest.jpg"
    try:
        with urllib.request.urlopen(url, timeout=config.EYES_FRAME_TIMEOUT_S) as reply:
            data = reply.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _exploring() -> bool:
    from mcp_server import hardware
    return hardware.lease_active() > 0


def _publish_scene(frame, seen: dict[str, float]) -> None:
    """scene.py stands down while EYES_ALWAYS, so what it fed the rest of her
    (situation, describe, "what do you see") keeps coming from here."""
    import cv2
    import scene
    ok, jpeg = cv2.imencode(".jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, config.SCENE_IMAGE_QUALITY])
    if not ok:
        return
    scene._atomic_write(Path(config.SCENE_IMAGE), jpeg.tobytes())
    scene._atomic_write(Path(config.SCENE_JSON), json.dumps({
        "taken_at": time.time(), "camera": "up",
        "objects": [{"label": label, "score": round(score, 2), "side": "ahead",
                     "bearing_deg": 0, "area": 0.0} for label, score in seen.items()],
        "capture_s": 0.0, "detect_ms": None, "image": str(config.SCENE_IMAGE),
    }).encode())


def _publish(sights: Sightings, last: list[str]) -> None:
    """state/explore_eyes.json, for the dashboard and for looking back."""
    try:
        path = Path(config.EYES_STATE_FILE)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({"at": time.time(), "last": last,
                                   "seen": sights.counts}))
        os.replace(tmp, path)
    except OSError as exc:
        log.debug("eyes: could not save sightings: %s", exc)


def _speaker(lines: queue.Queue) -> None:
    """One thread says everything, so a look never waits on Piper. A line that
    has waited too long is dropped: she has driven past that thing by now."""
    import tts
    while not _stop.is_set():
        try:
            said_at, text = lines.get(timeout=1.0)
        except queue.Empty:
            continue
        if time.time() - said_at >= config.EYES_SAY_STALE_S:
            log.info("eyes: dropped, too late: %s", text)
            continue
        started = time.time()
        try:
            tts.speak(text)
        except Exception as exc:                # noqa: BLE001
            log.warning("eyes: could not speak: %s", exc)
        log.info("eyes: %r waited %.2f s, spoken in %.2f s", text,
                 started - said_at, time.time() - started)


def _say(lines: queue.Queue, text: str) -> None:
    log.info("eyes: %s", text)
    try:
        lines.put_nowait((time.time(), text))
    except queue.Full:
        return
    try:
        import dashboard_log
        dashboard_log.emit({"type": "text", "text": f"👁 **{text}**"})
    except Exception as exc:                    # noqa: BLE001
        log.debug("eyes: no dashboard card: %s", exc)


def _end_of_run(sights: Sightings, lines: queue.Queue) -> None:
    labels = sights.everything()[:int(config.EYES_RUN_NAMES)]
    if not labels:
        return
    words = run_sentence(labels)
    _say(lines, words)
    try:
        import memory
        memory.remember(words, kind="exploration_sighting", source="explore_eyes")
    except Exception as exc:                    # noqa: BLE001
        log.warning("eyes: could not remember the run: %s", exc)


def rest_for_heat(celsius: float | None) -> float:
    """Seconds to wait before the next look. The chassis is glued shut: on
    2026-10-30 looking nonstop took the Pi from 43 to 84 C in 18 minutes and it
    throttled, which slows the control loop and the voice too. Naming things is
    worth less than the run, so eyes back off long before the Pi has to."""
    if celsius is None or celsius < config.EYES_WARM_C:
        return float(config.EYES_PERIOD_S)
    if celsius < config.EYES_HOT_C:
        return float(config.EYES_WARM_PERIOD_S)
    return float(config.EYES_HOT_PERIOD_S)


def _cpu_celsius() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def lines_to_prerecord() -> list[str]:
    """Every name she is likely to say, the judges' objects first. Rendered at
    start, a line plays at once. Live, Piper took 8.7 s for the first line and
    1.4 s each after it, on top of the look: "its speaking late"."""
    names = list(config.EYES_FOCUS) + ["person"] + [
        "chair", "table", "sofa", "bed", "laptop", "screen", "keyboard", "bag",
        "backpack", "suitcase", "umbrella", "clock", "vase", "scissors", "plant",
        "bowl", "fork", "knife", "spoon", "banana", "apple", "orange", "ball",
        "tie", "toothbrush", "sink", "microwave", "bench", "bicycle"]
    return [sentence([name]) for name in dict.fromkeys(names)]


def _run() -> None:
    lines: queue.Queue = queue.Queue(maxsize=2)
    # THE VOICE IS STARTED BEFORE THE PRIORITY DROP BELOW: a thread takes its nice
    # from the thread that starts it and cannot take it back, and at nice 10 every
    # line was queueing behind the detector.
    threading.Thread(target=_speaker, args=(lines,), name="friday-eyes-voice",
                     daemon=True).start()
    try:
        import tts
        tts.prerender(lines_to_prerecord())
    except Exception as exc:                    # noqa: BLE001
        log.warning("eyes: could not prerecord: %s", exc)

    # Below the run and everything else on this Pi: being late to name a chair
    # costs nothing, being late to a control tick costs a bump.
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), config.EYES_NICE)
    except OSError as exc:
        log.warning("eyes: could not lower my priority: %s", exc)

    # BOTH MODELS LOAD HERE, AT SERVER START, never while she drives: creating a
    # session holds the GIL for seconds (friday-model-loads-freeze-the-run).
    import scene
    detector = scene.shared_detector(wait_s=float(config.EYES_DETECTOR_WAIT_S))
    classifier = None
    try:
        classifier = Classifier.load()
    except Exception as exc:                    # noqa: BLE001
        log.warning("eyes: classifier OFF (%s: %s)", type(exc).__name__, exc)
    log.info("eyes ready: detector %s, classifier %s",
             "on" if detector is not None else "OFF",
             "on" if classifier is not None else "OFF")

    sights = Sightings()
    in_run = False
    scene_at = -1e12
    turn = 0
    cooling = False
    while not _stop.is_set():
        exploring = _exploring()
        if exploring and not in_run:
            sights, in_run = Sightings(), True
            log.info("eyes: a run started")
        elif in_run and not exploring:
            _end_of_run(sights, lines)
            sights, in_run = Sightings(), False
        if not (exploring or config.EYES_ALWAYS):
            _stop.wait(float(config.EYES_IDLE_POLL_S))
            continue
        for which in ("up", "down"):
            started = time.monotonic()
            try:
                frame = _frame(which)
                if frame is None:
                    continue
                turn = 1 - turn
                seen = looked_at(which, frame, detector, classifier, turn)
                now = time.monotonic()
                if seen:
                    log.info("eyes: %s sees %s", which,
                             {k: round(v, 2) for k, v in seen.items()})
                if which == "up" and now - scene_at >= config.SCENE_PERIOD_S:
                    scene_at = now
                    _publish_scene(frame, seen)
                fresh = sights.update(which, seen, now)[:1]
                if fresh:
                    sights.spoken(fresh, now)
                    _say(lines, sentence(fresh))
                    _publish(sights, fresh)
            except Exception as exc:            # noqa: BLE001
                log.warning("eyes: look failed: %s: %s", type(exc).__name__, exc)
            hot = _cpu_celsius()
            rest = rest_for_heat(hot)
            if rest >= config.EYES_HOT_PERIOD_S and not cooling:
                log.warning("eyes: %.0f C, looking every %.0f s until she cools",
                            hot, rest)
            cooling = rest >= config.EYES_HOT_PERIOD_S
            _stop.wait(rest)


def start() -> None:
    """Begin watching for runs. Idempotent; a no-op when EYES is False."""
    global _thread
    if not config.EYES:
        log.info("eyes off (EYES = False)")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="friday-eyes", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
