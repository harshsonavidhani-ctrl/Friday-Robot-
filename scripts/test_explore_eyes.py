"""explore_eyes: what counts as seen, what she says. No camera, no sound.

    ./venv/bin/python scripts/test_explore_eyes.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                               # noqa: E402
import explore_eyes as eyes                                 # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail if not ok else ''}")
    fails += 0 if ok else 1


# The words.
check("one name", eyes.sentence(["person"]) == "Person detected.")
check("two names", eyes.sentence(["person", "chair"]) == "Person and chair detected.")
check("never more than two", eyes.sentence(["a", "b", "c"]).count("and") == 1)
check("nothing to say", eyes.sentence([]) == "")
check("end of run, one", eyes.run_sentence(["person"]) == "On this run I saw a person.")
check("end of run, an", "an umbrella" in eyes.run_sentence(["shoe", "umbrella"]))
check("end of run, list",
      eyes.run_sentence(["shoe", "chair", "bag"])
      == "On this run I saw a shoe, a chair and a bag.")
check("end of run, nothing", eyes.run_sentence([]) == "")

# ImageNet to plain speech.
check("a running shoe is a shoe", eyes.spoken_name("running shoe") == "shoe")
check("a Loafer is a shoe", eyes.spoken_name("Loafer") == "shoe")
check("first synonym only",
      eyes.spoken_name("backpack, back pack, knapsack, packsack") == "backpack")
check("an everyday thing keeps its name", eyes.spoken_name("umbrella") == "umbrella")
check("a nonsense guess is not said", eyes.spoken_name("pretzel") == "")
check("a place is not a thing", eyes.spoken_name("prison, prison house") == "")
check("a wall is not a thing", eyes.spoken_name("stone wall") == "")
check("a floor texture is not a thing", eyes.spoken_name("doormat, welcome mat") == "")

# Seen twice, or sure.
s = eyes.Sightings()
check("one weak look is not enough", s.update("up", {"chair": 0.6}, 0.0) == [])
check("two of three is", s.update("up", {"chair": 0.6}, 1.0) == ["chair"])
s = eyes.Sightings()
check("sure at the first look", s.update("up", {"person": 0.9}, 0.0) == ["person"])
s = eyes.Sightings()
s.update("up", {"chair": 0.6}, 0.0)
for t in range(1, config.EYES_CONFIRM_OF):
    s.update("up", {}, float(t))
check("a look that fell out of the window does not count",
      s.update("up", {"chair": 0.6}, 9.0) == [])
s = eyes.Sightings()
s.update("up", {"cup": 0.37}, 0.0)
s.update("up", {}, 1.0)
s.update("up", {}, 2.0)
check("a cup seen twice, seconds apart, is said",
      s.update("up", {"cup": 0.42}, 3.0) == ["cup"])
s = eyes.Sightings()
s.update("up", {"box": 0.6}, 0.0)
check("each camera confirms for itself", s.update("down", {"box": 0.6}, 1.0) == [])
s = eyes.Sightings()
check("strongest first",
      s.update("up", {"chair": 0.8, "book": 0.95}, 0.0) == ["book", "chair"])

# Once per EYES_SAY_AGAIN_S.
s = eyes.Sightings()
s.spoken(s.update("up", {"cup": 0.9}, 0.0), 0.0)
check("not said again at once", s.update("up", {"cup": 0.9}, 5.0) == [])
check("said again later",
      s.update("up", {"cup": 0.9}, config.EYES_SAY_AGAIN_S + 1) == ["cup"])
s = eyes.Sightings()
s.spoken(s.update("up", {"person": 0.9}, 0.0), 0.0)
check("a person waits much longer",
      s.update("up", {"person": 0.9}, config.EYES_SAY_AGAIN_S + 1) == []
      and s.update("up", {"person": 0.9},
                   config.EYES_SAY_AGAIN_PERSON_S + 1) == ["person"])
s = eyes.Sightings()
for t, name in ((0, "shoe"), (50, "shoe"), (60, "person")):
    s.spoken([name], float(t))
check("most seen first", s.everything() == ["shoe", "person"])


# looked_at, with stand-ins. The frame only has to be croppable.
import numpy as np                                          # noqa: E402
FRAME = np.zeros((480, 640, 3), np.uint8)


class Box:
    def __init__(self, label, score):
        self.label, self.score = label, score


class Detector:
    def __init__(self, *found):
        self.found = found

    def _infer(self, frame, min_score=None):
        return [b for b in self.found if b.score >= (min_score or 0.55)]


class Named:
    def __init__(self, name, score):
        self.answer = (name, score)

    def best(self, frame):
        return self.answer


check("a weak detection is dropped",
      eyes.looked_at("up", FRAME, Detector(Box("chair", 0.4)), None) == {})
check("a good detection is kept",
      eyes.looked_at("up", FRAME, Detector(Box("chair", 0.7)), None) == {"chair": 0.7})
check("an aeroplane indoors needs to be very sure",
      eyes.looked_at("up", FRAME, Detector(Box("aeroplane", 0.6)), None) == {})
check("the classifier adds what the detector cannot name",
      eyes.looked_at("down", FRAME, Detector(), Named("shoe", 0.4)) == {"shoe": 0.4})
check("the up camera asks the classifier for more",
      eyes.looked_at("up", FRAME, Detector(), Named("umbrella", 0.4)) == {})
check("not a thing is not said",
      eyes.looked_at("down", FRAME, Detector(), Named("", 0.9)) == {})
check("a cup passes lower than a chair",
      eyes.looked_at("down", FRAME, Detector(Box("cup", 0.4)), None) == {"cup": 0.4}
      and eyes.looked_at("down", FRAME, Detector(Box("chair", 0.4)), None) == {})
check("she says phone, not cell phone",
      eyes.looked_at("up", FRAME, Detector(Box("cell phone", 0.5)), None) == {"phone": 0.5})
check("the classifier's cup passes low too",
      eyes.looked_at("up", FRAME, Detector(), Named("cup", 0.25)) == {"cup": 0.25})
check("the close-up is the middle of the picture",
      eyes.centre_view(FRAME).shape[:2] == (360, 360))
s = eyes.Sightings()
check("a thing goes first: the person showing it is not said",
      s.update("up", {"person": 0.9, "cup": 0.8}, 0.0) == ["cup"])
s = eyes.Sightings()
s.update("up", {"person": 0.9}, 0.0)
check("alone, a person is still said",
      s.update("up", {"person": 0.9}, 1.0) == ["person"])
check("a mouse or remote guess is never said",
      eyes.looked_at("up", FRAME, Detector(Box("mouse", 0.9), Box("remote", 0.9)), None) == {})
s = eyes.Sightings()
s.update("up", {"cup": 0.5}, 0.0)
s.update("up", {"cup": 0.5, "phone": 0.45}, 1.0)
check("a cup misread as a phone stays a cup",
      s.update("up", {"phone": 0.5}, 2.0) == [] and "cup" in s.update("up", {"cup": 0.5}, 3.0))
check("a pen is a thing she can name",
      eyes.spoken_name("ballpoint, ballpoint pen") == "pen"
      and eyes.looked_at("up", FRAME, Detector(), Named("pen", 0.25)) == {"pen": 0.25})
check("the television on an empty ceiling is not said",
      eyes.looked_at("up", FRAME, Detector(Box("television", 0.58)), None) == {})
check("the classifier is never sure on one look",
      eyes.looked_at("up", FRAME, Detector(), Named("cup", 0.95))["cup"]
      < config.EYES_FOCUS_SURE)
s = eyes.Sightings()
check("one of the judges' objects is said at a strong first look",
      s.update("up", {"phone": 0.72}, 0.0) == ["phone"])
check("a dog needs to be very sure",
      eyes.looked_at("up", FRAME, Detector(Box("dog", 0.65)), None) == {})
check("she takes turns: whole picture, then close-up",
      eyes.looked_at("up", FRAME, Detector(Box("cup", 0.5)), None, 0) == {"cup": 0.5}
      and eyes.looked_at("up", FRAME, Detector(Box("cup", 0.5)), None, 1) == {"cup": 0.5})
check("the judges' objects are prerecorded first",
      eyes.lines_to_prerecord()[:3] == ["Cup detected.", "Phone detected.",
                                        "Bottle detected."])
check("pen is prerecorded", "Pen detected." in eyes.lines_to_prerecord())
check("every prerecorded line is one name, as she will say it",
      all(line.count(" and ") == 0 for line in eyes.lines_to_prerecord()))
check("cool: she looks at full rate",
      eyes.rest_for_heat(50.0) == config.EYES_PERIOD_S
      and eyes.rest_for_heat(None) == config.EYES_PERIOD_S)
check("warm: less often", eyes.rest_for_heat(config.EYES_WARM_C + 1)
      == config.EYES_WARM_PERIOD_S)
check("hot: hardly at all", eyes.rest_for_heat(84.0) == config.EYES_HOT_PERIOD_S)
check("a mug is a cup", eyes.spoken_name("coffee mug") == "cup")
check("a smartphone-shaped guess is a phone",
      eyes.spoken_name("cellular telephone, cellular phone") == "phone"
      and eyes.spoken_name("hand-held computer") == "phone")
check("no models: nothing, and no crash",
      eyes.looked_at("up", FRAME, None, None) == {})

# She must never be able to steer from here.
tree = ast.parse((ROOT / "explore_eyes.py").read_text())
names = {a.name.split(".")[0] for n in ast.walk(tree)
         if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
mods = {n.module for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module}
check("no motor, no movement", not names & {"motor", "movement", "safety"}
      and not {m for m in mods if "tools" in m})

# A family adds up: stand-in session, nothing loaded.
class Session:
    def __init__(self, logits):
        self.logits = logits

    def get_inputs(self):
        return [type("I", (), {"name": "x"})()]

    def run(self, _, __):
        return [self.logits[None]]


names = ["n"] * 1000
names[504] = "coffee mug"
names[968] = "cup"
names[441] = "beer glass"
names[600] = "hook, claw"
logits = np.full(1000, -10.0, np.float32)
logits[[504, 441, 968]] = np.log([0.15, 0.13, 0.05])
logits[600] = np.log(0.2)
got = eyes.Classifier(Session(logits), names).best(FRAME)
check("a cup split over its family still wins", got[0] == "cup" and got[1] > 0.3, str(got))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
