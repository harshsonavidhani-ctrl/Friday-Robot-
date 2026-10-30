"""A person's left is her left; the map stack's left stays the map stack's.

Measured 2026-10-30 (camera and LiDAR): the drive's "left" turns her physically
right, and rl/ + slam/ read the LiDAR mirrored too, so those two cancel. The MCP
tools translate for people; the plain functions must NOT. No motors move here.

    ./venv/bin/python scripts/test_turn_words.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                               # noqa: E402
from mcp_server import embodiment, safety                   # noqa: E402
from mcp_server.tools import movement                       # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail if not ok else ''}")
    fails += 0 if ok else 1


drove: list = []


class Moved:
    def __init__(self, direction):
        self.direction = direction

    def as_text(self):
        return f"{self.direction} for 0.5s"


def fake_move(direction, seconds, **_):
    drove.append(direction)
    return Moved(direction)


safety.guarded_move = fake_move
embodiment.check = lambda direction: None
movement._gyro_turn = lambda degrees: f"Turned {'left' if degrees < 0 else 'right'} (gyro)."

check("the switch is on", config.TURN_WORDS_MIRRORED is True)

said = movement.move_tool("left", 0.5)
check("a person's left drives the drive's right", drove[-1] == "right", str(drove))
check("...and the answer says left", said == "left for 0.5s", said)
said = movement.move_tool("right", 0.5)
check("a person's right drives the drive's left", drove[-1] == "left", str(drove))
check("...and the answer says right", said == "right for 0.5s", said)

for word in ("forward", "backward"):
    said = movement.move_tool(word, 0.5)
    check(f"{word} is not touched", drove[-1] == word and said.startswith(word), said)

said = movement.turn_degrees_tool(-30)
check("turn_degrees: -30 is a person's left", said == "Turned left (gyro).", said)
said = movement.turn_degrees_tool(30)
check("turn_degrees: +30 is a person's right", said == "Turned right (gyro).", said)

# The plain functions are the map stack's door: nav_follower, the mission and
# home-facing call them, and they must keep meaning what those maps mean.
movement.move("left", 0.5)
check("movement.move() is unchanged", drove[-1] == "left", str(drove))
check("movement.turn_degrees() is unchanged",
      movement.turn_degrees(-30) == "Turned left (gyro).")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
