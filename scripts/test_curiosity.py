#!/usr/bin/env python3
"""Test curiosity.py on synthetic frames. No camera, no motors.

    scripts/test_curiosity.py

The interesting cases are the ones that must NOT fire: a brightness change, a
view she is turning back towards, and a dead camera. A curiosity detector that
fires on those is worse than none, because it spends real driving time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
from curiosity import Curiosity, _reduce                # noqa: E402

H, W = 120, 160
rng = np.random.default_rng(7)


def room(seed: int) -> np.ndarray:
    """A scene: smooth structure plus a little noise, like a real wall."""
    r = np.random.default_rng(seed)
    base = r.normal(120, 30, size=(12, 16))
    frame = np.kron(base, np.ones((H // 12, W // 16)))
    return np.clip(frame + r.normal(0, 2, size=(H, W)), 0, 255)


def with_object(frame: np.ndarray, col: int) -> np.ndarray:
    """The same scene with something bright standing in it."""
    out = frame.copy()
    out[40:100, col:col + 28] = 245
    return out


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def main() -> int:
    ok = True
    A, B = room(1), room(2)

    print("\n1. a genuinely new scene fires")
    c = Curiosity()
    c.consider(A, now=0)
    c.consider(A, now=1)
    v = c.consider(B, now=2)
    ok &= check("novel", v.novel, f"score {v.score:.2f}")

    print("\n2. BRIGHTNESS ALONE MUST NOT FIRE (the up camera auto-exposes)")
    c = Curiosity()
    c.consider(A, now=0)
    v = c.consider(np.clip(A * 1.6 + 30, 0, 255), now=1)
    ok &= check("same room, much brighter -> not novel", not v.novel,
                f"score {v.score:.2f}")

    print("\n3. a view she has recently seen must not fire (turning back)")
    c = Curiosity()
    c.consider(A, now=0)
    c.consider(B, now=1)
    v = c.consider(A, now=30)          # past the cooldown, so only memory saves it
    ok &= check("A is still remembered -> not novel", not v.novel,
                f"score {v.score:.2f}")

    print("\n4. a dead/blank camera must not fire")
    c = Curiosity()
    c.consider(np.zeros((H, W)), now=0)
    v = c.consider(np.zeros((H, W)), now=1)
    ok &= check("blank vs blank -> not novel", not v.novel)
    v = c.consider(rng.normal(0.02, 0.001, size=(H, W)), now=2)
    ok &= check("near-flat noise -> not novel", not v.novel,
                f"score {v.score:.2f}")

    print("\n5. it turns TOWARDS the change, and only as far as configured")
    c = Curiosity()
    c.consider(A, now=0)
    right = c.consider(with_object(A, W - 40), now=100)
    c2 = Curiosity()
    c2.consider(A, now=0)
    left = c2.consider(with_object(A, 6), now=100)
    ok &= check("something on the right -> turns right",
                right.novel and right.turn_degrees > 0,
                f"{right.turn_degrees:+.1f} deg")
    ok &= check("something on the left -> turns left",
                left.novel and left.turn_degrees < 0,
                f"{left.turn_degrees:+.1f} deg")
    ok &= check("never turns further than configured",
                abs(right.turn_degrees) <= config.CURIOSITY_LOOK_MAX_DEG
                and abs(left.turn_degrees) <= config.CURIOSITY_LOOK_MAX_DEG)

    print("\n6. the cooldown stops curiosity becoming a loop")
    c = Curiosity(cooldown_s=20.0)
    c.consider(room(10), now=0)
    first = c.consider(room(11), now=1)
    second = c.consider(room(12), now=2)
    ok &= check("first look fires", first.novel)
    ok &= check("a second, seconds later, does not", not second.novel,
                second.reason)
    third = c.consider(room(13), now=100)
    ok &= check("after the cooldown it may fire again", third.novel)

    print(f"\n{'ALL PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
