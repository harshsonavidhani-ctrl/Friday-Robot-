"""Which Hill Climb Racing screen is up, from a picture (2026-10-25). Used to get
from wherever the game is to a new run without clicking an ad by mistake:
box shares of "button green" and "X white" at the spots the screens put them.
Coordinates are fractions of the Google Play Games window (side bar included),
measured on his laptop's screenshots."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _share(a: np.ndarray, box, test) -> float:
    h, w, _ = a.shape
    x0, y0, x1, y1 = box
    sub = a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(int)
    return float(test(sub[..., 0], sub[..., 1], sub[..., 2]).mean()) if sub.size else 0.0


GREEN = lambda r, g, b: (g > 110) & (g - r > 35) & (g - b > 50)               # noqa: E731
WHITE = lambda r, g, b: (r > 215) & (g > 215) & (b > 215)                      # noqa: E731
LIGHT = lambda r, g, b: (r > 185) & (g > 185) & (b > 185)                      # noqa: E731

# screen -> (box, test, minimum share); checked in this order
SIGNS = [
    ("second_chance", (0.715, 0.160, 0.750, 0.210), WHITE, 0.30),   # its X
    ("bonus", (0.670, 0.180, 0.700, 0.230), WHITE, 0.30),           # its X
    ("choose_run", (0.470, 0.640, 0.590, 0.710), GREEN, 0.30),      # Normal mode START
    ("menu", (0.730, 0.645, 0.850, 0.745), GREEN, 0.30),            # START
    ("end", (0.580, 0.630, 0.840, 0.690), LIGHT, 0.04),             # TOUCH TO CONTINUE
]
CLICK = {"second_chance": (0.731, 0.183), "bonus": (0.685, 0.205), "choose_run": (0.531, 0.675),
         "menu": (0.79, 0.69), "end": (0.71, 0.66)}


def pedals(png: bytes) -> bool:
    """The game is being DRIVEN: its grey GAS pedal is on screen (0.58-0.60 of
    the box while driving, 0.0 on menus and end screens)."""
    a = np.array(Image.open(io.BytesIO(png)).convert("RGB").resize((480, 270)))
    metal = lambda r, g, b: ((np.abs(r - g) < 14) & (np.abs(g - b) < 14)        # noqa: E731
                             & (r > 120) & (r < 235))
    return _share(a, (0.85, 0.74, 0.95, 0.95), metal) > 0.45


def which(png: bytes) -> str:
    a = np.array(Image.open(io.BytesIO(png)).convert("RGB").resize((480, 270)))
    for name, box, test, least in SIGNS:
        if _share(a, box, test) >= least:
            return name
    return "other"            # driving, an ad, loading
