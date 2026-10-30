"""What the camera is actually LOOKING AT, as named objects with bearings.

Harsh, 2026-09-04, prepping for a competition: "cameras will give idea about
things like it will in object detection as well as navigation make like
proffesional robots".

WHAT THIS IS, AND WHAT IT IS DELIBERATELY NOT
=============================================
This is PERCEPTION, not control. It names things and says roughly where they
are; nothing in it touches the wheels, and it is not wired into the arc search,
the governor or the reflex. That is a deliberate line and it is worth stating
plainly, because the obvious next thought is "so let her steer around the
chair":

  * It runs at about 1 Hz against a 10 Hz control loop. A steering input that
    is 900 ms stale is worse than none.
  * A detector box gives an ANGLE, not a RANGE. Turning a box into a distance
    needs the camera height and tilt, and this robot's are estimated rather
    than measured (see friday_freespace.CAM_HEIGHT_M).
  * The LiDAR, the ToF and the free-space profile already answer "is there
    something there". They answer it at 10 Hz with real geometry. This answers
    a question none of them can -- WHAT it is -- and that answer belongs in the
    narration and the log, where a person reads it.

So: obstacle avoidance stays on the sensors that measure distance, and this
gives Friday something to say about the room she is driving through.

THE MODEL
=========
SSD-MobileNet v1, COCO, from the ONNX model zoo. Chosen by measurement on this
Pi, not by reputation:

    320x320   355-375 ms per frame     2 onnxruntime threads
    300x300   467-524 ms

so it runs on its own thread at ~1 Hz for about 40% of one core, and
onnxruntime releases the GIL for the duration, which is why it can sit beside a
control loop that must not miss its 100 ms deadline. Anything YOLO-shaped was
several times slower on this hardware.

The model takes uint8 NHWC directly -- no normalisation, no float conversion --
which is why the camera thread's raw BGR frame can be handed to it almost as-is.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("friday.vision")

try:
    import cv2
except ImportError:                                       # pragma: no cover
    cv2 = None

# The square the frame is letterboxed into. See the module note for the timings
# that chose it: 320 is both faster AND the model's better size here.
INPUT_SIZE = 320
# Below this the detector is guessing. COCO SSD is confidently wrong at 0.3 in
# cluttered indoor scenes -- it will call a doorway a refrigerator -- and a robot
# announcing furniture that is not there is worse than one saying nothing.
MIN_SCORE = 0.55
# How many to keep per frame. The narrator can say one thing at a time and the
# log wants the gist, not an inventory.
MAX_OBJECTS = 5
# Horizontal field of view of the up camera (Logitech C270, measured spec 60 deg
# diagonal -> ~52 deg horizontal at 4:3). Used ONLY to turn a box centre into a
# bearing for narration ("on your left"). Nothing steers on it.
CAM_HFOV_DEG = 52.0

# COCO 90-class ids as the TF/ONNX SSD models emit them: 1-based, with gaps.
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "aeroplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "sofa", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "television", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "mobile phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


class Detection:
    """One named thing, with where it is in the picture."""

    __slots__ = ("label", "score", "bearing_deg", "area", "box")

    def __init__(self, label: str, score: float, bearing_deg: float,
                 area: float, box) -> None:
        self.label, self.score = label, score
        self.bearing_deg = bearing_deg      # + is LEFT, matching omega's sign
        self.area = area                    # fraction of frame: a crude "how near"
        self.box = box

    def side(self) -> str:
        if self.bearing_deg > 12.0:
            return "on my left"
        if self.bearing_deg < -12.0:
            return "on my right"
        return "ahead"

    def __repr__(self) -> str:
        return f"<{self.label} {self.score:.2f} {self.bearing_deg:+.0f}deg>"


class ObjectDetector(threading.Thread):
    """Names what the up camera sees, on its own thread, at about 1 Hz.

    NEVER RAISES INTO THE CONTROL LOOP and never blocks it. A missing model file,
    a missing camera or a bad frame all degrade to "I currently see nothing",
    which is exactly the state the rest of the system already handles: this is
    advisory narration, and its absence must cost nothing at all.
    """

    def __init__(self, model_path: Path, slot, period_s: float = 1.0,
                 threads: int = 2) -> None:
        super().__init__(name="vision", daemon=True)
        self.model_path, self.slot, self.period = Path(model_path), slot, period_s
        self.threads = threads
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._session = None
        self._input = None
        self.objects: list[Detection] = []
        self.frames = 0
        self.failures = 0
        self.last_ms = 0.0
        self.mean_ms = 0.0
        self.enabled = False

    # ---- lifecycle ---------------------------------------------------------------
    def _load(self) -> bool:
        if cv2 is None:
            LOGGER.warning("Object detection OFF: cv2 is not importable.")
            return False
        if not self.model_path.exists():
            LOGGER.warning("Object detection OFF: no model at %s.", self.model_path)
            return False
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            # Two threads, not all four. The control loop, the mapper and the
            # frontier planner are all on this Pi and the loop has a 100 ms
            # deadline it has historically missed; a detector that wins the CPU
            # fight has broken the robot to describe it.
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), so, providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0].name
        except Exception as exc:                          # noqa: BLE001
            LOGGER.warning("Object detection OFF: %s: %s", type(exc).__name__, exc)
            return False
        LOGGER.info("Object detection ON: %s at %dx%d, ~%.1f Hz on %d threads",
                    self.model_path.name, INPUT_SIZE, INPUT_SIZE,
                    1.0 / self.period, self.threads)
        return True

    def stop(self) -> None:
        self._stop.set()

    # ---- the work ----------------------------------------------------------------
    def _infer(self, bgr, min_score: float | None = None) -> list[Detection]:
        floor = MIN_SCORE if min_score is None else min_score
        h, w = bgr.shape[:2]
        # RGB, and letterboxed rather than stretched: SSD's aspect assumptions are
        # baked into its anchors, and squashing a 4:3 frame into a square moves
        # every box.
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        ox, oy = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh),
                                                    interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        boxes, classes, scores, _ = self._session.run(
            None, {self._input: canvas[None, ...]})
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        # Running mean, so one slow frame under load does not read as the cost.
        self.frames += 1
        self.mean_ms += (self.last_ms - self.mean_ms) / self.frames

        out: list[Detection] = []
        boxes, classes, scores = boxes[0], classes[0], scores[0]
        for i in range(len(scores)):
            s = float(scores[i])
            if s < floor:
                break                       # SSD emits these already sorted
            label = COCO_LABELS.get(int(classes[i]))
            if label is None:
                continue
            # ymin, xmin, ymax, xmax, normalised to the LETTERBOXED square.
            ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
            cx = ((xmin + xmax) * 0.5 * INPUT_SIZE - ox) / max(nw, 1)
            area = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))
            # +cx is right in image space, and Friday's omega is + to the LEFT.
            bearing = -(cx - 0.5) * CAM_HFOV_DEG
            out.append(Detection(label, s, bearing, area,
                                 (xmin, ymin, xmax, ymax)))
            if len(out) >= MAX_OBJECTS:
                break
        return out

    def run(self) -> None:
        self.enabled = self._load()
        if not self.enabled:
            return
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                frame, age = self.slot.read()
                if frame is not None and age < 2.0:
                    found = self._infer(frame)
                    with self._lock:
                        self.objects = found
            except Exception as exc:                      # noqa: BLE001
                self.failures += 1
                if self.failures in (1, 10) or self.failures % 100 == 0:
                    LOGGER.warning("Object detection frame failed (%d): %s: %s",
                                   self.failures, type(exc).__name__, exc)
            # Pace by the period, but never busy-spin if inference overran it.
            time.sleep(max(0.05, self.period - (time.monotonic() - start)))

    # ---- what everyone else reads ------------------------------------------------
    def snapshot(self) -> list[Detection]:
        with self._lock:
            return list(self.objects)

    def describe(self) -> str:
        """One short sentence, or "" when there is nothing worth saying."""
        objs = self.snapshot()
        if not objs:
            return ""
        # Biggest first: in a picture with no depth, the thing filling the most
        # frame is the best available proxy for the thing she is closest to.
        objs.sort(key=lambda d: -d.area)
        head = objs[0]
        if len(objs) == 1:
            return f"I can see a {head.label} {head.side()}."
        others = ", ".join(dict.fromkeys(o.label for o in objs[1:]))
        return f"I can see a {head.label} {head.side()}, and also {others}."
