"""Single source of truth for Friday's configuration.

Rules:
  - Every key, device index, device name, threshold, and model name lives HERE.
    No hard-coded values scattered across modules.
  - Hardware values start as None on purpose. Do NOT guess an index or a port.
    Run scripts/check_hardware.sh on the real robot, then fill in what you saw.
  - Secrets come from the environment, never from this file.
"""

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
ASSET_DIR = ROOT / "assets"
MODEL_DIR = ASSET_DIR / "models"

# --- Secrets (from env only) ---------------------------------------------
# Unset while BUILDING Friday (Claude Code runs on the subscription).
# Set while RUNNING Friday (the brain calls the API).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- Brain ---------------------------------------------------------------
MODEL_MAIN = "claude-sonnet-5"              # reasoning, new situations
MODEL_FAST = "claude-haiku-4-5-20251001"    # reflexes, simple classification
SYSTEM_PROMPT_FILE = ROOT / "personality.md"
MAX_REPLY_SENTENCES = 3
ENABLE_WEB_SEARCH = True
ENABLE_PROMPT_CACHING = True

# Cost safety — the brain must refuse to spend past this in a single day.
DAILY_SPEND_CAP_USD = 2.00
COST_LOG = LOG_DIR / "spend.jsonl"

# --- Voice pipeline ------------------------------------------------------
WAKE_WORD_MODEL = MODEL_DIR / "friday.onnx"  # openWakeWord, custom-trained
WAKE_THRESHOLD = None        # TODO: tune on the real mic, in real room noise
STT_MODEL = "base"           # faster-whisper
STT_COMPUTE_TYPE = "int8"
VAD_SILENCE_MS = None        # TODO: tune — how much silence ends an utterance
TTS_VOICE = None             # TODO: choose a Piper voice, then pin it here

# --- Audio devices -------------------------------------------------------
# TODO: fill from `arecord -l` / `aplay -l` on the real Pi. Never guess.
MIC_DEVICE = None            # USB shotgun mic (top)
SPEAKER_DEVICE = None        # via USB soundcard

# --- Cameras -------------------------------------------------------------
# TODO: fill from `ls /dev/video*`. Indices are NOT stable across reboots —
# prefer a /dev/v4l/by-id/ path once verified.
CAM_UP = None                # forward vision, above screen
CAM_DOWN = None              # floor / cliff / optical flow, above bumper

# --- Range sensors -------------------------------------------------------
LIDAR_PORT = None            # TODO: LDROBOT D500, from `ls /dev/serial/by-id/`
TOF_I2C_ADDR = None          # TODO: DFRobot 8x8 ToF, from `i2cdetect -y 1`

# --- Drive ---------------------------------------------------------------
# L298N, 4-motor differential drive. Pins are BCM numbering.
# TODO: fill from the actual wiring before ANY motor command is sent.
MOTOR_PINS = None
MOTOR_MAX_DUTY = 0.6         # start conservative; raise only after testing
MOTOR_WATCHDOG_S = 0.5       # stop if no new command within this window

# --- Face ----------------------------------------------------------------
FACE_DISPLAY_SIZE = (800, 480)   # 7" front display
FACE_DEFAULT_EMOTION = "neutral"
FACE_FPS = 30


def missing_hardware_settings() -> list[str]:
    """Names of settings still unverified. Modules should check before use."""
    required = (
        "WAKE_THRESHOLD", "VAD_SILENCE_MS", "TTS_VOICE",
        "MIC_DEVICE", "SPEAKER_DEVICE", "CAM_UP", "CAM_DOWN",
        "LIDAR_PORT", "TOF_I2C_ADDR", "MOTOR_PINS",
    )
    return [name for name in required if globals()[name] is None]
