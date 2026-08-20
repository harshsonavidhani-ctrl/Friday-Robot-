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
# GeneralPlus USB soundcard (1b3f:2008) — carries BOTH the shotgun mic in and
# the speaker out. Addressed by CARD NAME, not index: card indices shift when
# a USB device drops off the bus (verified 2026-08-20, both directions tested).
MIC_DEVICE = "plughw:CARD=Device,DEV=0"
SPEAKER_DEVICE = "plughw:CARD=Device,DEV=0"
MIC_SAMPLE_RATE = 16000      # what faster-whisper and openWakeWord expect

# --- Cameras -------------------------------------------------------------
# by-id paths, not indices — indices are not stable across reboots.
CAM_C270 = "/dev/v4l/by-id/usb-046d_0825_9B5B4130-video-index0"
CAM_C270_MAX = (1280, 720)   # MJPG

# TODO: which physical position is the C270 in? Assign once confirmed.
CAM_UP = None                # forward vision, above screen
CAM_DOWN = None              # floor / cliff / optical flow, above bumper

# The second camera ("Generic HD", 349c:2317) dropped off the USB bus on
# 2026-08-20 and did not re-enumerate. Vision code must treat a second camera
# as OPTIONAL and degrade cleanly when it is absent.
CAM_SECOND = None

# --- Range sensors -------------------------------------------------------
# LDROBOT D500 via CP2102. Verified streaming: LD19-family binary protocol,
# 0x54 header / 0x2C verlen / 12 points per 47-byte packet, ~9.9 Hz.
LIDAR_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIDAR_BAUD = 230400

# DFRobot 8x8 ToF, bridged through a Raspberry Pi Pico over USB CDC.
# NOT on I2C — /dev/i2c-1 is disabled and enabling it is off-limits (see
# CLAUDE.md Part 1). The Pico makes that unnecessary.
# Protocol: ASCII lines "yN:d0,d1,...,d7," for N in 0..7, one 8x8 frame
# per 8 lines, distances in mm, ~15 fps. 4000 = no return / out of range.
TOF_PORT = "/dev/serial/by-id/usb-Raspberry_Pi_Pico_E66554A14B3F7423-if00"
TOF_BAUD = 115200
TOF_NO_RETURN_MM = 4000
TOF_ROWS = 8
TOF_COLS = 8

# Unidentified: an FTDI FT232R (serial A5069RR4) sits on /dev/ttyUSB1.
# Do NOT open it until Harsh confirms what it drives — if it reaches the
# motors, probing it could move the robot.
FTDI_PORT_UNKNOWN = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"

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
        "LIDAR_PORT", "TOF_PORT", "MOTOR_PINS",
    )
    return [name for name in required if globals()[name] is None]
