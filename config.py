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

# Spoken replies are 1-3 sentences, so this is small on purpose -- it is a
# ceiling against runaway output, not a target.
MAX_REPLY_TOKENS = 2048

# Effort controls how hard the model thinks. "low" suits a companion robot
# answering conversationally; raise it if Friday starts giving shallow answers
# to things that deserve thought.
BRAIN_EFFORT = "low"

# How many rounds of tool-calling one turn may take before giving up. Enough
# for scan -> move -> scan -> speak, not enough to loop forever spending money.
MAX_TOOL_ROUNDS = 8

# Messages of history kept. Friday is a companion, not a research assistant:
# old turns cost money on every request. Trimmed from the front so the cached
# system+tools prefix survives.
BRAIN_HISTORY_MESSAGES = 20

WEB_SEARCH_MAX_USES = 3

# Cost safety — the brain must refuse to spend past this in a single day.
DAILY_SPEND_CAP_USD = 2.00
COST_LOG = LOG_DIR / "spend.jsonl"

# --- Voice pipeline ------------------------------------------------------
WAKE_WORD_MODEL = MODEL_DIR / "friday.onnx"  # openWakeWord, custom-trained
WAKE_THRESHOLD = None        # TODO(step 2): tune on the real mic, in real room
                             # noise. Unused until openWakeWord is wired in.

# Which trigger starts a turn: "enter" (press Enter), "voice" (the word
# "friday" in a short transcript), or "both".
#
# "both" is the default deliberately. The voice path is a Step 1 STOPGAP and it
# is DEAF WHILE IT TRANSCRIBES, so it drops wakes -- exactly the
# record-transcribe-check loop CLAUDE.md warns about. Keeping Enter alive
# alongside it means there is always a way in when it misses. openWakeWord
# replaces the voice half in Step 2.
WAKE_SOURCE = "both"
WAKE_WINDOW_S = 4.0          # longest single wake-listen pass
WAKE_POLL_TIMEOUT_S = 3.0    # how long a pass waits for any speech at all
STT_MODEL = "base"           # faster-whisper
STT_COMPUTE_TYPE = "int8"
STT_BEAM_SIZE = 1            # 1 is greedy: fastest, and on a Pi that matters
                             # far more than the small accuracy gain from 5
STT_LANGUAGE = None          # None = auto-detect. Harsh mixes Hindi and
                             # English, so pinning "en" would mangle "aage jao"

# Endpointing. VAD_SILENCE_MS is how long a pause has to last before Friday
# decides Harsh has finished. Too short and it interrupts him mid-sentence;
# too long and it feels slow to answer. 700 ms is a normal conversational gap.
VAD_SILENCE_MS = 700
VAD_AGGRESSIVENESS = 2       # webrtcvad 0-3; 3 is most aggressive at calling
                             # things noise, which drops quiet speech
VAD_START_FRAMES = 3         # consecutive 30 ms voiced frames before Friday
                             # believes speech started — one frame is a cough
VAD_PREROLL_MS = 300         # audio kept from BEFORE speech was detected, so
                             # the first syllable is not clipped off
MAX_UTTERANCE_S = 20.0       # hard ceiling on one utterance
LISTEN_START_TIMEOUT_S = 8.0 # give up if nobody speaks after waking
# Downloaded 2026-08-23 into assets/models/ (gitignored, fetched with
# `python -m piper.download_voices --download-dir assets/models/piper NAME`).
#
# LOW, not medium, and that is a speed decision measured on this Pi:
#   low     x1.14 realtime
#   medium  x1.60 realtime
# Synthesis is barely faster than speech itself here, so 40% matters far more
# than the quality difference does through a small robot speaker.
TTS_VOICE = MODEL_DIR / "piper" / "en_US-lessac-low.onnx"
TTS_TIMEOUT_S = 20           # a wedged audio device must not freeze the robot

# Speaking rate as a multiplier: 1.0 is the voice's natural pace, higher is
# faster. tts.py converts this to Piper's inverse length_scale.
TTS_SPEED = 1.15
TTS_VOLUME = 1.0             # per-stream software gain. Leave at 1.0 -- above
                             # that it clips rather than getting louder. If
                             # Friday is quiet, the two settings below are the
                             # ones that actually matter.

# THE SPEAKER LEVELS. Friday was barely audible because these two INDEPENDENT
# attenuations multiplied: the pipewire sink sat at 0.40 and the USB card's own
# hardware gain at 15/30 (-22.5 dB), which is ~20% together. Both are runtime
# levels that reset, so tts.set_output_volume() asserts them at start-up.
SPEAKER_SINK_VOLUME = 1.0    # pipewire sink, 0.0-1.0
SPEAKER_ALSA_CARD = "Device" # the GeneralPlus USB card, BY NAME not index.
                             # Card indices shift when a USB device drops off
                             # the bus -- the same reason MIC_DEVICE and the
                             # camera paths are not indices either. Verified:
                             # `amixer -c Device sget Speaker` works.
SPEAKER_ALSA_CONTROL = "Speaker"   # None disables the ALSA step entirely
SPEAKER_ALSA_PERCENT = 100

# --- Audio devices -------------------------------------------------------
# GeneralPlus USB soundcard (1b3f:2008) — carries BOTH the shotgun mic in and
# the speaker out.
#
# Addressed as PIPEWIRE NODES, not ALSA devices. This is not a preference:
# pipewire holds the card open exclusively while the desktop is running, so
# aplay/arecord on plughw fail with "Device or resource busy" (hit 2026-08-23).
# Going through pipewire is the fix that changes nothing on the system --
# touching the audio server itself is off-limits (CLAUDE.md Part 1).
#
# These node names come from the USB descriptor, so they are stable across
# reboots and across card-index shuffles. Both verified 2026-08-23: a tone
# played out, and a 1.8 s capture came back at peak 4170 / rms 539.
#   pw-play  --target=$SPEAKER_DEVICE file.wav
#   pw-cat --record --target=$MIC_DEVICE --rate 16000 --channels 1 --format s16
MIC_DEVICE = "alsa_input.usb-GeneralPlus_USB_Audio_Device-00.mono-fallback"
SPEAKER_DEVICE = "alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo"
MIC_SAMPLE_RATE = 16000      # what faster-whisper and openWakeWord expect

# pipewire needs this to find the user's session bus when Friday runs from a
# systemd unit rather than from a login shell.
AUDIO_RUNTIME_DIR = "/run/user/1000"

# The ALSA fallbacks, for the case where pipewire is NOT running (a headless
# boot with no desktop). Audio code should try the pipewire node first and
# fall back to these rather than assuming either one.
MIC_DEVICE_ALSA = "plughw:CARD=Device,DEV=0"
SPEAKER_DEVICE_ALSA = "plughw:CARD=Device,DEV=0"

# --- Cameras -------------------------------------------------------------
# by-id paths, not indices — indices are not stable across reboots.
CAM_C270 = "/dev/v4l/by-id/usb-046d_0825_9B5B4130-video-index0"
CAM_C270_MAX = (1280, 720)   # MJPG

# CONFIRMED 2026-08-23 by capturing a frame and looking at it: the C270 sees
# the top of the wardrobe, the curtain rail and the window -- a horizon view,
# not the floor. So the C270 is the UP cam, and the second camera is the DOWN
# cam by elimination (it was off the bus at the time, so it is inferred, not
# seen). If a stitched look() ever comes back with the floor on top, flip
# CAM_SWAP_UP_DOWN rather than editing these.
CAM_UP = CAM_C270            # forward vision, above screen
CAM_DOWN = None              # None = "the other one", resolved per access;
                             # see CAM_SECOND for why it cannot be a fixed path

# The second camera is UNSTABLE on the USB bus. Observed 2026-08-23: it
# enumerates as "HD camera" (349c:2317) -- NOT the "Generic HD camera"
# descriptor it used on 2026-08-20 -- and drops off again within a minute.
# Because the descriptor changes, its /dev/v4l/by-id/ path changes with it,
# so a hardcoded path here goes stale. None means "auto": pick whichever
# by-id capture node is not the C270, re-checked on every access, so the
# camera is found again whenever it comes back without editing this file.
# Vision code must still treat a second camera as OPTIONAL.
CAM_SECOND = None
CAM_SECOND_MAX = (640, 480)  # MJPG

# Which physical position each camera sits in is still unconfirmed, so the
# dashboard shows both and this flag swaps the labels without touching code.
# Flip it if UP CAM and DOWN CAM are the wrong way round on screen.
CAM_SWAP_UP_DOWN = False

# --- Range sensors -------------------------------------------------------
# PRESENT and confirmed 2026-08-23 21:49 at /dev/ttyUSB1 (10c4:ea60), matching
# the by-id path below exactly. It had been unplugged earlier in the session;
# sensor code must still treat it as OPTIONAL and say so when it is missing,
# because it clearly does get unplugged.
# LDROBOT D500 via CP2102. Verified streaming: LD19-family binary protocol,
# 0x54 header / 0x2C verlen / 12 points per 47-byte packet, ~9.9 Hz.
LIDAR_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIDAR_BAUD = 230400

# Which RAW LiDAR bearing points at the robot's front. NOT MEASURED YET, and
# None means exactly that: sensors.py refuses to name a direction rather than
# guessing one, so scan_environment reports left/right/backward as "unknown"
# instead of confidently wrong. The forward safety stop does not depend on
# this -- it trusts the ToF, which is bolted to the bumper facing forward.
#
# To measure: put a box ~50 cm directly in front of the robot, run
# `python sensors.py`, and note the bearing whose distance drops. Set that
# number here and every direction lights up.
LIDAR_FORWARD_DEG = None

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

# RESOLVED 2026-08-23: Harsh confirmed the FTDI FT232R (serial A5069RR4) IS
# the Arduino Uno driving the motors -- an Uno clone using an FT232R instead
# of the usual CH340. It is no longer unidentified, so the old "never open
# this" warning no longer applies. See MOTOR_PORT below.

# --- Drive ---------------------------------------------------------------
# L298N, 4-motor differential drive, wired as Harsh built it (2026-08-23).
# The Pi does NOT drive these pins -- an Arduino does, running
# firmware/friday_motors/friday_motors.ino, and the Pi talks to it over USB
# serial. So these are ARDUINO pin numbers, not BCM.
#
#   left side, 2 motors  -> channel A: reds to OUT1, blacks to OUT2
#   right side, 2 motors -> channel B: reds to OUT3, blacks to OUT4
MOTOR_PINS = {
    "ENA": 9,    # PWM, left speed
    "IN1": 8,    # left direction
    "IN2": 7,    # left direction
    "IN3": 6,    # right direction
    "IN4": 5,    # right direction
    "ENB": 3,    # PWM, right speed
}

# Serial port of the Arduino running the motor firmware.
#
# Set EXPLICITLY, and deliberately not left to auto-detect. This Uno clone
# uses an FT232R (0403:6001), which is a generic USB-serial chip that any
# number of unrelated adapters also use -- widening motor.ARDUINO_IDS to
# accept 0403:6001 would let a future FTDI cable be picked as "the motors".
# The by-id path below carries the board's unique serial number A5069RR4, so
# it identifies THIS board and no other.
#
# motor.py still refuses to drive it unless the firmware answers the "P" ping
# with FRIDAY-MOTORS, so a wrong board is caught even if this path goes stale.
MOTOR_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"

MOTOR_MAX_DUTY = 0.6         # start conservative; raise only after testing
MOTOR_WATCHDOG_S = 0.5       # stop if no new command within this window

# --- Face ----------------------------------------------------------------
# The 7" HDMI panel (EDID "Mediatrix Peripherals Inc MPI7002") reports
# 1920x1080 natively -- confirmed via wlr-randr 2026-08-23. friday_face.py
# lays itself out proportionally from the real screen size, so this is
# informational rather than something it reads.
FACE_DISPLAY_SIZE = (1920, 1080)  # 7" front display, native mode
FACE_DEFAULT_EMOTION = "neutral"
FACE_FPS = 30


# --- Dashboard -----------------------------------------------------------
# Friday's live-log web UI. Served over HTTPS so it can install as a PWA
# (a service worker needs a secure context); scripts/make_dashboard_cert.sh
# generates the self-signed cert.
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8443
DASHBOARD_HTTP_PORT = 8080       # plain HTTP, for localhost / first setup
# The local CA that signs DASHBOARD_CERT. Install it on a phone once and the
# browser stops calling the dashboard insecure. Served at /ca.crt.
DASHBOARD_CA_CERT = ROOT / "certs" / "friday-ca.crt"
DASHBOARD_CERT = ROOT / "certs" / "dashboard.crt"
DASHBOARD_KEY = ROOT / "certs" / "dashboard.key"

# Camera streams. Both cameras output MJPEG natively, so ffmpeg copies the
# stream instead of re-encoding -- that is what keeps this off the CPU.
DASHBOARD_CAM_SIZE = (640, 480)
DASHBOARD_CAM_FPS = 10
DASHBOARD_CAM_QUALITY = 6        # ffmpeg -q:v, only used if a re-encode is
                                 # unavoidable; lower is better quality

# How many recent events a newly-connected browser is replayed.
DASHBOARD_HISTORY = 300

# How long the dashboard waits on an MCP tool call before giving up. Generous
# because a move tool blocks for the whole duration of the move.
DASHBOARD_ROBOT_TIMEOUT_S = 15.0

# Manual drive presets offered in the dashboard's motor panel, in seconds.
# Short on purpose -- a tap should nudge, not launch.
DASHBOARD_MOTOR_STEPS = (0.3, 0.6, 1.0)

# Motor calibration the dashboard panel edits. Written to state/, NOT to
# this file, so config stays declarative and the values survive edits here.
CALIBRATION_FILE = ROOT / "state" / "calibration.json"
CALIBRATION_DEFAULTS = {
    "move_speed": 150,           # 0-255, default straight-line speed
    "turn_speed": 130,           # 0-255, default pivot speed
    "forward_cm_per_s": 0.0,     # measured; 0 = not calibrated yet
    "turn_deg_per_s": 0.0,       # measured; 0 = not calibrated yet
}


# --- MCP server ----------------------------------------------------------
# friday-robot: the ONE place hardware is exposed. The brain calls these tools
# and nothing else in the project talks to a device directly. That is not a
# style preference -- it is what makes the safety interrupt below impossible
# for the brain to route around.
MCP_SERVER_NAME = "friday-robot"

# Transport. The brain can either spawn the server over stdio, or connect to
# a long-running one over HTTP; friday-mcp.service runs the HTTP form, because
# a stdio server running as a daemon has nothing attached to its stdin and is
# therefore useless as a service.
#
# BOUND TO LOOPBACK ON PURPOSE, and do not change this to 0.0.0.0. These tools
# drive the motors. On 0.0.0.0 anyone on the WiFi could make the robot move,
# with no authentication in front of it.
MCP_HTTP_HOST = "127.0.0.1"
MCP_HTTP_PORT = 8770
MCP_HTTP_PATH = "/mcp"
MCP_HTTP_URL = f"http://{MCP_HTTP_HOST}:{MCP_HTTP_PORT}{MCP_HTTP_PATH}"

# DEBUG: silence Friday. speak() still logs and still returns success, it just
# does not reach the soundcard -- so tool-use flows can be tested without the
# robot talking over you. Turn OFF for real use.
DEBUG_SILENT_SPEECH = False

# Decision log: every tool call, its arguments, its result and a timestamp.
# SQLite rather than a flat file so the dashboard can query it by time or by
# tool without parsing the whole history.
DECISION_LOG = ROOT / "state" / "decisions.sqlite3"

# Where capture_image() writes. Gitignored -- these are runtime output.
CAPTURE_DIR = ROOT / "state" / "captures"
CAPTURE_KEEP = 200           # newest N kept; older ones deleted on write.
                             # The SD card is what dies on this robot, so the
                             # capture directory is never allowed to grow.
CAPTURE_QUALITY = 85         # JPEG quality for saved dual-cam stitches

# A USB webcam's first frame is taken before auto-exposure settles and comes
# back near-black. Grab this many and keep the last. Measured on the C270:
# one frame was unusable, ten is enough.
CAM_WARMUP_FRAMES = 10
CAM_TIMEOUT_S = 12           # a wedged camera must not hang a tool call

# A USB webcam is single-access, and the dashboard streams the same cameras.
# Retry briefly so a stream that is mid-start does not fail a capture; a
# PERSISTENT conflict is reported with the name of the holding process.
CAM_OPEN_ATTEMPTS = 3
CAM_OPEN_RETRY_S = 0.7

# --- Motion safety -------------------------------------------------------
# Enforced inside the MCP server, in code, on every movement -- before it
# starts and repeatedly WHILE it runs. The brain cannot disable, raise, or
# argue with any of these. A prompt can be talked around; this cannot.
SAFETY_STOP_MM = 250         # stop this far short of anything ahead
SAFETY_CAUTION_MM = 450      # below this, forward moves are capped slow
SAFETY_POLL_S = 0.1          # how often clearance is re-checked mid-move
CAUTION_SPEED_FACTOR = 0.5   # speed multiplier inside the caution band, so
                             # the stop happens with less momentum to shed
MAX_MOVE_SECONDS = 5.0       # longest single commanded move
MAX_SPEED_PERCENT = 100      # set_speed ceiling; scales onto MOTOR_MAX_DUTY,
                             # so 100% is still only MOTOR_MAX_DUTY of full PWM
DEFAULT_SPEED_PERCENT = 60

# Backing away from something is allowed closer than driving into it -- the
# down cam and bumper cannot see behind, so reverse is time-limited instead.
MAX_REVERSE_SECONDS = 2.0

# Honesty about odometry. The TT gear motors have NO encoders, so distance and
# angle are timed approximations from state/calibration.json. This is the error
# margin every distance-based tool must quote back to the brain.
ODOMETRY_ERROR_PERCENT = 20  # measured; re-measure after changing surface,
                             # battery level, or MOTOR_MAX_DUTY

# --- Status --------------------------------------------------------------
# No battery fuel gauge is wired on this robot. get_status() reports battery as
# unknown rather than inventing a percentage. If a gauge is added, put its
# device here and get_status() will start reporting it.
BATTERY_SENSOR = None
CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"
CPU_TEMP_WARN_C = 75.0


def load_calibration() -> dict:
    """Current motor calibration, falling back to the defaults for any key
    that is missing or unreadable. Motor code should call this per command
    rather than caching, so dashboard edits take effect without a restart."""
    import json
    values = dict(CALIBRATION_DEFAULTS)
    try:
        with open(CALIBRATION_FILE) as handle:
            stored = json.load(handle)
        for key in CALIBRATION_DEFAULTS:
            if key in stored and isinstance(stored[key], (int, float)):
                values[key] = stored[key]
    except (OSError, ValueError):
        pass
    return values


def missing_hardware_settings() -> list[str]:
    """Names of settings still unverified. Modules should check before use."""
    # CAM_DOWN and CAM_SECOND are deliberately None -- None means "resolve the
    # other camera on every access", which is the fix for a device whose by-id
    # path changes. They are not unverified, so they are not listed here.
    required = (
        "WAKE_THRESHOLD", "VAD_SILENCE_MS", "TTS_VOICE",
        "MIC_DEVICE", "SPEAKER_DEVICE", "CAM_UP",
        "LIDAR_PORT", "TOF_PORT", "MOTOR_PINS",
    )
    return [name for name in required if globals()[name] is None]
