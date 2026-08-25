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

# Friday's self-development (mcp_server/tools/dev.py). A dev_task runs Claude
# Code headless, which costs real money and takes real time, so both are
# bounded. The estimate is deliberately rough and deliberately high: the point
# of the cap is to stop a runaway loop, not to do accounting.
DEV_TASK_TIMEOUT_S = 600     # a change that takes longer than 10 minutes has
                             # gone wrong; it is stopped and the branch dropped
DEV_TASK_EST_USD = 0.25      # charged against the daily cap per task
COST_LOG = LOG_DIR / "spend.jsonl"

# --- Voice pipeline ------------------------------------------------------
# --- Wake word (openWakeWord) --------------------------------------------
# openWakeWord runs a small classifier over a CONTINUOUS 16 kHz stream, so it
# is never deaf -- unlike the Step 1 record->transcribe->check stopgap it
# replaced, which spent seconds inside whisper and dropped anything said during
# that gap. Measured on this Pi: 17 ms per 80 ms chunk, ~21% of one core.
#
# ⚠ WAKE_WORD_MODEL IS A STAND-IN. friday.onnx does not exist yet, so this
# points at openWakeWord's bundled "hey jarvis" model. Everything else about
# the pipeline is final -- when the custom model lands, change THIS LINE ONLY.
WAKE_WORD_MODEL = (ROOT / "venv/lib/python3.13/site-packages/openwakeword"
                   / "resources/models/hey_jarvis_v0.1.onnx")

# The score above which a frame counts as a wake, 0-1. Higher = fewer false
# triggers but more misses. Tune against the real room, not in the abstract:
#   ./venv/bin/python wake_word.py --tune
WAKE_THRESHOLD = 0.5

# How long after a wake before another one can fire. Stops one "hey Friday"
# from registering two or three times as the score stays high across frames.
WAKE_COOLDOWN_S = 2.0

# openWakeWord's own built-in Silero VAD gate, 0-1, or 0 to disable. A frame
# whose speech probability is below this is never scored, which is what keeps
# music, fans and door noise from ever reaching the classifier.
WAKE_VAD_THRESHOLD = 0.5

# Audio kept from before the wake fired and handed to the VAD capture as
# pre-roll. Covers "hey Friday move forward" said in one breath -- without it
# the capture starts mid-word and the transcript is wrong.
WAKE_HANDOVER_MS = 400

WAKE_BEEP = True             # short tone on wake, so Harsh knows it heard him
                             # without having to look at the face
WAKE_BEEP_HZ = 880
WAKE_BEEP_MS = 90

# Which trigger starts a turn: "voice" (openWakeWord), "enter" (press Enter),
# or "both". "both" keeps a keyboard way in while the wake model is still the
# stand-in -- it costs nothing and there is always a way to talk to Friday.
WAKE_SOURCE = "both"

# --- Microphone stream ---------------------------------------------------
# ONE pw-cat process feeds every consumer (see stt.AudioStream). Two readers on
# one USB card is the single-access trap the ToF, LiDAR and cameras have each
# already sprung, and re-opening the device at wake time would burn latency at
# the worst possible moment.
MIC_CAPTURE_BUFFER_S = 3.0   # how much recent audio the capture subscription
                             # holds, so a wake can reach back for pre-roll
MIC_RESTART_BACKOFF_S = 0.5  # first retry delay if the mic stream dies
MIC_RESTART_BACKOFF_MAX_S = 10.0
# faster-whisper. TINY, not base — measured on this Pi 4 against 1.78 s of
# speech, WITH Friday's own services running (see the CPU note below):
#     base    threads=4   x13.3 realtime   (23.6 s — unusable)
#     tiny    threads=4   x5.3  realtime
#     tiny.en threads=4   x4.4  realtime   (English only)
# tiny.en is fastest but monolingual, and Harsh mixes Hindi and English, so
# "aage jao" needs the multilingual model.
#
# ⚠ HINDI IS UNVERIFIED WITH REAL SPEECH. In a loopback test tiny heard
# "What is the capital of Japan?" perfectly but turned "aage jao" into
# "odds chill" -- however that test fed it PIPER'S ENGLISH VOICE attempting
# Hindi, which sounds nothing like Harsh saying it. Test with his actual voice
# before concluding anything. If Hindi really is poor, "base" is the next step
# up and costs about 2x the latency.
STT_MODEL = "tiny"
STT_COMPUTE_TYPE = "int8"
# Explicit: ctranslate2's default did not use all four cores (x16.5 vs x13.3).
STT_CPU_THREADS = 4
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
# hfc_female, chosen by ear 2026-08-24 against amy / kathleen / jenny /
# kristin / ljspeech: American, female, warm and friendly, which is the voice
# Harsh wants Friday to have. lessac (the old default) was rejected.
#
# This IS a medium model, against the earlier speed finding that low is x1.14
# realtime and medium x1.60. It is affordable here because speaking starts on
# the FIRST sentence while the rest is still being made, so synthesis only has
# to beat speech, not finish ahead of it -- and hfc_female measured the fastest
# of the medium voices tried (6.9s to load and render a full sentence).
# If Friday ever starts stuttering mid-reply, this is the first thing to
# suspect, and en_US-amy-low is the fallback.
TTS_VOICE = MODEL_DIR / "piper" / "en_US-hfc_female-medium.onnx"
TTS_TIMEOUT_S = 20           # a wedged audio device must not freeze the robot

# Speaking rate as a multiplier: 1.0 is the voice's natural pace, higher is
# faster. tts.py converts this to Piper's inverse length_scale.
#
# BELOW 1.0 DELIBERATELY. This was 1.15 -- chosen when the goal was getting an
# answer out quickly -- and Harsh found it too fast to listen to. A companion
# that has to be replayed in your head is not faster in any way that counts.
TTS_SPEED = 0.95

# LOUDNESS. Both hardware stages below are already pinned at maximum, so there
# is no knob left -- and Piper's output already touches full scale on its
# loudest sample while sitting at only ~-15 dBFS RMS. It is peaky, not loud, so
# plain gain would clip the peaks and gain nothing. tts.louden() instead
# applies this gain and soft-limits what goes over, which raises PERCEIVED
# loudness. Set TTS_GAIN_DB to 0 to disable the stage entirely.
# TONE. hfc_female is warm but sits light, and a light voice through a small
# robot speaker -- which has almost no output down low -- reads as thin and
# less clear. Two stages fix that, both in tts.py:
#   TTS_PITCH  lowers pitch by resampling (Piper has no pitch control).
#              1.0 is unchanged, lower is heavier. Below ~0.85 it stops
#              sounding like a lower voice and starts sounding like slowed
#              tape. The speed loss is compensated at synthesis time.
#   TTS_BODY_* a low shelf that puts the chest back in the voice.
#   TTS_AIR_*  a small high shelf; slightly negative keeps the added weight
#              from reading as harsh.
TTS_PITCH = 1.03             # SLIGHTLY UP. This ran 0.95 -> 0.91 -> 0.88 while
                             # the complaint was that Friday sounded thin, and
                             # it overshot: on 2026-08-25 Harsh said she sounds
                             # "manly" and asked for cute and warm. Pitching a
                             # female voice ~2.2 semitones DOWN is precisely how
                             # you get manly, so the whole chain is reversed
                             # here -- up a little, no chest boost, air added
                             # back. The speed loss from resampling is
                             # compensated in _synthesis_config, so this does
                             # NOT change her pace; TTS_SPEED alone does.
TTS_BODY_DB = 0.0            # was +3.0. The low shelf was there to add chest,
                             # which is the opposite of what is wanted now.
TTS_BODY_HZ = 320
TTS_AIR_DB = 2.0             # was -1.0. Air is the brightness that reads as
                             # light and friendly; cutting it was part of what
                             # made her sound heavy.
TTS_AIR_HZ = 4500

# ⚠ GAIN IS DELIBERATELY MODEST. +8 dB was tried and rejected as piercing.
# The hardware is already at maximum underneath this, so this is the only
# loudness control there is -- but loudness past the point where it sounds bad
# is not loudness worth having.
TTS_GAIN_DB = 6.5

# The compressor in tts.louden(). It is an ENVELOPE compressor, not a
# waveshaper: an earlier tanh soft-knee version added harmonics and Harsh heard
# it as buzzing. These control how gently it moves.
TTS_CEILING = 0.93           # never let output exceed this, so nothing clips
TTS_LOOKAHEAD_MS = 5.0       # see peaks coming and duck before they land
TTS_RELEASE_MS = 90.0        # how slowly gain returns. Too fast pumps and
                             # buzzes; too slow and one loud word makes the
                             # next few quiet.

TTS_VOLUME = 1.0             # per-stream software gain. Leave at 1.0 -- above
                             # that it clips rather than getting louder. If
                             # Friday is quiet, the two settings below are the
                             # ones that actually matter.

# THE SPEAKER LEVELS. Friday was barely audible because these two INDEPENDENT
# attenuations multiplied: the pipewire sink sat at 0.40 and the USB card's own
# hardware gain at 15/30 (-22.5 dB), which is ~20% together. Both are runtime
# levels that reset, so tts.set_output_volume() asserts them at start-up.
# ⚠ PLAYBACK BUFFER. Harsh heard the speaker "buzzing, like something bursts".
# It was not the signal -- every preset measures 0.0% clipped. It is BUFFER
# UNDERRUN. The sink runs an ALSA period of 64 frames (~1.3 ms) while this Pi
# sits at load average 4.5 on 4 cores with the face alone eating ~96% of one,
# at 79 degC with the throttle bits already set. A 1.3 ms buffer on a CPU that
# starved cannot be refilled in time, and each miss is a discontinuity in the
# waveform -- which is exactly what a burst or a buzz sounds like.
#
# pw-play's --latency asks for a bigger buffer FOR FRIDAY'S OWN STREAM. It is a
# per-stream request, so nothing about the audio server or any other client
# changes: no pipewire config is edited, no device property is touched. Raise
# it if the bursting is still there; the cost is only that speech starts this
# much later, which at 200 ms is not perceptible in conversation.
# ⚠⚠ THE BUZZING IS A USB PROBLEM, AND THE REAL FIX IS PHYSICAL ⚠⚠
#
# Diagnosed 2026-08-25 with scripts/audio_doctor.py. Harsh heard buzzing on
# EVERY test tone equally, including a -30 dBFS sine. That single fact rules
# out almost everything software can cause: a -30 dBFS sine cannot overdrive an
# analog stage, it is not loud enough to clip, the sweep would have buzzed in
# one band only if it were mechanical resonance, and the 22050 vs 48000 test
# would have separated a resampler fault. If a perfectly quiet perfect sine
# buzzes, the waveform is being corrupted in TRANSPORT, before the driver.
#
# The topology says why (lsusb -t):
#
#   hub Dev 006 (4 ports)
#     |__ port 3: the GeneralPlus USB audio card   12 Mbps  (full speed)
#     |__ port 4: the Logitech C270 webcam        480 Mbps  (high speed)
#
# The two share one hub, and that camera had been reset TWENTY-ONE times
# ("reset high-speed USB device number 10 using xhci_hcd"). USB audio is
# ISOCHRONOUS: it needs a guaranteed slot every millisecond and has no
# retransmission. A camera storming the hub delays those slots, and every
# missed slot is a discontinuity in the waveform -- which is what the buzz is.
# The audio card itself was seen disconnecting once too (1-1.1.2.3).
#
# THE FIX IS TO MOVE THE AUDIO CARD ONTO A DIFFERENT HUB FROM THE CAMERAS,
# ideally straight into a Pi port. Nothing in this file can repair a bus that
# is dropping packets; the settings below only make Friday more tolerant of it.
# Stopping friday-dashboard.service (which streams the cameras) is the quick
# way to confirm: the resets stop immediately when the camera is released.
AUDIO_PLAY_LATENCY = "200ms"

# Piper renders at 22050 Hz and the sink runs at 48000, so every playback was
# resampling in real time on a CPU that has none to spare. tts.py now resamples
# ONCE while shaping -- off the realtime path, with scipy, which is already
# imported for the tone stages -- so pw-play gets audio at the rate the device
# already wants and has nothing left to do but hand it over.
AUDIO_SINK_RATE = 48000

# ⚠ BOTH OF THESE WERE AT MAXIMUM AND THAT IS WHAT BUZZES. Harsh confirmed on
# 2026-08-25 that the speaker did not do this before the levels were raised.
# That is the decisive clue, and it explains the one thing the USB theory could
# not: why even a -30 dBFS sine buzzed. A quiet signal cannot overdrive an amp
# -- but an amp already pinned at maximum gain distorts and hisses on
# EVERYTHING it is handed, quiet material included. The card is a cheap
# GeneralPlus part and its output stage is simply not clean at full tilt.
#
# So the analog gain comes down and the loudness is made up digitally, where
# there is a real limiter (tts.louden) and measurable headroom. This is the
# normal way round: set the analog stage where it sounds clean, then drive it.
# If Friday is too quiet now, raise TTS_GAIN_DB first, not these.
SPEAKER_SINK_VOLUME = 0.95   # pipewire sink, 0.0-1.0. Was 1.0, then 0.75
                             # which Harsh found too quiet. Still short of
                             # the maximum that was buzzing.
SPEAKER_ALSA_CARD = "Device" # the GeneralPlus USB card, BY NAME not index.
                             # Card indices shift when a USB device drops off
                             # the bus -- the same reason MIC_DEVICE and the
                             # camera paths are not indices either. Verified:
                             # `amixer -c Device sget Speaker` works.
SPEAKER_ALSA_CONTROL = "Speaker"   # None disables the ALSA step entirely
SPEAKER_ALSA_PERCENT = 85    # was 100 (30/30, its ceiling) which buzzed,
                             # then 70 which was too quiet. The output stage is
                             # the thing that distorts, so it stays off its
                             # ceiling -- loudness is made up digitally below,
                             # where there is a real limiter.

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

# ⚠ THE RIGHT MOTOR IS WIRED BACKWARDS. Measured on the real robot
# 2026-08-24, wheels off the ground: commanded forward, the left wheel turned
# forward and the right wheel turned backward. That one fault made every
# dashboard button wrong in a way that looked baffling until it was written
# out -- "right" drove forward, "left" drove backward, and forward/backward
# pivoted, because (+,-) and (-,+) become (+,+) and (-,-) when one side flips.
#
# Fixed in software rather than by rewiring: the chassis is glued shut and
# opening it risks destroying it. motor.drive() applies this at the last point
# before the serial write, so every caller above is corrected at once.
#
# If the motor leads are ever physically corrected, set this back to False.
MOTOR_INVERT_LEFT = False
MOTOR_INVERT_RIGHT = True

# KICKSTART. From rest the motors need more torque to start than to keep
# going; below that threshold they sit and buzz without moving, which is worst
# when pivoting because all four wheels scrub sideways at once. A move that
# starts from a standstill gets a brief pulse at MOTOR_KICKSTART_FLOOR of the
# duty cap, then settles to the requested speed. Set the duration to 0 to
# disable. This borrows headroom already permitted by MOTOR_MAX_DUTY -- it
# cannot exceed it.
MOTOR_KICKSTART_MS = 130
MOTOR_KICKSTART_FLOOR = 1.0  # fraction of the duty cap to pulse at

# The hard ceiling on motor output, as a fraction of full PWM. EVERY path is
# capped by this in motor._clamp, so it is the real limit on how fast Friday
# can drive regardless of what calibration or set_speed ask for.
#
# Raised 0.6 -> 0.75 on 2026-08-25 with Harsh's explicit approval, because
# turning was weak and 0.6 capped output at 153/255 -- which meant turn_speed
# above ~180 did nothing at all. Deliberately short of full power: the L298N
# heats and the battery sags under load, and a brownout mid-drive could reset
# the Pi. If turning is ever still too weak, ASK before going higher.
MOTOR_MAX_DUTY = 0.75
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
    # Lives HERE, in the file, rather than in a module global -- it used to be
    # one, and every restart of friday-mcp silently threw away whatever speed
    # Harsh had set from the dashboard. Anything the dashboard can change has
    # to outlive the process that changes it.
    "speed_percent": 60,         # 1-MAX_SPEED_PERCENT, trims the two above
    # Per-side balance: multiplies the commanded PWM for that side alone, to
    # correct a physical mismatch (worn brushes, gearing) between the two
    # motors that no amount of equal-speed commands can fix. 1.0 = no
    # correction. Tune by driving straight and nudging the slower side up.
    "trim_left": 1.0,
    "trim_right": 1.0,
}


# --- Offline fallback ----------------------------------------------------
# With no internet there is no brain, but there must still be a robot. offline.py
# handles forward/back/left/right/stop/time locally so a dropped connection
# costs Friday its conversation, not its ability to be told what to do.
#
# A TCP connect, not a ping and not an HTTP request: ICMP is often filtered and
# an HTTP request costs a round trip and possibly money. Cached, because it is
# asked once per turn.
ONLINE_CHECK_HOST = "api.anthropic.com"
ONLINE_CHECK_PORT = 443
ONLINE_CHECK_TIMEOUT_S = 2.0
ONLINE_CHECK_TTL_S = 20.0

# --- Face reactions ------------------------------------------------------
# face_control.py writes an emotion and friday_face.py tweens between them
# (TRANSITION_S = 0.28) and blinks on its own. What face_director.py adds is
# WHEN: which of several competing requests actually reaches the screen.
FACE_ENABLED = True

# Minimum time an expression is held before anything of the same or lower
# priority may replace it. Without this a fast turn -- listening, thinking,
# happy, speaking inside two seconds -- reads as a flicker, not a sequence.
# Safety expressions are exempt and are never made to wait.
FACE_MIN_DWELL_S = 0.55

# How long a layer-2 emotion lasts before easing back to neutral. An emotion
# is a moment, not a state; without an expiry the face stays stuck on the last
# thing that happened.
FACE_EMOTION_HOLD_S = 4.0

# How long a safety face (scared/dizzy) is held. Long enough to be seen and
# understood, short enough not to leave Friday looking frightened of a wall it
# already stopped for.
FACE_SAFETY_HOLD_S = 2.5

# Idle drift: with nothing happening the face sinks through these in turn, so
# a robot sitting alone looks asleep rather than switched off.
FACE_BORED_AFTER_S = 45.0
FACE_SLEEPY_AFTER_S = 150.0
FACE_SLEEPING_AFTER_S = 300.0

FACE_TICK_S = 0.2            # how often decay and idle drift are checked

# --- Memory --------------------------------------------------------------
# SQLite for the facts, the event log and the summaries; a TF-IDF index over
# the facts for recall. See memory.py for why not embeddings -- briefly, this
# Pi is CPU-bound and a transformer would compete with the wake word and STT
# for the same cores, making Friday slower at hearing in order to be slightly
# better at remembering.
MEMORY_DB = ROOT / "state" / "memory.db"

MEMORY_RECALL_LIMIT = 5      # facts pulled into the prompt per turn. Small on
                             # purpose: every one costs tokens on every turn
MEMORY_EXPERIENCE_LIMIT = 3  # past outcomes shown for a similar task
MEMORY_MIN_SCORE = 0.12      # below this a "match" is coincidence. Tuned so
                             # an unrelated question recalls nothing at all
                             # rather than the least-unrelated fact
# How much of the score comes from WORD overlap vs character n-grams.
# Characters alone were confidently wrong: "what is the capital of Japan"
# scored 0.405 against "the kitchen is through the left door" on shared letter
# sequences. Words carry the meaning and gate the result; characters only
# forgive typos and Hindi/English mixing.
MEMORY_WORD_WEIGHT = 0.75
MEMORY_DUPLICATE_SCORE = 0.82  # at or above this, being told something again
                             # UPDATES the fact instead of adding a copy
MEMORY_MAX_FACT_CHARS = 400  # a "fact" longer than this is a transcript
MEMORY_EVENTS_KEPT = 40      # events kept verbatim; older ones are collapsed
                             # into a counted summary so context stays bounded
MEMORY_TURNS_KEPT = 12       # conversation turns held in RAM

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

# If retrying is not enough because the dashboard is genuinely streaming that
# camera, vision.py BORROWS it: it asks the dashboard's own on/off endpoint to
# release the device, captures, and hands it straight back. Friday was blind to
# the exact camera Harsh could see live on his phone, and telling him to flick
# a switch first is not a fix.
DASHBOARD_LOCAL_URL = f"http://127.0.0.1:{DASHBOARD_HTTP_PORT}"
CAM_BORROW_SETTLE_S = 0.35   # v4l2 needs a breath between readers

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

# --- The fast loop (mcp_server/reflex.py) --------------------------------
# Friday's spinal cord: a local thread that watches for hazards while the
# wheels turn and stops the motors itself. It does not ask the brain, cannot be
# overruled by it, and keeps working with the network down -- which is the
# whole point of splitting fast from slow. There is deliberately NO off switch.
REFLEX_HZ = 20.0             # how often the reflex checks. 50 ms is well
                             # inside the distance covered at these speeds,
                             # and cheap: it is a dict lookup off the sensor
                             # pollers, not a serial round trip.

# A CLIFF IS THE HAZARD NOTHING ELSE CATCHES. The LiDAR sits on the top dome
# and reads a staircase as clear air; the bumper ToF looks ahead, not down. Its
# bottom rows see the floor a few hundred mm out, so when the nearest return
# from those rows suddenly gets FURTHER than the floor should be, the floor has
# ended. Measured on this robot's ride height -- re-measure if the chassis or
# the ToF mounting angle changes, because it is geometry, not a preference.
CLIFF_DROP_MM = 400          # floor further than this = a drop, stop now.
                             # 0 disables the cliff check (it should not be 0)

# --- Embodied reasoning --------------------------------------------------
# "Look before you move", enforced in code rather than asked for in the prompt.
# A movement tool refuses unless the robot has actually observed the world
# within OBSERVATION_TTL_S. The point is not politeness: a plan built on a
# reading from a minute ago is a plan about a room that may no longer exist,
# and the model has no way to know the difference. observe() and
# scan_environment() are what refresh it.
#
# This also buys "verify by looking again" for free. Because the freshness
# clock keeps running while the robot drives, a second move needs a second
# observation -- so the loop is forced to be observe -> move -> observe ->
# move, which is exactly the discipline wanted. It is not possible to chain
# blind moves.
REQUIRE_OBSERVATION_BEFORE_MOVING = True
OBSERVATION_TTL_S = 12.0     # how long an observation is trusted for
OBSERVATION_TTL_MOVING_S = 6.0   # ...and after the robot itself has moved,
                             # which invalidates a view faster than the world
                             # changing on its own does

# Act in small increments. A long blind commitment is the failure mode: you
# cannot correct a move that has already finished, and a small error repeated
# is recoverable where one large one is a collision. Anything longer is not
# refused -- it is CLIPPED to the increment and reported as partial, so the
# model re-observes and continues rather than being stonewalled.
MAX_INCREMENT_CM = 40.0      # longest single forward/backward step
MAX_INCREMENT_DEGREES = 90.0 # largest single pivot

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


def save_calibration(values: dict) -> dict:
    """Write calibration atomically. Returns what was written.

    Temp file in the SAME directory then rename, so motor code reading this
    per command can never catch it half written. Shared by the dashboard's
    calibration panel and by set_speed, which must agree: they write the same
    file and there is exactly one source of truth for how fast Friday drives.
    """
    import json
    import os
    import tempfile

    merged = load_calibration()
    merged.update({key: values[key] for key in CALIBRATION_DEFAULTS
                   if key in values})

    path = CALIBRATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".calib.",
                                   suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(merged, out, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return merged


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

# Attach relevant memories to each question automatically. The recall tool
# still exists for deliberate lookups, but Harsh should not have to ask Friday
# to consult its own memory any more than a person would.
MEMORY_IN_CONTEXT = True
