"""tts.py — Friday's voice: Piper through the USB soundcard.

WHY THIS IS BUILT THE WAY IT IS. Measured on this Pi, 2026-08-23:

    piper CLI, one call          17.3 s  to speak 1.8 s of audio
      of which model load         9.8 s  paid AGAIN on every single call
    voice kept loaded in-process  1.14x realtime (low), 1.60x (medium)

So the CLI was spending ten seconds loading a 63 MB model before every
sentence. The voice is loaded ONCE here and kept, which is the whole
difference between a robot that answers and one that appears to have hung.

Two more things follow from those numbers:

  * Synthesis is barely faster than realtime, so the first sentence is
    SYNTHESISED AND PLAYED while the second is still being made. Speaking
    starts as soon as the first sentence is ready rather than after the whole
    reply — on a long answer that is the difference between 2 s and 10 s
    before Friday makes a sound.
  * preload() exists so the ten-second load happens at start-up, in the
    background, instead of in front of the first thing Harsh asks.

DEBUG SILENCE. config.DEBUG_SILENT_SPEECH suppresses every word at the lowest
level in this file, not at the call sites. A caller that forgets to check the
flag still cannot make the robot talk.

PLAYBACK GOES THROUGH PIPEWIRE. pipewire holds the USB card open exclusively
while the desktop is running, so aplay on the raw ALSA device fails with
"Device or resource busy". config.SPEAKER_DEVICE is a pipewire node name; the
ALSA fallback is only for a headless boot with no audio server.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import audio_routing
import config

log = logging.getLogger("friday.tts")

# Sentence-ish split. Deliberately crude: the goal is to start speaking sooner,
# not to be a linguist. Includes the Devanagari danda, because Harsh mixes
# Hindi and English and a reply can come back with either terminator.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")

_speak_lock = threading.Lock()      # one voice, one utterance at a time
_voice = None
_voice_lock = threading.Lock()
_voice_error: str | None = None


class TTSError(RuntimeError):
    pass


# --- the voice -----------------------------------------------------------

def voice():
    """The Piper voice, loaded once and kept.

    Loading costs ~9.5 s, so this is guarded by a lock rather than a simple
    None check: two threads arriving together must not both pay it.
    """
    global _voice, _voice_error
    if _voice is not None:
        return _voice
    with _voice_lock:
        if _voice is not None:
            return _voice
        model = Path(str(config.TTS_VOICE)) if config.TTS_VOICE else None
        if model is None or not model.exists():
            _voice_error = (
                f"Piper voice not found at {model}. Fetch one with: "
                f"./venv/bin/python -m piper.download_voices "
                f"--download-dir assets/models/piper en_US-lessac-low")
            raise TTSError(_voice_error)
        try:
            from piper import PiperVoice
        except ImportError as exc:
            _voice_error = f"piper-tts is not installed in the venv ({exc})"
            raise TTSError(_voice_error) from exc
        log.info("loading Piper voice %s (takes ~10s)", model.name)
        _voice = PiperVoice.load(str(model))
        _voice_error = None
        log.info("Piper voice ready")
        return _voice


def set_output_volume() -> str:
    """Turn the speaker up to the level config asks for.

    WHY THIS EXISTS. Friday was barely audible, and the cause was two
    INDEPENDENT attenuations multiplying: the pipewire sink sat at 0.40 and the
    USB card's own ALSA gain sat at 15/30 (-22.5 dB). Together that is about
    20% — and no amount of TTS_VOLUME fixes it, because software gain above 1.0
    only clips. Both are runtime levels that reset, so Friday asserts them at
    start-up rather than relying on anyone having set them by hand.

    Turning a volume knob, not touching audio routing or any system config.
    Best-effort: a failure here must never stop Friday from talking.
    """
    environment = dict(os.environ)
    environment.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)
    done = []

    if shutil.which("wpctl"):
        try:
            # @DEFAULT_AUDIO_SINK@ rather than a node id: ids are assigned at
            # enumeration and change when a USB device drops off the bus.
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@",
                            str(config.SPEAKER_SINK_VOLUME)],
                           check=True, capture_output=True, env=environment,
                           timeout=5)
            subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
                           check=True, capture_output=True, env=environment,
                           timeout=5)
            done.append(f"sink {config.SPEAKER_SINK_VOLUME}")
        except Exception as exc:                    # noqa: BLE001
            log.warning("could not set the pipewire sink volume: %s", exc)

    if shutil.which("amixer") and config.SPEAKER_ALSA_CONTROL:
        try:
            subprocess.run(
                ["amixer", "-c", str(config.SPEAKER_ALSA_CARD), "sset",
                 config.SPEAKER_ALSA_CONTROL,
                 f"{config.SPEAKER_ALSA_PERCENT}%", "unmute"],
                check=True, capture_output=True, timeout=5)
            done.append(f"{config.SPEAKER_ALSA_CONTROL} "
                        f"{config.SPEAKER_ALSA_PERCENT}%")
        except Exception as exc:                    # noqa: BLE001
            log.warning("could not set the ALSA hardware gain: %s", exc)

    result = ", ".join(done) or "no mixer available"
    log.info("output volume: %s", result)
    return result


def preload() -> threading.Thread:
    """Start loading the voice in the background. Returns immediately.

    Called at server start-up so the ten-second model load is over before
    anyone asks Friday a question, rather than being charged to the first one.
    """
    def load() -> None:
        try:
            set_output_volume()
            voice()
        except TTSError as exc:
            log.warning("TTS preload failed: %s", exc)

    thread = threading.Thread(target=load, name="tts-preload", daemon=True)
    thread.start()
    return thread


def ready() -> bool:
    return _voice is not None


def engine_name() -> str:
    """Which voice would be used, without raising if there is none."""
    if _voice is not None:
        return f"piper {Path(str(config.TTS_VOICE)).stem} (loaded)"
    if _voice_error:
        return f"none ({_voice_error})"
    return f"piper {Path(str(config.TTS_VOICE)).stem} (not loaded yet)"


def _synthesis_config():
    """Speed and volume, from config.

    length_scale is INVERSE to speed — 0.9 is slightly faster than natural —
    so config carries the intuitive number and it is converted here.
    """
    from piper import SynthesisConfig
    # The pitch factor is folded in here, not applied afterwards on its own.
    # deepen() lowers pitch by RESAMPLING, which stretches the audio by the
    # same factor -- so synthesis has to come out proportionally shorter for
    # the finished sentence to land at the speed config actually asks for.
    return SynthesisConfig(
        length_scale=config.TTS_PITCH / max(0.1, config.TTS_SPEED),
        volume=config.TTS_VOLUME,
    )


# --- tone ----------------------------------------------------------------

def deepen(samples):
    """Lower the pitch a little, so the voice sits heavier in the room.

    Piper has no pitch control -- length_scale changes duration only. So the
    pitch is moved the honest way: resample the audio to play back over a
    longer span, which lowers every frequency by the same factor, and make up
    the lost speed at synthesis time (see _synthesis_config).

    Small shifts only. Past roughly 0.85 the formants move far enough that it
    stops sounding like a lower voice and starts sounding like a slowed tape.

    ⚠ USE A REAL RESAMPLER. The first version of this interpolated linearly
    (np.interp) and it sounded buzzy and gritty -- linear interpolation is a
    poor low-pass, so it aliases, and on a voice that reads as a permanent
    faint rasp. resample_poly filters properly and the artefact is gone.
    """
    from fractions import Fraction

    import numpy as np
    from scipy import signal

    factor = config.TTS_PITCH
    if factor == 1.0 or len(samples) < 2:
        return samples
    # Stretching by 1/factor lowers pitch by factor. A rational ratio keeps the
    # polyphase filter small; 1/200 resolution is far finer than the ear.
    ratio = Fraction(1.0 / factor).limit_denominator(200)
    return np.asarray(
        signal.resample_poly(samples, ratio.numerator, ratio.denominator),
        dtype=np.float32)


def add_body(samples, rate: int):
    """Boost the low mids and ease off the top, for weight and clarity.

    A thin voice through a small robot speaker is the usual complaint, and the
    cause is that the speaker has almost no output below a few hundred Hz while
    the top end comes through fine. Lifting the low mids puts the chest back in
    the voice; taking a little off the highs stops that lift reading as harsh.

    A shelf, not a peak: a narrow boost would make it honk on one vowel.
    """
    import numpy as np
    from scipy import signal

    out = samples
    if config.TTS_BODY_DB:
        gain = 10.0 ** (config.TTS_BODY_DB / 40.0)
        w = 2.0 * config.TTS_BODY_HZ / rate
        # A one-pole low shelf: cheap, phase-gentle, and plenty for this.
        b, a = signal.butter(2, min(0.99, w), btype="low")
        low = signal.lfilter(b, a, out)
        out = out + (gain - 1.0) * low
    if config.TTS_AIR_DB:
        gain = 10.0 ** (config.TTS_AIR_DB / 40.0)
        w = 2.0 * config.TTS_AIR_HZ / rate
        b, a = signal.butter(2, min(0.99, w), btype="high")
        high = signal.lfilter(b, a, out)
        out = out + (gain - 1.0) * high
    return np.asarray(out, dtype=np.float32)


# --- loudness ------------------------------------------------------------

def louden(samples, rate: int):
    """Make speech carry across a room, without distorting it.

    WHY GAIN ALONE CANNOT WORK. The pipewire sink and the USB card's hardware
    gain are both already pinned at maximum by set_output_volume(), so there is
    no knob left to turn. And Piper's output already touches full scale on its
    loudest sample while sitting at only ~-14 dBFS RMS — it is peaky, not loud.

    ⚠ DO NOT WAVESHAPE. The first version of this applied gain and then bent
    anything over a knee with tanh, per sample. That is waveshaping: it
    manufactures harmonics that were never in the voice, and on speech it reads
    as a buzzy rasp. Measured, it was pushing 3.2% of samples through the bend.
    Harsh heard it immediately and he was right.

    What works instead is a real compressor. The gain is driven by the signal's
    ENVELOPE and moves over milliseconds, not per sample, so the waveform keeps
    its shape and no harmonics are added. Lookahead means the gain is already
    down before a transient lands, so nothing has to be clipped after the fact.

    Takes and returns float32 in [-1, 1].
    """
    import numpy as np
    from scipy import ndimage, signal

    gain = 10.0 ** (config.TTS_GAIN_DB / 20.0)
    boosted = np.asarray(samples, dtype=np.float32) * gain
    if not len(boosted):
        return boosted

    ceiling = config.TTS_CEILING
    look = max(1, int(rate * config.TTS_LOOKAHEAD_MS / 1000))

    # 1. Peak envelope, looking ahead by `look` samples so the gain is already
    #    down by the time the peak arrives.
    envelope = ndimage.maximum_filter1d(np.abs(boosted), size=look * 2 + 1,
                                        mode="nearest")

    # 2. The gain needed to keep that envelope under the ceiling. Never above
    #    1: this only ever turns things down.
    needed = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-9))

    # 3. Smooth it. An abrupt gain change is itself a click, so the reduction
    #    is filtered to a release of TTS_RELEASE_MS. This is the step that
    #    keeps the result free of the harmonics tanh was adding.
    release = max(1, int(rate * config.TTS_RELEASE_MS / 1000))
    kernel = np.ones(release, dtype=np.float32) / release
    smoothed = signal.fftconvolve(needed, kernel, mode="same")
    smoothed = np.minimum(smoothed, needed)     # never louder than allowed

    return np.clip(boosted * smoothed, -1.0, 1.0).astype(np.float32)


# --- "she is talking", for anything that is listening ---------------------
# One small file, written the way face_control writes the emotion: a temp file
# in the same directory and a rename, so a reader never sees half of it. The
# wake detector reads it (see wake_word._own_voice_until) and drops audio while
# it says her voice is in the room. See config.SPEAKING_FILE for the two
# measured turns that made this necessary.
#
# NEVER RAISES, and never blocks speech. If the mark cannot be written, Friday
# still talks -- she is just audible to her own microphone again, which is
# exactly where this started.

def _mark(state: str, until: float = 0.0) -> None:
    try:
        path = config.SPEAKING_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"state": state, "at": time.time(),
                              "until": until, "pid": os.getpid()})
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".speaking.",
                                   suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not mark the speaking state: %s", exc)


@contextlib.contextmanager
def _audible():
    """Hold the "she is talking" mark for the length of an utterance."""
    _mark("speaking")
    try:
        yield
    finally:
        # Stamped INTO THE FUTURE: the player has returned but the tail is
        # still in the sink and in the room. config.AUDIO_DRAIN_S is how much
        # of it is left; the detector settles on top of that.
        _mark("quiet", until=time.time() + config.AUDIO_DRAIN_S)


class _FromFirstSound:
    """The "she is talking" mark from her FIRST SOUND, not from the top of
    speak() -- the same rule as the mouth (mouth.start() before play()).

    Everything before the first play() is loading the voice and synthesis:
    6 s measured on this Pi, and no sound in the room. Marked from the top,
    every line kept the wake word deaf for synthesis AND playback: on
    2026-10-04 16:29-16:31 the run narrated every 7-10 s and the wake word was
    deaf ~9 s of every ~9.5 s (scored 4-26 of 375 chunks per 30 s), so
    "Friday, stop" could not be heard while she explored.

    The first start() waits until the detector, which polls the mark every
    WAKE_SPEECH_POLL_S, has surely seen it before any sound arrives (the sound
    itself lags play() by AUDIO_START_LAG_S). Nothing played, nothing marked."""

    def __init__(self) -> None:
        self.on = False

    def start(self) -> None:
        if self.on:
            return
        self.on = True
        _mark("speaking")
        lead = (float(config.WAKE_SPEECH_POLL_S) + 0.05
                - float(config.AUDIO_START_LAG_S))
        if lead > 0.0:
            time.sleep(lead)

    def __enter__(self) -> "_FromFirstSound":
        return self

    def __exit__(self, *exc) -> bool:
        if self.on:
            _mark("quiet", until=time.time() + config.AUDIO_DRAIN_S)
        return False


@contextlib.contextmanager
def _voice_turn(path: Path | None = None, max_wait_s: float | None = None):
    """ONE VOICE AT A TIME, ACROSS PROCESSES.

    _speak_lock only serialises one process. Friday talks from several -- the
    mission and File 14 inside friday-mcp, the RL run's own narrator, friday-brain
    -- and Harsh heard two lines at once (2026-10-04 15:26:31: the run's
    "Exploration completed. Returning to where I started." over the mission's
    "I've finished exploring this area. What should I call it?"). An flock on one
    file makes them take turns. The kernel drops the lock when a process dies, so
    a crash cannot silence her; and after max_wait_s she speaks anyway -- a lost
    turn is better than a lost warning.
    """
    import fcntl
    target = Path(path) if path is not None else Path(config.SPEAK_TURN_FILE)
    wait = config.SPEAK_TURN_WAIT_S if max_wait_s is None else float(max_wait_s)
    fd = None
    try:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(target), os.O_CREAT | os.O_RDWR, 0o644)
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        log.warning("another voice held the turn for %.0f s; "
                                    "speaking anyway", wait)
                        break
                    time.sleep(0.05)
        except OSError as exc:
            log.warning("voice turn unavailable (%s); speaking anyway", exc)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def quiet_at() -> float:
    """When her own voice will have left the room (wall clock), or 0.0.

    Returns a moment in the future while she is speaking or has just stopped,
    and 0.0 when the mark is missing, unreadable, or too old to believe.
    """
    try:
        mark = json.loads(config.SPEAKING_FILE.read_text())
        stamped = float(mark.get("at", 0.0))
        if time.time() - stamped > config.SPEAKING_MARK_STALE_S:
            return 0.0                  # left behind by a process that died
        if mark.get("state") == "speaking":
            # No end is known yet, so believe it until the mark goes stale.
            return stamped + config.SPEAKING_MARK_STALE_S
        return float(mark.get("until", 0.0))
    except FileNotFoundError:
        return 0.0
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the speaking mark: %s", exc)
        return 0.0


# --- playback ------------------------------------------------------------

def play(path: Path) -> None:
    """Play a WAV through the USB soundcard.

    pipewire first, ALSA second. Neither may hang the caller — a wedged audio
    device must not be able to freeze the robot.
    """
    environment = dict(os.environ)
    environment.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)

    # ALSA FIRST WHEN PIPEWIRE HAS NO USABLE SINK FOR HER SPEAKER.
    #
    # pipewire will not always offer an output profile for this card. MEASURED
    # 2026-09-05: it advertised only `off`, `pro-audio` and `input:mono-fallback`
    # -- no ordinary output at all -- because ACP probed the card while the
    # webcam was streaming and the playback interface could not be allocated, so
    # the probe concluded there was no output to have.
    #
    # The card itself is fine: `aplay -D hw:3,0` plays a tone the moment nothing
    # is streaming. Going through pipewire in that state costs a 20 s timeout per
    # sentence and then falls back here anyway, which is the difference between a
    # robot that sounds slow and one that sounds broken.
    #
    # So: if the resolver cannot find a sink on her card, do not ask pipewire at
    # all -- go straight to the device.
    if audio_routing.speaker(config.SPEAKER_DEVICE) is None and shutil.which("aplay"):
        try:
            subprocess.run(["aplay", "-q", "-D", config.SPEAKER_DEVICE_ALSA,
                            str(path)],
                           check=True, capture_output=True,
                           timeout=config.TTS_TIMEOUT_S)
            return
        except Exception as exc:                      # noqa: BLE001
            # Her own card is unreachable too -- fall through to pipewire, which
            # will at least put the sound out of SOMETHING (the built-in jack).
            log.warning("her speaker is not reachable through ALSA either (%s); "
                        "trying pipewire's default output", exc)

    if shutil.which("pw-play"):
        # RESOLVED, NOT HARD-CODED. config.SPEAKER_DEVICE names a card AND a
        # profile, and this Pi's USB audio card re-enumerates under whichever
        # profile it feels like. pw-play does not fail fast on a target that is
        # not there -- it WAITS, so a stale name is twenty seconds of silence
        # per sentence. See audio_routing for the night this cost.
        result = subprocess.run(
            ["pw-play", *audio_routing.target_args(
                audio_routing.speaker(config.SPEAKER_DEVICE)),
             f"--volume={config.TTS_VOLUME}",
             # A bigger buffer for THIS STREAM only. The sink's own period is
             # ~1.3 ms, which this CPU cannot keep fed -- see config.
             f"--latency={config.AUDIO_PLAY_LATENCY}", str(path)],
            capture_output=True, env=environment, timeout=config.TTS_TIMEOUT_S)
        if result.returncode == 0:
            return
        log.warning("pw-play failed (%s), falling back to ALSA: %s",
                    result.returncode,
                    result.stderr.decode(errors="replace").strip()[:200])

    if shutil.which("aplay"):
        subprocess.run(["aplay", "-q", "-D", config.SPEAKER_DEVICE_ALSA,
                        str(path)],
                       check=True, capture_output=True,
                       timeout=config.TTS_TIMEOUT_S)
        return
    raise TTSError("no way to play audio — neither pw-play nor aplay exists")


def sentences(text: str) -> list[str]:
    """Split into speakable chunks, so the first one can start immediately."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _wav_seconds(path: Path) -> float:
    """How long this WAV plays for. 0.0 if it cannot be read -- a bad answer here
    must cost the mouth's timing, never the speech."""
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            return source.getnframes() / rate if rate else 0.0
    except Exception as exc:                        # noqa: BLE001
        log.debug("could not measure %s: %s", path, exc)
        return 0.0


def _render(chunk: str) -> Path:
    """One sentence to a temp WAV, loudened."""
    handle, name = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    path = Path(name)
    with wave.open(str(path), "wb") as out:
        voice().synthesize_wav(chunk, out, syn_config=_synthesis_config())
    try:
        _shape_wav(path)
    except Exception as exc:                    # noqa: BLE001
        # A failure here must cost tone, never speech.
        log.warning("could not shape %r: %s", chunk[:30], exc)
    return path


def _shape_wav(path: Path) -> None:
    """Pitch, tone and loudness, applied to a WAV in place.

    In that order deliberately: pitch first because it resamples and would
    otherwise smear the EQ, tone before loudness because the limiter must see
    the levels that will actually be played.
    """
    import numpy as np

    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        raw = source.readframes(source.getnframes())
    if params.sampwidth != 2 or not raw:
        return                                  # only 16-bit is expected here
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    samples = deepen(samples)
    samples = add_body(samples, params.framerate)
    samples = louden(samples, params.framerate)

    # Resample to the sink's native rate HERE, once, rather than leaving it to
    # pw-play to do on every playback in real time. The tone stages already pay
    # for scipy, this happens while the next sentence is still being spoken,
    # and it takes the last piece of realtime work off a CPU that is the reason
    # the audio was breaking up in the first place.
    rate = params.framerate
    if config.AUDIO_SINK_RATE and rate != config.AUDIO_SINK_RATE:
        samples, rate = _resample(samples, rate, config.AUDIO_SINK_RATE)

    out = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(params.nchannels)
        sink.setsampwidth(params.sampwidth)
        sink.setframerate(rate)
        sink.writeframes(out)


def _resample(samples, from_rate: int, to_rate: int):
    """Resample with a real polyphase filter. Returns (samples, rate).

    Same reasoning as deepen(): a proper filter, never linear interpolation,
    which aliases and reads as a permanent faint rasp on a voice.
    """
    from fractions import Fraction

    import numpy as np
    from scipy import signal

    if from_rate == to_rate or len(samples) < 2:
        return samples, from_rate
    ratio = Fraction(to_rate, from_rate).limit_denominator(1000)
    resampled = signal.resample_poly(samples, ratio.numerator,
                                     ratio.denominator)
    return np.asarray(resampled, dtype=np.float32), to_rate


class _Mouth:
    """The speaking face, switched on by the AUDIO rather than by the intent.

    The panel has a "speaking" expression and nothing was ever switching to
    it, so Friday talked with a neutral face -- which reads as a stuck screen
    rather than a robot with something to say.

    WHY IT IS NOT SIMPLY THE WHOLE OF speak(). It used to be, and the mouth
    therefore started moving the instant speak() was called -- before the voice
    was loaded and before Piper had synthesised a single sample. MEASURED on
    this Pi 2026-09-03, with the voice already resident so the ~9.5 s model load
    is not even in the number: the face went to "speaking" at +0.002 s and the
    first audio came out at +6.088 s. Six seconds of silent mouth-flapping,
    every time she spoke. That is the desync Harsh reported.

    So the mouth is armed here and started from inside the playback loop, one
    statement before the audio is handed to the device. The face process polls
    its state file every 80 ms and cross-fades over 280 ms, so the animation
    comes up while pw-play is still opening the stream -- which lands the mouth
    on the sound instead of a beat behind it.

    WHAT IT RESTORES MATTERS. The previous emotion, not neutral: if she was
    pleased about something before she said it, she should still be pleased
    afterwards. Forcing neutral would make every sentence flatten her. It is
    read at start(), not at construction, so it is whatever was actually showing
    when the mouth took over rather than whatever was showing when some caller
    decided to speak.

    BEST-EFFORT, ALWAYS. The face is a separate process and may not be running.
    Losing the animation must never cost the words, so every call here is
    wrapped and failures are logged, not raised. stop() is idempotent and runs
    from a finally, so a synthesis error cannot leave her stuck mid-sentence
    wearing a talking face forever.
    """

    def __init__(self) -> None:
        self._previous: str | None = None
        self._on = False

    def start(self) -> None:
        if self._on:
            return
        self._on = True
        import face_control              # local: face_control imports nothing
        try:
            previous = face_control.get_emotion()
            # NEVER RESTORE INTO "speaking". If the face is somehow already
            # showing it when we get here, something else set it and putting it
            # back would leave Friday talking forever with nothing to say.
            # Settle to neutral instead -- wrong for one beat, rather than
            # stuck for good.
            if previous == "speaking":
                log.debug("face was already 'speaking' — will settle to neutral")
                previous = config.FACE_DEFAULT_EMOTION
            self._previous = previous
            face_control.set_emotion("speaking")
        except Exception as exc:                    # noqa: BLE001
            log.debug("could not show the speaking face: %s", exc)

    def stop(self) -> None:
        """Put the face back. Idempotent, and safe to call when never started."""
        if not self._on:
            return
        self._on = False
        if self._previous is None:
            return
        import face_control
        try:
            face_control.set_emotion(self._previous)
        except Exception as exc:                    # noqa: BLE001
            log.warning("could not restore the %s face after speaking: %s",
                        self._previous, exc)


@contextlib.contextmanager
def _mouth_moving():
    """Arm the speaking face; the caller starts it when audio actually begins."""
    mouth = _Mouth()
    try:
        yield mouth
    finally:
        mouth.stop()


def speak(text: str) -> str:
    """Say something out loud. Returns what actually happened.

    Blocking — returns when Friday has finished speaking. Synthesis of the
    NEXT sentence overlaps playback of the current one, so the mouth starts
    moving as soon as the first sentence exists.
    """
    text = (text or "").strip()
    if not text:
        return "nothing to say"

    if config.DEBUG_SILENT_SPEECH:
        log.info("[silent] would say: %s", text)
        return f"[debug silence — not spoken] {text}"

    chunks = sentences(text)
    with _speak_lock, _voice_turn(), _FromFirstSound() as heard, \
            _mouth_moving() as mouth:
        voice()                         # raises TTSError if unavailable
        rendered: queue.Queue = queue.Queue(maxsize=2)

        def produce() -> None:
            for chunk in chunks:
                try:
                    rendered.put(_render(chunk))
                except Exception as exc:            # noqa: BLE001
                    log.error("synthesis failed for %r: %s", chunk[:40], exc)
                    rendered.put(None)
            rendered.put(StopIteration)

        worker = threading.Thread(target=produce, name="tts-synth", daemon=True)
        worker.start()

        spoken = 0
        produced = 0                    # every chunk yields exactly one queue item
        while True:
            item = rendered.get()
            if item is StopIteration:
                break
            produced += 1
            if item is None:
                continue
            # ANTICIPATE THE END OF THE LAST SENTENCE. The face polls its state file
            # every 80 ms and then cross-fades for 280 ms, so restoring it when
            # play() returns leaves the speaking mouth on screen for about a third
            # of a second of silence -- the tail Harsh reported. The producer puts
            # exactly one item per chunk, so the last item is known before it is
            # played; the restore is scheduled to complete as the audio ends rather
            # than to begin there. If playback overruns the estimate the mouth stops
            # a fraction early, which is the right way round to be wrong.
            timer = None
            if produced == len(chunks):
                # + the stream-setup lag, because the sound this is anticipating
                # does not begin until that much after play() is called.
                lead = (_wav_seconds(item) - config.FACE_SETTLE_S
                        + config.AUDIO_START_LAG_S)
                if lead > 0.0:
                    timer = threading.Timer(lead, mouth.stop)
                    timer.daemon = True
                    timer.start()
            try:
                # THE MOUTH STARTS HERE, not at the top of speak(): everything
                # above this line is model loading and synthesis, which on this
                # Pi measured 6.088 s before the first sample reached the device.
                # Idempotent, so only the first sentence pays for it and the
                # face is not rewritten between sentences.
                heard.start()
                mouth.start()
                play(item)
                spoken += 1
            except Exception as exc:                # noqa: BLE001
                log.error("playback failed: %s", exc)
            finally:
                if timer is not None:
                    timer.cancel()      # no-op once it has already fired
                try:
                    item.unlink()
                except OSError:
                    pass
        # BEFORE joining the synthesis worker, not after. The worker has already
        # finished producing by the time the queue yields StopIteration, but the
        # join is still up to a second of the mouth moving over silence if
        # anything goes wrong in there -- and the whole point of this change is
        # that the animation ends when the sound does.
        mouth.stop()
        worker.join(timeout=1.0)

    return (f"said {spoken} sentence{'s' if spoken != 1 else ''} "
            f"at speed {config.TTS_SPEED}x, volume {config.TTS_VOLUME}")


def main() -> int:
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Speak text through Friday.")
    ap.add_argument("text", nargs="*", default=["Friday online"])
    ap.add_argument("--engine", action="store_true", help="report and exit")
    ap.add_argument("--bench", action="store_true",
                    help="time the load and the speech")
    ap.add_argument("--volume", action="store_true",
                    help="reassert the speaker volume and exit")
    args = ap.parse_args()

    if args.volume:
        print(set_output_volume())
        return 0

    if args.engine:
        print(engine_name())
        return 0
    try:
        started = time.monotonic()
        voice()
        loaded = time.monotonic()
        result = speak(" ".join(args.text))
        done = time.monotonic()
        print(result)
        if args.bench:
            print(f"  model load {loaded - started:.2f}s, "
                  f"speech {done - loaded:.2f}s")
    except TTSError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
