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
MODEL_FAST = "claude-haiku-4-5"            # reflexes, simple classification
# ⚠ NO DATE SUFFIX. This was "claude-haiku-4-5-20251001", which the API
# rejects -- the model IDs are complete as written and a date-suffixed
# variant is a 404, not an older pin. Nothing caught it because the brain
# has never had an API key to call with.

# THE VOICE BRAIN RUNS ON HAIKU ONLY. Harsh, 2026-09-16: "switch the brain
# model to only haiku only for less consumption and fast answers". Haiku 4.5
# is half Sonnet 5's price per token (1/5 vs 2/10) and measured 1.31 s to the
# first token with the tools against Sonnet's 2.61 s (table below). Every turn
# -- the fast path AND the full turn with tools -- now uses it. brain_router's
# exploration decisions (PLAN/TAG/STUCK) still route to MODEL_MAIN.
# Set this back to MODEL_MAIN to put Sonnet behind the full turn again.
BRAIN_MODEL = MODEL_FAST
SYSTEM_PROMPT_FILE = ROOT / "personality.md"
# File 08 step 1: every turn ends with {"reply", "emotion", "memory_note"} -- see
# reply_contract.py. The reply is still spoken as it streams. emotion replaces
# main._pick_face's second Haiku call whenever it is valid; memory_note is saved
# with memory.remember(source=BRAIN_NOTE_SOURCE). False = plain spoken answers
# again, exactly as before (the parser passes plain text straight through).
BRAIN_STRUCTURED_REPLY = True
BRAIN_NOTE_SOURCE = "brain-note"
# Plain characters (spaces not counted) spoken since the last tool call after
# which a reply object's "reply" is treated as the same answer AGAIN and not
# spoken -- the guard for a PARAPHRASED echo; an exact one is caught at any
# length. Live on Haiku it wrote the gold price in plain words, then the object
# with the gold price. "Here's what I found." is 17 -- under it, so a lead-in
# never swallows the real answer. See reply_contract.ReplyStream.
BRAIN_REPLY_ECHO_CHARS = 40
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

# --- Making the brain answer faster -------------------------------------
# Measured on this Pi 2026-09-12, same question ("capital of Japan"), warm
# connection, time to the FIRST token -- which is what Harsh actually waits for,
# because Piper starts on sentence one:
#
#     sonnet-5  effort=low  + 41 tools     2.61 s   (this was every turn)
#     sonnet-5  effort=low  no tools       2.09 s
#     haiku-4-5              + 41 tools    1.31 s
#     haiku-4-5              no tools      0.75 s
#
# THE TOOL LIST COSTS HALF A SECOND ON ITS OWN, and most of what Harsh says to
# her needs no tool at all -- personality.md already tells her so ("A question
# is just a question. If he asks the capital of Japan, say Tokyo.").
#
# So a question that plainly does not touch the robot goes to Haiku with no
# tools: 0.75 s instead of 2.61 s, three and a half times faster to the first
# spoken word. Anything that might need her body, her sensors or her memory
# goes to Sonnet with the tools exactly as before.
#
# TWO INDEPENDENT GUARDS, because a router that silently answers the wrong way
# is worse than a slow one:
#   1. BRAIN_FAST_NEVER below -- a local word check. Anything matching it never
#      takes the fast path at all, so "what do you see" is never answered by a
#      model with no eyes.
#   2. The UNSURE sentinel. The fast model is told to reply with exactly UNSURE
#      when the answer needs anything it hasn't got, and that escalates to the
#      full Sonnet turn. The pattern is brain_router's and is already tested.
# The cost of a wrong guess is therefore 0.75 s, then the normal turn.
BRAIN_FAST_PATH = True
BRAIN_FAST_MODEL = MODEL_FAST
BRAIN_FAST_MAX_TOKENS = 300
# A fast answer is a SPOKEN answer, so it is capped hard. Anything needing more
# room than this needed the real brain.
BRAIN_FAST_UNSURE = "UNSURE"
# What either brain replies when the words were NOT said to her (a video, the
# room talking). main._ignored() never speaks it and the turn ends silently.
BRAIN_IGNORE = "IGNORE"
# HOW LONG A CONVERSATION STAYS OPEN. Inside this window, a QUESTION is
# answered even if the addressed test called it a fragment -- see
# brain._QUESTION_RE. 2026-09-19 23:57, thirty seconds after she had answered
# him: "So, how do you know me like what do you like what's it?" was his, the
# transcriber mangled it, and she said nothing. Harsh: "the brain is denying
# some of the ques". Only questions, only while her own voice is this recent,
# and only after the addressed test has already said no.
BRAIN_CONVERSATION_WINDOW_S = 90.0
# Words that mean the answer depends on the robot, the room, this house, or an
# action -- anything the fast model cannot honestly know. A false match costs
# about 1.9 s of latency; a miss costs a wrong answer spoken out loud with
# confidence, so when in doubt a word belongs in here.
#
# ⚠ MATCHED ON WHOLE WORDS, NOT SUBSTRINGS, and that is not a detail. As a
# substring list "time" caught "what's 17 TIMES 23?", which then took 14.4 s on
# the big model to answer 391 -- the exact kind of question the fast path
# exists for. Multi-word entries match as phrases.
#
# NOT IN THIS LIST, DELIBERATELY: time, weather, mood, and whether she is
# exploring. situation.py now hands all four to every request, fast path
# included, so the small model can answer them from what it was given rather
# than needing a tool. "timer", "alarm", "reminder" and "schedule" ARE here --
# those SET something, which is an action and needs the real brain.
BRAIN_FAST_NEVER = (
    "see", "look", "camera", "photo", "picture", "watching", "front of you",
    "move", "drive", "go", "come", "forward", "back", "backward", "turn",
    "spin", "wheel", "wheels", "motor", "motors", "stop", "park",
    "map", "room", "wall", "obstacle", "distance", "how far", "clear",
    "lidar", "sensor", "sensors", "battery", "charge", "power",
    "temperature", "hot", "cold",
    "remember", "forget", "memory", "memories", "recall", "you said",
    "earlier", "last time", "yesterday", "we talked",
    "alarm", "timer", "reminder", "remind", "schedule", "news",
    "here", "there", "behind", "around you", "where are you",
    "what are you doing", "yourself", "your name",
    # HER OWN BODY. 2026-09-20: "roughly how much does your chassis weigh"
    # was answered by the fast path with "About two to three kilograms,
    # depending on the battery — the aluminium frame and motors are pretty
    # light." Nobody has ever weighed her, there is no spec in the repo, and
    # `uncertain` stayed null, so the guess was not even logged as one. This
    # is the "cat" failure with a number instead of an animal: a model with
    # no access to the robot inventing a fact about the robot.
    "weigh", "weighs", "weight", "heavy", "how big", "how tall", "how wide",
    "how long", "size", "dimensions", "chassis", "frame", "body", "made of",
    "parts", "hardware", "screen", "display", "speaker", "microphone", "mic",
    "raspberry", "arduino", "servo", "cpu",
    # SEARCH COMMANDS. Harsh, 2026-09-16: "give her the ability to go and
    # search the web for anything when commanded". The fast path has no tools,
    # so "search the web for X" sent there is answered from memory -- the one
    # thing he asked her not to do. These always get the brain that can search.
    "search", "google", "internet", "online", "web", "website", "browse",
    "find out", "latest",
)

# Warm the API connection while she is still booting. A COLD DNS LOOKUP COSTS
# 5.04 s ON THIS PI (measured; warm is 0.02-0.15 s), and without this the first
# question of the evening pays all of it -- which is exactly the question
# somebody is watching her answer. models.list() is a GET, bills no tokens, and
# warms DNS, TLS and the connection pool in one go.
BRAIN_WARM_ON_START = True

# HOW LONG A PHOTOGRAPH IS STILL THE PRESENT TENSE.
# ⚠ THIS IS THE "SHE DESCRIBES OLD OBJECTS" BUG, and it is worth reading before
# changing. Measured from the live log 2026-09-13:
#
#   00:27:23  "can you recognize this object in front of you?"  -> observe()
#             -> "That's you, Harsh, leaning right over me, glasses and all,
#                 with that blue chair headrest in between us"
#   00:28:48  "can you see this color in front of you?"         -> NO TOOL CALL
#             -> "mostly that dark navy-blue of the chair headrest, with your
#                 dark hair above it"
#   00:29:24  "colour in front of you."                         -> NO TOOL CALL
#             -> "Blue - the chair headrest right in my face is a dark navy blue"
#
# She looked ONCE and then answered from that one frame for two minutes. And
# she was not wrong to: the image was still sitting in the conversation with
# nothing to say it was old, so the most recent picture she had WAS that one.
# No prompt wording fixes this reliably, because the evidence in front of the
# model genuinely does show a chair.
#
# So the image is REMOVED from the history once it is older than this, and
# replaced by a line saying when it was taken and that it is no longer current.
# Then answering "what do you see" from memory is not a judgement call she can
# get wrong -- the old picture is not there to answer from.
#
# 25 s: long enough that a follow-up in the same breath ("what colour is it?")
# still uses the frame she just took, short enough that nothing across two
# questions is ever described from a stale one. It also cuts cost and latency,
# because a stale image is re-uploaded on every subsequent turn until trimmed.
BRAIN_IMAGE_TTL_S = 25.0

# LOOK FIRST, THEN ANSWER — IN ONE ROUND INSTEAD OF TWO.
# Making her look every time (above) fixed the stale answers and cost real
# seconds: measured 8.9-9.4 s for "what can you see", because the turn is now
# a model round to decide to look, then the camera, then a second round to
# describe the picture. Two round trips for a question whose answer was always
# going to need the camera.
#
# So when the question is plainly about what is in front of her, the picture is
# fetched BEFORE the first request and attached to the question. One round.
# It also means freshness no longer depends on the model choosing to look --
# for these questions the current frame is simply there.
#
# The word list is deliberately NARROW, the opposite of BRAIN_FAST_NEVER: a
# miss here costs nothing (she calls observe herself, exactly as she does now),
# while a false match spends about 2 s on a camera nobody asked about. So only
# phrasings that are unambiguously "describe what is in front of you".
# ⚠ OFF — BUILT AND NOT VALIDATED, so it is not what runs tonight. Three live
# turns with it on came back 22.7 / 12.6 / 11.4 s against 19.9 / 9.4 / 8.9 s
# with her calling observe herself, but the comparison is confounded (two of
# the three slow turns also spent a round trip on set_face) and the Pi was at
# 95% CPU labelling clips at the time, so neither set of numbers is worth
# trusting. It may well be a win once measured on an idle Pi: turn it on, ask
# "what can you see" a few times, and compare against this line.
#
# THE CORRECTNESS FIX IS NOT THIS ONE — it is BRAIN_IMAGE_TTL_S above plus the
# "look before you say what is there" rule in personality.md, both verified
# working. This is only ever a latency optimisation on top.
#
# ON SINCE 2026-09-13 with the scene watcher: brain._cached_view() answers from
# scene.py's frame when it is fresh (no camera wait at all), and only a stale
# one falls back to observe-before-asking.
BRAIN_EYES_FIRST = True

# --- The scene watcher (scene.py, inside the MCP server) -------------------
# Harsh, 2026-09-13: describe the environment continuously, with object
# detection. One up-camera frame every SCENE_PERIOD_S, named by SSD-MobileNet
# (rl/friday_vision.py) on ONE thread -- this Pi is CPU-bound and whisper is the
# stage he waits on. Skips, never fights: autonomous runs, the dashboard stream.
SCENE_WATCH = True
SCENE_PERIOD_S = 8.0
SCENE_MODEL = ROOT / "assets" / "models" / "vision" / "ssd_mobilenet_v1_10.onnx"
SCENE_THREADS = 1
SCENE_CAPTURE_SIZE = (640, 480)   # MJPG; not CAM_C270_MAX -- see scene._tick
SCENE_WARMUP_FRAMES = 5
SCENE_MIN_SCORE = 0.6
SCENE_IMAGE_WIDTH = 640
SCENE_IMAGE_QUALITY = 80
SCENE_JSON = ROOT / "state" / "scene.json"
SCENE_IMAGE = ROOT / "state" / "scene_latest.jpg"
SCENE_MAX_AGE_S = 30.0       # situation.py names objects no older than this
SCENE_FRESH_IMAGE_S = 12.0   # the brain answers from the cached frame within this

# --- Her own voice must never reach the wake detector --------------------
# THE PAUSE IN main.py IS NOT ENOUGH, because main.py is not the only thing
# that talks. exploration_mode narrates a run through the MCP speak tool,
# pi_deploy and the RL scripts speak, say_what_you_see speaks from inside the
# MCP server, and the dashboard can speak on demand -- all in OTHER processes,
# all while the wake detector in main.py is wide open.
#
# MEASURED, in the turn log, during an exploration run:
#   2026-09-05 01:16:05  TRANSCRIBED "...Can I just talk? Continuing exploration."
#   2026-09-05 01:17:15  TRANSCRIBED "...Stop! Stop! I'd detected an obstacle 0.7 meters ahead."
# Both are her OWN narration, heard back through the microphone and answered
# as if Harsh had said them -- and the first one called start_exploration.
#
# So tts.speak() leaves a mark on disk while it plays, whatever process it is
# in, and the detector drops audio for as long as that mark says her voice is
# in the room. Same idea as the pause, just one that crosses processes.
SPEAKING_FILE = ROOT / "state" / "speaking.json"

# How long a "she is speaking" mark is believed before it is treated as
# rubbish left by a process that died mid-sentence. A deaf robot is a much
# worse failure than a self-wake, so this is deliberately short: one utterance
# is seconds, and TTS_TIMEOUT_S already caps a single sentence at 20 s.
SPEAKING_MARK_STALE_S = 45.0

# How often the detector re-reads the mark. Not per frame: a frame is tens of
# milliseconds and this Pi has no spare core (the face already eats one).
WAKE_SPEECH_POLL_S = 0.25

# What the player does not tell us. pw-play exits when it has written the last
# sample, not when the speaker has finished moving: --latency=AUDIO_PLAY_LATENCY
# of it is still queued in pipewire, and the room keeps it a little longer. The
# mark is stamped this far into the future when speech ends, so "quiet" means
# the sound has actually stopped.
AUDIO_DRAIN_S = 0.30

# --- She remarks on something new in view (file 08 step 7) ---------------
# scene_bridge.py reads scene.json (above) and, rarely, hands a NEW object to
# the brain for one short remark. Every gate must pass or no call is made; the
# worst case is one fast Haiku call (with the frame, ~$0.003) per cooldown.
# The numbers are behaviour knobs chosen, not measured -- change them freely.
SCENE_BRIDGE = True
SCENE_BRIDGE_POLL_S = 2.0            # how often scene.json is re-read (a file stat)
SCENE_BRIDGE_MIN_SCORE = 0.6         # SCENE_MIN_SCORE already drops below this
SCENE_BRIDGE_CONFIRM_TICKS = 2       # ticks in a row before a label is real
SCENE_BRIDGE_FORGET_S = 1800.0       # unseen this long -> new again
SCENE_BRIDGE_RESEED_GAP_S = 120.0    # not watching this long -> view is baseline
# QUIETER THAN IT WAS, after Harsh heard it on 2026-09-19: "the vision
# description, when no need it starts speaking". A remark every 3 minutes to
# anyone who happened to be in the room in the last QUARTER OF AN HOUR is a
# robot that narrates the furniture; once every 10 minutes, to someone who is
# actually there, is a robot that noticed something.
SCENE_BRIDGE_COOLDOWN_S = 600.0      # between remarks
SCENE_BRIDGE_QUIET_AFTER_TURN_S = 20.0   # after Harsh last spoke to her
SCENE_BRIDGE_AUDIENCE_S = 300.0      # he spoke, or a person was seen, this recently
SCENE_BRIDGE_IGNORE = ("person",)    # somebody, not something

# --- When Harsh says that was wrong, keep it (file 09 step 1) ------------
# corrections.py. Two ways in, because they fail in different places:
#   1. the brain fills `correction` in the reply object -- it read the whole
#      turn, so it knows what the right behaviour is, in Hinglish or English
#   2. a phrase tripwire here -- it still works when the brain is offline,
#      over the spend cap, or simply did not notice it was being corrected
# Stored in the facts table under kind=LEARNED_PREFERENCE_KIND, which is what
# makes it queryable separately from ordinary memory notes. The table has had
# a `kind` column since it was written; no schema change was needed.
CORRECTION_CAPTURE = True
LEARNED_PREFERENCE_KIND = "learned_preference"
CORRECTION_SOURCE = "correction"
CORRECTION_MAX_CHARS = 240

# File 09 step 2: the corrections are pulled back into every turn's context as
# instructions, in their own section, or being corrected changes the log and
# nothing else. BOUNDED ON PURPOSE -- the most relevant few, not the pile. A
# section that grows without limit is one the model learns to skim, and it is
# paid for on every single turn.
PREFERENCE_IN_CONTEXT = True
PREFERENCE_CONTEXT_MAX = 3

# File 09 step 3: an answer she was not confident about is logged and NOT
# spoken. Events table, task="uncertain_response", no outcome -- so it can
# never be read back to her as a fact she knows or a lesson that worked.
UNCERTAINTY_LOGGING = True
UNCERTAINTY_MAX_CHARS = 160
UNCERTAINTY_FRESH_S = 86400.0        # the idle look-up only chases today's

# ⚠ NO BARE "wrong". "What is wrong with the wifi" is a question, not a
# correction, and a tripwire that stores it teaches Friday a preference nobody
# expressed -- and junk facts are read back to her as things she knows (Harsh
# has already had to delete seven of those). Every pattern below needs the
# correcting CONTEXT, not just the word. "galat" is the exception: in this
# house it is only ever said about something that was wrong.
CORRECTION_PHRASES = (
    r"\bgalat\b",
    r"\b(nahi|nahin)[ ,]+(ye|yeh|woh|wo|that)\b",
    r"\b(aise|aisa|waise|vaise)[ ]+(mat|nahi|nah)\b",
    r"\b(mat|nahi)[ ]+(bolo|bol|kaho|kaha|karo)\b",
    r"\b(that'?s|that is|thats|this is)[ ]+(not right|wrong|incorrect)\b",
    r"\b(you'?re|you are|u r)[ ]+wrong\b",
    r"\bnot[ ]+(correct|right)\b",
    r"\bthat'?s not what i[ ]+(said|meant|asked|wanted)\b",
    r"\b(don'?t|do not|never)[ ]+(say|call me|speak|talk)\b",
    r"\bstop[ ]+(saying|calling)\b",
    r"\bi[ ]+(didn'?t|did not)[ ]+ask\b",
    r"\bwrong[ ]+(answer|name|one|again)\b",
)

# --- Self-reflection (file 09 step 4) ------------------------------------
# Harsh's spec: "One function, run_self_reflection(), called from three
# places: end of a conversation session, a daily scheduled time, and robot
# idle detection. Before running, check a last_reflection_at timestamp -- if a
# reflection already ran within the last few hours, skip."
#
# THE MINIMUM GAP IS WHAT MAKES THREE TRIGGERS SAFE. On a normal evening all
# three come true within minutes of each other -- he stops talking, the robot
# goes idle, and it is late -- so without it that is three Haiku calls for one
# quiet room. It is stored on DISK, because the brain restarts far more often
# than four hours (eight times on 2026-09-19).
SELF_REFLECTION = True
REFLECTION_FILE = ROOT / "state" / "reflection.json"
REFLECTION_MODEL = MODEL_FAST        # reading two short lists; Haiku is plenty
REFLECTION_MIN_GAP_S = 4 * 3600.0    # "a few hours", the spec's debounce
REFLECTION_SESSION_END_S = 600.0     # quiet this long = the conversation ended
REFLECTION_IDLE_S = 1800.0           # ...and this long = the robot is idle
REFLECTION_DAILY_AT = "23:30"        # local time; "" turns the daily one off
REFLECTION_POLL_S = 60.0
REFLECTION_MAX_TOKENS = 500
REFLECTION_MAX_PREFERENCES = 25
REFLECTION_MAX_GAPS = 25
REFLECTION_MAX_FLAGS = 12            # the list he is asked to rule on, bounded
# What step 5 puts in front of it, and how much of it is kept afterwards. A
# SAMPLE of the log, not the log: "not a full transcript replay".
REFLECTION_MAX_EVENTS = 12
REFLECTION_MAX_MEMORIES = 10
REFLECTION_LOG_MAX = 20              # rolling; the spec: don't append forever
# How long the latest note rides along in the turn context (situation.py).
# A day: after that it is about a conversation neither of them remembers.
REFLECTION_CONTEXT_FRESH_S = 86400.0
# The Brain Growth panel (step 7): how many learned preferences it lists, and
# how long growth.report() may be reused. The dashboard polls /api/state every
# 4 s and the counts are nine COUNT(*) queries.
DASHBOARD_PREFERENCES = 8
# Voice turns on the dashboard's live log (file 09 step 7, and Harsh:
# "if i am talking to friday the brain also replies should come in dashboard
# like what tool call and everything"). The stream carried only
# exploration_mode before this, so a conversation showed nothing at all.
# Fire and forget: dashboard_log posts on its own thread and drops events
# when the queue is full.
DASHBOARD_TURNS = True

# --- Companion or professional (file 09 step 8) --------------------------
# "companion (default — warm, casual, human-like) or professional (crisp,
# precise, minimal filler, confident — for demos, technical explanations, or
# formal requests)". The mode rides on the QUESTION, never in the system
# prompt: personality.md is the cached prefix of every request and changing
# it per turn would throw that cache away every turn.
#
# ⚠ MODE_STRANGER_QUIET_S IS A PROXY FOR A STRANGER, NOT A STRANGER. The spec
# wants "someone Friday hasn't talked to before asking it to explain itself"
# to sound professional; she cannot tell who is speaking until face
# recognition exists, so an explain-yourself question arriving when nobody
# has heard her voice for this long is treated as one. Replace it with real
# identity when that module lands.
MODE_DEFAULT = "companion"
MODE_FILE = ROOT / "state" / "mode.json"
MODE_AUTO = True                     # the automatic half; the command always works
MODE_STRANGER_QUIET_S = 600.0
GROWTH_CACHE_S = 20.0

# --- Nothing that costs money may run in a loop (file 08 step 10) --------
# rate_limit.py enforces these. DAILY_SPEND_CAP_USD above is the ceiling --
# the wall at the end of the corridor -- and these are the speed limit, so one
# tool called in a loop cannot spend the whole day's money in a minute.
#
# NOT EVERY PAYING PATH IS LISTED HERE, because some already carry their own
# gates and double-limiting them would just hide where the real one lives:
#   * a voice turn        -- the wake word is the gate, plus the daily cap
#   * scene remarks       -- SCENE_BRIDGE_COOLDOWN_S + the audience gate
#   * idle study          -- IDLE_STUDY_* (quiet, interval, per day)
#   * web_search          -- WEB_SEARCH_MAX_USES, enforced by the API itself
# What is listed is what had NO limit of its own.
RATE_LIMIT_FILE = ROOT / "state" / "rate_limits.json"
RATE_LIMITS = {
    # say_what_you_see: one Haiku call with a picture, ~$0.005. The brain can
    # call it as often as it likes -- "look again" is a reasonable-sounding
    # thought to have twice a second -- and nothing used to stop it.
    "vision_describe": {"min_interval_s": 15.0, "per_hour": 20, "per_day": 120},
    # Not money: wheels. The same three numbers keep the idle drive (step 9,
    # physical_curiosity.py) from becoming a robot that paces the room.
    "physical_curiosity": {"min_interval_s": 1800.0, "per_hour": 2,
                           "per_day": 12},
    # reflect_now: the ON-DEMAND path (file 09 step 4) skips the four-hour
    # debounce on purpose -- he asked for it -- so the debounce is not its
    # gate and it needs one of its own. ~$0.004 a call, and "let me think
    # about that again" is exactly the sort of thought a model has twice.
    "reflection": {"min_interval_s": 120.0, "per_hour": 6, "per_day": 20},
}

# --- She goes and looks for herself (file 08 step 9) ---------------------
# ⛔ OFF BY DEFAULT, AND IT SHIPS OFF. Harsh's own words in the spec: "Make
# this toggleable with a simple on/off flag, default off, so it can be tested
# in a controlled, open area before leaving it on unsupervised", and again in
# step 10: "don't ship it defaulting to on". scripts/test_smoke.py fails if
# this line ever says True, so turning it on is a thing HE does, on the robot,
# in a room he is standing in -- never something a change quietly does for him.
#
# physical_curiosity.py decides ONE thing: whether now is a good moment. The
# driving is explore_room() in the MCP server, unchanged -- the same clearance
# interrupt, duration cap, embodiment check and did-the-turn-happen test that
# every other move goes through. No stop-distance logic is written here.
PHYSICAL_CURIOSITY = False

PHYSICAL_CURIOSITY_AFTER_S = 600.0   # quiet this long before she wanders
PHYSICAL_CURIOSITY_BUDGET_M = 1.5    # a short leash: metres for one outing
PHYSICAL_CURIOSITY_BUDGET_S = 45.0   # and seconds
# Daylight only, in whole hours [start, end). An unsupervised robot setting
# off across a dark room at 3 am is the failure this avoids; she is asleep at
# that hour anyway (FACE_SLEEPING_AFTER_S).
PHYSICAL_CURIOSITY_HOURS = (9, 21)
PHYSICAL_CURIOSITY_TICK_S = 20.0     # how often the gates are re-asked

# --- She looks things up in the quiet (file 08 step 8) -------------------
# idle_study.py, on its own thread with its own client -- never in the wake
# loop, where a 3-15 s search would swallow a "hey Friday" (see the module).
# Silent: it searches, saves ONE sentence through memory.remember(), and the
# next turn may mention it (situation.py reads idle_study.pending()).
IDLE_STUDY = True
IDLE_STUDY_AFTER_S = 300.0           # quiet this long before she wonders
IDLE_STUDY_MIN_INTERVAL_S = 1800.0   # ...and at most one look-up that often
IDLE_STUDY_MAX_PER_DAY = 8           # measured ~$0.02 each when it searches
IDLE_STUDY_POLL_S = 20.0
IDLE_STUDY_MAX_SEARCHES = 2          # searches within one look-up
IDLE_STUDY_MAX_ROUNDS = 4            # requests, including pause_turn resumes
IDLE_STUDY_MAX_TOKENS = 400
IDLE_STUDY_FRESH_S = 7200.0          # after this she no longer brings it up
IDLE_STUDY_SOURCE = "idle-study"     # source= in the facts table
IDLE_STUDY_FILE = ROOT / "state" / "idle_study.json"
BRAIN_EYES_CAMERA = "up"     # the room, not the floor: 2.0 s against 3.3 s for
                             # "both", which returns the same single image
BRAIN_EYES_WORDS = (
    "what do you see", "what can you see", "what you see", "can you see",
    "do you see", "what is in front", "what's in front", "in front of you",
    "look at", "have a look", "what is this", "what's this", "what is that",
    "what's that", "what colour", "what color", "describe what",
    "recognise this", "recognize this", "what object", "who is this",
    "who's this", "am i holding", "holding", "point at", "pointing at",
)

# Cost safety — the brain must refuse to spend past this in a single day.
DAILY_SPEND_CAP_USD = 2.00

# Friday's self-development (mcp_server/tools/dev.py). A dev_task runs Claude
# Code headless, which costs real money and takes real time, so both are
# bounded. The estimate is deliberately rough and deliberately high: the point
# of the cap is to stop a runaway loop, not to do accounting.
DEV_TASK_TIMEOUT_S = 600     # a change that takes longer than 10 minutes has
                             # gone wrong; it is stopped and the branch dropped
DEV_TASK_EST_USD = 0.25      # charged against the daily cap per task

# The self-healing watchdog (scripts/friday_watchdog.py). Restraint is the hard
# part: an automatic fixer that reacts to everything burns money, churns the
# code, and can turn one broken service into a broken robot. Most crashes are a
# one-off -- a USB device that dropped, a camera that was busy -- and systemd's
# own restart has already fixed them by the time anyone looks.
WATCHDOG_INTERVAL_S = 60
WATCHDOG_FAILURES_BEFORE_FIX = 3   # must fail this many passes in a row first
WATCHDOG_FIXES_PER_DAY = 3         # over and above dev_task's spend cap
WATCHDOG_COOLDOWN_S = 1800         # no retry of the same unit inside this
WATCHDOG_MAX_ATTEMPTS_PER_UNIT = 2 # after this it STOPS and asks for Harsh.
                                   # Two fixes that did not hold means the
                                   # problem is not one it understands, and
                                   # continuing is just damage with a schedule.
COST_LOG = LOG_DIR / "spend.jsonl"

# --- Voice pipeline ------------------------------------------------------
# --- Wake word (openWakeWord) --------------------------------------------
# openWakeWord runs a small classifier over a CONTINUOUS 16 kHz stream, so it
# is never deaf -- unlike the Step 1 record->transcribe->check stopgap it
# replaced, which spent seconds inside whisper and dropped anything said during
# that gap. Measured on this Pi: 17 ms per 80 ms chunk, ~21% of one core.
#
# TRAINED ON THIS PI, 2026-08-27, by scripts/train_wake.py. openWakeWord ships
# alexa / hey_jarvis / hey_mycroft / hey_marvin and there is no "hey friday" to
# download, so this one was made: nine Piper voices saying it at three speeds,
# augmented for distance, room and noise, against 1830 negatives -- hard ones
# ("friday" alone, "hey" alone, "good friday", "hey fred") plus 45 s of this
# actual room. Held-out AUC 0.9958.
#
# Until it existed the detector ran on the bundled hey_jarvis, which means
# "hey Friday" could not have worked at ANY threshold. If wake detection ever
# looks broken again, check FIRST that this file is what is loaded.
#
# It is trained on SYNTHETIC speech. Measure it against a real voice --
#   ./venv/bin/python scripts/test_wake.py --live
# -- and if it is weak the fix is real recordings of Harsh, not more Piper.
WAKE_WORD_MODEL = ROOT / "assets/models/wake/friday.onnx"

# The score above which a frame counts as a wake, 0-1. Higher = fewer false
# triggers but more misses. Tune against the real room, not in the abstract:
#   ./venv/bin/python wake_word.py --tune
# 0.4, not 0.5, and the 0.1 was bought with measurement rather than nerve.
# At 0.5 the real room produced ZERO false fires in 2.5 minutes (peak score
# 0.000 -- openWakeWord's VAD gate means room tone never even reaches the
# classifier), while a real "hey Friday" was logged at 0.480 and missed by two
# hundredths. Spending unused false-positive headroom on catch rate is the
# right trade for a wake word: a spurious wake is a moment of silence, a
# missed one is a robot that ignores you.
# RAISED 0.4 -> 0.5 on 2026-08-28, and it cost nothing. After the retrain the
# catch rate on Harsh's 40 recordings is FLAT at 80% from 0.2 all the way to
# 0.5 -- the model stopped being ambiguous, so the threshold no longer trades
# catches against false fires the way it did when it was picked. What forced
# it: a false wake logged at 0.419, nineteen thousandths over the old bar,
# while every genuine wake that night scored 0.69, 0.70, 0.72, 0.76, 0.89,
# 0.97, 0.98, 0.99, 1.00, 1.00 and the live test's WEAKEST was 0.595. There is
# a wide empty gap between the two, and 0.5 sits in it.
# Re-measure with `scripts/test_wake.py --live` before moving it again.
WAKE_THRESHOLD = 0.5

# HOW MANY CONSECUTIVE CHUNKS MUST CLEAR THE THRESHOLD BEFORE IT FIRES.
# The validation layer, and the only part of the false-fire fix that works
# without retraining. It exists because the score cannot separate a real wake
# from a conversation -- both reach 1.00 -- but the SHAPE can.
#
# MEASURED 2026-09-06 on 40 real recordings and the 37 false fires the
# harvester caught in one hour of Harsh talking to a friend, longest run of
# consecutive chunks at or above 0.5:
#
#   real  : 0,0,0,0,0,1,1,1,1,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,8,8,9,9,9,10,11,11
#   fires : 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,3,3,4,5,5,6,6,6,7,8,8,15
#
# TWENTY-TWO OF THE THIRTY-SEVEN FALSE FIRES ARE A SINGLE FRAME. A real "hey
# Friday" holds the score for a median of five. So:
#
#   patience 1 (off):  35/40 caught,  37/37 false fires
#   patience 2:        31/40 caught,  15/37 false fires   <-- 59% fewer
#   patience 3:        29/40 caught,  13/37 false fires
#
# 2, not 3: 3 buys 2 more rejections for 2 more misses, and a missed wake is
# worse than a spurious one. This is a MITIGATION, not the fix -- 15 an hour is
# still 15 -- and the fix is retraining on the harvested negatives. Set to 1 to
# turn it off and get the old behaviour back.
#
# ⚠ DO NOT PUT IT BACK TO 1 NOW THAT THE VERIFIER EXISTS. That is the obvious
# move -- the numbers above say patience costs 4 catches (35/40 -> 31/40) and
# the verifier does precision far better, so the mitigation looks redundant.
# MEASURED 2026-09-12 with the verifier IN THE LOOP, his 40 takes and 70 known
# false fires:
#
#   patience 2 + verifier    REAL 32/40    FALSE 17/70   <-- shipped
#   patience 1 + verifier    REAL 32/40    FALSE 24/70
#
# THE CATCH RATE IS IDENTICAL AND THE FALSE FIRES GO UP BY 41%. The 35/40 was
# measured with NO verifier; the extra fires patience 1 lets through are ones
# the verifier now rejects anyway, so there is no catch rate left to buy back
# and the only thing changing the knob does is cost. Re-measure with
# scripts/test_wake.py before moving it, and measure WITH the verifier.
#
# ⚠ BACK TO 1 ON 2026-09-13 06:35 -- THE VERIFIER IS OFF, so the reasoning above
# no longer holds, and Harsh from a normal distance was being dropped by it:
# live 06:29 his attempts held 0.57 / 0.91 / 0.68 for ONE chunk and never
# fired. Without the verifier, 1 is the 35/40 row above.
WAKE_PATIENCE = 1

# THE VERIFIER -- A SECOND OPINION ON EVERY FIRE, BEFORE SHE BEEPS.
# Neither knob above can fix the false fires, because openWakeWord scores a
# conversation as high as "hey Friday" (415 turns: false fires median 0.98,
# real ones 0.83). Its classifier never heard a real conversation in training.
# So a second classifier, trained by scripts/train_verifier.py on the very
# clips that fooled the first, looks at the SAME 16x96 features at the moment
# it fires and says yes or no. It can only reject -- a wake openWakeWord misses
# stays missed -- and it is one dot product, so it adds no latency.
#
# Speech recognisers were tried first as the second opinion and measured
# useless on Harsh's voice (2026-09-12, his 40 takes): sherpa-onnx keyword
# spotting caught 15/40 at its loosest, Vosk Indian-English 25/40 at 3.6 s.
#
# If the file is missing or broken she wakes exactly as before and says so in
# the log. False turns it off.
#
# ⚠ OFF SINCE 2026-09-13 06:25. Live, it was turning HARSH away: 57 of his
# attempts in 13 minutes (06:10-06:23) were rejected at 0.01-0.15 while
# openWakeWord scored them 0.9-1.0 -- no threshold reaches that. Over 344 live
# wakes a higher cutoff dropped real and silent turns at the same rate (0.5
# kept 51/120 vs 70/224), so it was not separating them either. It needs a
# retrain on LABELLED live clips before it comes back.
#
# ✅ BACK ON 2026-09-13 10:20, RETRAINED (md5 6d335fcb) on 16 takes Harsh
# recorded from his normal distance in the humming room + 58 of that morning's
# video/conversation wakes. Held out, at 0.4: live wakes 6/6, his far takes 5/5,
# his close takes 6/8 (same as no verifier), that morning's false wakes 22 -> 6
# of 28, labelled false fires 20 -> 2 of 20. ship_verifier picked 0.7, which
# drops live wakes to 4/6 and far takes to 4/5 -- 0.4 is the highest cutoff
# that loses none of his. Old model: state/friday_verifier.before-20260913.npz.
# Then 0.4 -> 0.35 after re-scoring WITH TODAY'S HUM in front (the harness puts
# an old quiet-room recording there): far takes 13/16 at every level 0.2-0.5
# (detector alone 14), close 15/20 at 0.35 vs 14/20 at 0.4, today's false wakes
# 10/52 vs 9/52 (detector alone 24). A miss costs more than one false wake.
WAKE_VERIFY = True
WAKE_VERIFIER_MODEL = ROOT / "assets/models/wake/friday_verifier.npz"
WAKE_VERIFY_THRESHOLD = 0.6

# HEARING HIM FROM ACROSS THE ROOM. The misses in his 40 recordings are not
# scattered -- they are exactly the quiet takes, with clean separation:
# caught RMS -24.2 dBFS / peak -7.0, missed RMS -32.2 / peak -18.5. Every miss
# peaks below -14.5 and the quietest catch is -9.5. The cause is in the
# TRAINING (augment() adds an absolute noise level, so a quiet take is buried
# and a loud one barely touched), and the proper fix is an SNR-relative
# retrain. This is the runtime half of it, and it needs no retrain: lift a
# quiet chunk toward the level the model was trained to hear before scoring it.
#
# THE HARDWARE HAS NOTHING LEFT TO GIVE -- `amixer -c Device sget Mic` is
# already 28/28, +30.00 dB, maxed. So the gain has to happen here.
#
# THE THREE BOUNDS ARE THE WHOLE DESIGN, because a naive normaliser would
# amplify the room's fan straight into the classifier:
#   - nothing below WAKE_GAIN_FLOOR_DBFS is touched at all (that is room tone,
#     and a "wake word" found in amplified room tone is a false fire),
#   - the gain never exceeds WAKE_GAIN_MAX_DB, so a whisper cannot be inflated
#     into a shout, and
#   - it is driven by a running peak over ~1 s, not per chunk, so the ENVELOPE
#     of the phrase survives -- openWakeWord reads the shape over time, and a
#     per-chunk normaliser would flatten exactly the thing it keys on.
# Scoring only. The audio SAVED and the audio HANDED TO THE CAPTURE stay raw,
# and last_rms still reports the true level, or the mic-is-dead diagnostic
# would start lying.
#
# ⚠ OFF, BECAUSE IT DOES NOT DO THE THING IT WAS BUILT FOR. Measured
# 2026-09-12 on his 40 takes and 70 known false fires, verifier in the loop:
#
#   patience 2, no gain    REAL 32/40    FALSE 17/70   <-- shipped
#   patience 2 + gain      REAL 31/40    FALSE 10/70
#
# It was built to recover the QUIET takes. It recovered none and lost one. The
# likely reason is in the mechanism and not in the tuning: the gain starts at
# 1.0 and ramps toward its target over several chunks, so across a phrase as
# short as "hey Friday" (about twelve chunks) it is still climbing while the
# phrase is being spoken. That TILTS the envelope -- the exact thing the model
# reads -- and a real-time detector cannot look ahead to avoid it.
#
# The 41% cut in false fires is real and is NOT understood, which is why this
# is off rather than on: an unexplained mechanism is not something to put in
# front of judges, and Harsh's priority is not missing his name. Worth
# investigating properly later -- if the effect holds, the right form of it is
# probably SNR-relative augmentation in the TRAINING (see the note in
# friday-voice-pipeline), not a runtime normaliser. One line to try it.
WAKE_AUTO_GAIN = False
WAKE_GAIN_TARGET_DBFS = -10.0   # the peak level his CAUGHT takes sit at
WAKE_GAIN_MAX_DB = 12.0         # 4x. -18.5 dBFS (a typical miss) lands at -6.5
WAKE_GAIN_FLOOR_DBFS = -42.0    # below this it is the room, not him
WAKE_GAIN_WINDOW_S = 1.0        # the running peak the gain is derived from
# Fires the verifier turned down, kept apart from the unreviewed pile so
# --review-fires is not buried in them: the conversations that fooled
# openWakeWord, which are the next verifier's training set. One clip per
# rejected utterance, not one per 80 ms chunk of it.
WAKE_REJECT_DIR = ROOT / "state" / "wake_fires" / "rejected"
WAKE_REJECT_SAVE_GAP_S = 2.0

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
# openWakeWord fires at the END of the phrase, so audio from BEFORE the fire
# is the wake word itself, not the command -- "hey Friday move forward" in one
# breath has the command entirely after the fire. This only has to cover the
# model's own detection latency, so it is small. It was 400 ms, which dragged
# most of "hey Friday" into the capture and got transcribed as an empty string.
WAKE_HANDOVER_MS = 0            # see WAKE_DEADTIME_MS: any pre-roll here is
                             # pre-roll OF THE WAKE WORD itself

# Every scored frame at or above this is logged, so a threshold can be picked
# from REAL DATA rather than guessed at. Set it a little under WAKE_THRESHOLD
# and the log fills with the near-misses -- the "I said it and nothing
# happened" frames, which are the ones worth seeing. openWakeWord's VAD gate
# means only speech is ever scored, so this does not flood.
# Set to 1.1 to silence it entirely.
WAKE_LOG_SCORE_ABOVE = 0.2

# How long the wake detector may receive NO audio at all before it says so.
# A starved detector and a detector that simply heard nothing look identical
# in the log otherwise -- both are silence -- and that cost an hour once.
# Audio to discard when the detector re-arms after a turn. wait_until_quiet()
# returns when the PLAYER exits, but the tail of Friday's own sentence is
# still in flight through pipewire and still in the room, and it arrives a few
# frames later -- she woke herself at 1.000 half a second after re-arming.
# Deliberately small: anything longer starts eating a real "hey Friday" spoken
# the instant she stops talking.
WAKE_RESUME_SETTLE_S = 0.35

WAKE_STALL_WARN_S = 20.0

# How often the detector says it is alive when it has not fired. Prints
# paused=, how many chunks it actually scored, the last score and the last
# microphone level -- the four numbers that separate "left paused", "starved",
# "deaf to him" and "the room is quiet", which are otherwise one silence.
WAKE_HEARTBEAT_S = 30.0

WAKE_BEEP = True             # short tone on wake, so Harsh knows it heard him
                             # without having to look at the face
WAKE_BEEP_HZ = 880
WAKE_BEEP_MS = 90

# Which trigger starts a turn: "voice" (openWakeWord), "enter" (press Enter),
# or "both". "both" keeps a keyboard way in while the wake model is still the
# stand-in -- it costs nothing and there is always a way to talk to Friday.
WAKE_SOURCE = "both"

# --- Hard-negative harvesting --------------------------------------------
# WHY THIS EXISTS. On 2026-09-05 Friday fired 97 times in one evening while
# Harsh talked to a friend, median score 0.88 and MAX 1.00 -- the false fires
# are indistinguishable from real ones BY SCORE, so no threshold can separate
# them and tuning WAKE_THRESHOLD is wasted effort. The cause is in the
# training set, not the code: the only negatives the model ever saw were Piper
# TTS phrases and 45 s of an EMPTY room, while 60% of the positives were 1280
# augmentations of 40 clips of ONE voice. It learned "a human voice in this
# room", which is exactly what a conversation is.
#
# The fix needs the clips that actually fool it, and guessing at them produces
# a model tuned against a guess. So every fire writes the audio that caused it
# to WAKE_FIRE_DIR, and a conversation in front of Friday becomes a labelled
# pile of the hardest negatives there are. Review them with
#   scripts/train_wake.py --review-fires
# and retrain. This is the loop, not a debug flag: leave it on.
WAKE_SAVE_FIRES = True
WAKE_FIRE_DIR = ROOT / "state" / "wake_fires"
# How much audio to keep from BEFORE the fire. openWakeWord fires at the END
# of a phrase, so the whole utterance is behind the trigger, never ahead of it.
# 2.0 s is the window the model itself scores over; 3.0 gives a human enough
# context to tell "hey Friday" from "see you Friday" when reviewing.
WAKE_FIRE_SECONDS = 3.0
# Stop the pile eating the SD card. 500 x 3 s x 16 kHz mono = ~48 MB.
WAKE_FIRE_KEEP = 500

# --- Do not disturb ------------------------------------------------------
# A wake word that fires wrongly is a robot that interrupts a conversation,
# and until the model is retrained Harsh needs a way to say "not now" that
# does not mean killing the service. "Friday, go to sleep for ten minutes"
# pauses the DETECTOR only -- the loop, the face and the MCP server stay up,
# and she comes back on her own.
#
# There is no voice way out of it, deliberately: a detector that still listens
# for "wake up" is a detector that can still false-fire on "wake". Use the
# dashboard, or wait. The default is short for the same reason -- an
# open-ended sleep is how a robot ends up silently off for a day.
SLEEP_DEFAULT_MINUTES = 10.0
SLEEP_MAX_MINUTES = 480.0

# --- Microphone stream ---------------------------------------------------
# ONE pw-cat process feeds every consumer (see stt.AudioStream). Two readers on
# one USB card is the single-access trap the ToF, LiDAR and cameras have each
# already sprung, and re-opening the device at wake time would burn latency at
# the worst possible moment.
MIC_CAPTURE_BUFFER_S = 3.0   # how much recent audio the capture subscription
                             # holds, so a wake can reach back for pre-roll

# HOW MUCH THE WAKE DETECTOR CAN FALL BEHIND BEFORE AUDIO IS LOST. Each
# subscription is a bounded deque that silently discards its OLDEST frame when
# full, so this is not "how much history" -- it is how long the detector thread
# may be descheduled before a phrase gets torn in half.
#
# This was 2.0 s and it was measurably too small. On a loaded Pi the dropped
# count climbed 53 -> 127 -> 348 -> 567 across ten tries and the catch rate
# collapsed to 4/10, with the misses scoring 0.000-0.109 -- not near misses,
# audio that never arrived. On an idle Pi the same test drops 14 frames total
# (a startup burst) and catches 10/10. The classifier was never the problem.
#
# The cost of a bigger buffer is staleness if the detector falls behind for
# real. A late wake beats a missed one.
MIC_WAKE_BUFFER_S = 6.0
# HOW LONG A SILENT MICROPHONE IS ALLOWED TO STAY SILENT before the stream is
# assumed wedged and torn down for the pump to rebuild. pw-cat does not die
# when it fails -- measured on 2026-08-28 with Friday deaf after answering:
# the process was alive, had read 0 bytes in 5 s, and had used 0 seconds of
# CPU in 4m16s. Nothing raised, so nothing restarted. Comfortably longer than
# any real pause in a quiet room, because a needless restart costs the ~1 s of
# audio in flight.
MIC_STALL_RESTART_S = 12.0

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
# 3, NOT 4: one core must stay free for the face and the system. With 4,
# whisper running on false wakes took this Pi to load 13 and 84 degC on
# 2026-09-12 and every stage timed out. A slightly slower transcript beats
# a Pi that stalls.
# ⚠ THE REPETITION LOOP IS WHAT MAKES HER SLOW, NOT THE MODEL SIZE. Measured
# 2026-09-12 on false wakes: 81.8 s for 4.3 s of audio (x19), 141.7 s for 5.7 s
# (x25), and 265.2 s for 5.6 s (x47) -- whisper generating the same clause over
# and over until it runs out of room. transcribe() holds a PROCESS-WIDE lock, so
# a real question asked during one of those waits behind it and hits
# STT_QUEUE_TIMEOUT_S. Every one of those runs was a hallucination nobody asked
# for ("Danke schön.", Urdu, Italian gibberish).
#
# condition_on_previous_text=False is the fix: the decoder stops feeding its own
# output back in as context, which is what lets a loop bootstrap itself. Kept
# here rather than in stt.py so there is one place to change it.
STT_CONDITION_ON_PREVIOUS = False
# Belt and braces from faster-whisper's own degenerate-output detectors. They
# only engage when the temperature FALLBACK is on, so temperature is left at the
# library default (a tuple) on purpose -- pinning it to 0.0 silently disables
# both of these.
STT_NO_SPEECH_THRESHOLD = 0.6
STT_COMPRESSION_RATIO_THRESHOLD = 2.4

STT_CPU_THREADS = 3
STT_BEAM_SIZE = 1            # 1 is greedy: fastest, and on a Pi that matters
                             # far more than the small accuracy gain from 5

# THE THREE CEILINGS ON A RUNAWAY DECODE, added 2026-09-12.
# condition_on_previous_text=False above was meant to end the repetition loop
# and did not. Measured tonight on real false-fire clips: 165.0 s for 3.5 s of
# audio (x47), 126.2 s for 5.5 s (x22), 115.6 s for 4.2 s (x27). Whisper still
# loops INSIDE one segment -- "Noem, noem, noem", "ndweilay. ndweilay." -- and a
# loop that never repeats a whole SEGMENT never trips the compression-ratio
# detector either. Every one of those was room noise from a false wake, and each
# one held the process-wide transcribe lock while a real question queued behind
# it and timed out. So this is not one better detector; it is three independent
# ceilings, none of which real speech can reach.
#
# 1. DON'T DECODE WHAT ISN'T SPEECH. faster-whisper bundles Silero VAD
#    (assets/silero_vad_v6.onnx, 1.2 MB, already on disk -- no download), which
#    judges level and voicing rather than webrtcvad's spectral shape alone. It
#    was off here with the note "webrtcvad already endpointed this": true, and
#    the point is that webrtcvad's endpoint is the thing that is WRONG. It opens
#    captures on the room's fan (45.6% of empty-room frames read as speech to
#    it). Silero gets a second, better say before a single token is generated,
#    and finding nothing costs a fraction of a second instead of two minutes.
STT_VAD_FILTER = True
STT_VAD_MIN_SILENCE_MS = 300   # shorter than VAD_SILENCE_MS on purpose: this
                               # splits an already-endpointed clip, it does not
                               # decide when Harsh stopped talking
STT_VAD_SPEECH_PAD_MS = 120    # keep the edges. A clipped first phoneme is a
                               # wrong transcript, which is worse than a slow one

# 2. THE LOOP CANNOT FORM. A 3-gram already emitted may not be emitted again,
#    which is the exact shape of every hallucination in the log. Real speech
#    repeats words; it rarely repeats whole trigrams and never four times in
#    one breath.
STT_NO_REPEAT_NGRAM = 3

# 3. AND IF IT SOMEHOW DOES, IT ENDS. MAX_UTTERANCE_S is 6 s, and six seconds of
#    speech is about 15 words -- 25 tokens. 64 is more than double that and
#    still cuts tonight's worst case (a 448-token "noem") by seven times. No
#    real command has ever come close to it.
STT_MAX_NEW_TOKENS = 64

# The temperature FALLBACK is what arms no_speech_threshold and
# compression_ratio_threshold, so it cannot simply be switched off -- but the
# library default is SIX attempts (0.0 … 1.0), and a clip that fails at 0.0
# because it is noise fails at all six, for six times the cost. Two keeps both
# detectors armed and bounds the retry.
STT_TEMPERATURES = (0.0, 0.2)

# Timestamps are tokens too, and nothing downstream reads them -- main.py takes
# the text and drops the segments.
STT_WITHOUT_TIMESTAMPS = True
# WHISPER HALLUCINATES CONFIDENTLY ON NEAR-SILENCE, AND AUTO-DETECT IS HOW IT
# SHOWS. Measured 2026-09-12, turns 27-34: a false wake captures room noise and
# faster-whisper returns fluent text in a language nobody in the room speaks --
# "Danke schön." (de), "und eine Klimapose." (de), "We hebben hier een beetje
# uit." (nl) -- or the same English sentence twice. Those completed as SUCCESSFUL
# turns, so Friday answered noise out loud. That is worse than the old empty
# transcript, which at least ABANDONED.
#
# The fix is not to pin STT_LANGUAGE: Harsh mixes Hindi and English mid-sentence
# and pinning either one breaks the other. It is to refuse a transcript in a
# language she does not speak. A real command is en or hi; noise is whatever
# whisper felt like.
#
# ⚠ THE RISK IS A REAL COMMAND MISDETECTED AS FOREIGN AND DROPPED, and "stop" is
# the one that matters. Set STT_REJECT_FOREIGN = False to turn all of this off in
# one place if that is ever seen -- the log line says `rejected: language de` so
# it is visible rather than mysterious.
STT_REJECT_FOREIGN = True
STT_ALLOWED_LANGUAGES = ("en", "hi")
# A hallucination loop is the same short phrase emitted two or more times. Real
# speech repeats sometimes, so this needs to be near-identical, not merely
# similar, and it only ever applies when there is more than one sentence.
STT_REJECT_REPEATS = True
STT_REPEAT_SIMILARITY = 0.90
# LINES NOBODY SAYS TO A ROBOT. Whisper's best-known hallucinations are YouTube
# outros, and on 2026-09-13 a real video in the room said them too: turn 30
# heard "Thank you for watching." and Sonnet answered "Anytime, Harsh" -- which
# is what "she talks to herself" was. Lowercase substrings; a transcript
# containing one is discarded exactly like a hallucination.
STT_REJECT_PHRASES = ("thank you for watching", "thanks for watching",
                      "subscribe to", "like and subscribe",
                      "see you in the next video", "don't forget to subscribe")

# ⚠ PINNED TO ENGLISH 2026-09-13 FOR SPEED. Harsh asked for answers in 3-5 s,
# and auto-detect runs the encoder a second time on every command: measured on
# his own clips with the brain running, x2.9 realtime auto vs x1.3 pinned -- a
# 2 s command goes from ~6 s of transcription to ~2.6 s. The cost is Hindi:
# "aage jao" may come back mangled or translated. None restores auto-detect.
STT_LANGUAGE = "en"

# A SPOKEN ORDER GOES STALE. Harsh, 2026-09-13: "it should not execute any
# previously given command after 5 seconds". Transcription alone has taken
# 7-17 s on this Pi, so "turn left" could move her long after he had moved on.
# The clock starts when he STOPS talking (the capture's end) and is checked at
# the moment a tool below is called -- by a reflex or by the brain. Past the
# limit the tool is refused and never reaches the motors. stop() is never in
# this list: a late stop is still a stop.
# 5 -> 11 s the same morning, Harsh: "increase the time limit for output to 10
# to 11 seconds". The age is fixed when she starts ACTING (main._FreshOrders).
COMMAND_MAX_AGE_S = 11.0
COMMAND_FRESH_TOOLS = ("move", "move_distance", "turn_degrees", "go_to",
                       "start_exploration", "explore_room")

# Endpointing. VAD_SILENCE_MS is how long a pause has to last before Friday
# decides Harsh has finished. Too short and it interrupts him mid-sentence;
# too long and it feels slow to answer. 700 ms is a normal conversational gap.
VAD_SILENCE_MS = 700
VAD_AGGRESSIVENESS = 2       # webrtcvad 0-3; 3 is most aggressive at calling
                             # things noise, which drops quiet speech
VAD_START_FRAMES = 3         # consecutive 30 ms voiced frames before Friday
                             # believes speech started — one frame is a cough

# WEBRTCVAD ALONE CANNOT ENDPOINT IN THIS ROOM, and that one fact caused three
# separate bugs. Measured on 8.8 s of the EMPTY room: at aggressiveness 2,
# 45.6% of frames come back "speech" and there is exactly ONE gap of 700 ms in
# the whole recording. So the capture never closed, ran to MAX_UTTERANCE_S
# every single time, and handed Whisper ~10 s of fan noise. That produced an
# empty transcript, which tripped the RE-LISTENING retry -- Friday appearing to
# start listening again with no wake word -- and on the retry Whisper
# hallucinated "I'm sorry. I'm sorry." out of the noise. One turn took 81 s.
#
# webrtcvad has no idea how LOUD anything is; it only judges spectral shape,
# and a steady broadband fan at -33 dBFS looks enough like speech to it. So a
# frame now has to be BOTH vad-speech AND this many dB above the running noise
# floor. The floor is measured only from frames webrtcvad calls non-speech, so
# it cannot be dragged upward by Harsh actually talking.
# Measured in this room: floor -33 dBFS, loudest noise frame -23.7, Harsh's own
# wake recordings peak at -7. 10 dB clears the noise and leaves him ~13 dB.
# Set to 0 to disable the gate and go back to webrtcvad on its own.
#
# ⚠ RETUNED 2026-09-13 06:40: 10 dB FULL-BAND -> 6 dB IN 300-4000 Hz. The room
# had become a steady low hum (155 Hz and below) with its floor at -20 dBFS,
# 13 dB above the -33 it was tuned on. Harsh from a normal distance measured
# only 3-5 dB over that full-band, so no capture ever started: every wake ended
# "nothing said within 4s" and she answered nothing. In the speech band the same
# eight takes sit 7-12 dB over the floor (p95), the room 1.7 dB (max 7.0, one
# frame -- VAD_START_FRAMES needs three in a row). None = full band, as before.
VAD_LEVEL_BAND_HZ = (300, 4000)
VAD_SPEECH_MARGIN_DB = 6.0
# How many CONSECUTIVE below-full-margin frames may still count as speech
# after a strong one. 30 ms each, so 25 is 0.75 s -- long enough to carry the
# quiet tail of a word, short enough that room noise cannot hold a capture
# open on its own. Unbounded, it did exactly that.
VAD_SUSTAIN_FRAMES = 25

VAD_SUSTAIN_DROP_DB = 4.0    # once speech HAS started, this many dB is
                             # forgiven. Starting needs the full margin so the
                             # fan can never open a capture; sustaining needs
                             # less so the quiet tail of a word is not cut.
VAD_FLOOR_RISE_DB = 0.03     # how fast the floor creeps UP per 30 ms frame.
                             # It falls instantly. 0.03 dB/frame = 1 dB/second,
                             # slow enough that a sentence cannot drag it up,
                             # fast enough to follow a room that really changes.
VAD_PREROLL_MS = 300         # audio kept from BEFORE speech was detected, so
                             # the first syllable is not clipped off
# THE WAKE WORD'S OWN TAIL USED TO END THE CAPTURE. openWakeWord fires at the
# END of "hey Friday", so a short tail of it bleeds into the capture through
# the handover pre-roll. That tail is speech, it trips VAD_START_FRAMES, and
# then the perfectly normal pause before Harsh actually says anything exceeds
# VAD_SILENCE_MS -- so the capture endpointed on the wake word, transcribed
# ~nothing, and the command that followed was never heard. Measured: a 1.9 s
# capture containing only the tail, then "" as the transcript.
#
# So an endpoint carrying less voiced audio than this is treated as that tail
# rather than as an utterance, and listening continues until LISTEN_START_
# TIMEOUT_S. 300 ms sits above the tail and below any real word: "stop" runs
# 400-600 ms with its plosive. Raise it and short commands start disappearing.
VAD_MIN_UTTERANCE_MS = 300

# DEAD TIME AFTER THE WAKE, during which the capture refuses to believe speech
# has started. openWakeWord fires at the END of "hey Friday", so the first
# couple of hundred milliseconds of the capture are the wake word's own tail --
# measured at ~450 ms here, which sailed straight past the 300 ms length guard
# above and endpointed the capture at 1.3 s, before Harsh had said anything.
#
# Length could not separate them: a real "stop" is 400-600 ms too. TIME can --
# the tail is contiguous with the wake, and a command never is, because Friday
# beeps first and Harsh waits for it.
#
# The cost is "hey Friday move forward" said in one unbroken breath, where
# "move" could start inside this window. That is the rarer way to talk to her,
# and the beep actively teaches the other one.
WAKE_DEADTIME_MS = 350

# AN EMPTY TRANSCRIPT IS NOT AN ANSWER, IT IS A MISSED TURN. The capture can
# close on the wake word's own tail before Harsh has started speaking -- more
# so on a borderline score, which fires earlier in the phrase and leaves more
# of it behind. Whisper turns that into "", and Friday used to shrug and go
# back to sleep, which from across the room is indistinguishable from being
# ignored. So she listens once more instead, briefly, before giving up.
# One retry only: two would be a robot that will not stop listening.
#
# OFF SINCE 2026-09-10, BY MEASUREMENT. Every RE-LISTENING in logs/friday.log
# -- 145 of them, 2026-08-27 to 09-10 -- scored by what the retry produced:
#   nothing at all                     130
#   words, none of them a command       14   "Thank you for watching.",
#                                            "Oh my god.", Korean, Dutch
#   a command                            1   "emergency stop.", mid-run, 73 s
#                                            after the wake -- most likely the
#                                            run's own narrator, which says it
# The tail it was built for is handled at the source now (WAKE_DEADTIME_MS),
# and 69 of the 145 "empty" first takes were never transcribed at all: that
# was the transcription lock giving up after STT_QUEUE_TIMEOUT_S, not Whisper
# hearing nothing. What was left was the complaint itself -- a false wake in
# the middle of a conversation, and then Friday listening to it AGAIN.
# Harsh, 2026-09-10: "it listens again again human conversations".
# True brings it back.
EMPTY_TRANSCRIPT_RETRY = False
RETRY_LISTEN_TIMEOUT_S = 4.0   # shorter than the first wait -- if he was going
                               # to speak, he is speaking by now
# 20 s was a ceiling on the AUDIO that became a 3-minute ceiling on the
# TRANSCRIPTION at x9 realtime -- so the stage timeout could never be met and
# every long sentence was abandoned. A spoken command is rarely over 10 s, and
# bounding it here bounds everything downstream.
# 6, not 10: this ceiling IS the worst-case whisper bill, and whisper is the
# hottest thing on the Pi. A spoken command is rarely over 4 s.
MAX_UTTERANCE_S = 6.0        # hard ceiling on one utterance
# 4, not 8: a wake nobody follows should cost 4 s of listening and NO
# transcription -- the Silence path never reaches whisper.
LISTEN_START_TIMEOUT_S = 4.0 # give up if nobody speaks after waking

# --- Pipeline stage deadlines ---------------------------------------------
# WHY THESE EXIST. The failure that prompted them looked like "Friday hears me
# and then does nothing": the wake word fired, the face changed, and the turn
# then sat inside some stage forever with nothing in the log to say which one.
# A stage that overruns now ABORTS, says so out loud, and re-arms the wake
# word -- a Friday that admits it lost the thread is worth far more than one
# that goes quiet and looks broken.
#
# Capture is deliberately absent: it is already bounded by MAX_UTTERANCE_S and
# LISTEN_START_TIMEOUT_S above, which are the same mechanism by another name.
#
# Every value is a CEILING, not a target. Set them well above the measured
# warm time or a slow-but-working turn gets killed for no reason. Measured on
# this Pi with the other services running:
#   STT     tiny/int8, 4 threads   ~x5 realtime, so ~10 s worst case at 20 s in
#   brain   Sonnet, streamed       5-25 s depending on tool rounds
#   tool    most are <1 s          scan_environment is the slow one at ~8 s
# Measured on this Pi WITH the face running: x3 realtime at best, x9 at worst.
# MAX_UTTERANCE_S caps the audio at 10 s, so the worst realistic case is ~90 s
# and 30 s was guaranteed to fire on any long sentence -- which it did, three
# turns in a row, and each abandoned transcription kept running and made the
# next one slower. See the lock in stt.transcribe.
STAGE_STT_TIMEOUT_S = 75.0   # transcription of one utterance
# How long a NEW transcription waits for an abandoned one to finish before it
# gives up. Beyond this, returning "" quickly is better than joining a queue
# that is already the reason nothing is responding.
STT_QUEUE_TIMEOUT_S = 20.0
STAGE_BRAIN_TIMEOUT_S = 90.0 # the whole tool-use loop, all rounds together
STAGE_TOOL_TIMEOUT_S = 25.0  # ONE MCP tool call. A tool that overruns comes
                             # back to the model as a tool error, so it can
                             # apologise or try something else, rather than
                             # taking the turn down with it.
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

# THE SERIAL IS THE ONE PART OF A by-id NAME THAT SURVIVES RE-ENUMERATION.
# udev builds those names from the USB descriptor STRINGS, which is exactly
# what changes on this camera -- "Generic_HD_camera_20181212000000" became
# "HD_camera_20181212000000" and the configured path went stale. The trailing
# serial did not change, and neither did the C270's.
#
# Matching on it kills a bug that "pick whichever node is not the C270" could
# not avoid: when the C270 ITSELF renames, it stops matching CAM_UP, and the
# down camera then resolves to the up camera's node. Nothing errors. look()
# just quietly returns the ceiling and calls it the floor -- which for §5
# semantic mapping means furniture logged at the wrong place forever.
#
# An unrecognised camera is left UNASSIGNED rather than guessed at. Being
# blind is recoverable; being confidently wrong about which way you are
# looking is not.
CAM_UP_SERIAL = "9B5B4130"              # Logitech C270, 046d:0825
CAM_DOWN_SERIAL = "20181212000000"      # the down cam, 349c:2317

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

# Which RAW LiDAR bearing points at the robot's front.
#
# MEASURED 2026-08-26: 51.8 degrees, by driving 250 mm and scan-matching the
# sweep before against the sweep after (`scripts/calibrate_lidar.py --drive`).
# The direction she travelled IS forward, by definition, so this is a
# measurement rather than an inference. Match score 0.22 with a peak-to-median
# confidence of 13.9x, and the peak held at 47-54 degrees across a six-fold
# range of matcher blur, which is what makes it believable.
#
# ⚠ DO NOT re-measure with the object method. The LiDAR plane is at leg
# height: a box on the floor is BELOW it and never appears at all (three
# attempts), and a person appears as TWO returns 12 degrees apart -- left leg
# and right leg -- whose midpoint is a guess worth several degrees. That
# method is why this sat at None for so long. See the module docstring in
# scripts/calibrate_lidar.py.
LIDAR_FORWARD_DEG = 51.8

# ⚠ FRIDAY CAN SEE HERSELF. Raw bearings 90-97 return a rock-steady 210-220 mm
# with a standard deviation UNDER 2 mm -- that is not a room, it is part of her
# own chassis sitting in the LiDAR's plane. Proved 2026-08-26 by pivoting 45
# degrees in place: the return did not move by a single degree, while
# everything else in the room did.
#
# It has to be masked or it poisons everything downstream. It is permanently
# the nearest thing in any sweep, so it makes every omnidirectional clearance
# check fail; and for SLAM it would be integrated into the map as an obstacle
# that orbits with the robot, which is exactly the input that makes a scan
# matcher diverge.
#
# Inclusive ranges of RAW bearings, widened by one degree either side of the
# solid run because 89, 97 and 98 flicker between the chassis and the wall
# behind it (stdev 796, 306 and 177 mm respectively, against under 2 mm inside
# the run). 10 of 360 bearings, so 2.8% of the sweep is blind -- re-measure
# with scripts/lidar_self_mask.py if anything is ever bolted to the dome.
#
# 71-78 is a second dome post, measured 2026-09-13: 207-223 mm on 71/72-77 in
# three sweeps while 70 and 78 saw 2.3-3.9 m, and it stayed put through pivots
# and a whole scene change. It sits ~22 deg right of the forward bearing, so
# unmasked it pinned `forward` at ~210 mm and the reflex cut every forward move
# at 197. Harsh confirmed it is part of the dome, not removable. The bumper ToF
# still covers straight ahead.
LIDAR_SELF_MASK = ((71, 78), (89, 98))

# The D500 is specified to 12 m, but indoors a long return is usually a
# doorway or a window rather than a wall, and those are the points that drag a
# scan match. Anything past this is dropped for mapping.
# ⚠ A "scan" IS NOT A SWEEP, AND MAPPING MUST NOT TREAT IT AS ONE.
# Lidar.scan() returns a rolling window: every bearing independently holds
# whatever the last revolution to reach it wrote. Standing still that is
# invisible and the wide window is a virtue -- a clearance check wants
# anything seen recently. Turning, it is the bug that wrecked the map: the
# spread of capture times inside one cloud measured 95-130 ms typically and
# 658 ms at worst, and slam/ transforms the whole cloud by ONE pose. At the
# measured 92 deg/s that is 9 degrees of smear normally and 60 at worst.
#
# 120 ms is one revolution of the LD19, and it is measured, not chosen: it
# keeps 347-349 of 350 bearings, where 100 ms keeps 314-349 and 80 ms only
# 265-281. Going wider buys no bearings at all and only more smear.
#
# This is the cap on how BLENDED a mapping cloud may be. It does not make a
# cloud captured mid-turn safe -- for that the caller must also let the robot
# settle a full revolution before sweeping, so no point predates the stop.
LIDAR_SWEEP_MAX_AGE_S = 0.12

LIDAR_MAX_RANGE_MM = 6000

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

# ⚠ THE BOTTOM HALF OF THE ToF LOOKS AT THE GROUND, NOT AHEAD.
#
# The sensor is tilted down on the bumper, and it matters more than it sounds
# like it should. Measured 2026-08-26 over 120 frames in an empty room with
# nothing in front of the robot:
#
#   row  cells returning   median    spread across the 8 columns
#    0      3.7 / 8         1812 mm      +/-142      <- the room ahead
#    1      2.2 / 8         1693 mm      +/-381      <- sparse and ragged,
#    2      3.3 / 8         1580 mm      +/-417         because far surfaces
#    3      0.1 / 8         1763 mm      +/-1871        return weakly
#    4      8.0 / 8          295 mm      +/-27       <- THE FLOOR. every cell,
#    5      8.0 / 8          223 mm      +/-8           every frame, and the
#    6      8.0 / 8          161 mm      +/-6           spread collapses to
#    7      8.0 / 8          120 mm      +/-3           a few millimetres
#
# forward_mm() used to take the nearest return ANYWHERE in the frame, so it
# always returned row 7 -- the ground about 12 cm in front of the bumper. That
# made check_forward_clearance() answer "115 mm" permanently, in any room, and
# FRIDAY COULD NEVER DRIVE FORWARD AT ALL: every move was refused as too close
# to something, and the something was the floor she was standing on. It read
# exactly like a box in the way, which is how it survived this long.
TOF_FLOOR_FROM_ROW = 4       # rows at or below this index look at the ground

# What each of those ground rows reads when the floor ahead is empty, in the
# same order (row 4 first). Re-measure with scripts/calibrate_tof_floor.py if
# the bumper is ever re-seated or the sensor's angle changes.
TOF_FLOOR_PROFILE_MM = (295, 223, 161, 120)

# A ground row reading this much SHORTER than its (pitch-corrected) floor
# distance is something standing on the floor, not the floor. Comfortably
# above the +/-27 mm of noise on the worst row, so an empty floor never reads
# as an obstacle.
TOF_FLOOR_OBSTACLE_MM = 45

# How far the common pitch factor may stretch the floor profile. The chassis
# noses down as the motors take up and every ground row reads short together;
# without this, Friday braked 0.2 s into her first move for a 246 mm "object"
# that was her own floor seen at a slightly steeper angle. Clamped so that a
# surface large enough to fill the whole lower field of view can never
# normalise itself into looking like the floor.
TOF_FLOOR_PITCH_MIN = 0.80
TOF_FLOOR_PITCH_MAX = 1.25

# How many recent frames forward_mm() takes its median over. About 0.7% of
# frames carry a spurious 250-315 mm return in the centre cells just above the
# horizon -- multipath off the floor echo below them -- and a single one of
# those braked the robot mid-move. Three frames is ~200 ms of history at 15
# fps; a real obstacle is in all of them, an artefact is in one.
TOF_CORROBORATE_FRAMES = 3

# How far a ground row may drift from its profile before the floor is no
# longer considered intact. Beyond this the answer is "cannot see", never
# "clear" -- a floor that stopped reading is what a cliff looks like.
TOF_FLOOR_INTACT_MM = 70

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
MOTOR_KICKSTART_MS = 150   # 130 before the ramp; see MOTOR_RAMP_FLOOR
MOTOR_KICKSTART_FLOOR = 1.0  # fraction of the duty cap to pulse at

# ...BUT DELIVERED AS A RAMP, NOT A SLAM. The pulse above used to go 0 -> full
# cap in one write, hold, then drop straight to the requested speed: two
# near-instant torque steps, which is felt as a jerk and is worst on a pivot
# where all four wheels break traction together. Harsh reported exactly that.
#
# The same total time and the same peak, split into steps: rise to the pulse
# level, then settle to the target. Stiction still gets broken -- the peak is
# unchanged and it is still reached inside the same window -- but the torque
# arrives over several steps instead of one edge.
#
# Cost is MOTOR_RAMP_STEPS extra serial writes inside MOTOR_KICKSTART_MS. The
# Arduino watchdog is MOTOR_WATCHDOG_S, far longer, so the extra traffic
# cannot starve it. Set to 1 for the old single-slam behaviour.
#
# ⚠ THIS RAMPS STARTING ONLY. stop() is untouched and must stay untouched:
# an emergency stop that eases off is not an emergency stop. Same reason the
# ramp is here in Python and not in the firmware, where it would share a code
# path with S.
# A pivot that happened moves every bearing onto new geometry. A pivot that
# was commanded and did not happen leaves the sweep where it was. This is the
# line between them, in median absolute range change across shared bearings.
# Sensor noise on this LiDAR is single-digit mm; a real 30 degree turn moves
# it by hundreds. Anything in between is a turn that partly stalled.
# --- Missions (step 9: the loop that drives) -----------------------------
# THREE budgets, and any one of them ending stops the run. A loop that
# explores "until finished" is one bad frontier away from running until the
# battery dies. Conservative on purpose -- these are what a caller gets if it
# does not choose, and the caller can always ask for less.
MISSION_BUDGET_M = 8.0        # metres driven
MISSION_BUDGET_S = 180.0      # wall clock
MISSION_MAX_CYCLES = 40       # decisions
MISSION_SETTLE_SWEEPS = 8     # sweeps before deciding anything, standing still
# Below this a "turn" is not worth the stop-start, and is folded into the
# drive instead.
# 10, because that is the smallest angle MEASURED to turn reliably
# (scripts/calibrate_turn.py: 10 deg moves the world 84-100 mm, and every
# angle from 10 to 60 moved on both sides, 14 pivots without a stall).
# Anything below it is untested, and an untested turn on a chassis that
# sometimes does not turn is not worth the risk of a false heading.
MISSION_MIN_TURN_DEG = 10.0

# A safety trip is the reflex doing its job, not the mission failing, so she
# turns away and re-plans instead of giving up where she stands. Capped:
# a robot that backs off and re-plans forever in a corner it cannot leave has
# hung, and should say so rather than look busy.
# ⚠ HOW FAR SHE MAY DRIVE IN ONE GO, AND WHY IT IS SHORT.
#
# Friday bumped something during an autonomous run on 2026-08-27. Nothing
# failed: the object sat in the ToF's documented blind band (0.31-0.75 m for
# anything short) AND below the LiDAR's plane, which is on the top dome at leg
# height. So forward clearance could not see it, the 200 mm LiDAR proximity
# backstop could not see it either, and the first thing that noticed was the
# reflex at SIXTEEN MILLIMETRES -- after contact. The stall backstop catches
# it too, but only after SAFETY_STALL_AFTER_S of pushing, which is ~9 cm.
#
# Software cannot give her an eye she does not have. What it can do is stop
# her building up a long unsupervised run into space nothing has confirmed:
# a short leg means she re-observes, re-plans and re-checks far more often,
# and any push is a short one. It is a mitigation, not a fix. THE FIX IS
# PHYSICAL -- a contact bumper, or a sensor that can see low and near.
MISSION_MAX_LEG_M = 0.25

MISSION_MAX_RECOVERIES = 4
MISSION_RECOVER_TURN_DEG = 60.0   # comfortably above the measured 10 deg floor

# --- Places (step 9 section 5) -------------------------------------------
# How close counts as "at" a named place. Roughly an arm's length past the
# robot's own body radius: close enough that "I am in the kitchen" is true,
# far enough that it does not flicker as she shuffles.
PLACE_RADIUS_M = 0.75

# --- Narration (step 9 section 4) ----------------------------------------
# How talkative Friday is while exploring. The explorer emits a line every
# cycle and a cycle is a couple of seconds; saying all of them is a stuck
# record. Events (a loop closed, finished, stuck, a room opening up) ignore
# this floor -- see slam/narration.py.
NARRATION_MIN_GAP_S = 12.0
# A gain this big in one cycle is a room opening up rather than a corridor
# creeping forward, and is worth interrupting for.
NARRATION_ROOM_M2 = 4.0

PIVOT_MIN_SCAN_SHIFT_MM = 40

# --- Language-guided navigation ("go to the door") -----------------------
# slam/navigate.py. Deliberately tighter than the exploration budgets: a
# mission wanders and is allowed to, but "go to the door" has a known
# destination, so a run that has driven twice the straight-line distance is
# lost rather than thorough, and stopping is the honest answer.

# How close counts as arrived. A place is remembered with a radius (default
# 0.75 m in slam/places.py) and that radius wins when it is set; this is the
# fallback. 0.45 m is a little over the body radius (0.22 m), so "at the door"
# means within reach of it, not on top of it.
GOTO_ARRIVE_M = 0.45

GOTO_BUDGET_M = 12.0          # metres driven before giving up
GOTO_BUDGET_S = 240.0         # wall clock
# One leg is at most MISSION_MAX_LEG_M (0.25 m), so 120 legs is 30 m of
# driving at the cap -- the distance budget is the real bound and this is the
# backstop for a robot turning on the spot without covering ground.
GOTO_MAX_LEGS = 120

# The least map she may try to place herself in before "go to X" refuses.
# Matching a sweep against a nearly-empty map does not fail loudly -- it
# returns a confident pose off a handful of cells, which is the worst possible
# outcome because everything downstream then drives to a wrong address. The
# saved state/map.npz was 2.2 KB on 2026-09-08, so this refuses today and is
# meant to: map the room first.
GOTO_MIN_MAP_M2 = 6.0

# --- Curiosity (step 9 §7) -----------------------------------------------
# curiosity.py. She glances at something that does not match anything she has
# seen in the last few frames. It only ever SUGGESTS a turn -- the caller drives
# through the guarded movement tools, so the safety interrupt is untouched.
CURIOSITY_GRID = (12, 16)      # rows, columns the frame is reduced to
CURIOSITY_MEMORY = 6           # frames kept as "recently seen"
# Mean absolute difference between standardised grids. Both frames are unit
# spread, so ~0.6 is "a different scene" and ~0.2 is "the same scene, moved".
CURIOSITY_THRESHOLD = 0.55
# Long enough that a look does not immediately justify another look: after
# turning, the view IS new, and without this curiosity is an infinite loop.
CURIOSITY_COOLDOWN_S = 20.0
# How far to turn when the change is at the very edge of the frame. A BEHAVIOUR
# KNOB, NOT A FIELD OF VIEW -- this camera's FOV has never been measured and
# deriving one would be the ToF mistake again. Small on purpose: a glance.
CURIOSITY_LOOK_MAX_DEG = 25.0

# --- brain-guided targeting during an autonomous run ---------------------
#
# The brain reads what the run's on-board detector can see and names one thing
# to approach; the run nudges toward it. See rl/brain_target.py.
#
# ⚠ THE SAME RULE AS CURIOSITY'S ABOVE, AND FOR THE SAME REASON. A detection's
# horizontal position is used as a FRACTION OF THE FRAME multiplied by the
# swing below -- never as geometry. rl/friday_vision.py does carry a
# CAM_HFOV_DEG, but it is a SPEC figure for the C270 that nobody has measured on
# this camera, and its own comment says "used ONLY ... for narration. Nothing
# steers on it." Steering on it would be wrong in a way nothing downstream could
# detect. Because the nudge is re-aimed on every detection (~1 Hz) the loop
# closes: an error in this number changes how SHARPLY she turns, never which
# way. If someone measures the real FOV, that is the moment to make it geometric.
TARGET_SWING_DEG = 25.0
# How long a target the brain named stays live without being renewed. Its own
# clock, deliberately -- the frontier planner would abandon it at 45 s
# (explorer.TARGET_MAX_AGE_S) and the supervisor at 25 s, neither of which knows
# the brain meant it.
TARGET_TTL_S = 90.0
# A sighting older than this stops steering. The detector runs at ~1 Hz, so this
# is a few missed frames -- long enough to survive one that dropped, short
# enough that she is never driving at where something used to be.
TARGET_SIGHTING_STALE_S = 3.0
# How often the run looks for a new command. One stat() per this many ticks, in
# a directory the loop already writes to.
TARGET_COMMAND_POLL_TICKS = 10
# How often the brain is allowed to pick a target, and the least it will settle
# for. The cap in brain_router (DAILY_SPEND_CAP_USD) is the hard limit; this is
# the thing that keeps an ordinary run nowhere near it.
TARGET_DECIDE_EVERY_S = 20.0
TARGET_MIN_SCORE = 0.55

# The least spread a reduced frame must have before it is worth comparing, in
# the units of the incoming frame (grey levels, so 0-255 for a camera). Below
# this it is treated as blank. NOT a tiny epsilon: standardising near-flat
# sensor noise divides by its own tiny spread and turns it into a convincing
# pattern -- measured at 0.80 against a 0.55 threshold, i.e. a dead camera
# would have looked like the most interesting thing in the room.
CURIOSITY_MIN_CONTRAST = 1.0

# ⚠ SET TO 1 = THE OLD SINGLE SLAM, WHICH IS WHERE THIS SITS UNTIL THE
# STICTION THRESHOLD IS MEASURED. Ramping is worth having -- Harsh feels the
# slam as a jerk -- but two attempts at it have now produced turns that were
# commanded and did not happen, and a robot whose turns silently fail is worse
# than a robot that turns roughly. The second attempt moved strongly for three
# pivots (123, 141, 84 mm of scan shift) and faded over the next three (50,
# 26, 25), which is the signature of an impulse sitting right on the threshold
# rather than clearly above it.
#
# What is missing is a NUMBER: the lowest PWM at which these motors actually
# break free, on this floor, with this battery. Guessing at it has cost two
# runs. Measure it, then set MOTOR_RAMP_FLOOR just above it and raise this
# back to 6.
MOTOR_RAMP_STEPS = 1
MOTOR_RAMP_RISE = 0.6        # fraction of the window spent rising to the peak

# ⚠ THE RAMP MUST NOT START AT ZERO, and the first attempt did. Below the
# stiction threshold these motors produce NO motion -- they buzz. Ramping up
# from 0 therefore spends its first steps in a dead band, contributing nothing
# but noise, and the impulse that actually reaches the wheels came out at
# roughly 73% of the old single slam. On the robot the first two pivots of a
# six-pivot test did not turn at all, and NOTHING in software noticed: the
# estimator agreed with odometry about a turn that never happened, because the
# truth was 30 degrees outside its search window. Harsh saw it; the test did
# not. scripts/pivot_test.py now has a gate for exactly this.
#
# So the ramp starts at a fraction of the pulse level that is already past the
# dead band and climbs from there. It is gentler than a single edge to the
# cap, and it does not waste any of the window below the threshold.
MOTOR_RAMP_FLOOR = 0.6       # ramp starts here, as a fraction of the pulse

# The hard ceiling on motor output, as a fraction of full PWM. EVERY path is
# capped by this in motor._clamp, so it is the real limit on how fast Friday
# can drive regardless of what calibration or set_speed ask for.
#
# Raised 0.6 -> 0.75 on 2026-08-25 with Harsh's explicit approval, because
# turning was weak and 0.6 capped output at 153/255 -- which meant turn_speed
# above ~180 did nothing at all. Deliberately short of full power: the L298N
# heats and the battery sags under load, and a brownout mid-drive could reset
# the Pi. If turning is ever still too weak, ASK before going higher.
#
# Raised 0.75 -> 0.85 on 2026-09-02, again with Harsh's explicit approval, after
# the ACT policy's stand runs. Ceiling duty goes 191 -> 217.
#
# RESOLVED 2026-09-03. Raising the ceiling invalidated rl/pi_deploy.py
# RobotConfig.wheel_speed_max, which is defined as the speed reached AT the
# ceiling duty:
#     duty = 255 * max_duty * (wheel_speed / wheel_speed_max)
# It has now been re-measured at duty 216 and is 0.403 m/s, not the 0.476 taken
# at the old duty-191 ceiling -- so the true speed was ~16% BELOW what the
# policy believed, and dead reckoning (which integrates the commanded value
# through the same linear map) was overstating distance by ~18%. See the comment
# on that field for the method and its corroboration.
# Watch `vcgencmd get_throttled` after loaded runs: bit 0 set means undervoltage
# and the ceiling is too high for this battery.
MOTOR_MAX_DUTY = 0.85
MOTOR_WATCHDOG_S = 0.5       # stop if no new command within this window

# ⚠ NOTHING ON THE MOTOR LINK MAY BLOCK FOREVER. THE STOP DEPENDS ON IT.
#
# Found 2026-08-26 with py-spy, on a server where every `stop` call hung: the
# front thread was parked in serial.flush() -> tcdrain(), with four more
# stacked behind it waiting for the port lock. tcdrain waits for the UART to
# physically drain and HAS NO TIMEOUT -- write_timeout does not apply to it --
# so when the FTDI stopped draining (the whole USB hub had re-enumerated
# minutes earlier) the emergency stop blocked indefinitely. During a move,
# guarded_move's finally-block stop() would hang exactly the same way and the
# wheels would keep turning until the Arduino's own watchdog cut them. That is
# the most likely explanation for Friday bumping something.
MOTOR_WRITE_TIMEOUT_S = 0.5  # a write that cannot go out fails instead of hanging

# How long any one caller waits for the port. A wedged command must not be
# able to hold the emergency stop out of the port forever -- better to fail
# fast and let the Arduino watchdog cut the motors than to queue behind it.
MOTOR_LOCK_TIMEOUT_S = 1.0

# --- Face ----------------------------------------------------------------
# The 7" HDMI panel (EDID "Mediatrix Peripherals Inc MPI7002") reports
# 1920x1080 natively -- confirmed via wlr-randr 2026-08-23. friday_face.py
# lays itself out proportionally from the real screen size, so this is
# informational rather than something it reads.
FACE_DISPLAY_SIZE = (1920, 1080)  # 7" front display, native mode
FACE_DEFAULT_EMOTION = "neutral"
FACE_FPS = 30
# How long the face takes to actually SHOW a change, end to end. It is a separate
# process that polls its state file every POLL_S (0.08 s) and then cross-fades over
# TRANSITION_S (0.28 s) -- both in friday_face.py, which cannot be imported here
# because it pulls in pygame. Keep this in step with those two by hand if they change.
#
# tts.speak() subtracts it from the end of the LAST sentence so the fade back out of
# the speaking face LANDS on the end of the sound instead of starting there. Harsh:
# "the speaking animation lasts a little longer when it stops speaking". At the start
# of a sentence the same lag is already absorbed by pw-play opening its stream, so it
# is only the tail that needs anticipating.
FACE_SETTLE_S = 0.36
# How long pw-play takes to open its stream before the first sample is audible.
# MEASURED 2026-09-03 over four back-to-back plays of the same 2.784 s WAV:
# 0.177, 0.204, 0.201, 0.210 s beyond the file's own duration. It is the stream
# setup, not the file, so it is a constant offset rather than a rate -- which is
# why it can be subtracted once rather than modelled.
#
# It applies to BOTH ends. tts.speak() adds it back when anticipating the end of the
# last sentence, because the audio it is anticipating starts this much after play()
# is called. Re-measure it if AUDIO_PLAY_LATENCY or the sink changes.
AUDIO_START_LAG_S = 0.20


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

# ⚠ ASK THE CAMERA FOR AN INTERVAL IT ACTUALLY OFFERS. The down camera
# (349c:2317) enumerates ONLY 30 and 25 fps, at 640x480 or 320x240, in both
# YUYV and MJPG -- v4l2-ctl --list-formats-ext says so. Asking it for 10
# means asking for a frame interval it does not have; v4l2 does not
# necessarily refuse, it negotiates something else, and this is a device that
# has spent the evening dropping off the USB bus. Give it one of its own
# numbers instead. 25 rather than 30 because it is the cheaper of the two and
# this Pi is CPU-bound.
CAM_SECOND_FPS = 25

# HOW LONG ffmpeg STAYS ALIVE AFTER THE LAST VIEWER LEAVES.
#
# It used to be zero: the last viewer closing the page killed ffmpeg, and the
# next one starting it re-opened the device. Every page refresh, tab switch or
# dropped mobile connection was therefore a full open/close cycle on a UVC
# device -- and this camera dies during exactly those. Measured: it vanished
# from the bus THE INSTANT its stream was released, 0 seconds later.
#
# Lingering costs one idle ffmpeg for half a minute and removes most of the
# churn, because a refresh reconnects inside the window and finds the stream
# still running. Set to 0 for the old behaviour.
DASHBOARD_CAM_LINGER_S = 30.0
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

# --- Screen apps ---------------------------------------------------------
# Small full-screen views -- clock, timer, alarm, weather -- that take the 7"
# panel for as long as they have something to say and hand it straight back to
# the face. They run INSIDE the face process (it owns the only pygame window);
# everything else drives them by writing state files, exactly like emotion.txt.
# See app_control.py for the doorway and apps/ for the views themselves.
APPS_ENABLED = True

# The command channel and the schedule. Written atomically by app_control.py,
# polled by the face at APP_POLL_S. The schedule holds alarms and timers, so
# both survive a restart of the face, the MCP server, or the whole Pi.
APP_COMMAND_FILE = ROOT / "state" / "app_command.json"
APP_STATUS_FILE = ROOT / "state" / "app_status.json"
APP_SCHEDULE_FILE = ROOT / "state" / "schedule.json"
APP_POLL_S = 0.08            # same as the face's own state poll -- an os.stat

# An alarm that came due while the Pi was off still rings if it is no more
# than this late. Older than that and it is marked missed rather than going
# off at breakfast for something wanted at midnight.
APP_ALARM_GRACE_S = 300.0

# How long an alarm rings before it gives up and returns the screen.
APP_ALARM_MAX_RING_S = 120.0

# The ring itself. Generated as a WAV on first use and played through pipewire
# (pw-play), the same path as the wake beep. Deliberately not a klaxon; this is
# a companion, not a fire panel.
#
# ⚠ IT USED TO BE TWO PURE SINE TONES, AND THAT IS WHY IT SOUNDED LIKE A
# MICROWAVE. A sine has no partials, so the ear gets no material to identify
# -- it reads as "a beep", never as "a note". A phone ringtone is recognisable
# because it is a STRUCK instrument: a fundamental plus a stack of partials
# that each decay at their own rate, the high ones dying first. That decay
# profile is the whole trick. So the synthesis below is additive, per-partial,
# and the phrase is a real arpeggio rather than two tones.
APP_RING_FILE = ROOT / "state" / "ring.wav"

# Rendered AT THE SINK'S RATE ON PURPOSE. It was 44100 while the card runs
# 48000, so every single repeat was resampled in realtime on a Pi that has no
# CPU to spare -- the same class of fault as the face's phantom mixer stream.
APP_RING_RATE = AUDIO_SINK_RATE

# The phrase, as note names. "-" is a rest. C6-E6-G6-E6 is a plain major
# arpeggio: it rises, it resolves, and it is unmistakable through a wall
# without being shrill. Keep it short -- the gap is what makes it a ring
# rather than a song.
APP_RING_PHRASE = ("C6", "E6", "G6", "E6", "-", "G6", "C7", "-")
APP_RING_NOTE_MS = 165       # how often a new note is STRUCK, not how long it
                             # sounds -- notes ring on into the next through
                             # APP_RING_DECAY_S, which is what makes it sound
                             # like an instrument instead of a sequence.
APP_RING_DECAY_S = 0.62      # how long one struck note takes to fade out
APP_RING_GAP_MS = 900        # silence after the phrase, before it repeats

# The timbre, as (frequency ratio, amplitude, decay multiplier) per partial.
# The ratios are slightly INHARMONIC above the octave -- real bells and bars
# are, and exact integer ratios are what make additive synthesis sound like a
# cheap organ. The decay multiplier is the important column: 6.0 on the top
# partial means it is gone in a tenth of the time the fundamental takes, and
# that fast-dying shimmer is heard as "struck".
APP_RING_PARTIALS = (
    (1.00, 1.00, 1.0),
    (2.00, 0.42, 1.9),
    (3.01, 0.20, 3.1),
    (4.17, 0.11, 4.5),
    (5.43, 0.055, 6.0),
)
APP_RING_STRIKE_MS = 4.0     # attack. Not zero -- a square edge on a sine is a
                             # click, and a click reads as a fault.
APP_RING_SPREAD_HZ = 0.7     # a second voice detuned by this much, panned the
                             # other way. Gives the ring a little width and
                             # stops it sounding like it is inside a box.

APP_RING_VOLUME = 0.55       # 0-1, before the sink's own volume. Was 0.35,
                             # which Harsh could not hear from the next room.
                             # The finished phrase is PEAK-NORMALISED to this,
                             # so raising it makes the ring genuinely louder
                             # instead of just adding partials on top of each
                             # other until they clip.
APP_RING_CEILING = 0.93      # hard cap on that normalisation, whatever
                             # APP_RING_VOLUME says. Same value as TTS_CEILING.

# Screensaver: with nothing else on screen and the face asleep this long, the
# clock comes up on its own. Any new expression or command takes it straight
# back down. 0 disables.
#
# OFF SINCE 2026-09-12. Harsh: "after sleeping mode the clock comes ... remove
# the clock after it reaches sleeping face, I will activate from dashboard when
# needed and will be in sleeping mode only". So she now STAYS on the sleeping
# face instead of turning into a wall clock. The was-90 s argument above --
# that closed eyes for an hour look switched off -- is his call to make, and he
# made it.
#
# THIS DOES NOT AFFECT LAUNCHING THE CLOCK YOURSELF. The dashboard's Screen
# panel, app_control.py and the voice command all call apps.launch() directly;
# only the automatic one is gated here. A clock launched by hand also STAYS up,
# because _end_screensaver() only dismisses one the screensaver itself started.
APP_CLOCK_SCREENSAVER_AFTER_S = 0

# 12- or 24-hour clock, everywhere a time is shown or spoken back.
APP_CLOCK_24H = False

# A timer or alarm label is drawn under the digits; anything longer is cut.
APP_LABEL_MAX_CHARS = 28

# How long a weather card stays up before handing the screen back.
# 0 = it never hands it back. Harsh, 2026-09-12: "he wants the raining cloud
# and the readings up, not gone in half a minute" -- so it was 0, and with the
# card finally being triggered by voice and by the dashboard he saw what 0
# actually means. Harsh, 2026-09-19, after asking her for the weather: "after
# showing the on screen and teling after it gets stuck in weather animation it
# should go back to neutral". His live word wins over the old one. 25 s is the
# middle of the two: the line takes about four seconds to say, the card is
# there to be looked at afterwards, and then her face comes back on its own.
# app_control.py --dismiss still puts it away early; a safety face or a ringing
# alarm draws over it; any other expression yields to it.
APP_WEATHER_HOLD_S = 25.0

# THE ARTWORK, NOT THE READING. Harsh asked for the cloud-with-rain animation
# whatever the sky is doing, so this forces the icon and NOTHING else: the
# place, the temperature, the condition word, the high and low, the wind and
# the humidity all still come from weather_source.py and stay true. Set to None
# and the icon follows the real condition again (sun, mist, storm, snow).
APP_WEATHER_FORCE_ICON = "rain"

# --- weather without a key -------------------------------------------------
# apps/weather.py draws the card and holds no key ON PURPOSE: the brain has web
# search and hands the numbers over (mcp_server/tools/screen.show_weather).
# That path stays and is still the better one -- it can answer "will it rain
# before I walk home", this can only report the sky. Harsh asked on 2026-09-12
# for the card to work without the brain too, so weather_source.py fills the
# same dict from wttr.in, which needs no key and no account.
#
# WITH NO PLACE SET THE SERVICE LOCATES BY IP, which means the request tells it
# the Pi's public address. That is what keyless geolocation costs. Name a city
# here and nothing is inferred from the address.
APP_WEATHER_FETCH = True
# ⚠ NAMED, NOT INFERRED. Left as None this located by IP, and on 2026-09-19
# two fetches four minutes apart came back "Chhindwara" and then "Connaught
# Place" -- Delhi, 800 km away -- because the ISP's egress moved. A weather
# card is only worth having if it is the sky outside THIS window, so the town
# is written down. Another city is still a fair question: the brain has web
# search and show_weather draws whatever it found.
APP_WEATHER_PLACE = "Chhindwara"
APP_WEATHER_URL = "https://wttr.in/{place}?format=j1"
APP_WEATHER_USER_AGENT = "friday-robot (weather card)"
APP_WEATHER_TIMEOUT_S = 12.0
# The sky does not change between two "show me the weather"s, and wttr.in is
# free and rate-limits. Ten minutes, and a cache older than this is refetched
# rather than served stale.
APP_WEATHER_CACHE_S = 600.0

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
# 250 -> 200 on 2026-09-13, Harsh: "decrease the safety threshold a little bit".
SAFETY_STOP_MM = 200         # stop this far short of anything ahead
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
# File 08 step 4: "pull 1-2 relevant recent entries into context so Friday
# references shared history instead of responding cold". Facts attached per
# turn -- on the fast path too, which answers most of what is said to her and
# had no memory at all. MEMORY_RECALL_LIMIT (5) stays for the recall TOOL, a
# deliberate lookup. Measured warm: 9-41 ms per turn.
MEMORY_CONTEXT_FACTS = 2

# Attach what she can already see about herself -- the time, her mood, whether
# a run is driving, the last clearance around her, the weather -- to each
# question the same way, and for the same reason. It saves a whole round trip
# on everything she is most often asked about herself: "are you exploring?"
# used to cost a tool call plus a second request to answer "no", about seven
# seconds, and the answer was in a file on disk the whole time.
#
# ⚠ EVERY FIELD IN situation.py CARRIES A MAXIMUM AGE, and that is the part
# that matters. state/rl_autonomy.json is written only while a run drives; left
# ungated it would have her describe this afternoon's wall as if it were in
# front of her now. Stale state spoken in the present tense is worse than no
# state at all, so nothing past its age is emitted.
SITUATION_IN_CONTEXT = True


# =====================================================================
# SLAM — the room map
# =====================================================================
# Pure Python on numpy/scipy. ROS2 was evaluated and rejected: there are no
# ROS2 binaries for Debian 13 (trixie), so it would mean a third-party apt
# repo or a source build -- exactly the class of change CLAUDE.md Part 1
# forbids -- and slam_toolbox alone wants about a core on a Pi that is already
# at load 2.3 of 4. Harsh approved this approach on 2026-08-26.

# Map resolution. 5 cm is fine enough to plan a path through a doorway and
# coarse enough that a 12 x 12 m room is only 240 x 240 cells.
SLAM_CELL_M = 0.05

# How far the likelihood field is smeared. This is the matcher's tolerance for
# "nearly right": too small and the score is a field of spikes with nothing to
# optimise, too large and every pose looks equally good. Two to three cells is
# the usual choice.
SLAM_BLUR_M = 0.12

# The matcher searches at SLAM_BLUR_M and then reads the answer out at this
# much finer resolution. A blur wide enough for a safe global search is also
# wide enough to make a 25 cm move look plausible in almost any direction --
# measured on the real robot, 127 of 360 bearings within 75% of the peak at
# 12 cm blur, versus a 13.6x peak-to-median ratio at 2 cm on the same data.
SLAM_FINE_CELL_M = 0.01
SLAM_FINE_BLUR_M = 0.02

# A match scoring below this is not believed, and the odometry prior is used
# unchanged rather than being corrected by a bad answer.
SLAM_MIN_SCORE = 0.15

# And a peak this many times the median across all candidates, or the geometry
# did not really constrain the answer however high the peak looked.
SLAM_MIN_CONFIDENCE = 5.0


# --- The occupancy map ---------------------------------------------------
# The map is kept in LOG-ODDS, not in probabilities. Two reasons, and the
# second is the one that matters on a robot: adding is cheaper than the
# Bayes product, and clamping the total is what lets the map CHANGE ITS MIND.
# A cell seen as a wall two hundred times would otherwise need two hundred
# contradicting looks to become free again -- so a chair that moved would stay
# on the map as a wall for the rest of the day.
SLAM_MAP_FILE = ROOT / "state" / "map.npz"

# One observation's worth of evidence. Deliberately asymmetric: a hit is
# stronger evidence than a pass-through, because a beam that goes past a cell
# only proves nothing was there IN THAT INSTANT at that exact height, while a
# return is a surface. p_occupied = 0.7, p_free = 0.4.
SLAM_LOGODDS_HIT = 0.85
SLAM_LOGODDS_MISS = -0.40

# The clamp. About four consistent looks either way saturates a cell. Measured
# in scripts/test_slam.py: an obstacle mapped from six poses and then taken
# away is gone from the map after TWELVE sweeps of it -- rather than never.
SLAM_LOGODDS_MIN = -2.0
SLAM_LOGODDS_MAX = 3.5

# Reading the map back out. Between the two it is UNCERTAIN, and uncertain is
# never treated as driveable -- the same rule as a null clearance in the
# safety interrupt, for the same reason.
SLAM_P_OCCUPIED = 0.65
SLAM_P_FREE = 0.35

# A hard cap on how far the map may grow from where it started, so a bad pose
# estimate or a mirrored corridor cannot allocate the Pi to death. 24 m square
# at 5 cm is 480 x 480 cells -- under a megabyte, and larger than the flat.
SLAM_MAP_MAX_M = 24.0

# --- Transient obstacles vs the permanent map ----------------------------
# A person walking past, or a chair moved this morning, must NOT become part
# of the room. So a return does not go into the permanent map on sight: the
# cell has to be hit in this many SEPARATE sweeps first. Below that the hit is
# real and is avoided -- it is just not believed to be furniture yet.
#
# Three sweeps at the mapping rate is about three seconds. A person standing
# still that long does get mapped; they also get UNMAPPED about twelve sweeps
# after they walk off, because the beams that then pass through carve the cell
# free. That asymmetry is deliberate -- being slow to add and slow to forget
# is how a map stays useful without lying about a doorway that is blocked.
SLAM_CONFIRM_SCANS = 3

# How long an unconfirmed return keeps blocking the path. This is an
# AVOIDANCE lifetime, not a map lifetime: it only has to outlast the gap
# between sweeps, because a thing still there will be seen again.
SLAM_TRANSIENT_TTL_S = 4.0

# --- Pose tracking -------------------------------------------------------
# Odometry is the prior and the scan matcher corrects it. There are no wheel
# encoders, so the prior is a TIMED conversion (state/calibration.json) worth
# about +/-20% -- the search window has to be wide enough to contain that
# error over one sweep interval, and no wider, because every extra centimetre
# is candidates to score.
SLAM_POSE_WINDOW_M = 0.20
SLAM_POSE_WINDOW_DEG = 12.0

# How fine the refinement pass may answer in ANGLE. It was hardcoded at 1.0,
# and a whole degree is not a rounding detail here: the matcher can then only
# answer 0 or +/-1, it is not unbiased between them, and every +1 it picks is
# written into the map -- which then leans a degree, which makes the next +1
# look right. Twelve sweeps of that with the robot BOLTED TO THE FLOOR came
# out as +5 degrees. At 3 m a degree is 5 cm, one whole cell, which is also
# enough to spread a wall across two cells and break it into fragments.
#
# ⚠ LOWERING THIS WAS TRIED AND IS WORSE. 0.25 deg, replayed over three
# recorded pivot runs: heading error 3.3 -> 3.7 deg, map fit 0.69 -> 0.64,
# and 117 -> 245 ms per sweep on this Pi. Finer answers are not better
# answers -- the field is blurred over SLAM_BLUR_M = 12 cm, and at 3 m a
# quarter degree moves a point 1.3 cm, so the extra candidates are reading
# the smoothing rather than the room. It is left configurable so the next
# person can re-measure rather than re-argue, but 1.0 is the measured best.
#
# The ratchet it was meant to fix is fixed by SLAM_HOLD_WHEN_STILL instead.
SLAM_FINE_STEP_DEG = 1.0

# Do not let the matcher correct a pose the robot was never told to change.
#
# Standing still, the correction the matcher wants is smaller than the step
# it can express, so it answers 0 or +/-1 degree -- and it is not unbiased
# between them. Each +1 is written into the map, the map then leans a degree,
# and the next +1 looks right. Twelve stationary sweeps ratcheted +5 degrees
# with the robot bolted to the floor.
#
# Measured over three recorded pivot runs: heading error 3.3 -> 0.7 deg, and
# it wins on EVERY run (5/2/3 -> 1/1/0). Map fit is unchanged within noise
# (0.69 vs 0.64, against a 0.15 run-to-run spread). It is also CHEAPER --
# 117 -> 71 ms per sweep -- because a stationary sweep skips the search.
#
# THE TRADE: if something shoves Friday while she is not driving, she will
# not notice until her next commanded move. That is the right way round --
# an unbidden shove is rare, and inventing rotation every single sweep is
# not. Transient obstacles are still tracked either way.
SLAM_HOLD_WHEN_STILL = True

# Matching against the accumulated MAP rather than the previous sweep is what
# stops drift compounding: the previous sweep has already drifted, the map has
# been averaged over many. Rebuilding the map's likelihood field costs a blur
# over the whole grid, so it is cached and rebuilt only this often.
SLAM_FIELD_REBUILD_SCANS = 5


# ⚠ POSE MATCHING NEEDS ITS OWN SHARPNESS GATE, NOT SLAM_MIN_CONFIDENCE.
# That one is 5.0 because match_translation scores 360 bearings, most of them
# badly wrong, so the median is low and a true peak towers over it. match_pose
# only ever scores a 20 cm box around the prior, where EVERY candidate is
# nearly right and the median is therefore high -- a correct match on the test
# room scores 2.1 to 3.7 here, so a gate of 5.0 would reject every good answer
# it was ever given. Near 1.0 still means the room did not constrain it: a
# bare corridor has a ridge along its length rather than a peak.
SLAM_POSE_MIN_SHARPNESS = 1.4

# Scan-to-map matching needs a map worth matching against. Below this many
# occupied cells the room is still mostly a rumour, so odometry is kept.
SLAM_MATCH_MIN_CELLS = 150

# And a match needs this many beams landing on ground the map has an opinion
# about. Without a floor, a pose that puts almost the whole sweep into
# unmapped space can score 1.0 on the three beams that happened to land on a
# known wall -- a perfect score computed from nearly nothing.
SLAM_MATCH_MIN_BEAMS = 40

# ⚠ THE MOTION PRIOR IS NOT OPTIONAL ONCE THE KNOWN MASK IS ON. Excusing the
# beams that land in unmapped space is what stops the matcher freezing during
# exploration -- but it also excuses a WRONG pose that throws its beams out
# past the walls into the unknown, and in a fully mapped room that cost 2.4 cm
# of accuracy turning into 41 cm, measured. The answer is not to drop the mask
# but to stop pretending odometry told us nothing: candidates are scored by
# likelihood TIMES a Gaussian on how far they stray from the odometry prior,
# which is the posterior rather than the likelihood.
#
# The width of that Gaussian is NOT a constant, because odometry's uncertainty
# is not: a sweep taken after a 25 cm hop deserves a much wider prior than one
# taken standing still. So it scales with the move that was just reported --
# SLAM_ODOM_ERROR of it, the same +/-20% state/calibration.json is worth --
# with a floor for the error that does not come from moving at all (the pose
# it started from was already approximate).
SLAM_ODOM_ERROR = 0.20
SLAM_POSE_PRIOR_FLOOR_M = 0.04
SLAM_POSE_PRIOR_FLOOR_DEG = 3.0

# A keyframe -- pose plus the sweep taken there -- is kept every time the
# robot has moved this far or turned this much since the last one. They are
# what loop closure re-matches against, and what the map is rebuilt from when
# a loop closes.
SLAM_KEYFRAME_M = 0.30
SLAM_KEYFRAME_DEG = 25.0
SLAM_KEYFRAME_MAX = 400          # ~120 m of travel; caps memory on the Pi

# --- Loop closure --------------------------------------------------------
# ⚠ SCAN-TO-MAP MATCHING CANNOT SEE ITS OWN DRIFT, and measuring that was the
# whole reason this exists. Over a 46 m exploration run the matcher reported
# "matched" with scores of 0.6-1.0 and sharpness comfortably over the gate on
# essentially every sweep -- while applying a correction of 0.000 m. It was
# not failing. The map is BUILT at the drifted pose, so the next sweep fits it
# perfectly, and the error is invisible from the inside. The flat came out
# 157% of its real floor area: not offset, which would be harmless, but
# stretched, because the corridor mapped early and the rooms mapped late no
# longer agreed with each other.
#
# Drift only becomes observable on RETURNING to geometry that was mapped
# before it accumulated. That is the whole trick, and it is why these
# thresholds are stricter than the per-sweep ones: a false loop closure does
# not nudge the pose, it rewrites the entire map around a lie.
SLAM_LOOP_RADIUS_M = 2.0         # "near a place we have been"
# ...but not one we were just at. Measured in TRAVEL, not keyframes: a
# keyframe is also cut every 25 degrees of turn, so turning on the spot lays
# down a dozen of them without going anywhere at all.
SLAM_LOOP_MIN_TRAVEL_M = 4.0
SLAM_LOOP_WINDOW_M = 1.0         # search wide: the drift is what we are hunting
SLAM_LOOP_WINDOW_DEG = 30.0
SLAM_LOOP_MIN_SCORE = 0.45       # vs SLAM_MIN_SCORE 0.15 for a routine sweep
SLAM_LOOP_MIN_SHARPNESS = 2.0    # vs SLAM_POSE_MIN_SHARPNESS 1.4
SLAM_LOOP_MIN_ERROR_M = 0.10     # below this there is nothing worth rewriting


# =====================================================================
# NAVIGATION — planning a path across the map
# =====================================================================
# The robot's own radius, and it is not a guess: the LiDAR sees Friday's own
# chassis at a rock-steady 210 mm on bearings 89-98 (LIDAR_SELF_MASK). That
# return is the widest part of the body measured from the sensor, which is
# exactly the number a planner needs.
NAV_ROBOT_RADIUS_M = 0.21

# Extra room on top, because the pose estimate is worth a few centimetres and
# a wall does not move out of the way. Obstacles are inflated by radius +
# clearance, so the robot can be planned as a POINT — the alternative is
# checking a footprint at every cell of every candidate path.
NAV_CLEARANCE_M = 0.09

# ⚠ INFLATION CAN SEAL A DOORWAY SHUT. At 30 cm of inflation a 70 cm doorway
# leaves a 10 cm gap, and one badly-placed cell closes it. If Friday starts
# refusing to plan through doors that are plainly wide enough, this is the
# number to look at first -- not the planner.
NAV_INFLATE_M = NAV_ROBOT_RADIUS_M + NAV_CLEARANCE_M

# Diagonal steps are only allowed when both orthogonal neighbours are clear,
# so a path cannot squeeze through the corner between two obstacles that
# touch. Cheap to enforce, and the failure it prevents is a wedged robot.
NAV_ALLOW_DIAGONAL = True

# ⚠ UNCONFIRMED RETURNS ONLY BLOCK PLANNING NEAR THE ROBOT. Every surface that
# has just come into view is unconfirmed -- that is what the confirmation gate
# MEANS -- so a robot exploring new space sees hundreds of "unexplained"
# returns per sweep, all of them ordinary walls the map has not finished
# believing in yet. Inflating that lot globally walls the robot into whatever
# it can already see: measured on the test flat, reachable cells collapsed
# from 12,423 to 3,613 and exploration stopped in the corridor insisting the
# flat was fully mapped.
#
# So the transient layer constrains the plan only within this radius, which is
# exactly the job it was built for: not driving into the person who is HERE.
# Further out the map is in charge, and anything real out there will have been
# seen many more times before the robot arrives.
NAV_LOCAL_OBSTACLE_M = 1.5


# =====================================================================
# FRONTIER EXPLORATION — the boundary between what is known and what is not
# =====================================================================
# A frontier is a cell the map knows is FREE that touches a cell it knows
# nothing about. Driving to one is the only move that is guaranteed to reveal
# something new, which is why exploration is done this way and not by picking
# a direction: random headings re-cover the same floor and never finish.

# Frontier groups smaller than this are ignored. Below a few cells they are
# usually the ragged edge of the sensor's own range or a gap behind a chair
# leg, and chasing them makes the robot fidget instead of explore.
FRONTIER_MIN_CELLS = 5

# ⚠ CHOOSE BY NEW AREA PER METRE TRAVELLED, NOT BY EITHER ALONE. Nearest-first
# leaves the robot nibbling at the edge of the room it is in; biggest-first
# sends it across the flat and back for every scrap. The score is
#
#     expected new cells / (path distance + FRONTIER_NEAR_M)
#
# and the constant is what stops a frontier one centimetre away scoring
# infinity. One metre means "a frontier 1 m away must reveal twice as much as
# one at 0 m to be worth the same".
FRONTIER_NEAR_M = 1.0

# Expected new area is counted as the UNKNOWN cells within this radius of the
# frontier, rather than the frontier's own width -- a two-cell gap into a
# whole unmapped room is worth far more than a wide frontier against the
# range limit in open floor, and width alone cannot tell them apart.
FRONTIER_GAIN_RADIUS_M = 1.5

# A frontier the robot has failed to reach this many times is dropped for the
# rest of the run. Some frontiers are real but unreachable -- behind a chair
# leg the inflation seals, or across a threshold too narrow -- and retrying
# forever is the classic way exploration livelocks.
FRONTIER_MAX_ATTEMPTS = 3

# ⚠ ONE CONNECTED BOUNDARY IS NOT ONE DESTINATION. Frontier cells are found by
# connected components, and early in a run the whole known/unknown boundary is
# a single arc -- measured on the test flat, ONE cluster of 559 cells. Its
# centroid is a point in the middle of everything, so "drive to the frontier"
# means drive to nowhere in particular, and the run stalls in the first room
# while insisting it is exploring. So any arc longer than this is chopped into
# chunks about a doorway wide, each of which is somewhere a robot can actually
# be sent.
FRONTIER_SPLIT_M = 0.75
FRONTIER_MAX_CELLS = 40


# =====================================================================
# ANTI-BUMP — the backstops that do not depend on knowing which way is forward
# =====================================================================
# ⚠ FRIDAY BUMPED SOMETHING ON 2026-08-26 WHILE THE ToF SAID 4000 mm CLEAR.
# Both halves of that are true and neither was a lie:
#
#  - The bumper ToF has a BLIND BAND from ~0.31 m to ~0.75 m for anything not
#    tall enough to reach its ahead-rows (see TOF_FLOOR_FROM_ROW). Its
#    down-rows hit floor nearer than that; its ahead-rows look over it. An
#    intact floor with nothing above it legitimately reports "clear".
#  - Reversing is completely blind. There is no rear sensor at all.
#
# So neither the ToF nor the movement gate could have prevented it, and the
# fix cannot live in either. These two backstops run during EVERY translation
# and need no knowledge of LIDAR_FORWARD_DEG, which is the point -- they had
# to work before the bearing was measured, not after.

# 1. PROXIMITY. Anything this close in ANY direction stops a translation. The
# LiDAR sees 360 degrees, so this does not care where the nose is. It can only
# ever stop a move sooner -- it never permits one the gate refused. Turns are
# exempt on purpose: a pivot sweeps only the robot's own footprint, and
# blocking turns on side clearance would leave a robot facing a wall with no
# legal way out of the room.
LIDAR_BUMP_STOP_MM = 200

# 2. STALL. If the wheels are turning but the world is not moving past the
# LiDAR, Friday is pushing against something -- which is exactly what a bump
# looks like from the inside, and it catches obstacles NEITHER sensor can see.
# At 18.5 cm/s she covers ~9 cm in half a second, which moves ranges by far
# more than the D500's few millimetres of jitter.
SAFETY_STALL_AFTER_S = 0.5   # give the motors time to actually start
SAFETY_STALL_MM = 15         # median range change below this means stuck
