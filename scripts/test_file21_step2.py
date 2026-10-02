#!/usr/bin/env python3
"""File 21 step 2: area_store -- areas, home anchor, missions, checkpoints.

    ./venv/bin/python scripts/test_file21_step2.py

Network blocked. Every database and file is a temp one: the real memory.db,
the real places table and the navigation bundle are never touched (checked at
the end). The crash tests SIGKILL real child processes mid-write: the JSON
checkpoint must always load as the old or the new version, and the SQLite
tables must stay consistent (integrity_check, and every area saved in a
mission is in that mission's areas_new and nowhere else).
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402

REAL_DB = Path(config.MEMORY_DB)
TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s2-"))
config.MEMORY_DB = TMP / "places_memory.db"           # slam.places -> temp

from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import area_store as st        # noqa: E402
from exploration_mission import durable                 # noqa: E402

PY = sys.executable
_results: list[bool] = []
SQUARE = [[0, 0], [4, 0], [4, 3], [0, 3]]               # a 12 m2 room
FAR_SQUARE = [[10, 10], [13, 10], [13, 13], [10, 13]]


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:                                     # noqa: F841
        return e or True
    except Exception:                                   # noqa: BLE001
        return False
    return False


def wal_db(path: Path, with_memory_table: bool = True) -> Path:
    """A DB shaped like memory.db: WAL, and some of Friday's own tables."""
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    if with_memory_table:
        con.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, "
                    "text TEXT)")
        con.execute("INSERT INTO facts (text) VALUES ('Harsh likes tea')")
    con.commit()
    con.close()
    return path


def store(name: str, places=None, **kw) -> st.AreaStore:
    db = wal_db(TMP / f"{name}.db")
    s = st.AreaStore(db_path=db, data_dir=TMP / f"{name}-data", places=places,
                     **kw)
    s.ensure_schema()
    return s


# =============================================================================
print("--- names (spec 3.5, 4.7) ---")

cases = [("  The Kitchen ", "kitchen", "Kitchen"), ("my room", "room", "room"),
         ("Harsh's room", "harsh's room", "Harsh's room"),
         ("our  living   room", "living room", "living room"),
         ("kitchen!", "kitchen", "kitchen"), ("Harsh’s room", "harsh's room",
                                              "Harsh's room"),
         ("home office", "home office", "home office"),
         ("pooja ghar", "pooja ghar", "pooja ghar"), ("area 3", "area 3", "area 3"),
         ("the the study", "study", "study")]
for text, name, display in cases:
    got = st.validate_name(text)
    check(f"{text!r} -> {name!r} / {display!r}", got == (name, display), str(got))
for text in ("", "   ", "a", "the", "home", "save", "stop", "Friday", "what",
             "kya", "123", "the big room near the main door",
             "x" * 31, "kitchen/bath", "room_1"):
    check(f"{text!r} is refused", bool(raises(lambda t=text: st.validate_name(t),
                                              st.InvalidName)))
check("unnamed numbering is parsed", st.unnamed_number("area 12") == 12
      and st.unnamed_number("area") is None and st.unnamed_number("kitchen") is None)
check("point in polygon", st.point_in_polygon(1, 1, SQUARE)
      and not st.point_in_polygon(5, 1, SQUARE))


# =============================================================================
print("--- schema: backup first, migrate, journal mode untouched ---")

db = wal_db(TMP / "schema.db")
s0 = st.AreaStore(db_path=db, data_dir=TMP / "schema-data")
check("a DB without our tables is version 0", s0.schema_version() == 0)
check("ensure_schema -> version 1", s0.ensure_schema() == 1)
check("memory.db was backed up before the tables were added",
      s0.backup_file is not None and s0.backup_file.exists())
bcon = sqlite3.connect(str(s0.backup_file))
check("...and the backup holds Friday's memory, without our tables",
      bcon.execute("SELECT text FROM facts").fetchone()[0] == "Harsh likes tea"
      and not bcon.execute("SELECT 1 FROM sqlite_master WHERE name='areas'")
      .fetchone())
bcon.close()
first_backup = s0.backup_file
check("a second ensure_schema does nothing and backs up nothing",
      s0.ensure_schema() == 1 and s0.backup_file == first_backup
      and len(list(TMP.glob("schema.db.bak-*"))) == 1)
jm = sqlite3.connect(str(db)).execute("PRAGMA journal_mode").fetchone()[0]
check("the DB is still WAL (journal mode never changed)", jm == "wal", jm)
tables = {r[0] for r in sqlite3.connect(str(db)).execute(
    "SELECT name FROM sqlite_master WHERE type='table'")}
check("areas, home_anchor, missions and the meta table exist",
      {"areas", "home_anchor", "missions", st.META_TABLE} <= tables, str(tables))
syncmode = s0._connect().execute("PRAGMA synchronous").fetchone()[0]
check("our connection writes with synchronous=FULL", syncmode == 2, str(syncmode))
fresh = st.AreaStore(db_path=TMP / "fresh.db", data_dir=TMP / "fresh-data")
fresh.ensure_schema()
check("a brand-new empty DB is not 'backed up'", fresh.backup_file is None)
s0._connect().execute(f"UPDATE {st.META_TABLE} SET value='9' "
                      "WHERE key='schema_version'")
check("a DB from a newer version is refused, not overwritten",
      bool(raises(s0.ensure_schema, st.StoreError)))
s0._connect().execute(f"UPDATE {st.META_TABLE} SET value='1' "
                      "WHERE key='schema_version'")


# =============================================================================
print("--- areas: save, names, queue ---")

places = ad.FakePlaces()
s = store("areas", places=places)
m = s.start_mission("envA", "map-1", (0.1, 0.2, 0.3))
k = s.add_area("envA", "map-1", polygon=SQUARE, center=(2, 1.5), area_m2=12.0,
               name="the Kitchen", source="voice", mission_id=m["id"])
check("a named area is saved with its name and display",
      k["name"] == "kitchen" and k["display_name"] == "Kitchen"
      and k["status"] == "named" and k["complete"] is True)
check("radius = clamp(sqrt(area/pi), 0.5, 2.0)",
      abs(k["radius_m"] - min(2.0, math.sqrt(12 / math.pi))) < 1e-9)
check("saved into the mission's areas_new in the same write",
      s.mission(m["id"])["areas_new"] == [k["id"]])
lm = places.find_exact("kitchen")
check("File 14 landmark created: kind area, centre, radius, map + area stamp",
      lm and lm["kind"] == "area" and (lm["x"], lm["y"]) == (2.0, 1.5)
      and "[map:map-1]" in lm["note"] and f"area:{k['id']}" in lm["note"],
      str(lm))
u1 = s.add_area("envA", "map-1", polygon=FAR_SQUARE, center=(11.5, 11.5),
                area_m2=9.0, name=None, source="auto")
u2 = s.add_area("envA", "map-1", polygon=FAR_SQUARE, center=(11.5, 11.5),
                area_m2=9.0, name=None, source="auto")
check("no answer -> 'area 1', 'area 2', unnamed",
      (u1["name"], u1["status"], u2["name"]) == ("area 1", "unnamed", "area 2"))
check("unnamed areas are not File 14 landmarks",
      places.find_exact("area 1") is None)
check("the naming queue holds the unnamed ones",
      [a["name"] for a in s.naming_queue("envA")] == ["area 1", "area 2"])
dup = raises(lambda: s.add_area("envA", "map-1", polygon=FAR_SQUARE,
                                center=(11, 11), area_m2=9, name="kitchen"),
             st.DuplicateName)
check("the same name elsewhere -> DuplicateName, suggesting 'kitchen 2'",
      isinstance(dup, st.DuplicateName) and dup.suggestion == "kitchen 2")
k2 = s.add_area("envA", "map-1", polygon=FAR_SQUARE, center=(11, 11), area_m2=9,
                name="kitchen 2")
dup = raises(lambda: s.add_area("envA", "map-1", polygon=FAR_SQUARE,
                                center=(11, 11), area_m2=9, name="Kitchen"),
             st.DuplicateName)
check("...and 'kitchen 3' once 'kitchen 2' is taken",
      isinstance(dup, st.DuplicateName) and dup.suggestion == "kitchen 3")
other_env = s.add_area("envB", "map-9", polygon=SQUARE, center=(2, 1.5),
                       area_m2=12, name="kitchen")
check("names are unique per environment, not globally",
      other_env["name"] == "kitchen" and other_env["env_id"] == "envB")
check("the DB itself refuses a duplicate name (unique index)",
      bool(raises(lambda: s._connect().execute(
          "INSERT INTO areas (id, env_id, name, status, center_x, center_y, "
          "radius_m, polygon, area_m2, source, created_at, updated_at, "
          "schema_version) VALUES ('x','envA','kitchen','named',0,0,1,'[]',1,"
          "'t','t','t',1)"), sqlite3.IntegrityError)))
check("by_name: exact, with 'the', never a substring",
      s.by_name("envA", "The kitchen")["id"] == k["id"]
      and s.by_name("envA", "kitch") is None)
s.add_alias(k["id"], "rasoi")
check("aliases resolve", s.by_name("envA", "rasoi")["id"] == k["id"])
s.forget(u1["id"])
u3 = s.add_area("envA", "map-1", polygon=FAR_SQUARE, center=(11, 11), area_m2=9)
check("unnamed numbering reuses the next FREE number",
      u3["name"] == "area 1", u3["name"])
check("a manual label before the area is finished is complete=False",
      s.add_area("envA", "map-1", polygon=FAR_SQUARE, center=(11, 11), area_m2=9,
                 name="study", complete=False)["complete"] is False)
s.count_visit(k["id"])
s.count_visit(k["id"])
s.count_obstacle(k["id"])
k_now = s.get(k["id"])
check("visit and obstacle counters", (k_now["visit_count"],
                                      k_now["obstacle_events"]) == (2, 1))


# =============================================================================
print("--- File 14 landmark sync ---")

p2 = ad.FakePlaces([{"name": "desk", "x": 1.0, "y": 1.0, "radius_m": 0.75,
                     "kind": "place", "note": "[map:map-1] saved by hand"},
                    {"name": "sofa", "x": 20.0, "y": 20.0, "radius_m": 0.75,
                     "kind": "place", "note": "[map:map-1] saved by hand"}])
s2 = store("sync", places=p2)
desk = s2.add_area("home", "map-1", polygon=SQUARE, center=(2, 1.5), area_m2=12,
                   name="desk")
check("Harsh's own place inside the polygon = the same place: linked",
      desk["hints"].get("linked_place") == "desk")
check("...and never moved or re-stamped",
      p2.find_exact("desk")["note"] == "[map:map-1] saved by hand"
      and (p2.find_exact("desk")["x"], p2.find_exact("desk")["y"]) == (1.0, 1.0))
chk = s2.check_name("home", "sofa", SQUARE)
check("Harsh's place of that name ELSEWHERE -> duplicate_landmark, 'sofa 2'",
      chk.kind == "duplicate_landmark" and chk.suggestion == "sofa 2")
check("check_name on a fresh name -> ok", s2.check_name("home", "garage",
                                                        SQUARE).kind == "ok")
check("check_name explains an invalid one",
      s2.check_name("home", "stop", SQUARE).kind == "invalid")
bed = s2.add_area("home", "map-1", polygon=FAR_SQUARE, center=(11.5, 11.5),
                  area_m2=9, name="bedroom")
s2.update_geometry(bed["id"], polygon=[[10, 10], [14, 10], [14, 14], [10, 14]],
                   center=(12.0, 12.0), area_m2=16.0)
check("growing map: our landmark follows the new centre",
      (p2.find_exact("bedroom")["x"], p2.find_exact("bedroom")["y"]) == (12.0, 12.0)
      and s2.get(bed["id"])["area_m2"] == 16.0)
s2.update_geometry(desk["id"], polygon=SQUARE, center=(3.0, 2.0), area_m2=12)
check("...a linked place of Harsh's does not",
      (p2.find_exact("desk")["x"], p2.find_exact("desk")["y"]) == (1.0, 1.0))
s2.rename(bed["id"], "Guest Room")
check("rename: the old landmark goes, the new one arrives",
      p2.find_exact("bedroom") is None and p2.find_exact("guest room") is not None
      and s2.get(bed["id"])["display_name"] == "Guest Room")
unnamed = s2.add_area("home", "map-1", polygon=FAR_SQUARE, center=(11, 11),
                      area_m2=9)
s2.rename(unnamed["id"], "store room", source="dashboard")
check("naming an unnamed area makes it named, with a landmark",
      s2.get(unnamed["id"])["status"] == "named"
      and p2.find_exact("store room") is not None
      and s2.get(unnamed["id"])["source"] == "dashboard")
trip = s2.add_area("home", "mission-frame", polygon=SQUARE, center=(2, 1.5),
                   area_m2=12, name="hallway", frame="mission:abc")
check("an area not placed on the saved map is never a landmark",
      p2.find_exact("hallway") is None and trip["frame"] == "mission:abc")
gone = s2.forget(bed["id"])
lines = (s2.env_dir("home") / "forgotten.jsonl").read_text().splitlines()
check("forget: written to forgotten.jsonl FIRST, with the area and landmark",
      gone and json.loads(lines[-1])["area"]["id"] == bed["id"]
      and json.loads(lines[-1])["landmark"]["name"] == "guest room")
check("...then the area and its landmark are gone",
      s2.get(bed["id"]) is None and p2.find_exact("guest room") is None)
check("forgetting nothing is harmless", s2.forget("no-such-id") is None)


class FlakyPlaces(ad.FakePlaces):
    fail = True

    def upsert(self, *a, **k):
        if self.fail:
            raise OSError("power cut between the commit and the landmark")
        return super().upsert(*a, **k)


fp = FlakyPlaces()
s3 = store("flaky", places=fp)
hall = s3.add_area("home", "map-1", polygon=SQUARE, center=(2, 1.5), area_m2=12,
                   name="hall")
check("a landmark failure never loses the area",
      s3.get(hall["id"])["name"] == "hall" and fp.find_exact("hall") is None)
fp.fail = False
check("resync_landmarks repairs it at boot",
      s3.resync_landmarks("home") == 1 and fp.find_exact("hall") is not None)
check("...and has nothing to do the second time", s3.resync_landmarks("home") == 0)


# =============================================================================
print("--- map_id guard: stale, never deleted ---")

stale_ids = s.mark_stale("envA", "map-2")
envA = s.list_areas("envA")
check("areas saved on another map become stale",
      set(stale_ids) == {a["id"] for a in envA} and len(stale_ids) >= 4
      and all(a["status"] == "stale" for a in envA))
check("...every one of them is still there, name and all",
      s.by_name("envA", "kitchen") is not None and len(s.list_areas("envA")) == len(envA))
check("...and says so only once", s.mark_stale("envA", "map-2") == [])
check("other environments are untouched",
      s.list_areas("envB")[0]["status"] == "named")
check("stale areas are in the naming queue",
      len(s.naming_queue("envA")) == len(envA))
check("an unknown current map id changes nothing", s.mark_stale("envB", None) == [])


# =============================================================================
print("--- home anchor and missions ---")

sig = [1.5] * 360
sig[10] = float("nan")
ha = s.set_home_anchor("envA", 0.5, -0.2, 3.1, sig, "map-1", set_by="auto")
check("home anchor saved; no return is NaN, kept as NaN",
      ha["set_by"] == "auto" and len(ha["signature"]) == 360
      and math.isnan(ha["signature"][10]) and ha["signature"][0] == 1.5)
s.set_home_anchor("envA", 1.0, 1.0, 0.0, None, "map-1", set_by="voice")
check("one row per environment; the newer anchor replaces it",
      s.home_anchor("envA")["x"] == 1.0 and s.home_anchor("envA")["signature"] is None)
check("set_by is auto or voice", bool(raises(
    lambda: s.set_home_anchor("envA", 0, 0, 0, None, None, set_by="robot"),
    st.StoreError)))
check("no anchor -> None", s.home_anchor("nowhere") is None)
mm = s.start_mission("envA", "map-1", (1, 2, 0.5), start_signature=sig,
                     explored_m2_start=10.0)
check("a mission starts running, with its start pose and signature",
      mm["status"] == "running" and (mm["start_x"], mm["start_y"]) == (1.0, 2.0)
      and math.isnan(mm["start_signature"][10]))
s.update_mission(mm["id"], status="returned", stop_reason="no_frontiers",
                 home_error_m=0.21, rungs_used=[1, 3], rth_attempts=2,
                 ended_at=st.now_iso())
mm = s.mission(mm["id"])
check("update_mission writes status, reason, rungs",
      (mm["status"], mm["stop_reason"], mm["rungs_used"], mm["rth_attempts"])
      == ("returned", "no_frontiers", [1, 3], 2))
check("an unknown field is refused",
      bool(raises(lambda: s.update_mission(mm["id"], colour="red"), st.StoreError)))
check("an unknown status is refused",
      bool(raises(lambda: s.update_mission(mm["id"], status="lost"), st.StoreError)))
check("an unknown mission is refused",
      bool(raises(lambda: s.update_mission("nope", status="aborted"), st.StoreError)))
capped = store("events", events_cap=5)
cm = capped.start_mission("home", None, (0, 0, 0))
for i in range(12):
    capped.add_event(cm["id"], {"i": i})
ev = capped.mission(cm["id"])["events"]
check("the event log keeps only the newest events_cap",
      [e["i"] for e in ev] == [7, 8, 9, 10, 11] and all("t" in e for e in ev))
check("missions() lists newest first",
      [x["id"] for x in s.missions("envA", limit=2)][0] == mm["id"])


# =============================================================================
print("--- checkpoints: .bak, corruption, migration ---")

cp = store("cp", checkpoint_keep=500)
check("no checkpoint -> (None, None)", cp.load_checkpoint("home") == (None, None))
cp.save_checkpoint("home", {"mission_id": "m1", "state": "EXPLORE",
                            "breadcrumbs": [[i, i, i] for i in range(800)]})
data, src = cp.load_checkpoint("home")
check("saved and loaded from the main file",
      src == "main" and data["state"] == "EXPLORE" and data["schema_version"] == 1)
check("breadcrumbs trimmed to the last checkpoint_keep",
      len(data["breadcrumbs"]) == 500 and data["breadcrumbs"][0] == [300, 300, 300])
cp.save_checkpoint("home", {"mission_id": "m1", "state": "ASK_LABEL"})
cpf = cp.checkpoint_file("home")
check("the previous good copy is kept as .bak",
      json.loads(durable.bak_path(cpf).read_text())["state"] == "EXPLORE")
cpf.write_text('{"state": "EXPL')                     # torn / corrupt
data, src = cp.load_checkpoint("home")
check("corrupt main file -> the .bak loads", src == "bak"
      and data["state"] == "EXPLORE")
cp.save_checkpoint("home", {"mission_id": "m1", "state": "RETURN_TO_START"})
check("a corrupt main file is never promoted to .bak",
      json.loads(durable.bak_path(cpf).read_text())["state"] == "EXPLORE")
cpf.write_text("garbage")
durable.bak_path(cpf).write_text("garbage")
check("both corrupt -> (None, None), no crash",
      cp.load_checkpoint("home") == (None, None))
durable.atomic_write_json(cpf, {"mission_id": "old", "state": "EXPLORE"})
data, _ = cp.load_checkpoint("home")
check("a v0 checkpoint (no schema_version) is migrated to v1",
      data["schema_version"] == 1 and data["breadcrumbs"] == []
      and data["blacklist"] == [] and data["pending_label"] is None)
durable.atomic_write_json(cpf, {"schema_version": 99})
check("a checkpoint from a newer version is refused",
      bool(raises(lambda: cp.load_checkpoint("home"), st.StoreError)))
tp = cp.save_trail("home", "m1", [[0, 0, 1.0], [0.25, 0, 2.0]])
check("the full trail is saved per mission",
      json.loads(tp.read_text())["crumbs"][1] == [0.25, 0, 2.0])
check("an env id cannot escape the data folder",
      cp.env_dir("../../etc").parent == cp.data_dir / "envs")


# =============================================================================
print("--- crash test: JSON checkpoint, SIGKILL at random moments ---")

CRASH_DIR = TMP / "crash"
CHILD_JSON = r'''
import sys
sys.path.insert(0, %(root)r)
from exploration_mission.area_store import AreaStore
s = AreaStore(db_path=%(db)r, data_dir=%(data)r, backup_before_create=False)
start = int(sys.argv[1])
for i in range(start, start + 200):
    s.save_checkpoint("home", {"counter": i, "blob": list(range(i, i + 3000))})
    print(i, flush=True)
''' % {"root": str(ROOT), "db": str(CRASH_DIR / "x.db"), "data": str(CRASH_DIR)}
rng = random.Random(21)
counter, trials, bad, mid_run = 0, 0, [], 0
reader = st.AreaStore(db_path=CRASH_DIR / "x.db", data_dir=CRASH_DIR)
while trials < 12:
    trials += 1
    proc = subprocess.Popen([PY, "-c", CHILD_JSON, str(counter)],
                            stdout=subprocess.PIPE, text=True)
    # A kill that lands after the child has finished proves nothing, so land
    # it on purpose: after a random write number, somewhere inside the next.
    target = rng.randint(1, 199)
    last = counter - 1
    for line in proc.stdout:
        last = int(line)
        if last - counter + 1 >= target:
            break
    time.sleep(rng.uniform(0.0, 0.015))
    proc.send_signal(signal.SIGKILL)
    rest = proc.communicate()[0].split()
    if rest:
        last = int(rest[-1])
    if last < counter + 199:
        mid_run += 1
    data, src = reader.load_checkpoint("home")
    if data is None:
        bad.append(f"trial {trials}: nothing loads")
        continue
    c = data["counter"]
    blob = data["blob"]
    if c not in (last, last + 1) or blob[0] != c or len(blob) != 3000:
        bad.append(f"trial {trials}: last done {last}, loaded {c} ({src})")
    bak = json.loads(durable.bak_path(reader.checkpoint_file("home")).read_text()) \
        if durable.bak_path(reader.checkpoint_file("home")).exists() else None
    if bak is not None and (bak["counter"] > c or bak["blob"][0] != bak["counter"]):
        bad.append(f"trial {trials}: .bak {bak['counter']} vs main {c}")
    counter = c + 1
print(f"        ({trials} kills, {mid_run} of them mid-run; {counter} "
      f"checkpoints written in all)")
check("every kill landed while the child was still writing", mid_run == trials,
      f"{mid_run}/{trials}")
check(f"{trials} kills during checkpoint writes: always the old or the new "
      f"version, .bak never ahead", not bad, "; ".join(bad))
check("after all the kills the main file itself loads",
      reader.load_checkpoint("home")[1] == "main")


# =============================================================================
print("--- crash test: SQLite transactions, kill -9 ---")

SQLDB = wal_db(TMP / "sqlcrash.db")
st.AreaStore(db_path=SQLDB, data_dir=TMP / "sqlcrash").ensure_schema()
CHILD_SQL = r'''
import sys
sys.path.insert(0, %(root)r)
from exploration_mission.area_store import AreaStore
s = AreaStore(db_path=%(db)r, data_dir=%(data)r)
s.ensure_schema()
m = s.start_mission("home", "map-x", (0, 0, 0))
for i in range(200):
    s.add_area("home", "map-x", polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
               center=(0.5, 0.5), area_m2=1.0, mission_id=m["id"])
    s.add_event(m["id"], {"i": i})
    print(i, flush=True)
''' % {"root": str(ROOT), "db": str(SQLDB), "data": str(TMP / "sqlcrash")}
problems = []
for trial in range(10):
    proc = subprocess.Popen([PY, "-c", CHILD_SQL], stdout=subprocess.PIPE, text=True)
    time.sleep(rng.uniform(0.5, 2.5))
    proc.send_signal(signal.SIGKILL)
    proc.communicate()
    con = sqlite3.connect(str(SQLDB))
    ok = con.execute("PRAGMA integrity_check").fetchone()[0]
    if ok != "ok":
        problems.append(f"trial {trial}: integrity {ok}")
    for mid, new in con.execute("SELECT id, areas_new FROM missions"):
        saved = {r[0] for r in con.execute(
            "SELECT id FROM areas WHERE first_mission_id=?", (mid,))}
        if saved != set(json.loads(new)):
            problems.append(f"trial {trial}: mission {mid[:6]} areas_new "
                            f"{len(json.loads(new))} vs saved {len(saved)}")
    con.close()
n_areas = sqlite3.connect(str(SQLDB)).execute("SELECT COUNT(*) FROM areas") \
    .fetchone()[0]
print(f"        ({n_areas} areas committed across 10 killed runs of 200)")
check("10 x kill -9 during transactions: integrity ok, every area in its "
      "mission's areas_new and vice versa", not problems, "; ".join(problems[:5]))
check("...and the kills landed mid-run (areas were being written)", n_areas > 10,
      str(n_areas))
names = [r[0] for r in sqlite3.connect(str(SQLDB)).execute(
    "SELECT name FROM areas")]
check("unnamed numbering stayed unique through the kills",
      len(names) == len(set(names)))


# =============================================================================
print("--- write here, read in a new process ---")

xs = store("xproc")
xa = xs.add_area("home", "map-1", polygon=SQUARE, center=(2, 1.5), area_m2=12,
                 name="Harsh's room", hints={"objects": ["bed"]})
xs.set_home_anchor("home", 0.1, 0.2, 0.3, [2.0] * 360, "map-1")
xs.save_checkpoint("home", {"mission_id": "m9", "state": "PAUSED",
                            "breadcrumbs": [[0, 0, 1.0]]})
xs.close()
READER = r'''
import json, sys
sys.path.insert(0, %(root)r)
from exploration_mission.area_store import AreaStore
s = AreaStore(db_path=%(db)r, data_dir=%(data)r)
print(json.dumps({"area": s.by_name("home", "harsh's room"),
                  "anchor": s.home_anchor("home"),
                  "cp": s.load_checkpoint("home")[0]}))
''' % {"root": str(ROOT), "db": str(TMP / "xproc.db"), "data": str(TMP / "xproc-data")}
done = subprocess.run([PY, "-c", READER], capture_output=True, text=True, timeout=60)
got = json.loads(done.stdout)
xs2 = st.AreaStore(db_path=TMP / "xproc.db", data_dir=TMP / "xproc-data")
check("a new process reads the identical area",
      got["area"] == json.loads(json.dumps(xs2.get(xa["id"]))), done.stderr[-300:])
check("...the identical home anchor and checkpoint",
      got["anchor"] == json.loads(json.dumps(xs2.home_anchor("home")))
      and got["cp"]["state"] == "PAUSED" and got["cp"]["breadcrumbs"] == [[0, 0, 1.0]])


# =============================================================================
print("--- the real File 14 places table (temp memory.db) ---")

import nav_state_machine as nsm                         # noqa: E402
nsm._bundle = lambda reason: None
from slam import places as real_places                  # noqa: E402
import landmarks                                        # noqa: E402

refreshed: list[str] = []
real = ad.PlacesAdapter(bundle=refreshed.append)
rs = store("realplaces", places=real)
ra = rs.add_area("home", "map-real", polygon=SQUARE, center=(2, 1.5), area_m2=12,
                 name="Kitchen")
row = real_places.find("kitchen")
check("the area is in slam/places.py, kind area, where File 14 looks",
      row and row["kind"] == "area" and (row["x"], row["y"]) == (2.0, 1.5))
check("File 14 reads the map session from its note",
      landmarks.session_of(row) == "map-real", str(row and row["note"]))
check("File 14's navigation bundle was refreshed", bool(refreshed))
rs.forget(ra["id"])
check("forget removes it from slam/places.py", real_places.find("kitchen") is None)


# =============================================================================
print("--- nothing real was touched ---")

# Since 2026-10-02 the REAL mission created its tables (after a backup); the
# point here is that THESE tests never did: every store in this suite is temp.
_real_tables = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True).execute(
    "SELECT 1 FROM sqlite_master WHERE name='missions'").fetchone()
check("the real memory.db has exploration tables only after its backup",
      not _real_tables or bool(sorted(REAL_DB.parent.glob(REAL_DB.name + ".bak-*-file21"))))
check("the network guard saw no attempt", not netguard.attempts,
      str(netguard.attempts))

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
