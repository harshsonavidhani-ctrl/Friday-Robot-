"""Files that survive a power cut (File 21, spec 3.6).

Friday's crashes are power cuts (the reset register says so), so every file the
mission keeps is written the one way that leaves either the old version or the
new one on disk, never half of each:

    temp file in the SAME directory -> flush -> fsync -> os.replace -> fsync dir

`save_json` also keeps `<file>.bak`, the previous GOOD copy: before a new
version replaces the main file, the main file is copied to .bak -- but only if
it still parses, so .bak can never be a corrupt file promoted by accident.
`load_json` falls back to .bak when the main file is missing or does not parse,
and says which one it used.

Nothing here ever deletes a saved file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("friday.exploration_mission.durable")


def _fsync_dir(folder: Path) -> None:
    try:
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(Path(path), json.dumps(payload).encode("utf-8"))


def bak_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".bak")


def _parse(path: Path) -> Any:
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def save_json(path: Path, payload: Any) -> None:
    """Atomic write, keeping the previous good copy as <file>.bak."""
    path = Path(path)
    if path.exists():
        try:
            old = path.read_bytes()
            json.loads(old.decode("utf-8"))           # only a GOOD copy is kept
        except (OSError, ValueError, UnicodeDecodeError):
            log.warning("%s does not parse; keeping the older .bak", path)
        else:
            atomic_write_bytes(bak_path(path), old)
    atomic_write_json(path, payload)


def load_json(path: Path) -> tuple[Any | None, str | None]:
    """(payload, "main" | "bak") or (None, None) when neither parses."""
    path = Path(path)
    try:
        return _parse(path), "main"
    except FileNotFoundError:
        pass
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log.error("%s is corrupt (%s); loading the .bak", path, exc)
    try:
        data = _parse(bak_path(path))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log.error("%s is corrupt too (%s)", bak_path(path), exc)
        return None, None
    log.warning("loaded %s from its .bak", path.name)
    return data, "bak"


def append_jsonl(path: Path, record: Any) -> None:
    """One line, flushed and fsynced BEFORE the caller goes on to delete the
    thing the line records (forget writes here first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)
