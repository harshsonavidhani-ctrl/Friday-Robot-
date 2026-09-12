#!/usr/bin/env python3
"""weather_source.py — real weather for the card, with no API key.

    ./venv/bin/python weather_source.py            # what it would show
    ./venv/bin/python weather_source.py --force    # ignore the cache
    ./venv/bin/python app_control.py weather       # ...and put it on the panel

WHY THIS EXISTS, AND WHAT IT DOES NOT REPLACE. apps/weather.py draws the card
and deliberately holds no key: "the brain has web search; it looks the weather
up and hands the numbers here". That is still the better path and it stays --
the brain can answer "will it rain before I walk home", and this cannot. But it
needs a key, credits and a question. Harsh asked on 2026-09-12 for the card to
work on its own as well, so this fills the same dict the brain fills.

WHAT IT TALKS TO. wttr.in, which needs no key and no account. With no place
configured it locates by IP, which means THE REQUEST TELLS THAT SERVICE THE
PI'S PUBLIC IP -- the ordinary cost of keyless geolocation, and exactly why
APP_WEATHER_PLACE exists: name a city and nothing is inferred from the address.
APP_WEATHER_FETCH = False turns the whole thing off and leaves the brain path.

IT NEVER RAISES AND IT NEVER INVENTS. Every failure returns (None, reason): no
internet, a slow reply, a service that changed its JSON. A card showing last
Tuesday's sky with today's confidence would be worse than no card, so the cache
carries its own age and stale entries are refused rather than shown.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.weather")

CACHE_FILE = config.ROOT / "state" / "weather.json"

# The fields apps/weather.py draws. Anything else is dropped rather than
# carried around: the card is the contract.
FIELDS = ("place", "condition", "temp_c", "feels_c", "high_c", "low_c",
          "wind_kph", "humidity", "summary")


def _read_cache() -> tuple[dict[str, Any] | None, float]:
    try:
        data = json.loads(CACHE_FILE.read_text())
        return data.get("weather"), time.time() - float(data.get("fetched_at", 0))
    except (OSError, ValueError, TypeError):
        return None, float("inf")


def _write_cache(weather: dict[str, Any]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(
            {"fetched_at": time.time(), "weather": weather}, indent=1))
    except OSError as exc:
        log.debug("could not cache the weather: %s", exc)


def _parse(payload: dict[str, Any]) -> dict[str, Any]:
    """wttr.in's j1 JSON -> the card's dict.

    THE CONDITION TEXT GOES THROUGH UNTOUCHED, deliberately. apps/weather.py
    already matches on keywords -- "Light rain shower" contains "rain", "Partly
    cloudy" contains "partly" -- so a translation table here would be a second
    place to keep in step with it, and the first one to go stale.
    """
    current = (payload.get("current_condition") or [{}])[0]
    today = (payload.get("weather") or [{}])[0]
    area = (payload.get("nearest_area") or [{}])[0]

    def number(source: dict, key: str) -> float | None:
        try:
            return float(source[key])
        except (KeyError, TypeError, ValueError):
            return None

    def text(source: dict, key: str) -> str:
        try:
            return str(source[key][0]["value"]).strip()
        except (KeyError, IndexError, TypeError):
            return ""

    place = text(area, "areaName") or config.APP_WEATHER_PLACE or ""
    return {
        "place": place,
        "condition": text(current, "weatherDesc"),
        "temp_c": number(current, "temp_C"),
        "feels_c": number(current, "FeelsLikeC"),
        "high_c": number(today, "maxtempC"),
        "low_c": number(today, "mintempC"),
        "wind_kph": number(current, "windspeedKmph"),
        "humidity": number(current, "humidity"),
    }


def fetch(place: str | None = None,
          timeout_s: float | None = None) -> tuple[dict[str, Any] | None, str]:
    """One live look at the sky. -> (weather, why) and never raises."""
    if not config.APP_WEATHER_FETCH:
        return None, "APP_WEATHER_FETCH is off — the brain supplies the numbers"
    place = place if place is not None else (config.APP_WEATHER_PLACE or "")
    timeout_s = config.APP_WEATHER_TIMEOUT_S if timeout_s is None else timeout_s
    url = config.APP_WEATHER_URL.format(place=urllib.parse.quote(place))
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": config.APP_WEATHER_USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:80]}"

    weather = _parse(payload)
    if weather.get("temp_c") is None:
        return None, "the reply carried no temperature — the format may have moved"
    log.info("weather: %s %s %s°C in %.1fs", weather.get("place"),
             weather.get("condition"), weather.get("temp_c"),
             time.monotonic() - started)
    _write_cache(weather)
    return weather, f"fetched in {time.monotonic() - started:.1f}s"


def current(place: str | None = None, max_age_s: float | None = None,
            force: bool = False) -> tuple[dict[str, Any] | None, str]:
    """Cached if it is fresh, fetched if it is not.

    The cache exists to be a good neighbour: wttr.in is free and rate-limits,
    and the sky does not change in the time it takes to say "show me the
    weather" twice. A cache OLDER than the window is not served stale -- it is
    refetched, and only if that fails is it offered with its age stated.
    """
    max_age_s = config.APP_WEATHER_CACHE_S if max_age_s is None else max_age_s
    cached, age = _read_cache()
    if cached and not force and age <= max_age_s:
        return cached, f"cached {age:.0f}s ago"
    weather, why = fetch(place)
    if weather is not None:
        return weather, why
    if cached:
        return cached, f"{why}; showing the cached reading from {age / 60:.0f} min ago"
    return None, why


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--place", default=None, help="override APP_WEATHER_PLACE")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    weather, why = current(args.place, force=args.force)
    if weather is None:
        print(f"no weather: {why}")
        return 1
    print(f"{why}\n" + json.dumps(weather, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
