"""File 21 — the exploration mission (the uploaded spec "15_exploration_mission.md";
15 was already taken by vision description, so it is numbered 21 here).

Mark the start, explore with the trained RL policy, stop and ask what to call
each area, finish when nothing reachable is left, drive back to the start, and
keep all of it across reboots -- 100% offline: no LLM, no API key, no network.

Additive. Only `adapters.py` imports existing Friday code, and only lazily, so
importing this package imports nothing of Friday's.
"""
