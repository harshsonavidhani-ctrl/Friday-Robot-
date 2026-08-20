# Friday

A Raspberry Pi companion robot. See `CLAUDE.md` for the full brief — read
**Part 1 (system safety)** before touching anything on this machine.

## Layout

```
CLAUDE.md      the brief; safety rules, working agreement, hardware, targets
config.py      single source of truth for keys, devices, thresholds, models
main.py        the orchestrator — the thin spine that wires modules together
modules/       one capability per file; modules never import each other
scripts/       read-only helpers (hardware survey, model fetch)
assets/        face art, voice + wake-word models (models are gitignored)
logs/          runtime logs and the API spend log (gitignored)
```

## Setup

```bash
cd ~/friday
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # created as modules are added
```

## Before wiring any hardware module

```bash
bash scripts/check_hardware.sh
```

Then put the values you actually saw into `config.py`. Never guess an index,
a port, or an I2C address — `config.missing_hardware_settings()` lists what
is still unverified.

## Run

```bash
source venv/bin/activate
python main.py
```

## Build order

Spine first, then organs. One module per session: build it, test it on the
real robot, commit it, stop. Nothing is committed untested.
