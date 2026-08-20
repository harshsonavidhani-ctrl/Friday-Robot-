"""Friday's orchestrator.

The spine. Modules get wired in here, one per session — the loop stays thin
and every capability lives in its own file under modules/.

Right now it does exactly one honest thing: report what is and isn't ready.
"""

import logging
import sys

import config


def setup_logging() -> logging.Logger:
    config.LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / "friday.log"),
        ],
    )
    return logging.getLogger("friday")


def main() -> int:
    log = setup_logging()
    log.info("Friday starting up")

    if config.ANTHROPIC_API_KEY:
        log.info("Brain: API key present, model %s", config.MODEL_MAIN)
    else:
        log.warning("Brain: ANTHROPIC_API_KEY unset — brain will not run")

    unverified = config.missing_hardware_settings()
    if unverified:
        log.warning(
            "Unverified hardware settings (%d): %s",
            len(unverified), ", ".join(unverified),
        )
        log.warning("Run scripts/check_hardware.sh, then fill these into config.py")

    log.info("No modules wired yet. Next: build one, test it, commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
