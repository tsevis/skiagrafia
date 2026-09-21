"""Where finished work is written.

The destination was reachable only from the Preferences window. That is not
where anyone decides what a run produces -- the output modes and the Process
button are in the run-setup panel, and where that output lands belongs beside
them. Both panels call through here so all three places agree on one value.

A folder that cannot be written to is refused at the moment of choosing
rather than when a finished run tries to save into it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from utils.preferences import DEFAULT_PREFERENCES, save_preferences

logger = logging.getLogger(__name__)

PREFERENCE_KEY = "output_directory"


class DestinationError(Exception):
    """A folder that runs could not write into."""


def default_destination() -> Path:
    """Where runs go when nothing has been chosen."""
    return Path(DEFAULT_PREFERENCES[PREFERENCE_KEY])


def current_destination(prefs: dict[str, Any]) -> Path:
    """The folder the next run will write into."""
    configured = str(prefs.get(PREFERENCE_KEY, "")).strip()
    return Path(configured).expanduser() if configured else default_destination()


def validate_destination(path: str | Path) -> Path:
    """Return the folder a run may write into, or say why it may not.

    A path that does not exist yet is accepted when it can be created: the
    default destination is made on first use, so refusing a merely absent
    folder would refuse the default itself.
    """
    candidate = Path(path).expanduser()
    if not str(candidate).strip():
        raise DestinationError("No destination folder was chosen.")

    if candidate.exists() and not candidate.is_dir():
        raise DestinationError(f"The destination is not a folder: {candidate}")

    anchor = candidate
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent

    if not anchor.is_dir() or not os.access(anchor, os.W_OK | os.X_OK):
        raise DestinationError(
            f"The destination cannot be written to: {candidate}"
        )
    return candidate


def choose_destination(prefs: dict[str, Any], path: str | Path) -> Path:
    """Adopt a destination and write the choice down.

    A refusal changes nothing: reporting the fault while quietly pointing
    runs at an unusable folder would be worse than either alone.
    """
    destination = validate_destination(path)
    prefs[PREFERENCE_KEY] = str(destination)
    save_preferences(prefs)
    logger.info("Destination folder set: %s", destination)
    return destination
