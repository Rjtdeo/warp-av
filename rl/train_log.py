"""
The per-episode training log, and the one rule that keeps it readable.

Round 4 widened this log from 5 columns to 7 (difficulty and start distance
were added). The CARLA machine still holds the rounds 1-3 file. Appending wide
rows under a narrow header does not fail loudly — it makes the WHOLE log
unreadable afterwards: pandas raises ParserError, and csv.DictReader quietly
drops the extra values into a None key, so every episode's difficulty reads as
missing. A night's evidence is lost in silence.

So: check the header, not just whether the file exists.

No CARLA and no stable-baselines3 imports, so this is testable on any machine.
"""
from __future__ import annotations

import csv
import os

COLUMNS = ["wall_s", "steps", "episode", "reward", "result", "p", "start_dist_m"]


def header_of(path):
    """First row of an existing CSV, or None if it is missing or unreadable."""
    try:
        with open(path, newline="") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


def prepare_log(path, columns=None, fresh=False):
    """Make `path` safe to append `columns`-shaped rows to.

    Returns (needs_header, backup_path). A log whose header does not match is
    moved aside rather than appended to, so the older round's rows stay
    loadable under their own header and the live log always matches its rows.

    fresh=True means a NEW training run rather than a resumed one: the existing
    log is set aside even when its header matches. Round 6 briefly appended to
    round 5's file because the columns agreed — two runs in one CSV, with the
    episode counter restarting in the middle. One file, one run.
    """
    columns = list(columns or COLUMNS)
    if not os.path.exists(path):
        return True, None
    if not fresh and header_of(path) == columns:
        return False, None
    backup = path + ".v1.bak"
    n = 2
    while os.path.exists(backup):
        backup = f"{path}.v{n}.bak"
        n += 1
    os.replace(path, backup)
    return True, backup
