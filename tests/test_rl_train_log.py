"""A night's training log must never be appended into an unreadable shape."""
import csv
import os

from rl.train_log import COLUMNS, header_of, prepare_log

ROUND3_HEADER = ["wall_s", "steps", "episode", "reward", "result"]
ROUND3_ROWS = [["12", "1024", "1", "-30.0", "timeout"],
               ["31", "2048", "2", "184.2", "parked"]]


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_a_brand_new_log_gets_a_header(tmp_path):
    log = str(tmp_path / "train_log.csv")
    needs_header, backup = prepare_log(log)
    assert needs_header and backup is None


def test_a_matching_log_is_appended_to_untouched(tmp_path):
    log = str(tmp_path / "train_log.csv")
    _write(log, COLUMNS, [["1", "2", "3", "4.0", "parked", "0.45", "7.1", "1", "0", "0"]])
    needs_header, backup = prepare_log(log)
    assert not needs_header and backup is None
    assert header_of(log) == COLUMNS, "the existing log is left alone"


def test_an_old_narrow_log_is_set_aside_not_corrupted(tmp_path):
    """The bug this exists to stop: round 4 writes 7-field rows, the CARLA
    machine's log has a 5-field header, and appending makes the whole file
    unreadable — pandas raises, csv.DictReader silently loses p/start_dist."""
    log = str(tmp_path / "train_log.csv")
    _write(log, ROUND3_HEADER, ROUND3_ROWS)

    needs_header, backup = prepare_log(log)

    assert needs_header, "a fresh header must be written"
    assert backup and os.path.exists(backup), "the old log must be kept"
    assert not os.path.exists(log), "the live path is cleared for the new log"

    with open(backup, newline="") as f:
        kept = list(csv.reader(f))
    assert kept == [ROUND3_HEADER] + ROUND3_ROWS, "rounds 1-3 stay loadable"


def test_every_row_still_matches_the_header_after_rotation(tmp_path):
    log = str(tmp_path / "train_log.csv")
    _write(log, ROUND3_HEADER, ROUND3_ROWS)
    needs_header, _ = prepare_log(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if needs_header:
            w.writerow(COLUMNS)
        w.writerow(["9", "10", "1", "201.3", "parked", "0.0", "16.3", "1", "1", "12"])

    with open(log, newline="") as f:
        rows = list(csv.reader(f))
    assert all(len(r) == len(COLUMNS) for r in rows), "no ragged rows"
    with open(log, newline="") as f:
        parsed = list(csv.DictReader(f))
    assert parsed[0]["p"] == "0.0" and parsed[0]["start_dist_m"] == "16.3"
    assert None not in parsed[0], "nothing spilled into an unnamed column"


def test_a_second_rotation_does_not_clobber_the_first_backup(tmp_path):
    log = str(tmp_path / "train_log.csv")
    _write(log, ROUND3_HEADER, ROUND3_ROWS)
    _, first = prepare_log(log)
    _write(log, ["something", "else"], [["a", "b"]])
    _, second = prepare_log(log)
    assert first and second and first != second
    assert os.path.exists(first) and os.path.exists(second)


def test_header_of_is_quiet_about_a_missing_file(tmp_path):
    assert header_of(str(tmp_path / "nope.csv")) is None


def test_a_fresh_run_never_appends_to_the_previous_runs_log(tmp_path):
    """Round 6 appended to round 5's log because the columns matched: two runs
    in one CSV with the episode counter restarting mid-file. A new run must
    always start its own file, whatever the old header looks like."""
    log = str(tmp_path / "train_log.csv")
    _write(log, COLUMNS, [["1", "2", "3", "4.0", "parked", "0.45", "7.1", "1", "0", "0"]])
    needs_header, backup = prepare_log(log, fresh=True)
    assert needs_header and backup and os.path.exists(backup)
    assert not os.path.exists(log)
    with open(backup, newline="") as f:
        assert len(list(csv.reader(f))) == 2, "the old run is kept intact"


def test_a_resumed_run_keeps_appending_to_its_own_log(tmp_path):
    log = str(tmp_path / "train_log.csv")
    _write(log, COLUMNS, [["1", "2", "3", "4.0", "parked", "0.45", "7.1", "1", "0", "0"]])
    assert prepare_log(log, fresh=False) == (False, None)
