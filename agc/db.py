"""SQLite store for the dataset inventory and the training-run log.

The queries live in `sql/` as real .sql files rather than as strings in here, so
the aggregation can be run against the database by hand — `sqlite3 runs.db
< sql/run_comparison.sql` — without going through Python at all.
"""
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(ROOT, "sql")
DEFAULT_DB = os.path.join(ROOT, "artifacts", "runs.db")

#: Reporting cohorts. Wide enough to hold a usable count, narrow enough that a
#: band failing on its own is visible instead of averaged into its neighbours.
BANDS = [(1, 5), (6, 12), (13, 19), (20, 29), (30, 39),
         (40, 49), (50, 59), (60, 69), (70, 200)]


def age_band(age):
    for lo, hi in BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else "70+"
    return "0"


def read_sql_file(name):
    with open(os.path.join(SQL_DIR, name), encoding="utf8") as fh:
        return fh.read()


def connect(path=DEFAULT_DB):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(read_sql_file("schema.sql"))
    return con


def write_faces(con, faces: pd.DataFrame):
    """Upsert the dataset inventory. `faces` needs path, age, gender, split."""
    df = faces.copy()
    df["age_band"] = df["age"].map(age_band)
    df = df[["path", "age", "gender", "age_band", "split"]]
    con.executemany(
        "INSERT INTO faces (path, age, gender, age_band, split) VALUES (?,?,?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET age=excluded.age, gender=excluded.gender, "
        "age_band=excluded.age_band, split=excluded.split",
        df.itertuples(index=False, name=None))
    con.commit()
    return len(df)


def start_run(con, arch, age_loss, balanced, epochs):
    cur = con.execute(
        "INSERT INTO runs (started_at, arch, age_loss, balanced, epochs) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"),
         arch, age_loss, int(bool(balanced)), int(epochs)))
    con.commit()
    return cur.lastrowid


def finish_run(con, run_id, params, val_mae, val_acc, preds: pd.DataFrame):
    """Record a run's headline metrics and its per-face validation predictions."""
    con.execute("UPDATE runs SET params=?, val_mae=?, val_acc=? WHERE run_id=?",
                (int(params), float(val_mae), float(val_acc), run_id))
    rows = preds[["path", "pred_age", "pred_gender"]].copy()
    rows.insert(0, "run_id", run_id)
    con.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, path, pred_age, pred_gender) "
        "VALUES (?,?,?,?)", rows.itertuples(index=False, name=None))
    con.commit()


def band_errors(con, run_id) -> pd.DataFrame:
    """The per-age-band report, straight out of sql/age_band_error.sql."""
    return pd.read_sql_query(read_sql_file("age_band_error.sql"), con,
                             params={"run_id": run_id})


def run_comparison(con) -> pd.DataFrame:
    return pd.read_sql_query(read_sql_file("run_comparison.sql"), con)
