import pandas as pd

from agc.db import (
    age_band,
    band_errors,
    connect,
    finish_run,
    run_comparison,
    start_run,
    write_faces,
)

FACES = pd.DataFrame({
    "path":   ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    "age":    [4, 25, 27, 72],
    "gender": [0, 1, 1, 0],
    "split":  ["val", "val", "val", "val"],
})


def _db(tmp_path):
    return connect(str(tmp_path / "t.db"))


def test_age_band_labels_the_reporting_cohorts():
    assert age_band(4) == "1-5"
    assert age_band(25) == "20-29"
    assert age_band(95) == "70+"


def test_faces_upsert_rather_than_duplicate(tmp_path):
    con = _db(tmp_path)
    write_faces(con, FACES)
    write_faces(con, FACES.assign(split="train"))
    assert con.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 4
    assert con.execute("SELECT DISTINCT split FROM faces").fetchone()[0] == "train"


def test_band_errors_aggregates_per_cohort(tmp_path):
    con = _db(tmp_path)
    write_faces(con, FACES)
    run = start_run(con, arch="deep", age_loss="ce", balanced=True, epochs=3)
    preds = pd.DataFrame({
        "path": FACES["path"],
        "pred_age": [6.0, 27.0, 23.0, 60.0],
        "pred_gender": [0, 1, 0, 0],
    })
    finish_run(con, run, params=1000, val_mae=4.0, val_acc=0.75, preds=preds)

    bands = band_errors(con, run).set_index("age_band")
    assert list(bands.index) == ["1-5", "20-29", "70+"]      # ordered by age
    assert bands.loc["1-5", "n"] == 1
    assert bands.loc["20-29", "mae"] == 3.0                  # |27-25| and |23-27|
    assert bands.loc["20-29", "gender_accuracy"] == 0.5
    assert bands.loc["70+", "mae"] == 12.0


def test_run_comparison_orders_by_age_error(tmp_path):
    con = _db(tmp_path)
    empty = pd.DataFrame(columns=["path", "pred_age", "pred_gender"])
    for arch, mae in [("shallow", 11.9), ("deep", 8.3), ("transfer", 14.7)]:
        r = start_run(con, arch=arch, age_loss="ce", balanced=False, epochs=1)
        finish_run(con, r, params=1, val_mae=mae, val_acc=0.8, preds=empty)
    assert list(run_comparison(con)["arch"]) == ["deep", "shallow", "transfer"]
