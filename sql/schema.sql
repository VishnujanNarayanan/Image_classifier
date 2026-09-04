-- Schema for the age/gender experiment log.
--
-- Two things were previously trapped in .npz files and printed console output:
-- what the pipeline kept from the raw dataset, and how each training run did on
-- it. Neither could be queried, so comparing runs meant re-reading stdout from a
-- terminal that had scrolled away. Both live here instead.

CREATE TABLE IF NOT EXISTS faces (
    path        TEXT PRIMARY KEY,   -- source filename, the natural key in UTKFace
    age         INTEGER NOT NULL,
    gender      INTEGER NOT NULL,   -- 0 male, 1 female
    age_band    TEXT    NOT NULL,   -- the reporting cohort, e.g. '20-29'
    split       TEXT    NOT NULL    -- 'train' or 'val'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    arch        TEXT    NOT NULL,   -- 'shallow' | 'deep' | 'transfer'
    age_loss    TEXT    NOT NULL,   -- 'ce' | 'emd'
    balanced    INTEGER NOT NULL,   -- inverse-frequency age weighting on/off
    epochs      INTEGER NOT NULL,
    params      INTEGER,
    val_mae     REAL,
    val_acc     REAL
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id      INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    path        TEXT    NOT NULL REFERENCES faces(path),
    pred_age    REAL    NOT NULL,
    pred_gender INTEGER NOT NULL,
    PRIMARY KEY (run_id, path)
);

CREATE INDEX IF NOT EXISTS idx_faces_band  ON faces(age_band, split);
CREATE INDEX IF NOT EXISTS idx_pred_run    ON predictions(run_id);
