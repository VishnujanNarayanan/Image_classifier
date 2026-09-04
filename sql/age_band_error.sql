-- Per-age-band error for one run.
--
-- The headline MAE hides where the model actually fails: an average of 8 years
-- can be 4 years on children and 12 on the band next to it. Averaging the bands
-- away is what let the old model saturate at 45 unnoticed, so the report is
-- always read band by band.
--
-- :run_id  the run to report on

SELECT
    f.age_band                                            AS age_band,
    COUNT(*)                                              AS n,
    ROUND(AVG(p.pred_age), 1)                             AS mean_prediction,
    ROUND(AVG(ABS(p.pred_age - f.age)), 1)                AS mae,
    ROUND(AVG(CASE WHEN p.pred_gender = f.gender
                   THEN 1.0 ELSE 0.0 END), 3)             AS gender_accuracy
FROM predictions p
JOIN faces f ON f.path = p.path
WHERE p.run_id = :run_id
GROUP BY f.age_band
ORDER BY MIN(f.age);
