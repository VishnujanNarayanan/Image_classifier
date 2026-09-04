-- Every run so far, best age error first.
--
-- The loss change, the balancing experiment and the transfer-learning attempt
-- were each judged by scrolling back through a terminal. This is the same
-- comparison, kept.

SELECT
    run_id,
    started_at,
    arch,
    age_loss,
    CASE balanced WHEN 1 THEN 'yes' ELSE 'no' END AS age_balanced,
    epochs,
    params,
    ROUND(val_mae, 2)  AS val_mae_years,
    ROUND(val_acc, 3)  AS val_gender_acc
FROM runs
ORDER BY val_mae ASC;
