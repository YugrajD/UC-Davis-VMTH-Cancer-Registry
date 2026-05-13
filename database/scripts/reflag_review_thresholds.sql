-- Re-apply the PetBERT diagnosis review gate after threshold calibration.
--
-- Edit the two values in params before running if backend/app/config.py has
-- changed. This intentionally touches only system-managed rows that have not
-- been reviewed by a person.

BEGIN;

WITH params AS (
    SELECT
        0.23::numeric AS review_auto_accept_confidence,
        0.15::numeric AS review_auto_accept_margin
),
eligible AS (
    SELECT
        cd.id,
        cd.review_status AS from_status,
        cd.cancer_type_id,
        cd.icd_o_code,
        CASE
            WHEN cd.prediction_method = 'low_confidence'
                OR COALESCE(cd.confidence, 0) < p.review_auto_accept_confidence
                OR (
                    cd.top2_margin IS NOT NULL
                    AND cd.top2_margin < p.review_auto_accept_margin
                )
                THEN 'pending'
            ELSE 'confirmed'
        END AS to_status
    FROM case_diagnoses cd
    CROSS JOIN params p
    WHERE cd.review_status IN ('pending', 'confirmed')
      AND cd.reviewed_by_email IS NULL
      AND cd.reviewed_at IS NULL
),
changed AS (
    SELECT *
    FROM eligible
    WHERE from_status <> to_status
),
updated AS (
    UPDATE case_diagnoses cd
    SET review_status = changed.to_status
    FROM changed
    WHERE cd.id = changed.id
    RETURNING cd.id
),
events AS (
    INSERT INTO diagnosis_review_events (
        case_diagnosis_id,
        actor_email,
        action,
        from_status,
        to_status,
        cancer_type_id_before,
        cancer_type_id_after,
        icd_o_code_before,
        icd_o_code_after,
        notes
    )
    SELECT
        changed.id,
        'system',
        'threshold_reflag',
        changed.from_status,
        changed.to_status,
        changed.cancer_type_id,
        changed.cancer_type_id,
        changed.icd_o_code,
        changed.icd_o_code,
        'Applied REVIEW_AUTO_ACCEPT_CONFIDENCE='
            || p.review_auto_accept_confidence::text
            || ', REVIEW_AUTO_ACCEPT_MARGIN='
            || p.review_auto_accept_margin::text
    FROM changed
    JOIN updated ON updated.id = changed.id
    CROSS JOIN params p
    RETURNING id
)
SELECT
    (SELECT count(*) FROM eligible) AS eligible_rows,
    (SELECT count(*) FROM changed) AS changed_rows,
    (SELECT count(*) FROM changed WHERE to_status = 'pending') AS moved_to_pending,
    (SELECT count(*) FROM changed WHERE to_status = 'confirmed') AS moved_to_confirmed,
    (SELECT count(*) FROM events) AS review_events_inserted;

COMMIT;
