-- Add 'downloaded' status to export_requests so approvals are single-use.
-- Drop the old CHECK and add the new one.
ALTER TABLE export_requests
    DROP CONSTRAINT IF EXISTS export_requests_status_check;
ALTER TABLE export_requests
    ADD CONSTRAINT export_requests_status_check
    CHECK (status IN ('pending', 'approved', 'denied', 'downloaded'));
