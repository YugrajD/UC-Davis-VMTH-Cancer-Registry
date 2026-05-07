-- Fix: the UNIQUE(email, requested_role, status) constraint prevents denying
-- a second request for the same user+role because a row with status='denied'
-- already exists.  Drop it so multiple denied/approved rows are allowed.
-- The application-level check for duplicate *pending* requests is sufficient.
ALTER TABLE role_requests
    DROP CONSTRAINT IF EXISTS role_requests_email_requested_role_status_key;
