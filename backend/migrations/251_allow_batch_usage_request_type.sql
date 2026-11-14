-- Request type 7 identifies Gemini Batch API usage while retaining token billing.
-- Type 6 is reserved by channel-monitor filters; do not reuse it.
-- NOT VALID avoids a full scan of existing historical usage logs at startup.
ALTER TABLE usage_logs
    DROP CONSTRAINT IF EXISTS usage_logs_request_type_check;

ALTER TABLE usage_logs
    ADD CONSTRAINT usage_logs_request_type_check
    CHECK (request_type IN (0, 1, 2, 3, 4, 5, 7)) NOT VALID;
