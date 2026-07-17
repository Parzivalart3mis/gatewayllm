-- GatewayLLM durable schema: config + ledger.
--
-- Postgres owns only what must survive a restart and be queried after the fact:
-- who may call, what they may spend, and what they actually spent. Hot-path
-- state (cache entries, rate-limit buckets, breaker state) lives in Redis, and
-- vectors live in Qdrant. Nothing is duplicated across the three.

BEGIN;

-- --- tenants -------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tenants (
    id         TEXT PRIMARY KEY,
    name       TEXT        NOT NULL,
    active     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- --- api_keys ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    tenant_id  TEXT        NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    -- SHA-256 hex digest. The raw key is never stored: a database leak must not
    -- hand over usable credentials.
    key_hash   TEXT        NOT NULL UNIQUE,
    -- Non-secret prefix (e.g. "glm_live_a1b2") so a human can identify a key in
    -- a list without the key itself being recoverable.
    key_prefix TEXT        NOT NULL,
    label      TEXT,
    -- Per-key rate limit; NULL falls back to the config default.
    rpm        INTEGER,
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT api_keys_rpm_positive CHECK (rpm IS NULL OR rpm > 0)
);

-- Auth runs on every request. This index is what keeps it a single index probe
-- rather than a scan that grows with the key table.
CREATE INDEX IF NOT EXISTS api_keys_lookup_idx
    ON api_keys (key_hash)
    WHERE revoked_at IS NULL;

CREATE INDEX IF NOT EXISTS api_keys_tenant_idx ON api_keys (tenant_id);

-- --- providers and pricing -----------------------------------------------

CREATE TABLE IF NOT EXISTS providers (
    name       TEXT PRIMARY KEY,
    kind       TEXT        NOT NULL,
    enabled    BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_pricing (
    provider           TEXT           NOT NULL REFERENCES providers (name) ON DELETE CASCADE,
    model              TEXT           NOT NULL,
    -- NUMERIC, not float: money. Rounding drift in a float column would make the
    -- cost ledger disagree with the provider's invoice.
    input_per_million  NUMERIC(12, 6) NOT NULL,
    output_per_million NUMERIC(12, 6) NOT NULL,
    -- Prices change. Keeping effective_from in the key preserves history so an
    -- old request's cost still reconciles against the price in force that day.
    effective_from     TIMESTAMPTZ    NOT NULL DEFAULT now(),

    PRIMARY KEY (provider, model, effective_from),
    CONSTRAINT model_pricing_nonneg
        CHECK (input_per_million >= 0 AND output_per_million >= 0)
);

-- --- usage_log -----------------------------------------------------------

CREATE TABLE IF NOT EXISTS usage_log (
    -- The gateway's request ID. Primary key so the meter's at-least-once
    -- retries cannot double-count a request (ON CONFLICT DO NOTHING).
    request_id        TEXT PRIMARY KEY,
    tenant_id         TEXT        NOT NULL,
    key_id            TEXT,
    -- NULL when no provider was reached: a cache hit, or a request that failed
    -- before routing.
    provider          TEXT,
    model             TEXT        NOT NULL,
    -- The alias the client asked for, kept alongside the model that served it
    -- so routing changes stay auditable.
    model_alias       TEXT        NOT NULL,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(12, 6) NOT NULL DEFAULT 0,
    -- What a cache hit would have cost. Summing this is the "cost saved" panel.
    saved_usd         NUMERIC(12, 6) NOT NULL DEFAULT 0,
    cache_status      TEXT        NOT NULL,
    streamed          BOOLEAN     NOT NULL DEFAULT FALSE,
    status_code       INTEGER     NOT NULL,
    latency_ms        BIGINT      NOT NULL,
    attempts          INTEGER     NOT NULL DEFAULT 1,
    error_kind        TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT usage_log_tokens_nonneg
        CHECK (prompt_tokens >= 0 AND completion_tokens >= 0),
    CONSTRAINT usage_log_cost_nonneg
        CHECK (cost_usd >= 0 AND saved_usd >= 0)
);

-- No FK to tenants: the ledger is an immutable record of what happened. Deleting
-- a tenant must not cascade away the history of what they spent.

-- Dashboards and invoices both slice by tenant over a time range.
CREATE INDEX IF NOT EXISTS usage_log_tenant_time_idx
    ON usage_log (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS usage_log_time_idx
    ON usage_log (created_at DESC);

-- Supports "hit rate over the last hour" without scanning the whole table.
CREATE INDEX IF NOT EXISTS usage_log_cache_time_idx
    ON usage_log (cache_status, created_at DESC);

COMMIT;
