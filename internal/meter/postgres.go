package meter

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresSink writes usage rows to Postgres.
type PostgresSink struct {
	pool *pgxpool.Pool
}

// NewPostgresSink builds a sink over an existing pool.
func NewPostgresSink(pool *pgxpool.Pool) *PostgresSink {
	return &PostgresSink{pool: pool}
}

// WriteBatch inserts rows in a single statement.
//
// pgx's CopyFrom would be faster, but batches here are small (tens of rows) and
// a multi-row INSERT keeps the write compatible with connection poolers that
// reject the COPY protocol — a real constraint when the free-tier Postgres a
// portfolio deploy runs on sits behind PgBouncer.
func (s *PostgresSink) WriteBatch(ctx context.Context, rows []Record) error {
	if len(rows) == 0 {
		return nil
	}

	const cols = 17
	args := make([]any, 0, len(rows)*cols)
	placeholders := make([]byte, 0, len(rows)*48)

	for i, r := range rows {
		if i > 0 {
			placeholders = append(placeholders, ',')
		}
		placeholders = append(placeholders, '(')
		for c := 0; c < cols; c++ {
			if c > 0 {
				placeholders = append(placeholders, ',')
			}
			placeholders = append(placeholders, fmt.Sprintf("$%d", i*cols+c+1)...)
		}
		placeholders = append(placeholders, ')')

		args = append(args,
			r.RequestID, r.TenantID, nullable(r.KeyID), nullable(r.Provider),
			r.Model, r.ModelAlias, r.PromptTokens, r.CompletionTokens,
			r.CostUSD, r.SavedUSD, r.CacheStatus, r.Streamed,
			r.StatusCode, r.LatencyMS, r.Attempts, nullable(r.ErrorKind),
			r.CreatedAt,
		)
	}

	sql := `INSERT INTO usage_log (
		request_id, tenant_id, key_id, provider, model, model_alias,
		prompt_tokens, completion_tokens, cost_usd, saved_usd,
		cache_status, streamed, status_code, latency_ms, attempts,
		error_kind, created_at
	) VALUES ` + string(placeholders) + `
	ON CONFLICT (request_id) DO NOTHING`

	if _, err := s.pool.Exec(ctx, sql, args...); err != nil {
		return fmt.Errorf("usage_log insert: %w", err)
	}
	return nil
}

// nullable maps an empty string to NULL, so absent values are not stored as
// empty strings that would then need special-casing in every query.
func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// Close is a no-op: the pool is owned by the process, not the sink, and closing
// it here would take down the authenticator that shares it.
func (s *PostgresSink) Close() error { return nil }
