package api

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// PostgresAuthenticator resolves API keys against the durable key table.
type PostgresAuthenticator struct {
	pool *pgxpool.Pool
}

// NewPostgresAuthenticator builds an authenticator over a pool.
func NewPostgresAuthenticator(pool *pgxpool.Pool) *PostgresAuthenticator {
	return &PostgresAuthenticator{pool: pool}
}

// Authenticate looks up a key by its digest.
//
// The lookup is by hash, never by raw key: the table stores only digests, so a
// database compromise yields nothing that can be replayed against the gateway.
func (a *PostgresAuthenticator) Authenticate(ctx context.Context, token string) (*Tenant, error) {
	const q = `
		SELECT k.id, k.tenant_id, t.name, COALESCE(k.rpm, 0)
		FROM api_keys k
		JOIN tenants t ON t.id = k.tenant_id
		WHERE k.key_hash = $1
		  AND k.revoked_at IS NULL
		  AND (k.expires_at IS NULL OR k.expires_at > now())
		  AND t.active`

	var (
		keyID    string
		tenantID string
		name     string
		rpm      int
	)
	err := a.pool.QueryRow(ctx, q, HashKey(token)).Scan(&keyID, &tenantID, &name, &rpm)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			// No row covers unknown, revoked, expired, and inactive-tenant
			// alike. They are deliberately indistinguishable to the caller: a
			// more specific message would tell an attacker which keys exist.
			return nil, ErrUnauthorized
		}
		// A real database failure must stay distinct from a rejected key, or an
		// outage would look to every client like their credentials were revoked.
		return nil, fmt.Errorf("authenticate: %w", err)
	}

	return &Tenant{ID: tenantID, Name: name, KeyID: keyID, RPM: rpm}, nil
}
