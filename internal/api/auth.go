package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Tenant is the authenticated caller behind a request.
type Tenant struct {
	// ID is the tenant's stable identifier, recorded in the usage log.
	ID string
	// Name is a human-readable label for logs and dashboards.
	Name string
	// KeyID identifies the specific API key used, so a single tenant's keys can
	// be rate-limited and revoked independently.
	KeyID string
	// RPM overrides the default rate limit when positive.
	RPM int
}

// ErrUnauthorized reports an unknown, malformed, or revoked key.
var ErrUnauthorized = errors.New("unauthorized")

// Authenticator resolves a bearer token into a Tenant. Implementations must be
// safe for concurrent use.
type Authenticator interface {
	Authenticate(ctx context.Context, token string) (*Tenant, error)
}

// HashKey derives the storage form of an API key. Keys are stored and compared
// as SHA-256 digests so a database leak does not hand over usable credentials.
//
// A fast hash is the right choice here, unlike for passwords: API keys are
// high-entropy random strings, so brute-forcing a digest is infeasible without a
// slow KDF, and auth sits on the hot path of every request.
func HashKey(raw string) string {
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

// StaticAuthenticator authenticates against an in-memory key map. It backs local
// development and tests, where standing up Postgres to make one request is
// friction with no benefit.
type StaticAuthenticator struct {
	// byHash maps a key's SHA-256 digest to its tenant.
	byHash map[string]*Tenant
}

// NewStaticAuthenticator builds an authenticator from raw key -> tenant pairs.
func NewStaticAuthenticator(keys map[string]*Tenant) *StaticAuthenticator {
	byHash := make(map[string]*Tenant, len(keys))
	for raw, t := range keys {
		byHash[HashKey(raw)] = t
	}
	return &StaticAuthenticator{byHash: byHash}
}

func (a *StaticAuthenticator) Authenticate(_ context.Context, token string) (*Tenant, error) {
	t, ok := a.byHash[HashKey(token)]
	if !ok {
		return nil, ErrUnauthorized
	}
	return t, nil
}

// AllowAllAuthenticator accepts any request and attributes it to a single
// tenant. Intended for local development only; the server refuses to use it
// unless auth is explicitly disabled in config.
type AllowAllAuthenticator struct{}

func (AllowAllAuthenticator) Authenticate(context.Context, string) (*Tenant, error) {
	return &Tenant{ID: "local", Name: "local-dev", KeyID: "local"}, nil
}

// CachingAuthenticator memoizes lookups for a short TTL. Without it, every
// request pays a database round trip before doing any work — which for a cache
// hit would be the only I/O on the path, and would dominate its latency.
type CachingAuthenticator struct {
	inner Authenticator
	ttl   time.Duration
	now   func() time.Time

	mu sync.RWMutex
	m  map[string]cachedAuth
}

type cachedAuth struct {
	tenant  *Tenant
	err     error
	expires time.Time
}

// NewCachingAuthenticator wraps inner with a TTL cache.
func NewCachingAuthenticator(inner Authenticator, ttl time.Duration) *CachingAuthenticator {
	return &CachingAuthenticator{inner: inner, ttl: ttl, now: time.Now, m: map[string]cachedAuth{}}
}

func (a *CachingAuthenticator) Authenticate(ctx context.Context, token string) (*Tenant, error) {
	// Key the cache by digest so raw tokens are never held in memory.
	h := HashKey(token)

	a.mu.RLock()
	e, ok := a.m[h]
	a.mu.RUnlock()
	if ok && a.now().Before(e.expires) {
		return e.tenant, e.err
	}

	t, err := a.inner.Authenticate(ctx, token)
	// Cache negative results too, so a flood of bad keys cannot turn into a
	// flood of database queries. The TTL bounds how long a revoked key keeps
	// working and how long a newly issued key is rejected.
	if err != nil && !errors.Is(err, ErrUnauthorized) {
		return nil, err // transient failure: do not cache
	}

	a.mu.Lock()
	a.m[h] = cachedAuth{tenant: t, err: err, expires: a.now().Add(a.ttl)}
	a.mu.Unlock()
	return t, err
}

// Invalidate drops a cached entry, letting key revocation take effect without
// waiting out the TTL.
func (a *CachingAuthenticator) Invalidate(token string) {
	a.mu.Lock()
	delete(a.m, HashKey(token))
	a.mu.Unlock()
}

// bearerToken extracts a token from the Authorization header. It also accepts
// the api-key header that some OpenAI-compatible clients send.
func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if h == "" {
		return r.Header.Get("api-key")
	}
	// Compare the scheme case-insensitively per RFC 7235.
	if len(h) >= 7 && strings.EqualFold(h[:7], "Bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}
