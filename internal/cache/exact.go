package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// ExactTier is the Redis-backed exact-match cache. A repeat of a byte-identical
// request is answered from here in a single round trip, with no embedding call
// and no provider call.
type ExactTier struct {
	rdb *redis.Client
}

// NewExactTier builds the Redis tier.
func NewExactTier(rdb *redis.Client) *ExactTier {
	return &ExactTier{rdb: rdb}
}

// Get returns the entry for key, or nil when absent. A miss is not an error:
// it is the expected outcome most of the time, and callers should not have to
// distinguish "nothing cached" from "cache broken" via error inspection.
func (t *ExactTier) Get(ctx context.Context, key Key) (*Entry, error) {
	raw, err := t.rdb.Get(ctx, key.Redis()).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, nil
		}
		return nil, fmt.Errorf("exact cache get: %w", err)
	}

	var e Entry
	if err := json.Unmarshal(raw, &e); err != nil {
		// A corrupt entry would otherwise resurface on every request for this
		// key. Drop it so the next call repopulates cleanly.
		t.rdb.Del(ctx, key.Redis())
		return nil, fmt.Errorf("exact cache: corrupt entry evicted: %w", err)
	}
	return &e, nil
}

// Set stores an entry under key with the given TTL.
func (t *ExactTier) Set(ctx context.Context, key Key, e *Entry, ttl time.Duration) error {
	raw, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("exact cache marshal: %w", err)
	}
	if err := t.rdb.Set(ctx, key.Redis(), raw, ttl).Err(); err != nil {
		return fmt.Errorf("exact cache set: %w", err)
	}
	return nil
}

// Delete removes an entry.
func (t *ExactTier) Delete(ctx context.Context, key Key) error {
	return t.rdb.Del(ctx, key.Redis()).Err()
}

// Ping reports whether Redis is reachable.
func (t *ExactTier) Ping(ctx context.Context) error {
	return t.rdb.Ping(ctx).Err()
}
