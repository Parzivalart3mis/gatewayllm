// Package store holds clients for the three backing stores. The gateway binary
// is stateless; everything durable or shared lives behind one of these.
package store

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// NewRedis connects to Redis and verifies reachability before returning, so a
// bad URL fails at startup instead of on the first cache lookup.
func NewRedis(ctx context.Context, url string) (*redis.Client, error) {
	opt, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("redis: parse url: %w", err)
	}
	// Redis sits on the request hot path, where a slow call is worse than a
	// missing one: the cache exists to save time, so its own timeouts must be
	// short enough that a degraded Redis cannot cost more than it saves.
	opt.DialTimeout = 2 * time.Second
	opt.ReadTimeout = 500 * time.Millisecond
	opt.WriteTimeout = 500 * time.Millisecond
	opt.PoolSize = 50
	opt.MinIdleConns = 5

	c := redis.NewClient(opt)

	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := c.Ping(pingCtx).Err(); err != nil {
		_ = c.Close()
		return nil, fmt.Errorf("redis: ping: %w", err)
	}
	return c, nil
}
