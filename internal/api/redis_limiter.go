package api

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

// RedisLimiter is a token bucket in Redis, shared across replicas so a tenant's
// limit is the limit — not the limit multiplied by however many replicas happen
// to be running.
type RedisLimiter struct {
	rdb        *redis.Client
	defaultRPM int
	burst      int
}

// NewRedisLimiter builds a distributed limiter.
func NewRedisLimiter(rdb *redis.Client, defaultRPM, burst int) *RedisLimiter {
	if burst <= 0 {
		burst = defaultRPM
	}
	return &RedisLimiter{rdb: rdb, defaultRPM: defaultRPM, burst: burst}
}

// limiterScript implements the token bucket atomically.
//
// Lua rather than Go-side logic because refill-check-decrement must be one
// operation: interleaved, two concurrent requests both read the same token count
// and both spend it, letting a tenant exceed their limit under exactly the
// concurrency the limit exists to control.
//
// Tokens are stored as a float and refilled continuously from elapsed time,
// which gives a smooth rate rather than the thundering edge of a fixed window.
//
// KEYS[1] = bucket hash
// ARGV    = rate_per_sec, capacity, now_ms, ttl_ms
// Returns {allowed, remaining, retry_after_ms}
var limiterScript = redis.NewScript(`
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local ttl      = tonumber(ARGV[4])

local tokens = tonumber(redis.call('HGET', key, 'tokens') or capacity)
local last   = tonumber(redis.call('HGET', key, 'last') or now)

-- Refill for elapsed time, capped at capacity.
local elapsed = math.max(0, now - last) / 1000
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0

if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_after = math.ceil((1 - tokens) / rate * 1000)
end

redis.call('HSET', key, 'tokens', tokens, 'last', now)
-- Idle buckets expire: a full bucket is indistinguishable from no bucket, so
-- keeping it would only leak memory for every key that ever made one request.
redis.call('PEXPIRE', key, ttl)

return {allowed, math.floor(tokens), retry_after}
`)

func limiterKey(keyID string) string { return "glm:v1:rl:" + keyID }

// Allow consumes a token for the tenant.
func (l *RedisLimiter) Allow(ctx context.Context, t *Tenant) (LimitResult, error) {
	rpm := l.defaultRPM
	capacity := l.burst
	// A per-key override replaces both rate and capacity: a key granted a higher
	// rate but left at the default burst would be throttled below its own limit.
	if t.RPM > 0 {
		rpm = t.RPM
		capacity = t.RPM
	}
	if rpm <= 0 {
		return LimitResult{Allowed: true}, nil
	}

	key := t.KeyID
	if key == "" {
		key = t.ID
	}

	// TTL comfortably exceeds a full refill, so a bucket is never dropped while
	// it still holds a deficit the tenant should be paying down.
	ttl := time.Duration(capacity)*time.Minute/time.Duration(rpm) + time.Minute

	res, err := limiterScript.Run(ctx, l.rdb,
		[]string{limiterKey(key)},
		strconv.FormatFloat(float64(rpm)/60, 'f', 6, 64),
		strconv.Itoa(capacity),
		strconv.FormatInt(time.Now().UnixMilli(), 10),
		strconv.FormatInt(ttl.Milliseconds(), 10),
	).Result()
	if err != nil {
		return LimitResult{}, fmt.Errorf("rate limit script: %w", err)
	}

	vals, ok := res.([]any)
	if !ok || len(vals) != 3 {
		return LimitResult{}, fmt.Errorf("rate limit: unexpected script result %T", res)
	}
	allowed, _ := vals[0].(int64)
	remaining, _ := vals[1].(int64)
	retryMS, _ := vals[2].(int64)

	return LimitResult{
		Allowed:    allowed == 1,
		Limit:      rpm,
		Remaining:  int(remaining),
		RetryAfter: time.Duration(retryMS) * time.Millisecond,
	}, nil
}
