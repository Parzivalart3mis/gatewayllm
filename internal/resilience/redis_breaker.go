package resilience

import (
	"context"
	"fmt"
	"strconv"

	"github.com/redis/go-redis/v9"
)

// RedisBreakerStore keeps circuit state in Redis so it is shared across
// replicas. This is the point of putting it there: when one replica discovers a
// provider is down, every other replica stops calling it immediately, instead of
// each independently burning its own failure budget to learn the same fact.
type RedisBreakerStore struct {
	rdb *redis.Client
	cfg BreakerConfig
}

// NewRedisBreakerStore builds a Redis-backed breaker store.
func NewRedisBreakerStore(rdb *redis.Client, cfg BreakerConfig) *RedisBreakerStore {
	return &RedisBreakerStore{rdb: rdb, cfg: cfg}
}

func breakerKey(provider string) string { return "glm:v1:breaker:" + provider }

// The state machine runs as Lua so each transition is atomic. Done as
// read-modify-write from Go, concurrent requests would interleave and a burst of
// failures could each read "4 failures" and none would cross the threshold.
//
// KEYS[1] = breaker hash
// ARGV    = event, threshold, open_ms, half_open_probes, now_ms
// Returns the resulting state.
var breakerScript = redis.NewScript(`
local key       = KEYS[1]
local event     = ARGV[1]
local threshold = tonumber(ARGV[2])
local open_ms   = tonumber(ARGV[3])
local probes    = tonumber(ARGV[4])
local now       = tonumber(ARGV[5])

local state     = redis.call('HGET', key, 'state') or 'closed'
local failures  = tonumber(redis.call('HGET', key, 'failures') or '0')
local successes = tonumber(redis.call('HGET', key, 'successes') or '0')
local opened_at = tonumber(redis.call('HGET', key, 'opened_at') or '0')

-- An open circuit becomes half-open once its cooldown elapses. Evaluated on
-- every access so no timer or sweeper is needed.
if state == 'open' and (now - opened_at) >= open_ms then
  state = 'half_open'
  successes = 0
end

if event == 'success' then
  if state == 'half_open' then
    successes = successes + 1
    if successes >= probes then
      state = 'closed'; failures = 0; successes = 0
    end
  else
    state = 'closed'; failures = 0
  end
elseif event == 'failure' then
  if state == 'half_open' then
    -- Still failing under probe: reopen at once rather than spending the
    -- remaining probe budget on a provider that is evidently still down.
    state = 'open'; opened_at = now; successes = 0
  else
    failures = failures + 1
    if failures >= threshold then
      state = 'open'; opened_at = now
    end
  end
end

redis.call('HSET', key, 'state', state, 'failures', failures,
           'successes', successes, 'opened_at', opened_at)
-- Expire idle keys: a provider removed from config should not leave state in
-- Redis forever. Any live provider refreshes this on every call.
redis.call('PEXPIRE', key, math.max(open_ms * 10, 600000))

return state
`)

func (s *RedisBreakerStore) run(ctx context.Context, provider, event string) (State, error) {
	now := nowMillis()
	res, err := breakerScript.Run(ctx, s.rdb,
		[]string{breakerKey(provider)},
		event,
		strconv.Itoa(s.cfg.FailureThreshold),
		strconv.FormatInt(s.cfg.OpenDuration.Milliseconds(), 10),
		strconv.Itoa(s.cfg.HalfOpenProbes),
		strconv.FormatInt(now, 10),
	).Result()
	if err != nil {
		return StateClosed, fmt.Errorf("breaker %s: %w", event, err)
	}
	str, ok := res.(string)
	if !ok {
		return StateClosed, fmt.Errorf("breaker: unexpected script result %T", res)
	}
	return State(str), nil
}

// Snapshot reports current state, applying any due open→half-open transition.
func (s *RedisBreakerStore) Snapshot(ctx context.Context, provider string) (State, error) {
	return s.run(ctx, provider, "peek")
}

// RecordSuccess reports a successful call.
func (s *RedisBreakerStore) RecordSuccess(ctx context.Context, provider string) (State, error) {
	return s.run(ctx, provider, "success")
}

// RecordFailure reports a failed call.
func (s *RedisBreakerStore) RecordFailure(ctx context.Context, provider string) (State, error) {
	return s.run(ctx, provider, "failure")
}
