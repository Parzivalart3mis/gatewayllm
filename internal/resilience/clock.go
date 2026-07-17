package resilience

import "time"

// nowMillis returns the current wall-clock time in milliseconds.
//
// Wall clock, not monotonic: the value is shared across replicas through Redis,
// and a monotonic reading is only meaningful within one process. This makes
// breaker timing sensitive to clock skew between replicas, which is acceptable
// because the cooldown is measured in tens of seconds and NTP keeps skew orders
// of magnitude below that.
func nowMillis() int64 {
	return time.Now().UnixMilli()
}
