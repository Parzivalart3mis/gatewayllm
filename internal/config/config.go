// Package config loads gateway configuration from a YAML file with environment
// variable interpolation, and validates it before the process starts serving.
package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration document.
type Config struct {
	Server     Server           `yaml:"server"`
	Providers  []ProviderConfig `yaml:"providers"`
	Router     Router           `yaml:"router"`
	Cache      Cache            `yaml:"cache"`
	Embed      Embed            `yaml:"embed"`
	Resilience Resilience       `yaml:"resilience"`
	RateLimit  RateLimit        `yaml:"rate_limit"`
	Stores     Stores           `yaml:"stores"`
	Meter      Meter            `yaml:"meter"`
	Obs        Obs              `yaml:"obs"`
	Pricing    map[string]Price `yaml:"pricing"`
}

// Server holds HTTP listener settings.
type Server struct {
	Addr            string        `yaml:"addr"`
	ReadTimeout     time.Duration `yaml:"read_timeout"`
	WriteTimeout    time.Duration `yaml:"write_timeout"`
	ShutdownTimeout time.Duration `yaml:"shutdown_timeout"`
	// MaxBodyBytes caps the request body the gateway will read.
	MaxBodyBytes int64 `yaml:"max_body_bytes"`
}

// ProviderConfig configures one upstream backend.
type ProviderConfig struct {
	// Name is the unique identifier referenced by router policies.
	Name string `yaml:"name"`
	// Kind selects the adapter implementation: openai, groq, gemini, or mock.
	Kind    string        `yaml:"kind"`
	BaseURL string        `yaml:"base_url"`
	APIKey  string        `yaml:"api_key"`
	Timeout time.Duration `yaml:"timeout"`
	// Models lists upstream model IDs this provider serves.
	Models []string `yaml:"models"`
	// Enabled allows keeping a provider in config without routing to it.
	Enabled *bool `yaml:"enabled"`
}

// IsEnabled defaults to true when unset.
func (p ProviderConfig) IsEnabled() bool { return p.Enabled == nil || *p.Enabled }

// Router maps client-facing model aliases onto ordered provider targets.
type Router struct {
	// Aliases maps the model name a client sends to a routing policy.
	Aliases map[string]Alias `yaml:"aliases"`
	// DefaultAlias serves requests whose model matches no alias. Empty means
	// unknown models are rejected.
	DefaultAlias string `yaml:"default_alias"`
}

// Alias is a routing policy: an ordered list of targets tried in sequence.
type Alias struct {
	// Strategy selects among targets: "priority" (in order) or "weighted".
	Strategy string   `yaml:"strategy"`
	Targets  []Target `yaml:"targets"`
}

// Target binds a provider to the upstream model it should be called with.
type Target struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
	// Weight is used by the weighted strategy; ignored by priority.
	Weight int `yaml:"weight"`
}

// Cache configures the two-tier cache.
type Cache struct {
	Enabled bool `yaml:"enabled"`
	// ExactTTL bounds how long an exact-match entry stays in Redis.
	ExactTTL time.Duration `yaml:"exact_ttl"`
	// Semantic is the Qdrant-backed similarity tier.
	Semantic SemanticCache `yaml:"semantic"`
	// MaxTemperature disables caching above this temperature: high-temperature
	// requests ask for variety, and serving a cached answer defeats the request.
	MaxTemperature float64 `yaml:"max_temperature"`
}

// SemanticCache configures the Qdrant tier.
type SemanticCache struct {
	Enabled bool `yaml:"enabled"`
	// Collection is the Qdrant collection holding cached prompts.
	Collection string `yaml:"collection"`
	// Threshold is the minimum cosine similarity for a hit. Tuned empirically;
	// too low and the cache returns answers to different questions.
	Threshold float64 `yaml:"threshold"`
	// TTL bounds entry age; entries older than this are ignored on read.
	TTL time.Duration `yaml:"ttl"`
	// MaxTempDelta caps how far a cached entry's temperature may differ from the
	// request's before the entry is rejected as parameter-incompatible.
	MaxTempDelta float64 `yaml:"max_temp_delta"`
	// VectorSize must match the embedder's output dimensionality.
	VectorSize int `yaml:"vector_size"`
}

// Embed configures the embedder used by the semantic tier.
type Embed struct {
	// Kind selects the implementation: "sidecar" or "api".
	Kind    string        `yaml:"kind"`
	BaseURL string        `yaml:"base_url"`
	Model   string        `yaml:"model"`
	APIKey  string        `yaml:"api_key"`
	Timeout time.Duration `yaml:"timeout"`
}

// Resilience configures the wrapper around every provider call.
type Resilience struct {
	// MaxAttempts counts the initial try plus retries against one provider.
	MaxAttempts int           `yaml:"max_attempts"`
	BaseBackoff time.Duration `yaml:"base_backoff"`
	MaxBackoff  time.Duration `yaml:"max_backoff"`
	Breaker     Breaker       `yaml:"breaker"`
}

// Breaker configures the per-provider circuit breaker.
type Breaker struct {
	Enabled bool `yaml:"enabled"`
	// FailureThreshold is the consecutive-failure count that opens the circuit.
	FailureThreshold int `yaml:"failure_threshold"`
	// OpenDuration is how long the circuit stays open before probing half-open.
	OpenDuration time.Duration `yaml:"open_duration"`
	// HalfOpenProbes is how many successes in half-open close the circuit.
	HalfOpenProbes int `yaml:"half_open_probes"`
}

// RateLimit configures per-tenant token buckets.
type RateLimit struct {
	Enabled bool `yaml:"enabled"`
	// DefaultRPM applies to keys without a per-key override.
	DefaultRPM int `yaml:"default_rpm"`
	// Burst is the bucket capacity; defaults to DefaultRPM when unset.
	Burst int `yaml:"burst"`
}

// Stores holds connection settings for the three backing stores.
type Stores struct {
	RedisURL string `yaml:"redis_url"`
	// QdrantURL is the base URL. Local Qdrant needs no auth; managed Qdrant
	// (Qdrant Cloud) requires an API key alongside it.
	QdrantURL    string `yaml:"qdrant_url"`
	QdrantAPIKey string `yaml:"qdrant_api_key"`
	PostgresURL  string `yaml:"postgres_url"`
}

// Meter configures the async usage/cost writer.
type Meter struct {
	Enabled bool `yaml:"enabled"`
	// BufferSize bounds the in-memory queue of pending usage rows. When full,
	// rows are dropped rather than blocking the response path.
	BufferSize int `yaml:"buffer_size"`
	// FlushInterval bounds how long a row waits before being written.
	FlushInterval time.Duration `yaml:"flush_interval"`
	// BatchSize is the max rows per INSERT.
	BatchSize int `yaml:"batch_size"`
}

// Obs configures tracing and metrics.
type Obs struct {
	ServiceName string `yaml:"service_name"`
	// OTLPEndpoint enables tracing when set.
	OTLPEndpoint string  `yaml:"otlp_endpoint"`
	SampleRatio  float64 `yaml:"sample_ratio"`
	MetricsPath  string  `yaml:"metrics_path"`
	// MetricsAddr is the listener for the Prometheus endpoint. The default
	// (:9090) serves metrics on a second port, which keeps them off the public
	// API port for the compose/VPS deployment. The sentinel "inline" instead
	// mounts /metrics on the main API server — required on single-port hosts
	// like Cloud Run, which route to exactly one port.
	MetricsAddr string `yaml:"metrics_addr"`
	LogLevel    string `yaml:"log_level"`
}

// MetricsInline reports whether metrics should be served on the main API mux
// rather than a dedicated listener.
func (o Obs) MetricsInline() bool { return o.MetricsAddr == "inline" }

// Price is the per-million-token cost for one model, used by the meter.
type Price struct {
	InputPerMillion  float64 `yaml:"input_per_million"`
	OutputPerMillion float64 `yaml:"output_per_million"`
}

// envPattern matches ${VAR} and ${VAR:-default}.
var envPattern = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}`)

// expandEnv substitutes ${VAR} and ${VAR:-default} from the environment. An
// unset variable with no default expands to empty, which validation then catches
// if the field was required.
func expandEnv(b []byte) []byte {
	return envPattern.ReplaceAllFunc(b, func(m []byte) []byte {
		groups := envPattern.FindSubmatch(m)
		if v, ok := os.LookupEnv(string(groups[1])); ok && v != "" {
			return []byte(v)
		}
		return groups[2]
	})
}

// Load reads, interpolates, defaults, and validates a config file.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	if err := yaml.Unmarshal(expandEnv(raw), &c); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	c.applyDefaults()
	if err := c.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config %s: %w", path, err)
	}
	return &c, nil
}

func (c *Config) applyDefaults() {
	setIfZero(&c.Server.Addr, ":8080")
	setIfZeroD(&c.Server.ReadTimeout, 30*time.Second)
	setIfZeroD(&c.Server.WriteTimeout, 10*time.Minute) // long: streaming responses
	setIfZeroD(&c.Server.ShutdownTimeout, 20*time.Second)
	if c.Server.MaxBodyBytes == 0 {
		c.Server.MaxBodyBytes = 4 << 20
	}

	for i := range c.Providers {
		setIfZeroD(&c.Providers[i].Timeout, 60*time.Second)
	}

	if c.Resilience.MaxAttempts == 0 {
		c.Resilience.MaxAttempts = 3
	}
	setIfZeroD(&c.Resilience.BaseBackoff, 200*time.Millisecond)
	setIfZeroD(&c.Resilience.MaxBackoff, 5*time.Second)
	if c.Resilience.Breaker.FailureThreshold == 0 {
		c.Resilience.Breaker.FailureThreshold = 5
	}
	setIfZeroD(&c.Resilience.Breaker.OpenDuration, 30*time.Second)
	if c.Resilience.Breaker.HalfOpenProbes == 0 {
		c.Resilience.Breaker.HalfOpenProbes = 2
	}

	setIfZeroD(&c.Cache.ExactTTL, time.Hour)
	if c.Cache.MaxTemperature == 0 {
		c.Cache.MaxTemperature = 0.3
	}
	setIfZero(&c.Cache.Semantic.Collection, "gatewayllm_semantic")
	if c.Cache.Semantic.Threshold == 0 {
		c.Cache.Semantic.Threshold = 0.95
	}
	setIfZeroD(&c.Cache.Semantic.TTL, 24*time.Hour)
	if c.Cache.Semantic.MaxTempDelta == 0 {
		c.Cache.Semantic.MaxTempDelta = 0.05
	}
	if c.Cache.Semantic.VectorSize == 0 {
		c.Cache.Semantic.VectorSize = 384 // all-MiniLM-L6-v2, the sidecar default
	}

	setIfZero(&c.Embed.Kind, "sidecar")
	setIfZeroD(&c.Embed.Timeout, 10*time.Second)

	if c.RateLimit.DefaultRPM == 0 {
		c.RateLimit.DefaultRPM = 60
	}
	if c.RateLimit.Burst == 0 {
		c.RateLimit.Burst = c.RateLimit.DefaultRPM
	}

	if c.Meter.BufferSize == 0 {
		c.Meter.BufferSize = 4096
	}
	setIfZeroD(&c.Meter.FlushInterval, 2*time.Second)
	if c.Meter.BatchSize == 0 {
		c.Meter.BatchSize = 128
	}

	setIfZero(&c.Obs.ServiceName, "gatewayllm")
	setIfZero(&c.Obs.MetricsPath, "/metrics")
	setIfZero(&c.Obs.MetricsAddr, ":9090")
	setIfZero(&c.Obs.LogLevel, "info")
	if c.Obs.SampleRatio == 0 {
		c.Obs.SampleRatio = 1.0
	}

	for name, a := range c.Router.Aliases {
		if a.Strategy == "" {
			a.Strategy = "priority"
			c.Router.Aliases[name] = a
		}
	}
}

// Validate enforces the invariants the rest of the process assumes. It reports
// every problem at once so a misconfigured deploy fails with a full list.
func (c *Config) Validate() error {
	var errs []string

	enabled := map[string]ProviderConfig{}
	for i, p := range c.Providers {
		switch {
		case p.Name == "":
			errs = append(errs, fmt.Sprintf("providers[%d]: name is required", i))
			continue
		case p.Kind == "":
			errs = append(errs, fmt.Sprintf("providers[%s]: kind is required", p.Name))
		}
		if _, dup := enabled[p.Name]; dup {
			errs = append(errs, fmt.Sprintf("providers[%s]: duplicate provider name", p.Name))
		}
		if p.IsEnabled() {
			if p.Kind != "mock" && p.APIKey == "" {
				errs = append(errs, fmt.Sprintf("providers[%s]: api_key is required (is the env var set?)", p.Name))
			}
			enabled[p.Name] = p
		}
	}
	if len(enabled) == 0 {
		errs = append(errs, "providers: at least one enabled provider is required")
	}

	if len(c.Router.Aliases) == 0 {
		errs = append(errs, "router.aliases: at least one alias is required")
	}
	for name, a := range c.Router.Aliases {
		if len(a.Targets) == 0 {
			errs = append(errs, fmt.Sprintf("router.aliases[%s]: at least one target is required", name))
		}
		if a.Strategy != "priority" && a.Strategy != "weighted" {
			errs = append(errs, fmt.Sprintf("router.aliases[%s]: strategy must be priority or weighted, got %q", name, a.Strategy))
		}
		for i, t := range a.Targets {
			p, ok := enabled[t.Provider]
			if !ok {
				errs = append(errs, fmt.Sprintf("router.aliases[%s].targets[%d]: unknown or disabled provider %q", name, i, t.Provider))
				continue
			}
			if t.Model == "" {
				errs = append(errs, fmt.Sprintf("router.aliases[%s].targets[%d]: model is required", name, i))
				continue
			}
			// A target naming a model the provider does not serve is a typo that
			// would otherwise surface as an upstream 404 under production load.
			if len(p.Models) > 0 && !contains(p.Models, t.Model) {
				errs = append(errs, fmt.Sprintf("router.aliases[%s].targets[%d]: provider %q does not list model %q", name, i, t.Provider, t.Model))
			}
			if a.Strategy == "weighted" && t.Weight <= 0 {
				errs = append(errs, fmt.Sprintf("router.aliases[%s].targets[%d]: weighted strategy requires a positive weight", name, i))
			}
		}
	}
	if c.Router.DefaultAlias != "" {
		if _, ok := c.Router.Aliases[c.Router.DefaultAlias]; !ok {
			errs = append(errs, fmt.Sprintf("router.default_alias: %q is not a defined alias", c.Router.DefaultAlias))
		}
	}

	if c.Cache.Enabled && c.Stores.RedisURL == "" {
		errs = append(errs, "stores.redis_url: required when cache.enabled is true")
	}
	if c.RateLimit.Enabled && c.Stores.RedisURL == "" {
		errs = append(errs, "stores.redis_url: required when rate_limit.enabled is true")
	}
	if c.Meter.Enabled && c.Stores.PostgresURL == "" {
		errs = append(errs, "stores.postgres_url: required when meter.enabled is true")
	}

	if c.Cache.Semantic.Enabled {
		if !c.Cache.Enabled {
			errs = append(errs, "cache.semantic.enabled: requires cache.enabled")
		}
		if c.Stores.QdrantURL == "" {
			errs = append(errs, "stores.qdrant_url: required when cache.semantic.enabled is true")
		}
		if c.Embed.BaseURL == "" {
			errs = append(errs, "embed.base_url: required when cache.semantic.enabled is true")
		}
		if c.Embed.Kind != "sidecar" && c.Embed.Kind != "api" {
			errs = append(errs, fmt.Sprintf("embed.kind: must be sidecar or api, got %q", c.Embed.Kind))
		}
		if c.Embed.Kind == "api" && c.Embed.APIKey == "" {
			errs = append(errs, "embed.api_key: required when embed.kind is api")
		}
		if t := c.Cache.Semantic.Threshold; t <= 0 || t > 1 {
			errs = append(errs, fmt.Sprintf("cache.semantic.threshold: must be in (0,1], got %v", t))
		}
		// Below this, cosine similarity stops implying the prompts mean the same
		// thing, and the cache starts answering questions nobody asked.
		if c.Cache.Semantic.Threshold < 0.80 {
			errs = append(errs, fmt.Sprintf("cache.semantic.threshold: %v is dangerously low; false hits become likely below 0.80", c.Cache.Semantic.Threshold))
		}
		if c.Cache.Semantic.VectorSize <= 0 {
			errs = append(errs, "cache.semantic.vector_size: must be positive")
		}
	}

	if c.Resilience.MaxAttempts < 1 {
		errs = append(errs, "resilience.max_attempts: must be at least 1")
	}
	if c.RateLimit.Enabled && c.RateLimit.DefaultRPM <= 0 {
		errs = append(errs, "rate_limit.default_rpm: must be positive")
	}

	if len(errs) > 0 {
		return fmt.Errorf("\n  - %s", strings.Join(errs, "\n  - "))
	}
	return nil
}

func contains(hay []string, needle string) bool {
	for _, h := range hay {
		if h == needle {
			return true
		}
	}
	return false
}

func setIfZero(s *string, v string) {
	if *s == "" {
		*s = v
	}
}

func setIfZeroD(d *time.Duration, v time.Duration) {
	if *d == 0 {
		*d = v
	}
}
