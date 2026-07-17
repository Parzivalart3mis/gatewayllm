// Command gateway is the GatewayLLM server: a single stateless binary fronting
// multiple LLM providers with policy routing, failover, and a two-tier cache.
//
// All state lives in Redis, Qdrant, and Postgres, so scaling means running more
// identical replicas behind a load balancer.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"

	"github.com/yash/gatewayllm/internal/api"
	"github.com/yash/gatewayllm/internal/cache"
	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/embed"
	"github.com/yash/gatewayllm/internal/meter"
	"github.com/yash/gatewayllm/internal/obs"
	"github.com/yash/gatewayllm/internal/provider"
	"github.com/yash/gatewayllm/internal/resilience"
	"github.com/yash/gatewayllm/internal/router"
	"github.com/yash/gatewayllm/internal/store"
)

// version is overridden at build time via -ldflags.
var version = "dev"

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "path to the config file")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println("gatewayllm", version)
		return nil
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return err
	}

	log := newLogger(cfg.Obs.LogLevel)
	slog.SetDefault(log)
	log.Info("starting gatewayllm", "version", version, "config", *configPath)

	// Signal-aware root context: every long-lived component derives from it, so
	// one Ctrl-C unwinds the whole process cleanly.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	app, err := build(ctx, cfg, log)
	if err != nil {
		return err
	}
	defer app.close(log)

	// Metrics listen on their own port so the scrape endpoint is not exposed on
	// whatever the API port is published as.
	go app.serveMetrics(ctx, cfg, log)

	return app.server.ListenAndServe(ctx)
}

// application holds constructed components and their teardown order.
type application struct {
	server  *api.Server
	meter   *meter.Meter
	rdb     *redis.Client
	pgClose func()
	tpClose func(context.Context) error
	reg     *prometheus.Registry
}

// build constructs every layer from config. Each store is optional: a config
// with caching, metering, and rate limiting off runs with no backing stores at
// all, which is what makes the early build stages independently deployable.
func build(ctx context.Context, cfg *config.Config, log *slog.Logger) (*application, error) {
	app := &application{}

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	metrics := obs.NewMetrics(reg)
	app.reg = reg

	shutdownTracing, err := obs.InitTracing(ctx, cfg.Obs, version)
	if err != nil {
		return nil, fmt.Errorf("tracing: %w", err)
	}
	app.tpClose = shutdownTracing
	if cfg.Obs.OTLPEndpoint != "" {
		log.Info("tracing enabled", "endpoint", cfg.Obs.OTLPEndpoint, "sample_ratio", cfg.Obs.SampleRatio)
	}

	// --- providers + router ---
	registry, err := provider.Build(cfg.Providers)
	if err != nil {
		return nil, err
	}
	log.Info("providers registered", "count", registry.Len(), "names", registry.Names())

	rt, err := router.New(cfg.Router, registry)
	if err != nil {
		return nil, err
	}
	log.Info("router configured", "aliases", rt.Aliases())

	// --- redis (cache, rate limits, breaker state) ---
	var rdb *redis.Client
	if cfg.Stores.RedisURL != "" && (cfg.Cache.Enabled || cfg.RateLimit.Enabled || cfg.Resilience.Breaker.Enabled) {
		rdb, err = store.NewRedis(ctx, cfg.Stores.RedisURL)
		if err != nil {
			return nil, err
		}
		app.rdb = rdb
		log.Info("redis connected")
	}

	// --- resilience ---
	breakerCfg := resilience.BreakerConfig{
		FailureThreshold: cfg.Resilience.Breaker.FailureThreshold,
		OpenDuration:     cfg.Resilience.Breaker.OpenDuration,
		HalfOpenProbes:   cfg.Resilience.Breaker.HalfOpenProbes,
	}
	var breakerStore resilience.BreakerStore
	switch {
	case rdb != nil && cfg.Resilience.Breaker.Enabled:
		// Shared state: one replica learning a provider is down protects them all.
		breakerStore = resilience.NewRedisBreakerStore(rdb, breakerCfg)
		log.Info("circuit breaker: redis-backed (shared across replicas)")
	default:
		breakerStore = resilience.NewLocalBreakerStore(breakerCfg)
		if cfg.Resilience.Breaker.Enabled {
			log.Warn("circuit breaker: in-memory (per-replica); set stores.redis_url to share state")
		}
	}
	breaker := resilience.NewBreaker(breakerStore, cfg.Resilience.Breaker.Enabled)

	exec := resilience.New(resilience.Config{
		MaxAttempts: cfg.Resilience.MaxAttempts,
		BaseBackoff: cfg.Resilience.BaseBackoff,
		MaxBackoff:  cfg.Resilience.MaxBackoff,
		Breaker:     breakerCfg,
	}, breaker)

	// Wire provider attempts into metrics here rather than having resilience
	// import Prometheus.
	exec.OnAttempt = func(info resilience.AttemptInfo) {
		result := "success"
		switch {
		case info.Skipped:
			result = "circuit_open"
		case info.Err != nil:
			result = "error"
			if pe := provider.AsError(info.Err); pe != nil {
				result = string(pe.Kind)
			}
		}
		metrics.ProviderAttempts.WithLabelValues(info.Provider, info.Model, result).Inc()
		if !info.Skipped {
			metrics.ProviderDuration.WithLabelValues(info.Provider, info.Model).Observe(info.Duration.Seconds())
		}
		metrics.BreakerState.WithLabelValues(info.Provider).Set(obs.BreakerStateValue(string(info.BreakerState)))
	}

	// --- cache ---
	var c *cache.Cache
	if cfg.Cache.Enabled {
		if rdb == nil {
			return nil, errors.New("cache.enabled requires stores.redis_url")
		}
		exact := cache.NewExactTier(rdb)

		var semantic *cache.SemanticTier
		var embedder embed.Embedder
		if cfg.Cache.Semantic.Enabled {
			embedder, err = embed.Build(cfg.Embed, cfg.Cache.Semantic.VectorSize)
			if err != nil {
				return nil, err
			}
			qc := store.NewQdrant(cfg.Stores.QdrantURL, "", 3*time.Second)
			semantic = cache.NewSemanticTier(qc, cfg.Cache.Semantic)

			// Fail startup on a dimension mismatch rather than discovering it as
			// a per-request error once traffic arrives.
			initCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
			err := semantic.EnsureCollection(initCtx, cfg.Cache.Semantic.VectorSize)
			cancel()
			if err != nil {
				return nil, fmt.Errorf("qdrant: %w", err)
			}
			log.Info("semantic cache ready",
				"collection", cfg.Cache.Semantic.Collection,
				"threshold", cfg.Cache.Semantic.Threshold,
				"embedder", embedder.Name(),
				"dims", cfg.Cache.Semantic.VectorSize)
		}

		c = cache.New(cache.Options{
			Config:   cfg.Cache,
			Exact:    exact,
			Semantic: semantic,
			Embedder: embedder,
			Metrics:  metrics,
			Logger:   log,
		})
		log.Info("cache enabled", "exact_ttl", cfg.Cache.ExactTTL, "max_temperature", cfg.Cache.MaxTemperature)
	}

	// --- postgres (auth + meter) ---
	var auth api.Authenticator = api.AllowAllAuthenticator{}
	var m *meter.Meter

	if cfg.Stores.PostgresURL != "" {
		pool, err := store.NewPostgres(ctx, cfg.Stores.PostgresURL)
		if err != nil {
			return nil, err
		}
		app.pgClose = pool.Close

		if err := store.Migrate(ctx, pool); err != nil {
			return nil, err
		}
		log.Info("postgres connected and migrated")

		// Short TTL: long enough to keep auth off the hot path, short enough
		// that a revoked key stops working promptly.
		auth = api.NewCachingAuthenticator(api.NewPostgresAuthenticator(pool), 30*time.Second)

		if cfg.Meter.Enabled {
			m = meter.New(meter.Options{
				Sink:          meter.NewPostgresSink(pool),
				BufferSize:    cfg.Meter.BufferSize,
				BatchSize:     cfg.Meter.BatchSize,
				FlushInterval: cfg.Meter.FlushInterval,
				Logger:        log,
			})
			app.meter = m
			log.Info("metering enabled", "batch_size", cfg.Meter.BatchSize, "flush", cfg.Meter.FlushInterval)
		}
	} else {
		log.Warn("no postgres configured: authentication is DISABLED and all requests run as tenant 'local'")
	}

	// --- rate limiting ---
	var limiter api.Limiter
	if cfg.RateLimit.Enabled {
		if rdb != nil {
			limiter = api.NewRedisLimiter(rdb, cfg.RateLimit.DefaultRPM, cfg.RateLimit.Burst)
			log.Info("rate limiting: redis-backed", "default_rpm", cfg.RateLimit.DefaultRPM)
		} else {
			limiter = api.NewLocalLimiter(cfg.RateLimit.DefaultRPM, cfg.RateLimit.Burst)
			log.Warn("rate limiting: in-memory (per-replica); the effective fleet limit is N times this")
		}
	}

	app.server = api.NewServer(api.Deps{
		Config: cfg,
		Router: rt,
		Exec:   exec,
		Auth:   auth,
		Limit:  limiter,
		Cache:  c,
		Meter:  m,
		Logger: log,
	})
	return app, nil
}

// serveMetrics runs the Prometheus scrape endpoint on its own listener.
func (a *application) serveMetrics(ctx context.Context, cfg *config.Config, log *slog.Logger) {
	mux := http.NewServeMux()
	mux.Handle(cfg.Obs.MetricsPath, promhttp.HandlerFor(a.reg, promhttp.HandlerOpts{}))

	srv := &http.Server{
		Addr:              ":9090",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	log.Info("metrics listening", "addr", srv.Addr, "path", cfg.Obs.MetricsPath)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("metrics server failed", "err", err)
	}
}

// close tears down components in reverse dependency order.
func (a *application) close(log *slog.Logger) {
	// The meter first: it must flush buffered usage rows before the pool it
	// writes through is closed.
	if a.meter != nil {
		if err := a.meter.Close(); err != nil {
			log.Error("meter close failed", "err", err)
		}
	}
	if a.pgClose != nil {
		a.pgClose()
	}
	if a.rdb != nil {
		_ = a.rdb.Close()
	}
	if a.tpClose != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := a.tpClose(ctx); err != nil {
			log.Error("tracing shutdown failed", "err", err)
		}
	}
}

// newLogger builds the structured logger.
func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = slog.LevelInfo
	}
	// JSON to stdout: containers log to stdout, and structured logs are what
	// make the access log queryable next to the traces.
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))
}
