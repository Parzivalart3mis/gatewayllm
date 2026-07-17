package obs

import (
	"context"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"

	"github.com/yash/gatewayllm/internal/config"
)

// tracerName namespaces spans this process creates.
const tracerName = "github.com/yash/gatewayllm"

// Tracer returns the gateway's tracer.
func Tracer() trace.Tracer { return otel.Tracer(tracerName) }

// InitTracing configures the global tracer provider and returns a shutdown
// function. When no OTLP endpoint is configured, tracing is a no-op and spans
// cost nothing beyond a few nil checks.
func InitTracing(ctx context.Context, cfg config.Obs, version string) (func(context.Context) error, error) {
	if cfg.OTLPEndpoint == "" {
		// Still install the propagator: even without an exporter, inbound trace
		// context must be forwarded so an upstream service's trace is not broken
		// by passing through the gateway.
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{}, propagation.Baggage{},
		))
		return func(context.Context) error { return nil }, nil
	}

	exp, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpoint(cfg.OTLPEndpoint),
		// Plaintext: the collector is a compose-network sibling, not a public
		// endpoint. A hosted collector would need WithTLSClientConfig.
		otlptracehttp.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("otlp exporter: %w", err)
	}

	res, err := resource.Merge(resource.Default(), resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(cfg.ServiceName),
		semconv.ServiceVersion(version),
	))
	if err != nil {
		return nil, fmt.Errorf("otel resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp, sdktrace.WithBatchTimeout(5*time.Second)),
		sdktrace.WithResource(res),
		// ParentBased keeps a trace intact: if the caller sampled a request, the
		// gateway's spans are kept too, rather than sampling independently and
		// producing traces with holes.
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(cfg.SampleRatio))),
	)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{}, propagation.Baggage{},
	))

	return tp.Shutdown, nil
}

// Span attribute keys. Defined once so a dashboard query cannot be broken by a
// typo at one call site.
const (
	AttrTenant      = attribute.Key("gateway.tenant")
	AttrModelAlias  = attribute.Key("gateway.model_alias")
	AttrModel       = attribute.Key("gateway.model")
	AttrProvider    = attribute.Key("gateway.provider")
	AttrCacheStatus = attribute.Key("gateway.cache_status")
	AttrSimilarity  = attribute.Key("gateway.cache_similarity")
	AttrAttempt     = attribute.Key("gateway.attempt")
	AttrBreaker     = attribute.Key("gateway.breaker_state")
	AttrStreamed    = attribute.Key("gateway.streamed")
	AttrTokensIn    = attribute.Key("gateway.tokens_in")
	AttrTokensOut   = attribute.Key("gateway.tokens_out")
	AttrCostUSD     = attribute.Key("gateway.cost_usd")
)
