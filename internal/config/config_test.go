package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// write drops a config file into a temp dir and returns its path.
func write(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	return path
}

const minimalConfig = `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    alias1:
      targets:
        - provider: p1
          model: m1
`

// TestLoad_Minimal asserts a minimal config loads and gets sensible defaults.
func TestLoad_Minimal(t *testing.T) {
	cfg, err := Load(write(t, minimalConfig))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Addr != ":8080" {
		t.Errorf("addr = %q, want the default :8080", cfg.Server.Addr)
	}
	// An unspecified strategy must default rather than fail validation.
	if got := cfg.Router.Aliases["alias1"].Strategy; got != "priority" {
		t.Errorf("strategy = %q, want the default 'priority'", got)
	}
	if cfg.Cache.MaxTemperature != 0.3 {
		t.Errorf("max_temperature = %v, want the default 0.3", cfg.Cache.MaxTemperature)
	}
}

// TestLoad_EnvInterpolation asserts secrets can come from the environment
// instead of being committed to the config file.
func TestLoad_EnvInterpolation(t *testing.T) {
	t.Setenv("TEST_GW_KEY", "sk-from-env")

	cfg, err := Load(write(t, `
providers:
  - name: p1
    kind: openai
    api_key: ${TEST_GW_KEY}
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Providers[0].APIKey != "sk-from-env" {
		t.Errorf("api_key = %q, want the interpolated value", cfg.Providers[0].APIKey)
	}
}

// TestLoad_EnvDefault asserts the ${VAR:-default} form works.
func TestLoad_EnvDefault(t *testing.T) {
	os.Unsetenv("TEST_GW_UNSET")

	cfg, err := Load(write(t, `
server:
  addr: ${TEST_GW_UNSET:-:9999}
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Addr != ":9999" {
		t.Errorf("addr = %q, want the default from ${VAR:-default}", cfg.Server.Addr)
	}
}

// TestLoad_CloudRunShape asserts the two settings a Cloud Run deploy depends on:
// the listener honors the injected $PORT, and metrics can be moved onto the main
// port. Both are single-port-host requirements, so a regression here would make
// the service unreachable or unscrapeable in the cloud.
func TestLoad_CloudRunShape(t *testing.T) {
	t.Setenv("PORT", "9987")

	cfg, err := Load(write(t, `
server:
  addr: ":${PORT:-8080}"
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
obs:
  metrics_addr: inline
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Addr != ":9987" {
		t.Errorf("addr = %q, want :9987 interpolated from $PORT", cfg.Server.Addr)
	}
	if !cfg.Obs.MetricsInline() {
		t.Errorf("MetricsInline() = false, want true for metrics_addr: inline")
	}
}

// TestLoad_PortDefault asserts the fallback when $PORT is unset (local runs).
func TestLoad_PortDefault(t *testing.T) {
	os.Unsetenv("PORT")

	cfg, err := Load(write(t, `
server:
  addr: ":${PORT:-8080}"
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Addr != ":8080" {
		t.Errorf("addr = %q, want the :8080 default when $PORT is unset", cfg.Server.Addr)
	}
	// Default metrics stay on their own port unless explicitly inlined.
	if cfg.Obs.MetricsInline() {
		t.Error("metrics must default to a dedicated port, not inline")
	}
	if cfg.Obs.MetricsAddr != ":9090" {
		t.Errorf("metrics_addr = %q, want the :9090 default", cfg.Obs.MetricsAddr)
	}
}

// TestLoad_SemanticDisabledDropsQdrant asserts the escape hatch for deploys with
// no embeddings backend: turning the semantic tier off must also lift the Qdrant
// requirement, so an exact-cache-only deploy validates without a vector store.
func TestLoad_SemanticDisabledDropsQdrant(t *testing.T) {
	t.Setenv("SEMANTIC_ENABLED", "false")

	cfg, err := Load(write(t, `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
stores:
  redis_url: redis://localhost:6379
cache:
  enabled: true
  semantic:
    enabled: ${SEMANTIC_ENABLED:-true}
`))
	if err != nil {
		t.Fatalf("an exact-cache-only config must validate without Qdrant: %v", err)
	}
	if cfg.Cache.Semantic.Enabled {
		t.Error("semantic tier must be off when SEMANTIC_ENABLED=false")
	}
	if !cfg.Cache.Enabled {
		t.Error("the exact tier must remain enabled")
	}
}

// TestValidate_Errors covers the misconfigurations that would otherwise surface
// as confusing runtime failures under production traffic.
func TestValidate_Errors(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		wantHint string
	}{
		{
			name: "alias references an unknown provider",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: nonexistent
          model: m1
`,
			wantHint: "unknown or disabled provider",
		},
		{
			// A typo here would otherwise become an upstream 404 in production.
			name: "target names a model the provider does not serve",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: typo-model
`,
			wantHint: "does not list model",
		},
		{
			name: "no providers",
			body: `
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`,
			wantHint: "at least one enabled provider",
		},
		{
			name: "non-mock provider without an api key",
			body: `
providers:
  - name: p1
    kind: openai
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`,
			wantHint: "api_key is required",
		},
		{
			name: "semantic cache without redis",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
cache:
  enabled: true
`,
			wantHint: "redis_url",
		},
		{
			// The guard that keeps the semantic cache from answering questions
			// nobody asked.
			name: "dangerously low similarity threshold",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
stores:
  redis_url: redis://localhost:6379
  qdrant_url: http://localhost:6333
embed:
  kind: sidecar
  base_url: http://localhost:8000
cache:
  enabled: true
  semantic:
    enabled: true
    threshold: 0.4
`,
			wantHint: "dangerously low",
		},
		{
			name: "weighted strategy without weights",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
  - name: p2
    kind: mock
    models: ["m2"]
router:
  aliases:
    a:
      strategy: weighted
      targets:
        - provider: p1
          model: m1
        - provider: p2
          model: m2
`,
			wantHint: "positive weight",
		},
		{
			name: "unknown default alias",
			body: `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
router:
  default_alias: nope
  aliases:
    a:
      targets:
        - provider: p1
          model: m1
`,
			wantHint: "not a defined alias",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Load(write(t, tc.body))
			if err == nil {
				t.Fatal("expected a validation error")
			}
			if !strings.Contains(err.Error(), tc.wantHint) {
				t.Errorf("error = %v\nwant it to mention %q", err, tc.wantHint)
			}
		})
	}
}

// TestValidate_ReportsAllErrors asserts a misconfigured deploy fails with the
// full list rather than one problem at a time.
func TestValidate_ReportsAllErrors(t *testing.T) {
	_, err := Load(write(t, `
providers:
  - name: p1
    kind: openai
    models: ["m1"]
router:
  aliases:
    a:
      targets:
        - provider: nonexistent
          model: m1
`))
	if err == nil {
		t.Fatal("expected validation errors")
	}
	// Both the missing key and the bad provider reference must be reported.
	msg := err.Error()
	if !strings.Contains(msg, "api_key") || !strings.Contains(msg, "unknown or disabled provider") {
		t.Errorf("want every problem reported at once, got: %v", msg)
	}
}

// TestValidate_DisabledProviderNotRoutable asserts a disabled provider cannot be
// a routing target.
func TestValidate_DisabledProviderNotRoutable(t *testing.T) {
	_, err := Load(write(t, `
providers:
  - name: p1
    kind: mock
    models: ["m1"]
  - name: p2
    kind: mock
    enabled: false
    models: ["m2"]
router:
  aliases:
    a:
      targets:
        - provider: p2
          model: m2
`))
	if err == nil || !strings.Contains(err.Error(), "unknown or disabled provider") {
		t.Errorf("routing to a disabled provider must fail validation, got: %v", err)
	}
}
