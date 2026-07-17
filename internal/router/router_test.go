package router

import (
	"testing"

	"github.com/yash/gatewayllm/internal/config"
	"github.com/yash/gatewayllm/internal/provider"
)

func testRegistry() *provider.Registry {
	return provider.NewRegistry(
		provider.NewMock("fast", "fast reply"),
		provider.NewMock("cheap", "cheap reply"),
		provider.NewMock("backup", "backup reply"),
	)
}

// TestRoute_PriorityOrder asserts the priority strategy returns targets in
// config order: the failover order is exactly what the operator wrote.
func TestRoute_PriorityOrder(t *testing.T) {
	r, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"gpt-4o": {Strategy: "priority", Targets: []config.Target{
				{Provider: "fast", Model: "m1"},
				{Provider: "cheap", Model: "m2"},
				{Provider: "backup", Model: "m3"},
			}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	targets, err := r.Route("gpt-4o")
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	want := []string{"fast", "cheap", "backup"}
	if len(targets) != len(want) {
		t.Fatalf("got %d targets, want %d", len(targets), len(want))
	}
	for i, w := range want {
		if got := targets[i].Provider.Name(); got != w {
			t.Errorf("target[%d] = %q, want %q", i, got, w)
		}
	}
	// The upstream model must travel with the target, not the alias.
	if targets[0].Model != "m1" {
		t.Errorf("model = %q, want the target's upstream model m1", targets[0].Model)
	}
}

// TestRoute_UnknownModel asserts unknown aliases are rejected when no default is
// configured, rather than silently routing somewhere surprising.
func TestRoute_UnknownModel(t *testing.T) {
	r, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"known": {Strategy: "priority", Targets: []config.Target{{Provider: "fast", Model: "m"}}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := r.Route("unknown"); err == nil {
		t.Error("expected an error for an unconfigured alias")
	}
}

// TestRoute_DefaultAlias asserts the fallback works when configured.
func TestRoute_DefaultAlias(t *testing.T) {
	r, err := New(config.Router{
		DefaultAlias: "fallback",
		Aliases: map[string]config.Alias{
			"fallback": {Strategy: "priority", Targets: []config.Target{{Provider: "cheap", Model: "m"}}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	targets, err := r.Route("something-nobody-configured")
	if err != nil {
		t.Fatalf("Route with a default alias should succeed: %v", err)
	}
	if targets[0].Provider.Name() != "cheap" {
		t.Errorf("provider = %q, want cheap via the default alias", targets[0].Provider.Name())
	}
}

// TestRoute_WeightedIncludesAllTargets asserts the weighted strategy still
// returns every target. Selection picks the primary by weight, but the rest must
// remain as fallbacks or a weighted alias would lose failover entirely.
func TestRoute_WeightedIncludesAllTargets(t *testing.T) {
	r, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"split": {Strategy: "weighted", Targets: []config.Target{
				{Provider: "fast", Model: "m1", Weight: 90},
				{Provider: "cheap", Model: "m2", Weight: 10},
			}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	for i := 0; i < 50; i++ {
		targets, err := r.Route("split")
		if err != nil {
			t.Fatalf("Route: %v", err)
		}
		if len(targets) != 2 {
			t.Fatalf("got %d targets, want 2: weighted routing must retain fallbacks", len(targets))
		}
		if targets[0].Provider.Name() == targets[1].Provider.Name() {
			t.Fatal("weighted selection returned the same provider twice")
		}
	}
}

// TestRoute_WeightedRespectsDistribution asserts weights actually bias
// selection. A 90/10 split must favor the heavy target by a wide margin.
func TestRoute_WeightedRespectsDistribution(t *testing.T) {
	r, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"split": {Strategy: "weighted", Targets: []config.Target{
				{Provider: "fast", Model: "m1", Weight: 90},
				{Provider: "cheap", Model: "m2", Weight: 10},
			}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const runs = 2000
	fastFirst := 0
	for i := 0; i < runs; i++ {
		targets, _ := r.Route("split")
		if targets[0].Provider.Name() == "fast" {
			fastFirst++
		}
	}

	// Expect ~90%. The bounds are wide enough that random variation will not
	// flake, but tight enough to catch weights being ignored entirely.
	ratio := float64(fastFirst) / runs
	if ratio < 0.85 || ratio > 0.95 {
		t.Errorf("fast selected first %.1f%% of the time, want ~90%%", ratio*100)
	}
}

// TestNew_UnregisteredProvider asserts a config referencing a missing provider
// fails at construction, not on the first request.
func TestNew_UnregisteredProvider(t *testing.T) {
	_, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"x": {Strategy: "priority", Targets: []config.Target{{Provider: "does-not-exist", Model: "m"}}},
		},
	}, testRegistry())
	if err == nil {
		t.Error("expected an error for an unregistered provider reference")
	}
}

// TestAliases asserts the /v1/models surface lists aliases sorted.
func TestAliases(t *testing.T) {
	r, err := New(config.Router{
		Aliases: map[string]config.Alias{
			"zeta":  {Strategy: "priority", Targets: []config.Target{{Provider: "fast", Model: "m"}}},
			"alpha": {Strategy: "priority", Targets: []config.Target{{Provider: "fast", Model: "m"}}},
		},
	}, testRegistry())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	got := r.Aliases()
	if len(got) != 2 || got[0] != "alpha" || got[1] != "zeta" {
		t.Errorf("Aliases() = %v, want [alpha zeta] sorted", got)
	}
}
