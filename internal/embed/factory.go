package embed

import (
	"fmt"

	"github.com/yash/gatewayllm/internal/config"
)

// Build constructs the configured Embedder. Swapping between a local sidecar and
// a hosted API is a config edit, which is the point of the interface.
func Build(cfg config.Embed, dims int) (Embedder, error) {
	switch cfg.Kind {
	case "sidecar":
		return NewSidecar(SidecarOptions{
			BaseURL:    cfg.BaseURL,
			Model:      cfg.Model,
			Timeout:    cfg.Timeout,
			Dimensions: dims,
		}), nil
	case "api":
		e := NewAPI(APIOptions{
			BaseURL:    cfg.BaseURL,
			Model:      cfg.Model,
			APIKey:     cfg.APIKey,
			Timeout:    cfg.Timeout,
			Dimensions: dims,
		})
		if e.Dimensions() == 0 {
			return nil, fmt.Errorf("embed: dimensions unknown for model %q; set cache.semantic.vector_size", cfg.Model)
		}
		return e, nil
	default:
		return nil, fmt.Errorf("embed: unknown kind %q (want sidecar or api)", cfg.Kind)
	}
}
