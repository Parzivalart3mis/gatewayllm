package api

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/yash/gatewayllm/internal/provider"
)

// errorBody is the OpenAI error envelope. Clients written against the OpenAI SDK
// parse this shape, so error responses must match it or SDK error handling
// breaks even though the happy path works.
type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// errorType maps a provider failure kind onto OpenAI's error `type` vocabulary.
func errorType(k provider.Kind) string {
	switch k {
	case provider.KindInvalidRequest, provider.KindContextLength:
		return "invalid_request_error"
	case provider.KindAuth:
		return "authentication_error"
	case provider.KindRateLimit:
		return "rate_limit_error"
	case provider.KindContentFilter:
		return "content_filter_error"
	default:
		return "api_error"
	}
}

// writeError renders an OpenAI-compatible error response.
func writeError(w http.ResponseWriter, status int, typ, msg, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	body := errorBody{Error: errorDetail{Message: msg, Type: typ}}
	if code != "" {
		body.Error.Code = &code
	}
	if err := json.NewEncoder(w).Encode(body); err != nil {
		slog.Debug("write error response failed", "err", err)
	}
}

// writeProviderError renders a provider failure using its own classification.
func writeProviderError(w http.ResponseWriter, err error) {
	if pe := provider.AsError(err); pe != nil {
		writeError(w, pe.HTTPStatus(), errorType(pe.Kind), pe.Message, string(pe.Kind))
		return
	}
	writeError(w, http.StatusInternalServerError, "api_error", err.Error(), "")
}

// writeBadRequest renders a client-side validation failure.
func writeBadRequest(w http.ResponseWriter, msg string) {
	writeError(w, http.StatusBadRequest, "invalid_request_error", msg, "")
}
