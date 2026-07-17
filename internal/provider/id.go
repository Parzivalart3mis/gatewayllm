package provider

import (
	"crypto/rand"
	"encoding/hex"
)

// NewID returns a random identifier, used to synthesize OpenAI-style response
// IDs for providers that do not supply one and to correlate requests.
func NewID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failing means the OS entropy source is broken; an ID
		// collision is harmless next to that, and the field is cosmetic.
		return "0000000000000000000000"
	}
	return hex.EncodeToString(b[:])
}
