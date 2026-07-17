// Command glmctl is the operator CLI: it seeds tenants and issues API keys.
//
// It exists because keys are stored only as hashes. There is deliberately no way
// to recover a key after issuance, so something has to generate one, show it
// once, and store only its digest.
package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/yash/gatewayllm/internal/api"
	"github.com/yash/gatewayllm/internal/store"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `glmctl — GatewayLLM operator CLI

Usage:
  glmctl create-tenant -id <id> -name <name>
  glmctl create-key    -tenant <id> [-label <label>] [-rpm <n>] [-expires <duration>]
  glmctl list-keys     [-tenant <id>]
  glmctl revoke-key    -id <key-id>
  glmctl usage         [-tenant <id>] [-since <duration>]

The database URL comes from -db or $POSTGRES_URL.
`)
}

func run() error {
	if len(os.Args) < 2 {
		usage()
		return fmt.Errorf("a subcommand is required")
	}

	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	dbURL := fs.String("db", os.Getenv("POSTGRES_URL"), "postgres connection URL")

	var (
		id      = fs.String("id", "", "identifier")
		name    = fs.String("name", "", "display name")
		tenant  = fs.String("tenant", "", "tenant id")
		label   = fs.String("label", "", "human-readable key label")
		rpm     = fs.Int("rpm", 0, "per-key requests per minute (0 = use the config default)")
		expires = fs.Duration("expires", 0, "key lifetime (0 = never expires)")
		since   = fs.Duration("since", 24*time.Hour, "usage window")
	)

	if err := fs.Parse(os.Args[2:]); err != nil {
		return err
	}
	if *dbURL == "" {
		return fmt.Errorf("set -db or $POSTGRES_URL")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool, err := store.NewPostgres(ctx, *dbURL)
	if err != nil {
		return err
	}
	defer pool.Close()

	if err := store.Migrate(ctx, pool); err != nil {
		return err
	}

	switch cmd {
	case "create-tenant":
		if *id == "" || *name == "" {
			return fmt.Errorf("create-tenant requires -id and -name")
		}
		_, err := pool.Exec(ctx,
			`INSERT INTO tenants (id, name) VALUES ($1, $2)
			 ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name`, *id, *name)
		if err != nil {
			return err
		}
		fmt.Printf("tenant %q ready\n", *id)
		return nil

	case "create-key":
		if *tenant == "" {
			return fmt.Errorf("create-key requires -tenant")
		}
		raw, err := generateKey()
		if err != nil {
			return err
		}
		keyID := "key_" + shortID()

		var expiresAt any
		if *expires > 0 {
			expiresAt = time.Now().Add(*expires)
		}
		var rpmVal any
		if *rpm > 0 {
			rpmVal = *rpm
		}

		_, err = pool.Exec(ctx,
			`INSERT INTO api_keys (id, tenant_id, key_hash, key_prefix, label, rpm, expires_at)
			 VALUES ($1, $2, $3, $4, $5, $6, $7)`,
			keyID, *tenant, api.HashKey(raw), prefix(raw), nullIfEmpty(*label), rpmVal, expiresAt)
		if err != nil {
			return fmt.Errorf("create key (does tenant %q exist?): %w", *tenant, err)
		}

		// Shown once. Only the digest is stored, so this cannot be recovered.
		fmt.Printf("key id:  %s\ntenant:  %s\n\n  %s\n\nStore it now — it is not recoverable.\n",
			keyID, *tenant, raw)
		return nil

	case "list-keys":
		q := `SELECT id, tenant_id, key_prefix, COALESCE(label,''), COALESCE(rpm,0),
		             created_at, revoked_at IS NOT NULL
		      FROM api_keys`
		var args []any
		if *tenant != "" {
			q += ` WHERE tenant_id = $1`
			args = append(args, *tenant)
		}
		q += ` ORDER BY created_at DESC`

		rows, err := pool.Query(ctx, q, args...)
		if err != nil {
			return err
		}
		defer rows.Close()

		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "KEY ID\tTENANT\tPREFIX\tLABEL\tRPM\tCREATED\tREVOKED")
		for rows.Next() {
			var (
				id, tid, pfx, lbl string
				rpmV              int
				created           time.Time
				revoked           bool
			)
			if err := rows.Scan(&id, &tid, &pfx, &lbl, &rpmV, &created, &revoked); err != nil {
				return err
			}
			rpmStr := "default"
			if rpmV > 0 {
				rpmStr = fmt.Sprint(rpmV)
			}
			fmt.Fprintf(w, "%s\t%s\t%s…\t%s\t%s\t%s\t%v\n",
				id, tid, pfx, lbl, rpmStr, created.Format("2006-01-02"), revoked)
		}
		return w.Flush()

	case "revoke-key":
		if *id == "" {
			return fmt.Errorf("revoke-key requires -id")
		}
		tag, err := pool.Exec(ctx,
			`UPDATE api_keys SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL`, *id)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return fmt.Errorf("key %q not found or already revoked", *id)
		}
		// The gateway caches auth lookups briefly, so revocation is not instant.
		fmt.Printf("key %s revoked (takes effect within the auth cache TTL, ~30s)\n", *id)
		return nil

	case "usage":
		q := `SELECT tenant_id,
		             count(*),
		             sum(prompt_tokens + completion_tokens),
		             sum(cost_usd),
		             sum(saved_usd),
		             count(*) FILTER (WHERE cache_status IN ('exact_hit','semantic_hit'))
		      FROM usage_log
		      WHERE created_at > now() - $1::interval`
		args := []any{fmt.Sprintf("%d seconds", int((*since).Seconds()))}
		if *tenant != "" {
			q += ` AND tenant_id = $2`
			args = append(args, *tenant)
		}
		q += ` GROUP BY tenant_id ORDER BY sum(cost_usd) DESC`

		rows, err := pool.Query(ctx, q, args...)
		if err != nil {
			return err
		}
		defer rows.Close()

		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "TENANT\tREQUESTS\tTOKENS\tSPENT\tSAVED\tHIT RATE\n")
		for rows.Next() {
			var (
				tid                string
				reqs, tokens, hits int64
				spent, saved       float64
			)
			if err := rows.Scan(&tid, &reqs, &tokens, &spent, &saved, &hits); err != nil {
				return err
			}
			rate := 0.0
			if reqs > 0 {
				rate = float64(hits) / float64(reqs) * 100
			}
			fmt.Fprintf(w, "%s\t%d\t%d\t$%.4f\t$%.4f\t%.1f%%\n", tid, reqs, tokens, spent, saved, rate)
		}
		return w.Flush()

	default:
		usage()
		return fmt.Errorf("unknown subcommand %q", cmd)
	}
}

// generateKey mints a high-entropy API key. 32 random bytes make brute force
// infeasible, which is what justifies storing only a fast SHA-256 digest rather
// than a slow password KDF.
func generateKey() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generate key: %w", err)
	}
	return "glm_live_" + base64.RawURLEncoding.EncodeToString(b[:]), nil
}

func shortID() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return base64.RawURLEncoding.EncodeToString(b[:])
}

// prefix returns the non-secret leading portion, stored so a human can identify
// a key in a list without the key itself being recoverable.
func prefix(raw string) string {
	const n = 16
	if len(raw) <= n {
		return raw
	}
	return raw[:n]
}

func nullIfEmpty(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}
