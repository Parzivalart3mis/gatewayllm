// Command glmbench replays a prompt corpus through a running gateway and
// reports latency and cache behaviour per X-Cache status.
//
// It exists to turn the cache's claims into measurements: a prime phase sends
// each prompt once (misses, real provider calls), then a replay phase sends the
// same prompts again for several rounds (hits). The per-status latency split is
// the number the README and dashboard promise — it is measured here, not
// estimated.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type result struct {
	status string // X-Cache header, or "error"
	code   int
	dur    time.Duration
}

func main() {
	var (
		url       = flag.String("url", "http://localhost:8099", "gateway base URL")
		key       = flag.String("key", "", "API key (any value when auth is disabled)")
		model     = flag.String("model", "fast", "model alias to request")
		prompts   = flag.String("prompts", "bench/prompts.txt", "file with one prompt per line")
		rounds    = flag.Int("rounds", 3, "replay rounds after the prime phase")
		conc      = flag.Int("conc", 8, "concurrency during replay")
		missDelay = flag.Duration("miss-delay", 2100*time.Millisecond, "delay between prime requests (respect provider RPM)")
		maxTokens = flag.Int("max-tokens", 60, "max_tokens per request")
	)
	flag.Parse()

	lines, err := readPrompts(*prompts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Printf("corpus: %d prompts | prime: sequential, %s apart | replay: %d rounds at concurrency %d\n\n",
		len(lines), *missDelay, *rounds, *conc)

	client := &http.Client{Timeout: 60 * time.Second}
	call := func(prompt string) result {
		body, _ := json.Marshal(map[string]any{
			"model":       *model,
			"temperature": 0, // must be under the cache's max_temperature
			"max_tokens":  *maxTokens,
			"messages":    []map[string]string{{"role": "user", "content": prompt}},
		})
		req, _ := http.NewRequest(http.MethodPost, *url+"/v1/chat/completions", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+*key)

		start := time.Now()
		resp, err := client.Do(req)
		dur := time.Since(start)
		if err != nil {
			return result{status: "error", dur: dur}
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body) // count full response time, not first byte

		status := resp.Header.Get("X-Cache")
		if resp.StatusCode != http.StatusOK {
			status = fmt.Sprintf("http_%d", resp.StatusCode)
		}
		return result{status: status, code: resp.StatusCode, dur: time.Since(start)}
	}

	var all []result
	var mu sync.Mutex
	record := func(r result) {
		mu.Lock()
		all = append(all, r)
		mu.Unlock()
	}

	// Phase 1 — prime. Sequential with a delay so the provider's free-tier RPM
	// is respected; each of these is a real upstream call.
	fmt.Println("phase 1: prime (expect miss)")
	for i, p := range lines {
		r := call(p)
		record(r)
		fmt.Printf("  %2d/%d  %-12s %8.0fms\n", i+1, len(lines), r.status, r.dur.Seconds()*1000)
		if i < len(lines)-1 {
			time.Sleep(*missDelay)
		}
	}

	// Phase 2 — replay. The same prompts again; every request should be served
	// from cache, so concurrency is not limited by the provider.
	fmt.Printf("\nphase 2: replay ×%d (expect exact_hit)\n", *rounds)
	jobs := make(chan string)
	var wg sync.WaitGroup
	replayStart := time.Now()
	var replayCount int
	for w := 0; w < *conc; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range jobs {
				record(call(p))
			}
		}()
	}
	for r := 0; r < *rounds; r++ {
		for _, p := range lines {
			jobs <- p
			replayCount++
		}
	}
	close(jobs)
	wg.Wait()
	replayWall := time.Since(replayStart)

	// Summary.
	byStatus := map[string][]time.Duration{}
	for _, r := range all {
		byStatus[r.status] = append(byStatus[r.status], r.dur)
	}
	statuses := make([]string, 0, len(byStatus))
	for s := range byStatus {
		statuses = append(statuses, s)
	}
	sort.Strings(statuses)

	fmt.Printf("\n%-14s %6s %10s %10s %10s %10s\n", "X-Cache", "count", "min", "p50", "p95", "max")
	for _, s := range statuses {
		d := byStatus[s]
		sort.Slice(d, func(i, j int) bool { return d[i] < d[j] })
		fmt.Printf("%-14s %6d %10s %10s %10s %10s\n",
			s, len(d), ms(d[0]), ms(pct(d, 50)), ms(pct(d, 95)), ms(d[len(d)-1]))
	}

	hits := len(byStatus["exact_hit"]) + len(byStatus["semantic_hit"])
	misses := len(byStatus["miss"])
	if hits+misses > 0 {
		fmt.Printf("\nhit rate (hits / cacheable): %.1f%%  (%d hits, %d misses)\n",
			float64(hits)/float64(hits+misses)*100, hits, misses)
	}
	if replayCount > 0 && replayWall > 0 {
		fmt.Printf("replay throughput: %.0f req/s (%d requests in %s at concurrency %d)\n",
			float64(replayCount)/replayWall.Seconds(), replayCount, replayWall.Round(time.Millisecond), *conc)
	}
	if n := len(byStatus["error"]); n > 0 {
		fmt.Printf("errors: %d\n", n)
	}
}

func readPrompts(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if l := strings.TrimSpace(sc.Text()); l != "" && !strings.HasPrefix(l, "#") {
			out = append(out, l)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no prompts", path)
	}
	return out, sc.Err()
}

// pct returns the p-th percentile of sorted durations (nearest-rank).
func pct(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := (len(sorted)*p + 99) / 100
	if idx < 1 {
		idx = 1
	}
	return sorted[idx-1]
}

func ms(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fms", d.Seconds()*1000)
	}
	return fmt.Sprintf("%.0fms", d.Seconds()*1000)
}
