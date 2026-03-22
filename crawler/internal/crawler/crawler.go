// Package crawler provides the main crawler orchestration.
package crawler

import (
        "context"
        "crypto/sha256"
        "encoding/hex"
        "fmt"
        "io"
        "math/rand"
        "net/http"
        "strings"
        "sync/atomic"
        "time"

        "crawler/internal/config"
        "crawler/internal/dedup"
        "crawler/internal/extractor"
        "crawler/internal/index"
        "crawler/internal/robots"
)

// Stats tracks crawl statistics.
type Stats struct {
        PagesCrawled    int64
        PagesFailed     int64
        PagesSkipped    int64
        BytesDownloaded int64
}

// Crawler orchestrates the crawling process.
type Crawler struct {
        cfg       *config.Config
        client    *http.Client
        robots    *robots.Checker
        dedup     *dedup.Store
        extractor *extractor.Extractor
        builder   *index.Builder
        stats     Stats
        startTime time.Time
}

// New creates a new crawler instance.
func New(cfg *config.Config) *Crawler {
        client := &http.Client{
                Timeout: cfg.Timeout,
                CheckRedirect: func(req *http.Request, via []*http.Request) error {
                        if len(via) >= cfg.MaxRedirects {
                                return fmt.Errorf("stopped after %d redirects", cfg.MaxRedirects)
                        }
                        return nil
                },
        }

        return &Crawler{
                cfg:       cfg,
                client:    client,
                robots:    robots.NewChecker(client, cfg.UserAgent),
                dedup:     dedup.New(),
                extractor: extractor.New(cfg.Domain()),
                builder:   index.NewBuilder(cfg.OutputDir),
        }
}

// Fetch retrieves a URL with retry logic.
func (c *Crawler) Fetch(ctx context.Context, targetURL string) (*http.Response, error) {
        var lastErr error

        for attempt := 0; attempt <= c.cfg.MaxRetries; attempt++ {
                if attempt > 0 {
                        // Exponential backoff with jitter: 1s, 2s, 4s with ±20% jitter
                        delay := time.Duration(1<<uint(attempt-1)) * time.Second
                        jitter := time.Duration(rand.Float64() * 0.4 * float64(delay))
                        delay = delay + jitter - time.Duration(float64(delay)*0.2)
                        select {
                        case <-ctx.Done():
                                return nil, ctx.Err()
                        case <-time.After(delay):
                        }
                }

                req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
                if err != nil {
                        return nil, err
                }

                req.Header.Set("User-Agent", c.cfg.UserAgent)
                req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
                req.Header.Set("Accept-Language", "en-US,en;q=0.5")

                resp, err := c.client.Do(req)
                if err != nil {
                        lastErr = err
                        continue
                }

                // Handle rate limiting (429)
                if resp.StatusCode == http.StatusTooManyRequests {
                        resp.Body.Close()
                        retryAfter := resp.Header.Get("Retry-After")
                        if retryAfter != "" {
                                if seconds, err := parseRetryAfter(retryAfter); err == nil && seconds > 0 {
                                        delay := time.Duration(seconds) * time.Second
                                        select {
                                        case <-ctx.Done():
                                                return nil, ctx.Err()
                                        case <-time.After(delay):
                                        }
                                }
                        }
                        lastErr = fmt.Errorf("rate limited (429)")
                        continue
                }

                // Handle server errors with retry
                if resp.StatusCode >= 500 {
                        resp.Body.Close()
                        lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
                        continue
                }

                return resp, nil
        }

        return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

// parseRetryAfter parses the Retry-After header value.
func parseRetryAfter(value string) (int, error) {
        // Try parsing as seconds
        var seconds int
        if _, err := fmt.Sscanf(value, "%d", &seconds); err == nil {
                return seconds, nil
        }

        // Try parsing as date
        t, err := http.ParseTime(value)
        if err != nil {
                return 0, err
        }

        return int(time.Until(t).Seconds()), nil
}

// Process handles a single URL.
func (c *Crawler) Process(ctx context.Context, targetURL string, enqueue func(string)) error {
        // Check robots.txt
        if !c.robots.IsAllowed(targetURL) {
                atomic.AddInt64(&c.stats.PagesSkipped, 1)
                return nil
        }

        // Deduplicate
        if !c.dedup.CheckAndMark(targetURL) {
                return nil // Already seen
        }

        // Check page limit
        if c.cfg.MaxPages > 0 && atomic.LoadInt64(&c.stats.PagesCrawled) >= int64(c.cfg.MaxPages) {
                return nil
        }

        // Fetch the page
        resp, err := c.Fetch(ctx, targetURL)
        if err != nil {
                atomic.AddInt64(&c.stats.PagesFailed, 1)
                return fmt.Errorf("fetch failed: %w", err)
        }
        defer resp.Body.Close()

        // Track bytes
        bodyBytes, err := io.ReadAll(resp.Body)
        if err != nil {
                atomic.AddInt64(&c.stats.PagesFailed, 1)
                return fmt.Errorf("failed to read body: %w", err)
        }
        atomic.AddInt64(&c.stats.BytesDownloaded, int64(len(bodyBytes)))

        // Handle non-200 status codes
        if resp.StatusCode != http.StatusOK {
                if resp.StatusCode == http.StatusNotFound {
                        atomic.AddInt64(&c.stats.PagesSkipped, 1)
                        return nil
                }
                if resp.StatusCode >= 400 && resp.StatusCode < 500 {
                        atomic.AddInt64(&c.stats.PagesSkipped, 1)
                        return nil
                }
                atomic.AddInt64(&c.stats.PagesFailed, 1)
                return fmt.Errorf("unexpected status: %d", resp.StatusCode)
        }

        // Check content type
        contentType := resp.Header.Get("Content-Type")
        if !extractor.IsHTML(contentType) {
                atomic.AddInt64(&c.stats.PagesSkipped, 1)
                return nil
        }

        // Extract content
        page, err := c.extractor.Extract(targetURL, bodyBytes)
        if err != nil {
                atomic.AddInt64(&c.stats.PagesFailed, 1)
                return fmt.Errorf("extraction failed: %w", err)
        }

        page.ContentType = contentType

        // Write page to disk
        doc := &index.PageDocument{
                URL:       page.URL,
                Title:     page.Title,
                Body:      page.Body,
                Links:     page.Links,
                CrawledAt: time.Now().Format(time.RFC3339),
        }

        if err := c.builder.WritePage(doc); err != nil {
                atomic.AddInt64(&c.stats.PagesFailed, 1)
                return fmt.Errorf("failed to write page: %w", err)
        }

        atomic.AddInt64(&c.stats.PagesCrawled, 1)

        // Enqueue discovered links
        for _, link := range page.Links {
                enqueue(link)
        }

        return nil
}

// Finalize builds the inverted index and writes the manifest.
func (c *Crawler) Finalize() error {
        if err := c.builder.BuildInvertedIndex(); err != nil {
                return fmt.Errorf("failed to build inverted index: %w", err)
        }

        stats := &index.CrawlStats{
                CrawlID:         generateCrawlID(),
                StartedAt:       c.startTime,
                CompletedAt:     time.Now(),
                PagesCrawled:    int(atomic.LoadInt64(&c.stats.PagesCrawled)),
                PagesFailed:     int(atomic.LoadInt64(&c.stats.PagesFailed)),
                BytesDownloaded: atomic.LoadInt64(&c.stats.BytesDownloaded),
        }

        if err := index.WriteManifest(c.cfg.OutputDir, stats); err != nil {
                return fmt.Errorf("failed to write manifest: %w", err)
        }

        return nil
}

// GetStats returns current crawl statistics.
func (c *Crawler) GetStats() Stats {
        return Stats{
                PagesCrawled:    atomic.LoadInt64(&c.stats.PagesCrawled),
                PagesFailed:     atomic.LoadInt64(&c.stats.PagesFailed),
                PagesSkipped:    atomic.LoadInt64(&c.stats.PagesSkipped),
                BytesDownloaded: atomic.LoadInt64(&c.stats.BytesDownloaded),
        }
}

// generateCrawlID creates a unique ID for the crawl run.
func generateCrawlID() string {
        timestamp := time.Now().UnixNano()
        hash := sha256.Sum256([]byte(fmt.Sprintf("%d", timestamp)))
        return hex.EncodeToString(hash[:8])
}

// IsSameDomain checks if a URL belongs to the same domain.
func IsSameDomain(rawURL, domain string) bool {
        parsed, err := parseURL(rawURL)
        if err != nil {
                return false
        }
        return strings.EqualFold(parsed.Host, domain)
}

func parseURL(rawURL string) (*parsedURL, error) {
        // Simple URL parsing without importing net/url heavily
        // This is a minimal implementation for domain checking
        if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
                return nil, fmt.Errorf("invalid URL scheme")
        }

        parts := strings.SplitN(rawURL[8:], "/", 2)
        host := parts[0]
        if strings.HasPrefix(rawURL, "https://") {
                host = strings.TrimPrefix(rawURL, "https://")
                host = strings.SplitN(host, "/", 2)[0]
        } else {
                host = strings.TrimPrefix(rawURL, "http://")
                host = strings.SplitN(host, "/", 2)[0]
        }

        return &parsedURL{Host: host}, nil
}

type parsedURL struct {
        Host string
}

// Run starts the crawling process with the given configuration.
func Run(ctx context.Context, cfg *config.Config) error {
        crawler := New(cfg)
        crawler.startTime = time.Now()

        // Cleanup any leftover temp files
        if err := index.CleanupTempFiles(cfg.OutputDir); err != nil {
                return fmt.Errorf("failed to cleanup temp files: %w", err)
        }

        // Fetch and parse robots.txt
        if err := crawler.robots.Fetch(ctx, cfg.Entry); err != nil {
                // Log but continue - robots.txt is optional
                fmt.Printf("Warning: failed to fetch robots.txt: %v\n", err)
        }

        // Create worker pool
        pool := newPool(cfg.Workers, cfg.DelayMS)
        pool.crawler = crawler

        // Start the pool
        pool.Start(ctx)

        // Add entry URL to frontier
        pool.Enqueue(cfg.Entry)

        // Wait for completion
        pool.Wait()

        // Finalize
        return crawler.Finalize()
}
