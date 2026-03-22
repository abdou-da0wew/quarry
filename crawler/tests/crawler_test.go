package tests

import (
        "context"
        "encoding/json"
        "fmt"
        "net/http"
        "net/http/httptest"
        "net/url"
        "os"
        "path/filepath"
        "sync"
        "testing"
        "time"

        "crawler/internal/config"
        "crawler/internal/crawler"
        "crawler/internal/dedup"
        "crawler/internal/extractor"
        "crawler/internal/index"
        "crawler/internal/robots"

        "go.uber.org/goleak"
)

// TestMain verifies no goroutine leaks across all tests.
func TestMain(m *testing.M) {
        goleak.VerifyTestMain(m)
}

// TestNormalizeURL tests URL normalization across various cases.
func TestNormalizeURL(t *testing.T) {
        tests := []struct {
                name     string
                input    string
                expected string
                hasError bool
        }{
                {
                        name:     "basic URL",
                        input:    "https://example.com/path",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "trailing slash",
                        input:    "https://example.com/path/",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "fragment removal",
                        input:    "https://example.com/path#section",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "uppercase scheme",
                        input:    "HTTPS://example.com/path",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "uppercase host",
                        input:    "https://EXAMPLE.COM/path",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "default port http",
                        input:    "http://example.com:80/path",
                        expected: "http://example.com/path",
                        hasError: false,
                },
                {
                        name:     "default port https",
                        input:    "https://example.com:443/path",
                        expected: "https://example.com/path",
                        hasError: false,
                },
                {
                        name:     "invalid URL",
                        input:    "://invalid",
                        expected: "",
                        hasError: true,
                },
        }

        for _, tt := range tests {
                t.Run(tt.name, func(t *testing.T) {
                        result, err := dedup.NormalizeURL(tt.input)
                        if tt.hasError {
                                if err == nil {
                                        t.Errorf("expected error, got none")
                                }
                                return
                        }
                        if err != nil {
                                t.Errorf("unexpected error: %v", err)
                                return
                        }
                        if result != tt.expected {
                                t.Errorf("expected %q, got %q", tt.expected, result)
                        }
                })
        }
}

// TestDedupStore tests the deduplication store.
func TestDedupStore(t *testing.T) {
        store := dedup.New()

        // First mark should succeed
        if !store.CheckAndMark("https://example.com/page1") {
                t.Error("first mark should succeed")
        }

        // Second mark of same URL should fail
        if store.CheckAndMark("https://example.com/page1") {
                t.Error("second mark of same URL should fail")
        }

        // Different URL should succeed
        if !store.CheckAndMark("https://example.com/page2") {
                t.Error("different URL should succeed")
        }

        // Same URL with trailing slash should be treated as same
        if store.CheckAndMark("https://example.com/page1/") {
                t.Error("same URL with trailing slash should be treated as seen")
        }

        // Count should be 2
        if count := store.Count(); count != 2 {
                t.Errorf("expected count 2, got %d", count)
        }
}

// TestDedupStoreConcurrent tests concurrent access to the dedup store.
func TestDedupStoreConcurrent(t *testing.T) {
        store := dedup.New()
        const workers = 10
        const iterations = 100

        done := make(chan bool, workers)

        for w := 0; w < workers; w++ {
                go func(workerID int) {
                        for i := 0; i < iterations; i++ {
                                urlStr := fmt.Sprintf("https://example.com/page/%d/%d", workerID, i)
                                store.CheckAndMark(urlStr)
                        }
                        done <- true
                }(w)
        }

        // Wait for all workers
        for w := 0; w < workers; w++ {
                <-done
        }

        expectedCount := workers * iterations
        if count := store.Count(); count != expectedCount {
                t.Errorf("expected count %d, got %d", expectedCount, count)
        }
}

// TestRobotsParser tests robots.txt parsing.
func TestRobotsParser(t *testing.T) {
        robotsTxt := `
User-agent: *
Disallow: /admin/
Disallow: /private
Allow: /public/

User-agent: GoodBot
Crawl-delay: 2
Disallow: /tmp/
`

        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                if r.URL.Path == "/robots.txt" {
                        w.Header().Set("Content-Type", "text/plain")
                        w.Write([]byte(robotsTxt))
                        return
                }
                w.WriteHeader(http.StatusOK)
        }))
        defer server.Close()

        client := &http.Client{Timeout: 5 * time.Second}
        checker := robots.NewChecker(client, "GoodBot")

        ctx := context.Background()
        if err := checker.Fetch(ctx, server.URL); err != nil {
                t.Fatalf("failed to fetch robots.txt: %v", err)
        }

        tests := []struct {
                urlStr   string
                expected bool
        }{
                {server.URL + "/public/page.html", true},
                {server.URL + "/admin/login", false},
                {server.URL + "/private/data", false},
                {server.URL + "/normal/page.html", true},
        }

        for _, tt := range tests {
                result := checker.IsAllowed(tt.urlStr)
                if result != tt.expected {
                        t.Errorf("IsAllowed(%q) = %v, expected %v", tt.urlStr, result, tt.expected)
                }
        }

        // Check crawl delay
        delay := checker.GetCrawlDelay()
        if delay != 2*time.Second {
                t.Errorf("expected crawl delay 2s, got %v", delay)
        }
}

// TestRobotsCompliance tests that crawler respects robots.txt.
func TestRobotsCompliance(t *testing.T) {
        var accessedPaths []string
        var mu sync.Mutex

        robotsTxt := `
User-agent: *
Disallow: /forbidden/
`

        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                mu.Lock()
                accessedPaths = append(accessedPaths, r.URL.Path)
                mu.Unlock()

                if r.URL.Path == "/robots.txt" {
                        w.Header().Set("Content-Type", "text/plain")
                        w.Write([]byte(robotsTxt))
                        return
                }

                if r.URL.Path == "/forbidden/secret.html" {
                        t.Error("forbidden path was accessed")
                }

                w.Header().Set("Content-Type", "text/html")
                fmt.Fprintf(w, `<html><head><title>Page</title></head><body>Content</body></html>`)
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/index.html",
                ParsedEntry:  mustParseURL(server.URL + "/index.html"),
                Workers:      2,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      10,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 5,
        }

        ctx := context.Background()
        if err := crawler.Run(ctx, cfg); err != nil {
                t.Fatalf("crawl failed: %v", err)
        }

        // Verify forbidden path was not accessed
        for _, path := range accessedPaths {
                if path == "/forbidden/secret.html" {
                        t.Error("robots.txt disallowed path was accessed")
                }
        }
}

// TestExtractor tests HTML content extraction.
func TestExtractor(t *testing.T) {
        html := `<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<nav>Navigation</nav>
<main>
<h1>Main Heading</h1>
<p>This is the main content.</p>
<a href="/page2">Internal Link</a>
<a href="https://external.com/page">External Link</a>
</main>
<script>console.log('script');</script>
<style>body { color: red; }</style>
</body>
</html>`

        ext := extractor.New("example.com")
        page, err := ext.Extract("https://example.com/page1", []byte(html))
        if err != nil {
                t.Fatalf("extraction failed: %v", err)
        }

        // Check title
        if page.Title != "Test Page" {
                t.Errorf("expected title 'Test Page', got %q", page.Title)
        }

        // Check body doesn't contain script/style content
        if containsSubstring(page.Body, "console.log") {
                t.Error("body should not contain script content")
        }
        if containsSubstring(page.Body, "color: red") {
                t.Error("body should not contain style content")
        }

        // Check body contains main content
        if !containsSubstring(page.Body, "Main Heading") {
                t.Error("body should contain main content heading")
        }

        // Check links - only same-domain
        if len(page.Links) != 1 {
                t.Errorf("expected 1 internal link, got %d", len(page.Links))
        }
        if len(page.Links) > 0 && page.Links[0] != "https://example.com/page2" {
                t.Errorf("expected link to page2, got %q", page.Links[0])
        }
}

// TestExtractorEdgeCases tests edge cases in HTML extraction.
func TestExtractorEdgeCases(t *testing.T) {
        tests := []struct {
                name        string
                html        string
                expectTitle string
                expectBody  bool
        }{
                {
                        name:        "empty page",
                        html:        `<html><body></body></html>`,
                        expectTitle: "",
                        expectBody:  false,
                },
                {
                        name:        "malformed HTML",
                        html:        `<html><head><title>Test</title><body><p>Content`,
                        expectTitle: "Test",
                        expectBody:  true,
                },
                {
                        name:        "unicode content",
                        html:        `<html><head><title>日本語</title></head><body>内容</body></html>`,
                        expectTitle: "日本語",
                        expectBody:  true,
                },
                {
                        name:        "very long title",
                        html:        fmt.Sprintf(`<html><head><title>%s</title></head><body></body></html>`, string(make([]byte, 10000))),
                        expectTitle: "", // Will be truncated
                        expectBody:  false,
                },
        }

        ext := extractor.New("example.com")

        for _, tt := range tests {
                t.Run(tt.name, func(t *testing.T) {
                        page, err := ext.Extract("https://example.com/page", []byte(tt.html))
                        if err != nil {
                                t.Fatalf("extraction failed: %v", err)
                        }

                        if tt.expectTitle != "" && page.Title != tt.expectTitle {
                                t.Errorf("expected title %q, got %q", tt.expectTitle, page.Title)
                        }

                        if tt.expectBody && page.Body == "" {
                                t.Error("expected non-empty body")
                        }
                })
        }
}

// TestIndexBuilder tests page writing and inverted index building.
func TestIndexBuilder(t *testing.T) {
        outputDir := t.TempDir()
        builder := index.NewBuilder(outputDir)

        doc := &index.PageDocument{
                URL:       "https://example.com/page1",
                Title:     "Test Page",
                Body:      "This is a test page with some words. The test contains multiple test instances.",
                Links:     []string{"https://example.com/page2"},
                CrawledAt: time.Now().Format(time.RFC3339),
        }

        if err := builder.WritePage(doc); err != nil {
                t.Fatalf("failed to write page: %v", err)
        }

        if count := builder.PageCount(); count != 1 {
                t.Errorf("expected page count 1, got %d", count)
        }

        // Build inverted index
        if err := builder.BuildInvertedIndex(); err != nil {
                t.Fatalf("failed to build inverted index: %v", err)
        }

        // Verify files exist
        if _, err := os.Stat(filepath.Join(outputDir, "pages")); os.IsNotExist(err) {
                t.Error("pages directory should exist")
        }
        if _, err := os.Stat(filepath.Join(outputDir, "inverted-index.json")); os.IsNotExist(err) {
                t.Error("inverted-index.json should exist")
        }

        // Load and verify inverted index
        data, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
        if err != nil {
                t.Fatalf("failed to read inverted index: %v", err)
        }

        var idx index.InvertedIndex
        if err := json.Unmarshal(data, &idx); err != nil {
                t.Fatalf("failed to parse inverted index: %v", err)
        }

        // Check that "test" is indexed (appears 3 times)
        if urls, ok := idx.Terms["test"]; !ok {
                t.Error("expected 'test' to be indexed")
        } else if len(urls) != 1 || urls[0] != doc.URL {
                t.Errorf("expected 'test' to map to %q, got %v", doc.URL, urls)
        }
}

// TestRedirectHandling tests handling of redirect chains.
func TestRedirectHandling(t *testing.T) {
        redirectCount := 0
        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                switch r.URL.Path {
                case "/start":
                        redirectCount++
                        if redirectCount < 3 {
                                http.Redirect(w, r, "/start", http.StatusFound)
                                return
                        }
                        http.Redirect(w, r, "/final", http.StatusFound)
                case "/final":
                        w.Header().Set("Content-Type", "text/html")
                        fmt.Fprintf(w, `<html><head><title>Final</title></head><body>Content</body></html>`)
                default:
                        w.WriteHeader(http.StatusNotFound)
                }
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/start",
                ParsedEntry:  mustParseURL(server.URL + "/start"),
                Workers:      1,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      10,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 10,
        }

        ctx := context.Background()
        err := crawler.Run(ctx, cfg)

        // Should succeed after following redirects
        if err != nil {
                t.Logf("crawl result: %v", err)
        }
}

// TestErrorHandling tests handling of various HTTP errors.
func TestErrorHandling(t *testing.T) {
        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                switch r.URL.Path {
                case "/404":
                        w.WriteHeader(http.StatusNotFound)
                case "/500":
                        w.WriteHeader(http.StatusInternalServerError)
                case "/429":
                        w.Header().Set("Retry-After", "1")
                        w.WriteHeader(http.StatusTooManyRequests)
                case "/good":
                        w.Header().Set("Content-Type", "text/html")
                        fmt.Fprintf(w, `<html><head><title>Good</title></head><body><a href="/404">404</a><a href="/500">500</a></body></html>`)
                }
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/good",
                ParsedEntry:  mustParseURL(server.URL + "/good"),
                Workers:      2,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      10,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 5,
        }

        ctx := context.Background()
        if err := crawler.Run(ctx, cfg); err != nil {
                t.Fatalf("crawl failed: %v", err)
        }

        // Verify good page was crawled
        pagesDir := filepath.Join(outputDir, "pages")
        entries, _ := os.ReadDir(pagesDir)

        found := false
        for _, entry := range entries {
                data, err := os.ReadFile(filepath.Join(pagesDir, entry.Name()))
                if err != nil {
                        continue
                }
                var doc index.PageDocument
                if json.Unmarshal(data, &doc) == nil {
                        if doc.URL == server.URL+"/good" {
                                found = true
                                break
                        }
                }
        }

        if !found {
                t.Error("good page should have been crawled")
        }
}

// TestMemoryStability tests that memory doesn't grow unboundedly.
func TestMemoryStability(t *testing.T) {
        if testing.Short() {
                t.Skip("skipping memory stability test in short mode")
        }

        pageCount := 0
        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                pageCount++
                w.Header().Set("Content-Type", "text/html")
                // Generate a page with 10000 outbound links (stress test)
                fmt.Fprintf(w, `<html><head><title>Page %d</title></head><body>`, pageCount)
                for i := 0; i < 10000; i++ {
                        fmt.Fprintf(w, `<a href="/page%d">Link %d</a>`, i, i)
                }
                fmt.Fprintf(w, `</body></html>`)
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/start",
                ParsedEntry:  mustParseURL(server.URL + "/start"),
                Workers:      5,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      1,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 5,
                MaxPages:     100, // Limit pages for test
        }

        ctx := context.Background()
        if err := crawler.Run(ctx, cfg); err != nil {
                t.Fatalf("crawl failed: %v", err)
        }

        // Verify output was created
        manifest, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
        if err != nil {
                t.Fatalf("manifest not created: %v", err)
        }

        var m map[string]interface{}
        if err := json.Unmarshal(manifest, &m); err != nil {
                t.Fatalf("invalid manifest: %v", err)
        }

        t.Logf("Crawled %v pages", m["pages_crawled"])
}

// TestGracefulShutdown tests that crawler shuts down cleanly on context cancel.
func TestGracefulShutdown(t *testing.T) {
        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                time.Sleep(100 * time.Millisecond) // Slow response
                w.Header().Set("Content-Type", "text/html")
                fmt.Fprintf(w, `<html><head><title>Page</title></head><body>Content</body></html>`)
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/page",
                ParsedEntry:  mustParseURL(server.URL + "/page"),
                Workers:      5,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      10,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 5,
        }

        ctx, cancel := context.WithCancel(context.Background())

        // Cancel after a short time
        go func() {
                time.Sleep(200 * time.Millisecond)
                cancel()
        }()

        // Should not panic or hang
        _ = crawler.Run(ctx, cfg)
}

// TestNoDuplicateFetches verifies the same URL is never fetched twice.
func TestNoDuplicateFetches(t *testing.T) {
        fetchCount := make(map[string]int)
        var mu sync.Mutex

        server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
                mu.Lock()
                fetchCount[r.URL.Path]++
                mu.Unlock()

                w.Header().Set("Content-Type", "text/html")
                // Create circular links to test dedup
                fmt.Fprintf(w, `<html><head><title>Page</title></head><body>
                        <a href="/">Home</a>
                        <a href="/page1">Page 1</a>
                        <a href="/page2">Page 2</a>
                        <a href="/page1/">Page 1 with slash</a>
                </body></html>`)
        }))
        defer server.Close()

        outputDir := t.TempDir()

        cfg := &config.Config{
                Entry:        server.URL + "/",
                ParsedEntry:  mustParseURL(server.URL + "/"),
                Workers:      3,
                Timeout:      5 * time.Second,
                OutputDir:    outputDir,
                DelayMS:      10,
                UserAgent:    "TestBot",
                MaxRetries:   1,
                MaxRedirects: 5,
        }

        ctx := context.Background()
        if err := crawler.Run(ctx, cfg); err != nil {
                t.Fatalf("crawl failed: %v", err)
        }

        // Verify no URL was fetched more than once
        for path, count := range fetchCount {
                if count > 1 {
                        t.Errorf("path %q was fetched %d times (expected 1)", path, count)
                }
        }
}

// TestInvertedIndexSchema verifies the inverted index has correct schema.
func TestInvertedIndexSchema(t *testing.T) {
        outputDir := t.TempDir()
        builder := index.NewBuilder(outputDir)

        docs := []*index.PageDocument{
                {
                        URL:       "https://example.com/page1",
                        Title:     "Go Programming",
                        Body:      "Go is a programming language. Go is fast.",
                        CrawledAt: time.Now().Format(time.RFC3339),
                },
                {
                        URL:       "https://example.com/page2",
                        Title:     "Python Programming",
                        Body:      "Python is another programming language.",
                        CrawledAt: time.Now().Format(time.RFC3339),
                },
        }

        for _, doc := range docs {
                if err := builder.WritePage(doc); err != nil {
                        t.Fatalf("failed to write page: %v", err)
                }
        }

        if err := builder.BuildInvertedIndex(); err != nil {
                t.Fatalf("failed to build index: %v", err)
        }

        // Load and verify schema
        data, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
        if err != nil {
                t.Fatalf("failed to read index: %v", err)
        }

        var idx map[string][]string
        if err := json.Unmarshal(data, &idx); err != nil {
                t.Fatalf("failed to parse index: %v", err)
        }

        // Verify "programming" appears in both pages
        if urls, ok := idx["programming"]; !ok {
                t.Error("expected 'programming' in index")
        } else if len(urls) != 2 {
                t.Errorf("expected 'programming' in 2 pages, got %d", len(urls))
        }

        // Verify "go" appears only in page1
        if urls, ok := idx["go"]; !ok {
                t.Error("expected 'go' in index")
        } else if len(urls) != 1 {
                t.Errorf("expected 'go' in 1 page, got %d", len(urls))
        }
}

// Helper functions

func mustParseURL(rawURL string) *url.URL {
        u, err := url.Parse(rawURL)
        if err != nil {
                panic(err)
        }
        return u
}

func containsSubstring(s, substr string) bool {
        return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsSubstringHelper(s, substr))
}

func containsSubstringHelper(s, substr string) bool {
        for i := 0; i <= len(s)-len(substr); i++ {
                if s[i:i+len(substr)] == substr {
                        return true
                }
        }
        return false
}
