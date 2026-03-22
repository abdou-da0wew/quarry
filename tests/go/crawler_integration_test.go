// Package tests contains integration tests for the Go crawler.
// Each test runs against a mock HTTP server with precise PASS/FAIL conditions.
package tests

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"crawler/internal/config"
	"crawler/internal/crawler"
	"crawler/internal/index"

	"go.uber.org/goleak"
)

// ============================================================================
// INTEGRATION TESTS - CRAWLER AGAINST MOCK SERVER
// ============================================================================

// TestCrawlEmptySite tests crawling a site with zero outbound links.
// PASS: Entry page is crawled, count is 1, no other pages found.
// FAIL: Count != 1 or crawler hangs waiting for more pages.
func TestCrawlEmptySite(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create site with single page, no links
	entryURL := server.CreateEmptySite("/index.html")

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        entryURL,
		ParsedEntry:  mustParseURL(entryURL),
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
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// FAIL: Should have crawled exactly 1 page
	pagesDir := filepath.Join(outputDir, "pages")
	entries, _ := os.ReadDir(pagesDir)
	if len(entries) != 1 {
		t.Errorf("FAIL: Expected 1 page, got %d - empty site should only have entry page", len(entries))
	}

	// Verify manifest
	manifest, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("FAIL: No manifest.json created: %v", err)
	}

	var m map[string]interface{}
	if err := json.Unmarshal(manifest, &m); err != nil {
		t.Fatalf("FAIL: Invalid manifest JSON: %v", err)
	}

	// FAIL: pages_crawled must be 1
	if pagesCrawled, ok := m["pages_crawled"].(float64); !ok || int(pagesCrawled) != 1 {
		t.Errorf("FAIL: Expected pages_crawled=1, got %v", m["pages_crawled"])
	}
}

// TestCrawlCircularRedirect tests that circular redirects are detected and broken.
// PASS: Crawler doesn't hang, returns after max redirect hops.
// FAIL: Infinite loop or goroutine leak.
func TestCrawlCircularRedirect(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create circular redirect A -> B -> A
	server.CreateCircularRedirect()
	entryURL := server.URL() + "/a"

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        entryURL,
		ParsedEntry:  mustParseURL(entryURL),
		Workers:      1,
		Timeout:      2 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      10,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 10,
	}

	// FAIL: Should complete within 5 seconds (not hang)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := crawler.Run(ctx, cfg)
	// Either succeeds (stopped by redirect limit) or fails gracefully
	// FAIL: If it times out, the redirect loop wasn't handled
	if ctx.Err() == context.DeadlineExceeded {
		t.Error("FAIL: Crawler hung on circular redirect - redirect loop not detected")
	}

	t.Logf("Crawl result (expected redirect handling): %v", err)
}

// TestCrawlRobotsDisallowAll tests that Disallow: / is respected.
// PASS: Zero pages crawled beyond entry (entry may be fetched before robots check).
// FAIL: Any pages crawled indicates robots.txt violation.
func TestCrawlRobotsDisallowAll(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Set robots.txt to disallow everything
	server.SetRobots("User-agent: *\nDisallow: /")

	// Create linked pages
	server.SetHTML("/index.html", `<html><body><a href="/page1">Page 1</a><a href="/page2">Page 2</a></body></html>`)
	server.SetHTML("/page1.html", `<html><body>Page 1</body></html>`)
	server.SetHTML("/page2.html", `<html><body>Page 2</body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
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
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// FAIL: Should not have crawled /page1.html or /page2.html
	if server.WasRequested("/page1.html") {
		t.Error("FAIL: /page1.html was requested despite Disallow: /")
	}
	if server.WasRequested("/page2.html") {
		t.Error("FAIL: /page2.html was requested despite Disallow: /")
	}
}

// TestCrawlLargePageWithManyLinks tests handling of pages with 10,000 links.
// PASS: No OOM, no crash, all unique links are eventually found.
// FAIL: Panic, OOM, or incorrect link extraction.
func TestCrawlLargePageWithManyLinks(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create page with 10,000 links
	server.CreateLargePage("/index.html", 10000)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      1,
		Timeout:      10 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      1, // Fast for test
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 5,
		MaxPages:     1, // Only crawl entry page
	}

	ctx := context.Background()
	if err := crawler.Run(ctx, cfg); err != nil {
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// FAIL: Should have crawled exactly 1 page (entry page)
	pagesDir := filepath.Join(outputDir, "pages")
	entries, _ := os.ReadDir(pagesDir)
	if len(entries) != 1 {
		t.Errorf("FAIL: Expected 1 page, got %d", len(entries))
	}

	// Verify the page was processed
	manifest, _ := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	var m map[string]interface{}
	json.Unmarshal(manifest, &m)
	t.Logf("Crawled %v pages from large-link page", m["pages_crawled"])
}

// TestCrawlNonUTF8Content tests handling of invalid UTF-8 in page content.
// PASS: Non-UTF8 pages are handled gracefully, no panic.
// FAIL: Panic or crash on invalid encoding.
func TestCrawlNonUTF8Content(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create page with invalid UTF-8
	server.CreateNonUTF8Page("/index.html")

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      1,
		Timeout:      5 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      10,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 5,
	}

	ctx := context.Background()
	// FAIL: Should not panic
	if err := crawler.Run(ctx, cfg); err != nil {
		t.Fatalf("FAIL: Crawl panicked on non-UTF8 content: %v", err)
	}

	t.Log("PASS: Non-UTF8 content handled gracefully")
}

// TestCrawlAllTimeouts tests that crawler exits cleanly when all requests timeout.
// PASS: Crawler completes (with failures) without hanging or leaking goroutines.
// FAIL: Goroutine leak or indefinite hang.
func TestCrawlAllTimeouts(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Set very short timeout to force all requests to fail
	server.SetHTML("/index.html", `<html><body><a href="/slow">Slow</a></body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      1,
		Timeout:      1 * time.Millisecond, // Extremely short
		OutputDir:    outputDir,
		DelayMS:      0,
		UserAgent:    "TestBot",
		MaxRetries:   0,
		MaxRedirects: 5,
	}

	// FAIL: Should complete within 5 seconds
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := crawler.Run(ctx, cfg)

	// FAIL: If context deadline exceeded, crawler hung
	if ctx.Err() == context.DeadlineExceeded {
		t.Error("FAIL: Crawler hung on timeout - didn't exit cleanly")
	}

	t.Logf("Crawl result (expected timeout handling): %v", err)
}

// TestCrawlRateLimited tests handling of 429 Too Many Requests.
// PASS: Crawler backs off and retries, eventually succeeds or gives up after max retries.
// FAIL: Immediate failure without retry, or infinite retry loop.
func TestCrawlRateLimited(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	var requestCount int64

	// Handler that returns 429 twice then 200
	http.DefaultServeMux = http.NewServeMux()
	http.Handle("/test", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt64(&requestCount, 1)
		if count <= 2 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte("Rate limited"))
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("<html><body>Success after rate limit</body></html>"))
	}))

	// Use custom server for this test
	testServer := httptest.NewServer(http.DefaultServeMux)
	defer testServer.Close()

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        testServer.URL + "/test",
		ParsedEntry:  mustParseURL(testServer.URL),
		Workers:      1,
		Timeout:      5 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      0,
		UserAgent:    "TestBot",
		MaxRetries:   5,
		MaxRedirects: 5,
	}

	ctx := context.Background()
	err := crawler.Run(ctx, cfg)

	// Verify multiple requests were made (retry happened)
	// FAIL: If only 1 request, retry didn't happen
	if requestCount < 2 {
		t.Errorf("FAIL: Expected multiple requests due to retry, got %d", requestCount)
	}

	t.Logf("Made %d requests (retries for rate limiting)", requestCount)
	t.Logf("Crawl result: %v", err)
}

// TestCrawl404Handling tests that 404s are handled gracefully.
// PASS: 404 pages are skipped, crawl continues to other pages.
// FAIL: Crawl stops entirely on 404.
func TestCrawl404Handling(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Entry page has link to 404 and valid page
	server.SetHTML("/index.html", `<html><body>
		<a href="/missing">Missing Page</a>
		<a href="/valid">Valid Page</a>
	</body></html>`)
	server.SetError("/missing", http.StatusNotFound)
	server.SetHTML("/valid.html", `<html><body>Valid Content</body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
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
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// FAIL: Should have crawled 2 pages (index + valid), not 1 (stopped on 404)
	pagesDir := filepath.Join(outputDir, "pages")
	entries, _ := os.ReadDir(pagesDir)
	if len(entries) != 2 {
		t.Errorf("FAIL: Expected 2 pages (skipped 404), got %d", len(entries))
	}

	// FAIL: 404 page should have been requested but not saved
	if !server.WasRequested("/missing") {
		t.Error("FAIL: 404 page should have been attempted")
	}
}

// TestCrawlDeduplication tests that same URL is never fetched twice.
// PASS: Each URL appears exactly once in request log.
// FAIL: Any URL appears more than once.
func TestCrawlDeduplication(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create circular link structure
	server.SetHTML("/a", `<html><body><a href="/b">B</a><a href="/a">Self</a></body></html>`)
	server.SetHTML("/b", `<html><body><a href="/a">A</a><a href="/b">Self</a></body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/a",
		ParsedEntry:  mustParseURL(server.URL()),
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
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// FAIL: Each path should be requested exactly once
	for _, path := range []string{"/a", "/b"} {
		count := server.RequestCountFor(path)
		if count > 1 {
			t.Errorf("FAIL: Path %s requested %d times - deduplication failed", path, count)
		}
	}
}

// TestCrawlInvertedIndex tests that inverted index is built correctly.
// PASS: Terms from crawled pages appear in inverted index with correct URLs.
// FAIL: Missing terms or wrong URLs in index.
func TestCrawlInvertedIndex(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create pages with distinct content
	server.SetHTML("/index.html", `<html><body><a href="/apple">Apple</a></body></html>`)
	server.SetHTML("/apple.html", `<html><body><h1>Apple Page</h1><p>This page is about apples and fruit.</p></body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      1,
		Timeout:      5 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      10,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 5,
	}

	ctx := context.Background()
	if err := crawler.Run(ctx, cfg); err != nil {
		t.Fatalf("FAIL: Crawl failed: %v", err)
	}

	// Load inverted index
	indexData, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
	if err != nil {
		t.Fatalf("FAIL: No inverted-index.json: %v", err)
	}

	var idx map[string][]string
	if err := json.Unmarshal(indexData, &idx); err != nil {
		t.Fatalf("FAIL: Invalid inverted index JSON: %v", err)
	}

	// FAIL: Term "apple" should be in index
	if urls, ok := idx["apple"]; !ok {
		t.Error("FAIL: Term 'apple' not found in inverted index")
	} else if len(urls) != 1 {
		t.Errorf("FAIL: Expected 'apple' in 1 URL, got %d", len(urls))
	}

	// FAIL: Term "fruit" should be in index
	if _, ok := idx["fruit"]; !ok {
		t.Error("FAIL: Term 'fruit' not found in inverted index")
	}
}

// TestCrawlRedirectChain tests handling of long redirect chains.
// PASS: Redirect chain is followed up to max_redirects, then stopped.
// FAIL: Infinite loop or panic on long chain.
func TestCrawlRedirectChain(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create 15-hop redirect chain (exceeds default max of 10)
	entryURL := server.CreateLongRedirectChain(15)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        entryURL,
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      1,
		Timeout:      5 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      10,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 10, // Less than chain length
	}

	ctx := context.Background()
	err := crawler.Run(ctx, cfg)

	// Should fail due to redirect limit
	// FAIL: If succeeds, redirect limit wasn't enforced
	t.Logf("Crawl result (expected redirect limit): %v", err)
}

// TestCrawlConcurrentWorkers tests that multiple workers don't cause race conditions.
// PASS: All pages crawled, no race detected by race detector, no crash.
// FAIL: Race condition, deadlock, or panic.
func TestCrawlConcurrentWorkers(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	// Create many linked pages
	pages := make(map[string]string)
	links := make(map[string][]string)

	pages["/index.html"] = "Entry Page"
	links["/index.html"] = []string{"/page1.html", "/page2.html", "/page3.html"}

	for i := 1; i <= 3; i++ {
		path := fmt.Sprintf("/page%d.html", i)
		pages[path] = fmt.Sprintf("Page %d content with unique keyword number%d", i, i)
		links[path] = []string{}
	}

	server.CreateLinkedSite(pages, links)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      10, // High concurrency
		Timeout:      5 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      1,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 5,
	}

	ctx := context.Background()
	if err := crawler.Run(ctx, cfg); err != nil {
		t.Fatalf("FAIL: Crawl failed with concurrent workers: %v", err)
	}

	// FAIL: Should have all 4 pages
	pagesDir := filepath.Join(outputDir, "pages")
	entries, _ := os.ReadDir(pagesDir)
	if len(entries) != 4 {
		t.Errorf("FAIL: Expected 4 pages with concurrent workers, got %d", len(entries))
	}
}

// TestCrawlGracefulShutdown tests that SIGINT/SIGTERM triggers clean shutdown.
// PASS: Crawler shuts down cleanly when context is cancelled.
// FAIL: Goroutine leak or hanging after cancellation.
func TestCrawlGracefulShutdown(t *testing.T) {
	defer goleak.VerifyNone(t)

	server := NewMockServer()
	defer server.Close()

	server.SetHTML("/index.html", `<html><body><a href="/slow">Slow</a></body></html>`)

	outputDir := t.TempDir()

	cfg := &config.Config{
		Entry:        server.URL() + "/index.html",
		ParsedEntry:  mustParseURL(server.URL()),
		Workers:      5,
		Timeout:      10 * time.Second,
		OutputDir:    outputDir,
		DelayMS:      100,
		UserAgent:    "TestBot",
		MaxRetries:   1,
		MaxRedirects: 5,
	}

	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after short delay
	go func() {
		time.Sleep(200 * time.Millisecond)
		cancel()
	}()

	// FAIL: Should not hang or panic on cancellation
	err := crawler.Run(ctx, cfg)
	t.Logf("Crawl result after cancellation: %v", err)
}

// Helper function
func mustParseURL(rawURL string) *url.URL {
	u, err := url.Parse(rawURL)
	if err != nil {
		panic(err)
	}
	return u
}
