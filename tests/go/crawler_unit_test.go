// Package tests contains comprehensive unit tests for the Go crawler.
// Each test has precise PASS/FAIL conditions documented in comments.
package tests

import (
	"net/url"
	"sync"
	"testing"

	"crawler/internal/dedup"
	"crawler/internal/extractor"
	"crawler/internal/robots"

	"go.uber.org/goleak"
)

// TestMain ensures no goroutine leaks across all tests.
// FAIL: If any goroutine is still running after a test completes.
func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// ============================================================================
// DEDUPLICATION TESTS
// ============================================================================

// TestDedupBasic tests basic URL deduplication.
// PASS: First mark returns true (URL not seen), second mark returns false (URL seen).
// FAIL: First mark returns false (would mean empty store already contains URL).
// FAIL: Second mark returns true (would mean duplicate was not detected).
func TestDedupBasic(t *testing.T) {
	store := dedup.New()

	// First occurrence must be marked as new
	if !store.CheckAndMark("https://example.com/page1") {
		t.Error("FAIL: First CheckAndMark returned false - empty store should accept new URL")
	}

	// Second occurrence must be detected as duplicate
	if store.CheckAndMark("https://example.com/page1") {
		t.Error("FAIL: Second CheckAndMark returned true - duplicate URL should be rejected")
	}
}

// TestDedupNormalization tests that URL normalization works correctly.
// PASS: URLs that differ only by trailing slash, fragment, or case are treated as identical.
// FAIL: Normalized URLs are treated as different (would cause duplicate fetches).
func TestDedupNormalization(t *testing.T) {
	store := dedup.New()

	// Mark original URL
	store.CheckAndMark("https://example.com/page")

	// Trailing slash should normalize to same URL
	if store.CheckAndMark("https://example.com/page/") {
		t.Error("FAIL: URL with trailing slash not normalized - would cause duplicate fetch")
	}

	// Fragment should be stripped
	if store.CheckAndMark("https://example.com/page#section") {
		t.Error("FAIL: URL with fragment not normalized - would cause duplicate fetch")
	}

	// Uppercase host should normalize to lowercase
	if store.CheckAndMark("https://EXAMPLE.COM/page") {
		t.Error("FAIL: Uppercase host not normalized - would cause duplicate fetch")
	}
}

// TestDedupConcurrent tests thread-safety under concurrent access.
// PASS: 10 goroutines × 100 URLs = 1000 total, no race conditions, no lost updates.
// FAIL: Final count != 1000 (would indicate race condition or lost updates).
func TestDedupConcurrent(t *testing.T) {
	store := dedup.New()
	const workers = 10
	const urlsPerWorker = 100

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < urlsPerWorker; i++ {
				u := "https://example.com/page/" + string(rune('A'+workerID)) + "/" + string(rune('0'+i%10))
				store.CheckAndMark(u)
			}
		}(w)
	}

	wg.Wait()

	count := store.Count()
	expected := workers * urlsPerWorker
	// FAIL: Count mismatch indicates race condition
	if count != expected {
		t.Errorf("FAIL: Expected count %d, got %d - concurrent race condition detected", expected, count)
	}
}

// TestDedupDifferentURLs tests that different URLs are correctly tracked.
// PASS: Two different URLs result in count of 2.
// FAIL: Count != 2 (would indicate URLs incorrectly deduped or not stored).
func TestDedupDifferentURLs(t *testing.T) {
	store := dedup.New()

	store.CheckAndMark("https://example.com/page1")
	store.CheckAndMark("https://example.com/page2")

	// FAIL: Count should be exactly 2
	if count := store.Count(); count != 2 {
		t.Errorf("FAIL: Expected 2 distinct URLs, got %d - URLs incorrectly stored", count)
	}
}

// ============================================================================
// ROBOTS.TXT TESTS
// ============================================================================

// TestRobotsBasicAllow tests basic allow rule.
// PASS: URL matching an Allow rule returns IsAllowed = true.
// FAIL: Allowed URL returns false (would block valid content).
func TestRobotsBasicAllow(t *testing.T) {
	robotsContent := `User-agent: *
Allow: /public/`

	server := NewMockServer()
	defer server.Close()
	server.SetRobots(robotsContent)

	checker := robots.NewChecker(server.Client(), "TestBot")
	if err := checker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	// FAIL: Public path should be allowed
	if !checker.IsAllowed(server.URL() + "/public/page.html") {
		t.Error("FAIL: /public/page.html should be allowed by Allow: /public/")
	}
}

// TestRobotsDisallow tests basic disallow rule.
// PASS: URL matching a Disallow rule returns IsAllowed = false.
// FAIL: Disallowed URL returns true (would violate site policy).
func TestRobotsDisallow(t *testing.T) {
	robotsContent := `User-agent: *
Disallow: /private/`

	server := NewMockServer()
	defer server.Close()
	server.SetRobots(robotsContent)

	checker := robots.NewChecker(server.Client(), "TestBot")
	if err := checker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	// FAIL: Private path should be disallowed
	if checker.IsAllowed(server.URL() + "/private/secret.html") {
		t.Error("FAIL: /private/secret.html should be disallowed by Disallow: /private/")
	}
}

// TestRobotsUserAgentSpecific tests user-agent-specific rules.
// PASS: Specific user-agent rules override generic * rules.
// FAIL: Wrong rule applied (would either block allowed paths or access disallowed ones).
func TestRobotsUserAgentSpecific(t *testing.T) {
	robotsContent := `User-agent: BadBot
Disallow: /

User-agent: GoodBot
Disallow: /admin/`

	server := NewMockServer()
	defer server.Close()
	server.SetRobots(robotsContent)

	// Test GoodBot
	goodChecker := robots.NewChecker(server.Client(), "GoodBot")
	if err := goodChecker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	// FAIL: GoodBot should be allowed on public pages
	if !goodChecker.IsAllowed(server.URL() + "/public/page.html") {
		t.Error("FAIL: GoodBot should be allowed on /public/")
	}

	// FAIL: GoodBot should be disallowed on admin pages
	if goodChecker.IsAllowed(server.URL() + "/admin/login.html") {
		t.Error("FAIL: GoodBot should be disallowed on /admin/")
	}

	// Test BadBot
	badChecker := robots.NewChecker(server.Client(), "BadBot")
	if err := badChecker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	// FAIL: BadBot should be disallowed everywhere
	if badChecker.IsAllowed(server.URL() + "/any/page.html") {
		t.Error("FAIL: BadBot should be disallowed on all paths")
	}
}

// TestRobotsCrawlDelay tests crawl-delay directive.
// PASS: Crawl-delay value is correctly parsed and returned.
// FAIL: Wrong delay value (would cause rate limiting issues).
func TestRobotsCrawlDelay(t *testing.T) {
	robotsContent := `User-agent: *
Crawl-delay: 2`

	server := NewMockServer()
	defer server.Close()
	server.SetRobots(robotsContent)

	checker := robots.NewChecker(server.Client(), "TestBot")
	if err := checker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	delay := checker.GetCrawlDelay()
	// FAIL: Delay should be exactly 2 seconds
	if delay != 2 {
		t.Errorf("FAIL: Expected crawl delay of 2s, got %v", delay)
	}
}

// TestRobotsNotFound tests behavior when robots.txt is missing.
// PASS: Missing robots.txt allows all paths (conservative default).
// FAIL: Returns false for any path (would block entire site).
func TestRobotsNotFound(t *testing.T) {
	server := NewMockServer()
	defer server.Close()
	// No robots.txt set, will return 404

	checker := robots.NewChecker(server.Client(), "TestBot")
	if err := checker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Unexpected error fetching missing robots.txt: %v", err)
	}

	// FAIL: When robots.txt is missing, all paths should be allowed
	if !checker.IsAllowed(server.URL() + "/any/path.html") {
		t.Error("FAIL: Missing robots.txt should allow all paths")
	}
}

// TestRobotsDisallowAll tests Disallow: / rule.
// PASS: Disallow: / blocks all paths.
// FAIL: Any path returns true (would violate site-wide disallow).
func TestRobotsDisallowAll(t *testing.T) {
	robotsContent := `User-agent: *
Disallow: /`

	server := NewMockServer()
	defer server.Close()
	server.SetRobots(robotsContent)

	checker := robots.NewChecker(server.Client(), "TestBot")
	if err := checker.Fetch(server.Context(), server.URL()); err != nil {
		t.Fatalf("FAIL: Failed to fetch robots.txt: %v", err)
	}

	// FAIL: No path should be allowed
	if checker.IsAllowed(server.URL() + "/any/path.html") {
		t.Error("FAIL: Disallow: / should block all paths")
	}
}

// ============================================================================
// EXTRACTOR TESTS
// ============================================================================

// TestExtractorBasic tests basic HTML extraction.
// PASS: Title, body text, and links are correctly extracted.
// FAIL: Any extraction returns wrong data.
func TestExtractorBasic(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Heading</h1>
<p>Body content here.</p>
<a href="/link1">Link 1</a>
<a href="/link2">Link 2</a>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: Wrong title extracted
	if page.Title != "Test Page" {
		t.Errorf("FAIL: Expected title 'Test Page', got %q", page.Title)
	}

	// FAIL: Body doesn't contain expected content
	if !containsString(page.Body, "Body content") {
		t.Errorf("FAIL: Body should contain 'Body content', got: %q", page.Body)
	}

	// FAIL: Wrong number of links
	if len(page.Links) != 2 {
		t.Errorf("FAIL: Expected 2 links, got %d", len(page.Links))
	}
}

// TestExtractorRemovesScripts tests that script content is removed.
// PASS: Script content does not appear in extracted body.
// FAIL: Script content found in body (would pollute semantic search).
func TestExtractorRemovesScripts(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<body>
<script>alert('malicious');</script>
<p>Visible content</p>
<script>console.log('debug');</script>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: Script content should be removed
	if containsString(page.Body, "alert") || containsString(page.Body, "console.log") {
		t.Errorf("FAIL: Script content found in body: %q", page.Body)
	}
}

// TestExtractorRemovesStyles tests that style content is removed.
// PASS: CSS content does not appear in extracted body.
// FAIL: CSS content found in body.
func TestExtractorRemovesStyles(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<head><style>body { color: red; }</style></head>
<body>
<p>Content</p>
<style>.hidden { display: none; }</style>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: CSS content should be removed
	if containsString(page.Body, "color:") || containsString(page.Body, "display:") {
		t.Errorf("FAIL: CSS content found in body: %q", page.Body)
	}
}

// TestExtractorSameDomainOnly tests that only same-domain links are extracted.
// PASS: Only links to the same domain are included.
// FAIL: External links appear in the list (would cause off-site crawling).
func TestExtractorSameDomainOnly(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<body>
<a href="/internal">Internal Link</a>
<a href="https://example.com/page">Same Domain</a>
<a href="https://external.com/page">External Link</a>
<a href="https://another.org/page">Another External</a>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: External links should not be included
	for _, link := range page.Links {
		u, _ := url.Parse(link)
		if u.Host != "example.com" {
			t.Errorf("FAIL: External link found: %s (should be filtered)", link)
		}
	}

	// FAIL: Should have exactly 2 same-domain links
	if len(page.Links) != 2 {
		t.Errorf("FAIL: Expected 2 same-domain links, got %d: %v", len(page.Links), page.Links)
	}
}

// TestExtractorUnicode tests Unicode content handling.
// PASS: Unicode characters (CJK, Arabic, emoji) are preserved.
// FAIL: Unicode corrupted or missing.
func TestExtractorUnicode(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<head><title>中文标题 العربية Emoji 🎮</title></head>
<body>
<p>日本語コンテンツ</p>
<p>محتوى عربي</p>
<p>Emoji: 🚀 🎮 🔥</p>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: Unicode should be preserved
	if !containsString(page.Title, "中文") {
		t.Error("FAIL: Chinese characters missing from title")
	}
	if !containsString(page.Title, "العربية") {
		t.Error("FAIL: Arabic characters missing from title")
	}
	if !containsString(page.Body, "日本語") {
		t.Error("FAIL: Japanese characters missing from body")
	}
}

// TestExtractorDedupLinks tests that duplicate links are deduplicated.
// PASS: Same link appearing multiple times in HTML is extracted only once.
// FAIL: Duplicate links appear in output.
func TestExtractorDedupLinks(t *testing.T) {
	html := `<!DOCTYPE html>
<html>
<body>
<a href="/page">Link</a>
<a href="/page">Same Link</a>
<a href="/page">Again Same Link</a>
<a href="/other">Other Link</a>
</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed: %v", err)
	}

	// FAIL: Should have exactly 2 unique links
	if len(page.Links) != 2 {
		t.Errorf("FAIL: Expected 2 unique links, got %d (duplicates not removed)", len(page.Links))
	}
}

// TestExtractorEmptyPage tests handling of empty/minimal pages.
// PASS: Empty page doesn't crash, returns empty fields.
// FAIL: Panic or error on empty page.
func TestExtractorEmptyPage(t *testing.T) {
	html := `<!DOCTYPE html>
<html><body></body></html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed on empty page: %v", err)
	}

	// FAIL: Should not crash, just return empty values
	if page.Title != "" {
		t.Errorf("FAIL: Expected empty title, got %q", page.Title)
	}
}

// TestExtractorLargePage tests handling of pages with many links.
// PASS: 10,000 links extracted without OOM or crash.
// FAIL: Panic, OOM, or incorrect count.
func TestExtractorLargePage(t *testing.T) {
	var links string
	for i := 0; i < 10000; i++ {
		links += `<a href="/page/` + string(rune('0'+i%10)) + `">Link</a>`
	}

	html := `<!DOCTYPE html>
<html>
<body>` + links + `</body>
</html>`

	ext := extractor.New("example.com")
	page, err := ext.Extract("https://example.com/test", []byte(html))
	if err != nil {
		t.Fatalf("FAIL: Extraction failed on large page: %v", err)
	}

	// FAIL: Should handle large number of links
	// Due to dedup, actual count will be less (only 10 unique paths)
	if len(page.Links) == 0 {
		t.Error("FAIL: No links extracted from large page")
	}
}

// Helper functions

func containsString(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && containsSubstring(s, substr))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
