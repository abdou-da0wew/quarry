package tests

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"crawler/internal/robots"
)

// TestRobotsParsing tests parsing of robots.txt content.
func TestRobotsParsing(t *testing.T) {
	tests := []struct {
		name           string
		robotsTxt      string
		userAgent      string
		urlChecks      map[string]bool
	}{
		{
			name: "basic disallow",
			robotsTxt: `User-agent: *
Disallow: /private/`,
			userAgent: "TestBot",
			urlChecks: map[string]bool{
				"http://example.com/private/secret": false,
				"http://example.com/public/page":   true,
			},
		},
		{
			name: "user-agent specific",
			robotsTxt: `User-agent: BadBot
Disallow: /

User-agent: GoodBot
Disallow: /admin/`,
			userAgent: "GoodBot",
			urlChecks: map[string]bool{
				"http://example.com/admin/login": false,
				"http://example.com/page":       true,
			},
		},
		{
			name: "allow overrides disallow",
			robotsTxt: `User-agent: *
Disallow: /private/
Allow: /private/public/`,
			userAgent: "TestBot",
			urlChecks: map[string]bool{
				"http://example.com/private/secret":  false,
				"http://example.com/private/public/": true,
			},
		},
		{
			name: "wildcard matching",
			robotsTxt: `User-agent: *
Disallow: /tmp/*
Disallow: /*.pdf`,
			userAgent: "TestBot",
			urlChecks: map[string]bool{
				"http://example.com/tmp/file":  false,
				"http://example.com/document.pdf": false,
				"http://example.com/page.html":    true,
			},
		},
		{
			name: "empty disallow means allow all",
			robotsTxt: `User-agent: *
Disallow:`,
			userAgent: "TestBot",
			urlChecks: map[string]bool{
				"http://example.com/anything": true,
			},
		},
		{
			name: "no matching user-agent defaults to *",
			robotsTxt: `User-agent: SpecificBot
Disallow: /specific/

User-agent: *
Disallow: /general/`,
			userAgent: "UnknownBot",
			urlChecks: map[string]bool{
				"http://example.com/specific/page": true,
				"http://example.com/general/page":  false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/robots.txt" {
					w.Header().Set("Content-Type", "text/plain")
					w.Write([]byte(tt.robotsTxt))
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := &http.Client{Timeout: 5 * time.Second}
			checker := robots.NewChecker(client, tt.userAgent)

			ctx := context.Background()
			if err := checker.Fetch(ctx, server.URL); err != nil {
				t.Fatalf("failed to fetch robots.txt: %v", err)
			}

			for url, expected := range tt.urlChecks {
				result := checker.IsAllowed(url)
				if result != expected {
					t.Errorf("IsAllowed(%q) = %v, expected %v", url, result, expected)
				}
			}
		})
	}
}

// TestRobotsNotFound tests behavior when robots.txt is not found.
func TestRobotsNotFound(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	client := &http.Client{Timeout: 5 * time.Second}
	checker := robots.NewChecker(client, "TestBot")

	ctx := context.Background()
	if err := checker.Fetch(ctx, server.URL); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All URLs should be allowed when robots.txt is not found
	if !checker.IsAllowed(server.URL + "/any/path") {
		t.Error("expected URL to be allowed when robots.txt is not found")
	}
}

// TestRobotsServerError tests behavior when robots.txt returns server error.
func TestRobotsServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := &http.Client{Timeout: 5 * time.Second}
	checker := robots.NewChecker(client, "TestBot")

	ctx := context.Background()
	if err := checker.Fetch(ctx, server.URL); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All URLs should be allowed on server error
	if !checker.IsAllowed(server.URL + "/any/path") {
		t.Error("expected URL to be allowed on robots.txt server error")
	}
}

// TestRobotsCrawlDelay tests extraction of crawl-delay directive.
func TestRobotsCrawlDelay(t *testing.T) {
	tests := []struct {
		name          string
		robotsTxt     string
		userAgent     string
		expectedDelay time.Duration
	}{
		{
			name: "explicit crawl delay",
			robotsTxt: `User-agent: *
Crawl-delay: 2`,
			userAgent:     "TestBot",
			expectedDelay: 2 * time.Second,
		},
		{
			name: "user-agent specific delay",
			robotsTxt: `User-agent: SlowBot
Crawl-delay: 5

User-agent: *
Crawl-delay: 1`,
			userAgent:     "SlowBot",
			expectedDelay: 5 * time.Second,
		},
		{
			name: "no crawl delay",
			robotsTxt: `User-agent: *
Disallow: /private/`,
			userAgent:     "TestBot",
			expectedDelay: 0,
		},
		{
			name: "fractional delay",
			robotsTxt: `User-agent: *
Crawl-delay: 0.5`,
			userAgent:     "TestBot",
			expectedDelay: 500 * time.Millisecond,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/robots.txt" {
					w.Header().Set("Content-Type", "text/plain")
					w.Write([]byte(tt.robotsTxt))
					return
				}
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			client := &http.Client{Timeout: 5 * time.Second}
			checker := robots.NewChecker(client, tt.userAgent)

			ctx := context.Background()
			if err := checker.Fetch(ctx, server.URL); err != nil {
				t.Fatalf("failed to fetch robots.txt: %v", err)
			}

			delay := checker.GetCrawlDelay()
			if delay != tt.expectedDelay {
				t.Errorf("expected delay %v, got %v", tt.expectedDelay, delay)
			}
		})
	}
}

// TestRobotsComments tests handling of comments in robots.txt.
func TestRobotsComments(t *testing.T) {
	robotsTxt := `# This is a comment
User-agent: *  # inline comment
Disallow: /private/  # another comment
# Another comment
Allow: /public/
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
	checker := robots.NewChecker(client, "TestBot")

	ctx := context.Background()
	if err := checker.Fetch(ctx, server.URL); err != nil {
		t.Fatalf("failed to fetch robots.txt: %v", err)
	}

	// Verify rules are parsed correctly despite comments
	if !checker.IsAllowed(server.URL + "/public/page") {
		t.Error("expected /public/page to be allowed")
	}
	if checker.IsAllowed(server.URL + "/private/secret") {
		t.Error("expected /private/secret to be disallowed")
	}
}

// TestRobotsMalformed tests handling of malformed robots.txt.
func TestRobotsMalformed(t *testing.T) {
	robotsTxt := `This is not valid robots.txt
Random text
No proper directives
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
	checker := robots.NewChecker(client, "TestBot")

	ctx := context.Background()
	if err := checker.Fetch(ctx, server.URL); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Malformed robots.txt should allow all
	if !checker.IsAllowed(server.URL + "/any/path") {
		t.Error("expected all URLs to be allowed with malformed robots.txt")
	}
}

// TestRobotsDifferentDomain tests that different domain URLs are always allowed.
func TestRobotsDifferentDomain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/robots.txt" {
			w.Header().Set("Content-Type", "text/plain")
			w.Write([]byte("User-agent: *\nDisallow: /"))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := &http.Client{Timeout: 5 * time.Second}
	checker := robots.NewChecker(client, "TestBot")

	ctx := context.Background()
	if err := checker.Fetch(ctx, server.URL); err != nil {
		t.Fatalf("failed to fetch robots.txt: %v", err)
	}

	// URL on different domain should be allowed
	if !checker.IsAllowed("http://different-domain.com/page") {
		t.Error("expected different domain URL to be allowed")
	}
}
