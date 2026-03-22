package tests

import (
	"testing"

	"crawler/internal/extractor"
)

// TestExtractTitle tests title extraction from various HTML structures.
func TestExtractTitle(t *testing.T) {
	tests := []struct {
		name     string
		html     string
		expected string
	}{
		{
			name:     "simple title",
			html:     `<html><head><title>Test Title</title></head><body></body></html>`,
			expected: "Test Title",
		},
		{
			name:     "title with whitespace",
			html:     `<html><head><title>  Test Title  </title></head><body></body></html>`,
			expected: "Test Title",
		},
		{
			name:     "no title tag",
			html:     `<html><body></body></html>`,
			expected: "",
		},
		{
			name:     "empty title",
			html:     `<html><head><title></title></head><body></body></html>`,
			expected: "",
		},
	}

	ext := extractor.New("example.com")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			page, err := ext.Extract("https://example.com/page", []byte(tt.html))
			if err != nil {
				t.Fatalf("extraction failed: %v", err)
			}
			if page.Title != tt.expected {
				t.Errorf("expected title %q, got %q", tt.expected, page.Title)
			}
		})
	}
}

// TestExtractBody tests body text extraction.
func TestExtractBody(t *testing.T) {
	tests := []struct {
		name              string
		html              string
		shouldContain     []string
		shouldNotContain  []string
	}{
		{
			name: "extracts main content",
			html: `<html><body><main>Main content here</main></body></html>`,
			shouldContain:    []string{"Main content here"},
			shouldNotContain: []string{},
		},
		{
			name: "removes script tags",
			html: `<html><body><script>alert('test');</script><main>Content</main></body></html>`,
			shouldContain:    []string{"Content"},
			shouldNotContain: []string{"alert", "script"},
		},
		{
			name: "removes style tags",
			html: `<html><body><style>body { color: red; }</style><main>Content</main></body></html>`,
			shouldContain:    []string{"Content"},
			shouldNotContain: []string{"color", "red"},
		},
		{
			name: "removes nav elements",
			html: `<html><body><nav>Navigation Menu</nav><main>Main content</main></body></html>`,
			shouldContain:    []string{"Main content"},
			shouldNotContain: []string{"Navigation Menu"},
		},
		{
			name: "fallback to body",
			html: `<html><body>Plain body content</body></html>`,
			shouldContain:    []string{"Plain body content"},
			shouldNotContain: []string{},
		},
	}

	ext := extractor.New("example.com")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			page, err := ext.Extract("https://example.com/page", []byte(tt.html))
			if err != nil {
				t.Fatalf("extraction failed: %v", err)
			}

			for _, s := range tt.shouldContain {
				if !containsSubstring(page.Body, s) {
					t.Errorf("body should contain %q", s)
				}
			}

			for _, s := range tt.shouldNotContain {
				if containsSubstring(page.Body, s) {
					t.Errorf("body should not contain %q", s)
				}
			}
		})
	}
}

// TestExtractLinks tests link extraction.
func TestExtractLinks(t *testing.T) {
	tests := []struct {
		name           string
		html           string
		baseURL        string
		domain         string
		expectedLinks  []string
		unexpectedLinks []string
	}{
		{
			name: "extracts internal links",
			html: `<html><body><a href="/page1">Link 1</a><a href="/page2">Link 2</a></body></html>`,
			baseURL: "https://example.com/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/page1",
				"https://example.com/page2",
			},
			unexpectedLinks: []string{},
		},
		{
			name: "ignores external links",
			html: `<html><body><a href="/internal">Internal</a><a href="https://external.com/page">External</a></body></html>`,
			baseURL: "https://example.com/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/internal",
			},
			unexpectedLinks: []string{
				"https://external.com/page",
			},
		},
		{
			name: "resolves relative links",
			html: `<html><body><a href="page1">Relative</a><a href="../page2">Parent relative</a></body></html>`,
			baseURL: "https://example.com/subdir/current/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/subdir/current/page1",
				"https://example.com/subdir/page2",
			},
			unexpectedLinks: []string{},
		},
		{
			name: "ignores non-http links",
			html: `<html><body><a href="mailto:test@example.com">Email</a><a href="javascript:void(0)">JS</a><a href="/page">Page</a></body></html>`,
			baseURL: "https://example.com/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/page",
			},
			unexpectedLinks: []string{
				"mailto:test@example.com",
				"javascript:void(0)",
			},
		},
		{
			name: "removes fragment",
			html: `<html><body><a href="/page#section">With fragment</a></body></html>`,
			baseURL: "https://example.com/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/page",
			},
			unexpectedLinks: []string{},
		},
		{
			name: "deduplicates links",
			html: `<html><body><a href="/page">Link 1</a><a href="/page">Link 2</a><a href="/page">Link 3</a></body></html>`,
			baseURL: "https://example.com/",
			domain: "example.com",
			expectedLinks: []string{
				"https://example.com/page",
			},
			unexpectedLinks: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ext := extractor.New(tt.domain)
			page, err := ext.Extract(tt.baseURL, []byte(tt.html))
			if err != nil {
				t.Fatalf("extraction failed: %v", err)
			}

			// Check expected links
			for _, expected := range tt.expectedLinks {
				found := false
				for _, link := range page.Links {
					if link == expected {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected link %q not found in %v", expected, page.Links)
				}
			}

			// Check unexpected links
			for _, unexpected := range tt.unexpectedLinks {
				for _, link := range page.Links {
					if link == unexpected {
						t.Errorf("unexpected link %q found", unexpected)
					}
				}
			}
		})
	}
}

// TestIsHTML tests content type detection.
func TestIsHTML(t *testing.T) {
	tests := []struct {
		contentType string
		expected    bool
	}{
		{"text/html", true},
		{"text/html; charset=utf-8", true},
		{"application/xhtml+xml", true},
		{"application/json", false},
		{"text/plain", false},
		{"image/png", false},
		{"", false},
	}

	for _, tt := range tests {
		result := extractor.IsHTML(tt.contentType)
		if result != tt.expected {
			t.Errorf("IsHTML(%q) = %v, expected %v", tt.contentType, result, tt.expected)
		}
	}
}
