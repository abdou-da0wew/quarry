// Package testutils provides a reusable mock HTTP server for crawler testing.
package testutils

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
)

// MockServer is a configurable HTTP server for testing.
type MockServer struct {
	*httptest.Server
	mu              sync.RWMutex
	responses       map[string]*MockResponse
	requestCount    int64
	requestLog      []string
	requestLogMu    sync.Mutex
	robotsContent   string
	delay           func(path string) int // milliseconds delay per path
	errorRate       float64               // 0.0 to 1.0
	statusOverrides map[string]int
}

// MockResponse represents a mock HTTP response.
type MockResponse struct {
	StatusCode int
	Headers    map[string]string
	Body       string
}

// NewMockServer creates a new mock HTTP server.
func NewMockServer() *MockServer {
	ms := &MockServer{
		responses:       make(map[string]*MockResponse),
		requestLog:      make([]string, 0),
		statusOverrides: make(map[string]int),
		robotsContent:   "User-agent: *\nAllow: /",
	}

	ms.Server = httptest.NewServer(http.HandlerFunc(ms.handler))
	return ms
}

// handler is the main HTTP handler for the mock server.
func (ms *MockServer) handler(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(&ms.requestCount, 1)

	// Log the request
	ms.requestLogMu.Lock()
	ms.requestLog = append(ms.requestLog, r.URL.Path)
	ms.requestLogMu.Unlock()

	path := r.URL.Path

	// Apply delay if configured
	if ms.delay != nil {
		if d := ms.delay(path); d > 0 {
			// Simulate delay (non-blocking in real scenario)
			// For testing, we just track it
		}
	}

	// Handle robots.txt
	if path == "/robots.txt" {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(ms.robotsContent))
		return
	}

	// Check for status override
	if status, ok := ms.statusOverrides[path]; ok {
		w.WriteHeader(status)
		return
	}

	// Look up response
	ms.mu.RLock()
	resp, ok := ms.responses[path]
	ms.mu.RUnlock()

	if !ok {
		// Return 404 for unknown paths
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, "<html><body><h1>404 Not Found</h1><p>Path: %s</p></body></html>", path)
		return
	}

	// Set headers
	for k, v := range resp.Headers {
		w.Header().Set(k, v)
	}

	// Write status and body
	w.WriteHeader(resp.StatusCode)
	w.Write([]byte(resp.Body))
}

// SetResponse sets a mock response for a path.
func (ms *MockServer) SetResponse(path string, resp *MockResponse) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.responses[path] = resp
}

// SetHTML sets an HTML response for a path.
func (ms *MockServer) SetHTML(path string, body string) {
	ms.SetResponse(path, &MockResponse{
		StatusCode: http.StatusOK,
		Headers: map[string]string{
			"Content-Type": "text/html; charset=utf-8",
		},
		Body: body,
	})
}

// SetRedirect sets a redirect response.
func (ms *MockServer) SetRedirect(path string, target string, code int) {
	ms.SetResponse(path, &MockResponse{
		StatusCode: code,
		Headers: map[string]string{
			"Location": target,
		},
		Body: "",
	})
}

// SetError sets an error status code for a path.
func (ms *MockServer) SetError(path string, statusCode int) {
	ms.SetResponse(path, &MockResponse{
		StatusCode: statusCode,
		Headers:    map[string]string{},
		Body:       fmt.Sprintf("<html><body><h1>%d Error</h1></body></html>", statusCode),
	})
}

// SetRobots sets the robots.txt content.
func (ms *MockServer) SetRobots(content string) {
	ms.robotsContent = content
}

// SetStatusOverride sets a status code override for a path (useful for timeouts, etc).
func (ms *MockServer) SetStatusOverride(path string, status int) {
	ms.statusOverrides[path] = status
}

// GetRequestCount returns the total number of requests received.
func (ms *MockServer) GetRequestCount() int64 {
	return atomic.LoadInt64(&ms.requestCount)
}

// GetRequestLog returns all requested paths in order.
func (ms *MockServer) GetRequestLog() []string {
	ms.requestLogMu.Lock()
	defer ms.requestLogMu.Unlock()
	result := make([]string, len(ms.requestLog))
	copy(result, ms.requestLog)
	return result
}

// WasRequested checks if a path was requested.
func (ms *MockServer) WasRequested(path string) bool {
	ms.requestLogMu.Lock()
	defer ms.requestLogMu.Unlock()
	for _, p := range ms.requestLog {
		if p == path {
			return true
		}
	}
	return false
}

// RequestCountFor returns how many times a specific path was requested.
func (ms *MockServer) RequestCountFor(path string) int {
	ms.requestLogMu.Lock()
	defer ms.requestLogMu.Unlock()
	count := 0
	for _, p := range ms.requestLog {
		if p == path {
			count++
		}
	}
	return count
}

// Reset clears all state for a fresh test.
func (ms *MockServer) Reset() {
	ms.mu.Lock()
	ms.responses = make(map[string]*MockResponse)
	ms.mu.Unlock()

	ms.requestLogMu.Lock()
	ms.requestLog = make([]string, 0)
	ms.requestLogMu.Unlock()

	atomic.StoreInt64(&ms.requestCount, 0)
	ms.robotsContent = "User-agent: *\nAllow: /"
	ms.statusOverrides = make(map[string]int)
}

// URL returns the base URL of the mock server.
func (ms *MockServer) URL() string {
	return ms.Server.URL
}

// Close shuts down the mock server.
func (ms *MockServer) Close() {
	ms.Server.Close()
}

// CreateLinkedSite creates a site with multiple linked pages.
// Returns a map of paths to their full URLs.
func (ms *MockServer) CreateLinkedSite(pages map[string]string, links map[string][]string) map[string]string {
	urls := make(map[string]string)
	baseURL := ms.URL()

	for path, content := range pages {
		// Build HTML with links
		html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><title>Page %s</title></head>
<body>
<nav>
%s
</nav>
<main>%s</main>
</body>
</html>`, path, buildNavLinks(links[path], baseURL), content)

		ms.SetHTML(path, html)
		urls[path] = baseURL + path
	}

	return urls
}

func buildNavLinks(paths []string, baseURL string) string {
	var links string
	for _, p := range paths {
		links += fmt.Sprintf(`<a href="%s">%s</a>`, p, p)
	}
	return links
}

// CreateCircularRedirect creates a circular redirect chain A -> B -> A.
func (ms *MockServer) CreateCircularRedirect() {
	baseURL := ms.URL()
	ms.SetRedirect("/a", baseURL+"/b", http.StatusFound)
	ms.SetRedirect("/b", baseURL+"/a", http.StatusFound)
}

// CreateLongRedirectChain creates a redirect chain of specified length.
func (ms *MockServer) CreateLongRedirectChain(length int) string {
	baseURL := ms.URL()
	for i := 0; i < length-1; i++ {
		from := fmt.Sprintf("/redirect/%d", i)
		to := fmt.Sprintf("%s/redirect/%d", baseURL, i+1)
		ms.SetRedirect(from, to, http.StatusFound)
	}
	// Final destination
	finalPath := fmt.Sprintf("/redirect/%d", length-1)
	ms.SetHTML(finalPath, "<html><body><h1>Final Page</h1></body></html>")
	return baseURL + "/redirect/0"
}

// CreateLargePage creates a page with many outbound links (stress test).
func (ms *MockServer) CreateLargePage(path string, linkCount int) {
	var links string
	for i := 0; i < linkCount; i++ {
		links += fmt.Sprintf(`<a href="/page/%d">Page %d</a>`, i, i)
	}

	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><title>Large Page with %d Links</title></head>
<body>
<h1>Large Page</h1>
<nav>%s</nav>
<p>This page has %d outbound links for testing.</p>
</body>
</html>`, linkCount, links, linkCount)

	ms.SetHTML(path, html)
}

// CreateNonUTF8Page creates a page with invalid UTF-8 sequences.
func (ms *MockServer) CreateNonUTF8Page(path string) {
	// Create response with invalid UTF-8 bytes
	ms.SetResponse(path, &MockResponse{
		StatusCode: http.StatusOK,
		Headers: map[string]string{
			"Content-Type": "text/html; charset=utf-8",
		},
		// Invalid UTF-8: 0xFF is never valid in UTF-8
		Body: "<html><body><h1>Non-UTF8 Page</h1><p>Invalid: \xff\xfe bytes</p></body></html>",
	})
}

// CreateRateLimitedEndpoint creates an endpoint that returns 429 initially, then 200.
func (ms *MockServer) CreateRateLimitedEndpoint(path string, failuresBeforeSuccess int) {
	var attempts int64
	ms.SetResponse(path, &MockResponse{
		StatusCode: http.StatusTooManyRequests,
		Headers: map[string]string{
			"Content-Type": "text/plain",
			"Retry-After":  "1",
		},
		Body: "Rate limited",
	})

	// Override the handler for this path to track attempts
	originalHandler := ms.Server.Config.Handler
	ms.Server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == path {
			attempt := atomic.AddInt64(&attempts, 1)
			if attempt <= int64(failuresBeforeSuccess) {
				w.Header().Set("Retry-After", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				w.Write([]byte("Rate limited"))
				return
			}
		}
		originalHandler.ServeHTTP(w, r)
	})
}

// TimeoutHandler creates an endpoint that never responds.
type TimeoutHandler struct{}

func (h *TimeoutHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Block forever - client should timeout
	select {}
}

// CreateEmptySite creates a site with a single page and no outbound links.
func (ms *MockServer) CreateEmptySite(entryPath string) string {
	ms.SetHTML(entryPath, `<!DOCTYPE html>
<html>
<head><title>Empty Site</title></head>
<body>
<h1>Empty Site</h1>
<p>This page has no outbound links.</p>
</body>
</html>`)
	return ms.URL() + entryPath
}
