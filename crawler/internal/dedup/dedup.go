// Package dedup provides thread-safe URL deduplication using sync.Map.
// Time complexity: O(1) amortized lookup. Space complexity: O(n) where n = pages.
package dedup

import (
	"net/url"
	"strings"
	"sync"
)

// Store provides thread-safe URL deduplication.
// Uses sync.Map for concurrent-safe operations with zero false positives.
// For 50-200 pages, memory overhead is ~32-48 KiB total.
type Store struct {
	// seen holds normalized URLs that have been processed.
	// sync.Map provides O(1) amortized lookup with thread safety.
	seen sync.Map
}

// New creates a new deduplication store.
func New() *Store {
	return &Store{}
}

// NormalizeURL canonicalizes a URL for consistent comparison.
// It removes fragments, trailing slashes, and normalizes the path.
func NormalizeURL(rawURL string) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}

	// Remove fragment
	parsed.Fragment = ""

	// Normalize scheme to lowercase
	parsed.Scheme = strings.ToLower(parsed.Scheme)

	// Normalize host to lowercase
	parsed.Host = strings.ToLower(parsed.Host)

	// Normalize path: remove trailing slash except for root
	if parsed.Path != "/" {
		parsed.Path = strings.TrimSuffix(parsed.Path, "/")
	}

	// Remove default port
	if parsed.Scheme == "http" && strings.HasSuffix(parsed.Host, ":80") {
		parsed.Host = strings.TrimSuffix(parsed.Host, ":80")
	}
	if parsed.Scheme == "https" && strings.HasSuffix(parsed.Host, ":443") {
		parsed.Host = strings.TrimSuffix(parsed.Host, ":443")
	}

	// Remove query parameters that don't affect content
	// (Keep query params for this crawler as they may affect page content)

	return parsed.String(), nil
}

// CheckAndMark attempts to mark a URL as seen.
// Returns true if the URL was newly marked (not seen before).
// Returns false if the URL was already seen.
func (s *Store) CheckAndMark(rawURL string) bool {
	normalized, err := NormalizeURL(rawURL)
	if err != nil {
		return false
	}

	// sync.Map.LoadOrStore is atomic and thread-safe
	_, loaded := s.seen.LoadOrStore(normalized, struct{}{})
	return !loaded
}

// Seen checks if a URL has been seen without marking it.
func (s *Store) Seen(rawURL string) bool {
	normalized, err := NormalizeURL(rawURL)
	if err != nil {
		return false
	}

	_, ok := s.seen.Load(normalized)
	return ok
}

// Count returns the number of unique URLs seen.
func (s *Store) Count() int {
	count := 0
	s.seen.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

// Clear removes all seen URLs from the store.
func (s *Store) Clear() {
	s.seen = sync.Map{}
}
