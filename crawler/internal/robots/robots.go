// Package robots provides robots.txt parsing and URL access checking.
// Implements RFC 9309 compliance for web crawlers.
package robots

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// Rule represents a single allow or disallow rule from robots.txt.
type Rule struct {
	Path     string
	Allow    bool
	Priority int // Longer paths have higher priority
}

// Group represents rules for a specific user-agent.
type Group struct {
	UserAgent string
	Rules     []Rule
	CrawlDelay time.Duration
}

// Checker handles robots.txt parsing and URL access verification.
type Checker struct {
	client    *http.Client
	userAgent string
	mu        sync.RWMutex
	groups    []*Group
	fetched   bool
	baseURL   *url.URL
}

// NewChecker creates a new robots.txt checker.
func NewChecker(client *http.Client, userAgent string) *Checker {
	return &Checker{
		client:    client,
		userAgent: userAgent,
		groups:    make([]*Group, 0),
	}
}

// Fetch retrieves and parses robots.txt from the given base URL.
func (c *Checker) Fetch(ctx context.Context, baseURL string) error {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}

	c.baseURL = parsed
	robotsURL := fmt.Sprintf("%s://%s/robots.txt", parsed.Scheme, parsed.Host)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, robotsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", c.userAgent)

	resp, err := c.client.Do(req)
	if err != nil {
		// If robots.txt is unavailable, allow all by default
		c.mu.Lock()
		c.fetched = true
		c.mu.Unlock()
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		// No robots.txt means everything is allowed
		c.mu.Lock()
		c.fetched = true
		c.mu.Unlock()
		return nil
	}

	if resp.StatusCode != http.StatusOK {
		// On error, be conservative and allow all
		c.mu.Lock()
		c.fetched = true
		c.mu.Unlock()
		return nil
	}

	groups, err := c.parse(resp.Body)
	if err != nil {
		// On parse error, allow all
		c.mu.Lock()
		c.fetched = true
		c.mu.Unlock()
		return nil
	}

	c.mu.Lock()
	c.groups = groups
	c.fetched = true
	c.mu.Unlock()

	return nil
}

// parse reads robots.txt content and extracts rules.
func (c *Checker) parse(r io.Reader) ([]*Group, error) {
	groups := make([]*Group, 0)
	var currentGroup *Group

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Remove inline comments
		if idx := strings.Index(line, "#"); idx != -1 {
			line = strings.TrimSpace(line[:idx])
		}

		// Split into directive and value
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}

		directive := strings.ToLower(strings.TrimSpace(parts[0]))
		value := strings.TrimSpace(parts[1])

		switch directive {
		case "user-agent":
			if currentGroup != nil && len(currentGroup.Rules) > 0 {
				groups = append(groups, currentGroup)
			}
			currentGroup = &Group{
				UserAgent: strings.ToLower(value),
				Rules:     make([]Rule, 0),
			}

		case "allow":
			if currentGroup != nil {
				currentGroup.Rules = append(currentGroup.Rules, Rule{
					Path:     value,
					Allow:    true,
					Priority: len(value),
				})
			}

		case "disallow":
			if currentGroup != nil {
				// Empty disallow means allow all
				if value == "" {
					currentGroup.Rules = append(currentGroup.Rules, Rule{
						Path:     "/",
						Allow:    true,
						Priority: 1,
					})
				} else {
					currentGroup.Rules = append(currentGroup.Rules, Rule{
						Path:     value,
						Allow:    false,
						Priority: len(value),
					})
				}
			}

		case "crawl-delay":
			if currentGroup != nil {
				var delaySeconds float64
				if _, err := fmt.Sscanf(value, "%f", &delaySeconds); err == nil {
					currentGroup.CrawlDelay = time.Duration(delaySeconds * float64(time.Second))
				}
			}
		}
	}

	// Add the last group
	if currentGroup != nil && len(currentGroup.Rules) > 0 {
		groups = append(groups, currentGroup)
	}

	return groups, scanner.Err()
}

// IsAllowed checks if a URL is allowed to be crawled according to robots.txt rules.
func (c *Checker) IsAllowed(targetURL string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.fetched {
		return true
	}

	parsed, err := url.Parse(targetURL)
	if err != nil {
		return false
	}

	// Only check URLs from the same domain
	if c.baseURL != nil && parsed.Host != c.baseURL.Host {
		return true
	}

	path := parsed.Path
	if parsed.RawQuery != "" {
		path = path + "?" + parsed.RawQuery
	}

	// Find matching groups (most specific user-agent first)
	userAgentLower := strings.ToLower(c.userAgent)

	var matchingGroups []*Group
	for _, group := range c.groups {
		if group.UserAgent == "*" || strings.Contains(userAgentLower, group.UserAgent) {
			matchingGroups = append(matchingGroups, group)
		}
	}

	// Check rules in matching groups
	// Default is to allow if no rules match
	allowed := true
	highestPriority := -1

	for _, group := range matchingGroups {
		for _, rule := range group.Rules {
			if matches(path, rule.Path) {
				if rule.Priority > highestPriority {
					highestPriority = rule.Priority
					allowed = rule.Allow
				} else if rule.Priority == highestPriority {
					// Allow takes precedence over disallow at same priority
					if rule.Allow {
						allowed = true
					}
				}
			}
		}
	}

	return allowed
}

// GetCrawlDelay returns the crawl delay specified in robots.txt.
func (c *Checker) GetCrawlDelay() time.Duration {
	c.mu.RLock()
	defer c.mu.RUnlock()

	userAgentLower := strings.ToLower(c.userAgent)

	for _, group := range c.groups {
		if group.UserAgent == "*" || strings.Contains(userAgentLower, group.UserAgent) {
			if group.CrawlDelay > 0 {
				return group.CrawlDelay
			}
		}
	}

	return 0
}

// matches checks if a path matches a robots.txt pattern.
func matches(path, pattern string) bool {
	if pattern == "/" {
		return true
	}

	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(path, prefix)
	}

	if strings.HasSuffix(pattern, "$") {
		exact := strings.TrimSuffix(pattern, "$")
		return path == exact
	}

	return strings.HasPrefix(path, pattern)
}
