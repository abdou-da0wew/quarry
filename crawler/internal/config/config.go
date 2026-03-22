// Package config provides configuration management for the crawler.
package config

import (
	"flag"
	"fmt"
	"net/url"
	"os"
	"time"
)

// Config holds all configurable parameters for the crawler.
type Config struct {
	// Entry is the starting URL for the crawl.
	Entry string
	// Workers is the number of concurrent crawl workers.
	Workers int
	// Timeout is the HTTP client timeout.
	Timeout time.Duration
	// OutputDir is the directory for output files.
	OutputDir string
	// MaxPages limits the total pages to crawl (0 = unlimited).
	MaxPages int
	// DelayMS is the minimum milliseconds between requests to the same domain.
	DelayMS int
	// UserAgent is the HTTP User-Agent header.
	UserAgent string
	// MaxRetries is the maximum number of retry attempts.
	MaxRetries int
	// MaxRedirects is the maximum redirect chain length.
	MaxRedirects int
	// ParsedEntry is the parsed entry URL.
	ParsedEntry *url.URL
}

// Load parses command-line flags and validates the configuration.
func Load() (*Config, error) {
	cfg := &Config{}

	flag.StringVar(&cfg.Entry, "entry", "", "Entry point URL (required)")
	flag.IntVar(&cfg.Workers, "workers", 10, "Number of concurrent workers")
	flag.DurationVar(&cfg.Timeout, "timeout", 10*time.Second, "HTTP timeout")
	flag.StringVar(&cfg.OutputDir, "output-dir", "./output", "Output directory for crawled pages")
	flag.IntVar(&cfg.MaxPages, "max-pages", 0, "Maximum pages to crawl (0 = unlimited)")
	flag.IntVar(&cfg.DelayMS, "delay-ms", 500, "Minimum milliseconds between requests")
	flag.StringVar(&cfg.UserAgent, "user-agent", "MinecraftLinuxCrawler/1.0 (+https://github.com/minecraft-linux)", "User-Agent header")
	flag.IntVar(&cfg.MaxRetries, "max-retries", 3, "Maximum retry attempts for transient failures")
	flag.IntVar(&cfg.MaxRedirects, "max-redirects", 10, "Maximum redirect chain length")

	flag.Parse()

	if cfg.Entry == "" {
		return nil, fmt.Errorf("--entry flag is required")
	}

	if cfg.Workers < 1 {
		return nil, fmt.Errorf("--workers must be at least 1")
	}

	if cfg.Timeout < time.Second {
		return nil, fmt.Errorf("--timeout must be at least 1s")
	}

	if cfg.DelayMS < 0 {
		return nil, fmt.Errorf("--delay-ms cannot be negative")
	}

	if cfg.MaxRetries < 0 {
		return nil, fmt.Errorf("--max-retries cannot be negative")
	}

	if cfg.MaxRedirects < 1 {
		return nil, fmt.Errorf("--max-redirects must be at least 1")
	}

	parsed, err := url.Parse(cfg.Entry)
	if err != nil {
		return nil, fmt.Errorf("invalid entry URL: %w", err)
	}

	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("entry URL must have http or https scheme")
	}

	if parsed.Host == "" {
		return nil, fmt.Errorf("entry URL must have a host")
	}

	cfg.ParsedEntry = parsed

	// Ensure output directory exists
	if err := os.MkdirAll(cfg.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}

	return cfg, nil
}

// Domain returns the domain (host) of the entry URL.
func (c *Config) Domain() string {
	return c.ParsedEntry.Host
}

// RobotsURL returns the URL for the robots.txt file.
func (c *Config) RobotsURL() string {
	return fmt.Sprintf("%s://%s/robots.txt", c.ParsedEntry.Scheme, c.ParsedEntry.Host)
}
