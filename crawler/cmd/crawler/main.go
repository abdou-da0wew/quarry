// Command crawler is the entry point for the site crawler.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"crawler/internal/config"
	"crawler/internal/crawler"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Configuration error: %v\n", err)
		os.Exit(1)
	}

	// Create context with cancellation for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nReceived interrupt signal, shutting down gracefully...")
		cancel()
	}()

	// Print configuration
	fmt.Printf("Starting crawler...\n")
	fmt.Printf("  Entry: %s\n", cfg.Entry)
	fmt.Printf("  Workers: %d\n", cfg.Workers)
	fmt.Printf("  Output: %s\n", cfg.OutputDir)
	fmt.Printf("  Delay: %dms\n", cfg.DelayMS)
	if cfg.MaxPages > 0 {
		fmt.Printf("  Max pages: %d\n", cfg.MaxPages)
	}
	fmt.Println()

	// Run the crawler
	startTime := time.Now()
	err = crawler.Run(ctx, cfg)
	elapsed := time.Since(startTime)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Crawl error: %v\n", err)
		os.Exit(1)
	}

	// Print summary
	fmt.Printf("\nCrawl completed in %v\n", elapsed.Round(time.Millisecond))

	// Read and print stats from manifest
	manifestPath := cfg.OutputDir + "/manifest.json"
	if data, err := os.ReadFile(manifestPath); err == nil {
		var manifest map[string]interface{}
		if json.Unmarshal(data, &manifest) == nil {
			fmt.Printf("Pages crawled: %v\n", manifest["pages_crawled"])
			fmt.Printf("Pages failed: %v\n", manifest["pages_failed"])
			fmt.Printf("Bytes downloaded: %v\n", manifest["bytes_downloaded"])
		}
	}

	fmt.Printf("\nOutput files:\n")
	fmt.Printf("  Pages: %s/pages/\n", cfg.OutputDir)
	fmt.Printf("  Index: %s/inverted-index.json\n", cfg.OutputDir)
	fmt.Printf("  Manifest: %s/manifest.json\n", cfg.OutputDir)
}
