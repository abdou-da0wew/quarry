// Package index provides inverted index building and page writing.
package index

import (
        "encoding/json"
        "fmt"
        "io/fs"
        "os"
        "path/filepath"
        "regexp"
        "strings"
        "sync"
        "time"
)

// PageDocument represents the output format for a crawled page.
type PageDocument struct {
        URL       string   `json:"url"`
        Title     string   `json:"title"`
        Body      string   `json:"body"`
        Links     []string `json:"links"`
        CrawledAt string   `json:"crawled_at"`
}

// InvertedIndex maps terms to URLs containing those terms.
type InvertedIndex struct {
        Terms map[string][]string `json:"terms"`
}

// Builder handles page writing and inverted index construction.
type Builder struct {
        outputDir string
        mu        sync.Mutex
        termIndex map[string]map[string]struct{} // term -> set of URLs
        pageCount int
}

// NewBuilder creates a new index builder.
func NewBuilder(outputDir string) *Builder {
        return &Builder{
                outputDir: outputDir,
                termIndex: make(map[string]map[string]struct{}),
        }
}

// WritePage writes a page document to disk and updates the term index.
// This is thread-safe and streams directly to disk without buffering in memory.
func (b *Builder) WritePage(doc *PageDocument) error {
        b.mu.Lock()
        defer b.mu.Unlock()

        // Create pages subdirectory if needed
        pagesDir := filepath.Join(b.outputDir, "pages")
        if err := os.MkdirAll(pagesDir, 0755); err != nil {
                return fmt.Errorf("failed to create pages directory: %w", err)
        }

        // Generate filename from URL hash
        filename := urlToFilename(doc.URL)
        filePath := filepath.Join(pagesDir, filename)

        // Write to temp file, then rename for atomicity
        tempPath := filePath + ".tmp"
        file, err := os.Create(tempPath)
        if err != nil {
                return fmt.Errorf("failed to create temp file: %w", err)
        }

        encoder := json.NewEncoder(file)
        encoder.SetEscapeHTML(false)
        if err := encoder.Encode(doc); err != nil {
                file.Close()
                os.Remove(tempPath)
                return fmt.Errorf("failed to encode document: %w", err)
        }

        if err := file.Close(); err != nil {
                os.Remove(tempPath)
                return fmt.Errorf("failed to close file: %w", err)
        }

        // Atomic rename
        if err := os.Rename(tempPath, filePath); err != nil {
                os.Remove(tempPath)
                return fmt.Errorf("failed to rename file: %w", err)
        }

        // Update term index
        b.indexTerms(doc.Body, doc.URL)
        b.pageCount++

        return nil
}

// indexTerms extracts and indexes terms from the body text.
func (b *Builder) indexTerms(body string, url string) {
        terms := tokenize(body)
        for _, term := range terms {
                if _, ok := b.termIndex[term]; !ok {
                        b.termIndex[term] = make(map[string]struct{})
                }
                b.termIndex[term][url] = struct{}{}
        }
}

// tokenize extracts lowercase alphanumeric tokens from text.
func tokenize(text string) []string {
        // Convert to lowercase
        text = strings.ToLower(text)

        // Extract alphanumeric tokens
        re := regexp.MustCompile(`[a-z0-9]+`)
        tokens := re.FindAllString(text, -1)

        // Filter short tokens and stopwords
        filtered := make([]string, 0, len(tokens))
        for _, token := range tokens {
                if len(token) >= 2 && !isStopword(token) {
                        filtered = append(filtered, token)
                }
        }

        return filtered
}

// isStopword checks if a token is a common stopword.
func isStopword(token string) bool {
        stopwords := map[string]bool{
                "a": true, "an": true, "the": true, "and": true, "or": true,
                "but": true, "in": true, "on": true, "at": true, "to": true,
                "for": true, "of": true, "with": true, "by": true, "from": true,
                "as": true, "is": true, "was": true, "are": true, "were": true,
                "been": true, "be": true, "have": true, "has": true, "had": true,
                "do": true, "does": true, "did": true, "will": true, "would": true,
                "could": true, "should": true, "may": true, "might": true, "must": true,
                "shall": true, "can": true, "need": true, "dare": true, "ought": true,
                "used": true, "it": true, "its": true, "this": true, "that": true,
                "these": true, "those": true, "i": true, "you": true, "he": true,
                "she": true, "we": true, "they": true, "what": true, "which": true,
                "who": true, "whom": true, "if": true, "then": true, "else": true,
                "when": true, "where": true, "why": true, "how": true, "all": true,
                "each": true, "every": true, "both": true, "few": true, "more": true,
                "most": true, "other": true, "some": true, "such": true, "no": true,
                "nor": true, "not": true, "only": true, "own": true, "same": true,
                "so": true, "than": true, "too": true, "very": true, "just": true,
                "also": true, "now": true, "here": true, "there": true, "about": true,
                "into": true, "through": true, "during": true, "before": true, "after": true,
                "above": true, "below": true, "between": true, "under": true, "again": true,
                "further": true, "once": true, "any": true, "your": true, "our": true,
        }
        return stopwords[token]
}

// urlToFilename creates a safe filename from a URL.
func urlToFilename(url string) string {
        // Replace problematic characters
        re := regexp.MustCompile(`[^a-zA-Z0-9_-]`)
        filename := re.ReplaceAllString(url, "_")

        // Limit length
        if len(filename) > 200 {
                filename = filename[:200]
        }

        return filename + ".json"
}

// BuildInvertedIndex writes the inverted index to disk.
func (b *Builder) BuildInvertedIndex() error {
        b.mu.Lock()
        defer b.mu.Unlock()

        // Create flat map for JSON output
        index := make(map[string][]string)

        // Convert sets to sorted slices
        for term, urlSet := range b.termIndex {
                urls := make([]string, 0, len(urlSet))
                for url := range urlSet {
                        urls = append(urls, url)
                }
                // Sort for consistent output
                sortStrings(urls)
                index[term] = urls
        }

        filePath := filepath.Join(b.outputDir, "inverted-index.json")
        tempPath := filePath + ".tmp"

        file, err := os.Create(tempPath)
        if err != nil {
                return fmt.Errorf("failed to create inverted index file: %w", err)
        }

        encoder := json.NewEncoder(file)
        encoder.SetEscapeHTML(false)
        encoder.SetIndent("", "  ")
        if err := encoder.Encode(index); err != nil {
                file.Close()
                os.Remove(tempPath)
                return fmt.Errorf("failed to encode inverted index: %w", err)
        }

        if err := file.Close(); err != nil {
                os.Remove(tempPath)
                return fmt.Errorf("failed to close file: %w", err)
        }

        if err := os.Rename(tempPath, filePath); err != nil {
                os.Remove(tempPath)
                return fmt.Errorf("failed to rename file: %w", err)
        }

        return nil
}

// PageCount returns the number of pages written.
func (b *Builder) PageCount() int {
        b.mu.Lock()
        defer b.mu.Unlock()
        return b.pageCount
}

// LoadPages reads all page documents from the output directory.
// Used for rebuilding the inverted index.
func LoadPages(outputDir string) ([]*PageDocument, error) {
        pagesDir := filepath.Join(outputDir, "pages")

        entries, err := os.ReadDir(pagesDir)
        if err != nil {
                return nil, fmt.Errorf("failed to read pages directory: %w", err)
        }

        pages := make([]*PageDocument, 0, len(entries))

        for _, entry := range entries {
                if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
                        continue
                }

                filePath := filepath.Join(pagesDir, entry.Name())
                data, err := os.ReadFile(filePath)
                if err != nil {
                        continue
                }

                var doc PageDocument
                if err := json.Unmarshal(data, &doc); err != nil {
                        continue
                }

                pages = append(pages, &doc)
        }

        return pages, nil
}

// RebuildInvertedIndex recreates the inverted index from existing page files.
func RebuildInvertedIndex(outputDir string) error {
        pages, err := LoadPages(outputDir)
        if err != nil {
                return err
        }

        builder := NewBuilder(outputDir)
        for _, page := range pages {
                builder.indexTerms(page.Body, page.URL)
                builder.pageCount++
        }

        return builder.BuildInvertedIndex()
}

// simple string sort without importing sort package
func sortStrings(s []string) {
        for i := 0; i < len(s)-1; i++ {
                for j := i + 1; j < len(s); j++ {
                        if s[i] > s[j] {
                                s[i], s[j] = s[j], s[i]
                        }
                }
        }
}

// WriteManifest writes a crawl manifest with statistics.
func WriteManifest(outputDir string, stats *CrawlStats) error {
        manifest := map[string]interface{}{
                "version":        "1.0",
                "crawl_id":       stats.CrawlID,
                "started_at":     stats.StartedAt.Format(time.RFC3339),
                "completed_at":   stats.CompletedAt.Format(time.RFC3339),
                "pages_crawled":  stats.PagesCrawled,
                "pages_failed":   stats.PagesFailed,
                "bytes_downloaded": stats.BytesDownloaded,
        }

        filePath := filepath.Join(outputDir, "manifest.json")
        tempPath := filePath + ".tmp"

        file, err := os.Create(tempPath)
        if err != nil {
                return fmt.Errorf("failed to create manifest file: %w", err)
        }

        encoder := json.NewEncoder(file)
        encoder.SetEscapeHTML(false)
        encoder.SetIndent("", "  ")
        if err := encoder.Encode(manifest); err != nil {
                file.Close()
                os.Remove(tempPath)
                return fmt.Errorf("failed to encode manifest: %w", err)
        }

        if err := file.Close(); err != nil {
                os.Remove(tempPath)
                return fmt.Errorf("failed to close file: %w", err)
        }

        return os.Rename(tempPath, filePath)
}

// CrawlStats holds statistics for the manifest.
type CrawlStats struct {
        CrawlID         string
        StartedAt       time.Time
        CompletedAt     time.Time
        PagesCrawled    int
        PagesFailed     int
        BytesDownloaded int64
}

// CleanupTempFiles removes any leftover .tmp files from interrupted runs.
func CleanupTempFiles(outputDir string) error {
        pagesDir := filepath.Join(outputDir, "pages")

        // Create directory if it doesn't exist
        if err := os.MkdirAll(pagesDir, 0755); err != nil {
                return err
        }

        return filepath.WalkDir(outputDir, func(path string, d fs.DirEntry, err error) error {
                if err != nil {
                        return err
                }
                if !d.IsDir() && strings.HasSuffix(path, ".tmp") {
                        os.Remove(path)
                }
                return nil
        })
}
