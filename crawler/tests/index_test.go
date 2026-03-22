package tests

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"crawler/internal/index"
)

// TestWritePage tests writing page documents to disk.
func TestWritePage(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	doc := &index.PageDocument{
		URL:       "https://example.com/test-page",
		Title:     "Test Page Title",
		Body:      "This is the body content of the test page.",
		Links:     []string{"https://example.com/link1", "https://example.com/link2"},
		CrawledAt: time.Now().Format(time.RFC3339),
	}

	err := builder.WritePage(doc)
	if err != nil {
		t.Fatalf("failed to write page: %v", err)
	}

	// Verify file was created
	pagesDir := filepath.Join(outputDir, "pages")
	entries, err := os.ReadDir(pagesDir)
	if err != nil {
		t.Fatalf("failed to read pages directory: %v", err)
	}

	if len(entries) != 1 {
		t.Fatalf("expected 1 file, got %d", len(entries))
	}

	// Verify content
	data, err := os.ReadFile(filepath.Join(pagesDir, entries[0].Name()))
	if err != nil {
		t.Fatalf("failed to read page file: %v", err)
	}

	var loaded index.PageDocument
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("failed to parse page JSON: %v", err)
	}

	if loaded.URL != doc.URL {
		t.Errorf("expected URL %q, got %q", doc.URL, loaded.URL)
	}
	if loaded.Title != doc.Title {
		t.Errorf("expected title %q, got %q", doc.Title, loaded.Title)
	}
	if loaded.Body != doc.Body {
		t.Errorf("expected body %q, got %q", doc.Body, loaded.Body)
	}
}

// TestWritePageAtomic verifies that writes are atomic (temp file renamed).
func TestWritePageAtomic(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	doc := &index.PageDocument{
		URL:       "https://example.com/test",
		Title:     "Test",
		Body:      "Content",
		CrawledAt: time.Now().Format(time.RFC3339),
	}

	err := builder.WritePage(doc)
	if err != nil {
		t.Fatalf("failed to write page: %v", err)
	}

	// No .tmp files should exist
	err = filepath.WalkDir(outputDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if filepath.Ext(path) == ".tmp" {
			t.Errorf("temp file should not exist: %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk directory: %v", err)
	}
}

// TestBuildInvertedIndex tests inverted index construction.
func TestBuildInvertedIndex(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	docs := []*index.PageDocument{
		{
			URL:       "https://example.com/page1",
			Title:     "Go Programming",
			Body:      "Go is a programming language. Go is efficient and fast.",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
		{
			URL:       "https://example.com/page2",
			Title:     "Python Programming",
			Body:      "Python is another programming language. Python is easy to learn.",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
		{
			URL:       "https://example.com/page3",
			Title:     "Rust Programming",
			Body:      "Rust is a systems programming language. Rust is safe and fast.",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
	}

	for _, doc := range docs {
		if err := builder.WritePage(doc); err != nil {
			t.Fatalf("failed to write page: %v", err)
		}
	}

	if err := builder.BuildInvertedIndex(); err != nil {
		t.Fatalf("failed to build inverted index: %v", err)
	}

	// Load and verify inverted index
	data, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
	if err != nil {
		t.Fatalf("failed to read inverted index: %v", err)
	}

	var idx map[string][]string
	if err := json.Unmarshal(data, &idx); err != nil {
		t.Fatalf("failed to parse inverted index: %v", err)
	}

	// "programming" should appear in all 3 pages
	if urls, ok := idx["programming"]; !ok {
		t.Error("expected 'programming' in index")
	} else if len(urls) != 3 {
		t.Errorf("expected 'programming' in 3 pages, got %d", len(urls))
	}

	// "go" should appear only in page1
	if urls, ok := idx["go"]; !ok {
		t.Error("expected 'go' in index")
	} else if len(urls) != 1 {
		t.Errorf("expected 'go' in 1 page, got %d", len(urls))
	}

	// "python" should appear only in page2
	if urls, ok := idx["python"]; !ok {
		t.Error("expected 'python' in index")
	} else if len(urls) != 1 {
		t.Errorf("expected 'python' in 1 page, got %d", len(urls))
	}
}

// TestInvertedIndexStopwords verifies stopwords are filtered.
func TestInvertedIndexStopwords(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	doc := &index.PageDocument{
		URL:       "https://example.com/test",
		Title:     "Test",
		Body:      "The quick brown fox jumps over the lazy dog. A an the is are was were.",
		CrawledAt: time.Now().Format(time.RFC3339),
	}

	if err := builder.WritePage(doc); err != nil {
		t.Fatalf("failed to write page: %v", err)
	}

	if err := builder.BuildInvertedIndex(); err != nil {
		t.Fatalf("failed to build index: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
	if err != nil {
		t.Fatalf("failed to read index: %v", err)
	}

	var idx map[string][]string
	if err := json.Unmarshal(data, &idx); err != nil {
		t.Fatalf("failed to parse index: %v", err)
	}

	// Stopwords should not be indexed
	stopwords := []string{"the", "a", "an", "is", "are", "was", "were"}
	for _, sw := range stopwords {
		if _, ok := idx[sw]; ok {
			t.Errorf("stopword %q should not be in index", sw)
		}
	}

	// Content words should be indexed
	contentWords := []string{"quick", "brown", "fox", "jumps", "lazy", "dog"}
	for _, word := range contentWords {
		if _, ok := idx[word]; !ok {
			t.Errorf("content word %q should be in index", word)
		}
	}
}

// TestPageCount tests page counting.
func TestPageCount(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	if count := builder.PageCount(); count != 0 {
		t.Errorf("expected count 0, got %d", count)
	}

	for i := 0; i < 5; i++ {
		doc := &index.PageDocument{
			URL:       "https://example.com/page" + string(rune('0'+i)),
			Title:     "Test",
			Body:      "Content",
			CrawledAt: time.Now().Format(time.RFC3339),
		}
		if err := builder.WritePage(doc); err != nil {
			t.Fatalf("failed to write page: %v", err)
		}
	}

	if count := builder.PageCount(); count != 5 {
		t.Errorf("expected count 5, got %d", count)
	}
}

// TestLoadPages tests loading pages from disk.
func TestLoadPages(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	original := []*index.PageDocument{
		{
			URL:       "https://example.com/page1",
			Title:     "Page 1",
			Body:      "Content 1",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
		{
			URL:       "https://example.com/page2",
			Title:     "Page 2",
			Body:      "Content 2",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
	}

	for _, doc := range original {
		if err := builder.WritePage(doc); err != nil {
			t.Fatalf("failed to write page: %v", err)
		}
	}

	loaded, err := index.LoadPages(outputDir)
	if err != nil {
		t.Fatalf("failed to load pages: %v", err)
	}

	if len(loaded) != len(original) {
		t.Errorf("expected %d pages, got %d", len(original), len(loaded))
	}

	// Verify content matches
	loadedMap := make(map[string]*index.PageDocument)
	for _, doc := range loaded {
		loadedMap[doc.URL] = doc
	}

	for _, orig := range original {
		loaded, ok := loadedMap[orig.URL]
		if !ok {
			t.Errorf("page %q not found in loaded pages", orig.URL)
			continue
		}
		if loaded.Title != orig.Title {
			t.Errorf("title mismatch: expected %q, got %q", orig.Title, loaded.Title)
		}
		if loaded.Body != orig.Body {
			t.Errorf("body mismatch: expected %q, got %q", orig.Body, loaded.Body)
		}
	}
}

// TestRebuildInvertedIndex tests rebuilding index from existing pages.
func TestRebuildInvertedIndex(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	docs := []*index.PageDocument{
		{
			URL:       "https://example.com/page1",
			Title:     "Test",
			Body:      "alpha beta gamma",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
		{
			URL:       "https://example.com/page2",
			Title:     "Test",
			Body:      "alpha delta epsilon",
			CrawledAt: time.Now().Format(time.RFC3339),
		},
	}

	for _, doc := range docs {
		if err := builder.WritePage(doc); err != nil {
			t.Fatalf("failed to write page: %v", err)
		}
	}

	// Build initial index
	if err := builder.BuildInvertedIndex(); err != nil {
		t.Fatalf("failed to build index: %v", err)
	}

	// Rebuild from files
	if err := index.RebuildInvertedIndex(outputDir); err != nil {
		t.Fatalf("failed to rebuild index: %v", err)
	}

	// Verify rebuilt index
	data, err := os.ReadFile(filepath.Join(outputDir, "inverted-index.json"))
	if err != nil {
		t.Fatalf("failed to read index: %v", err)
	}

	var idx map[string][]string
	if err := json.Unmarshal(data, &idx); err != nil {
		t.Fatalf("failed to parse index: %v", err)
	}

	// "alpha" should be in both pages
	if urls, ok := idx["alpha"]; !ok || len(urls) != 2 {
		t.Errorf("expected 'alpha' in 2 pages, got %v", urls)
	}
}

// TestWriteManifest tests manifest file generation.
func TestWriteManifest(t *testing.T) {
	outputDir := t.TempDir()

	stats := &index.CrawlStats{
		CrawlID:         "test-crawl-id",
		StartedAt:       time.Now().Add(-1 * time.Hour),
		CompletedAt:     time.Now(),
		PagesCrawled:    100,
		PagesFailed:     5,
		BytesDownloaded: 1024000,
	}

	if err := index.WriteManifest(outputDir, stats); err != nil {
		t.Fatalf("failed to write manifest: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("failed to read manifest: %v", err)
	}

	var manifest map[string]interface{}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("failed to parse manifest: %v", err)
	}

	if manifest["crawl_id"] != stats.CrawlID {
		t.Errorf("expected crawl_id %q, got %q", stats.CrawlID, manifest["crawl_id"])
	}

	// PagesCrawled is a number, check as float64 (JSON default)
	if pagesCrawled, ok := manifest["pages_crawled"].(float64); !ok || int(pagesCrawled) != stats.PagesCrawled {
		t.Errorf("expected pages_crawled %d, got %v", stats.PagesCrawled, manifest["pages_crawled"])
	}
}

// TestCleanupTempFiles tests removal of leftover temp files.
func TestCleanupTempFiles(t *testing.T) {
	outputDir := t.TempDir()
	pagesDir := filepath.Join(outputDir, "pages")
	os.MkdirAll(pagesDir, 0755)

	// Create some temp files
	tempFiles := []string{
		filepath.Join(pagesDir, "page1.json.tmp"),
		filepath.Join(pagesDir, "page2.json.tmp"),
		filepath.Join(outputDir, "index.json.tmp"),
	}

	for _, f := range tempFiles {
		if err := os.WriteFile(f, []byte("temp"), 0644); err != nil {
			t.Fatalf("failed to create temp file: %v", err)
		}
	}

	// Cleanup
	if err := index.CleanupTempFiles(outputDir); err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}

	// Verify temp files are gone
	for _, f := range tempFiles {
		if _, err := os.Stat(f); !os.IsNotExist(err) {
			t.Errorf("temp file should be removed: %s", f)
		}
	}
}

// TestConcurrentWrites tests concurrent page writing.
func TestConcurrentWrites(t *testing.T) {
	outputDir := t.TempDir()
	builder := index.NewBuilder(outputDir)

	const workers = 10
	const pagesPerWorker = 10

	done := make(chan bool, workers)

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			for i := 0; i < pagesPerWorker; i++ {
				doc := &index.PageDocument{
					URL:       "https://example.com/page/" + string(rune('0'+workerID)) + "/" + string(rune('0'+i)),
					Title:     "Test",
					Body:      "Content",
					CrawledAt: time.Now().Format(time.RFC3339),
				}
				if err := builder.WritePage(doc); err != nil {
					t.Errorf("worker %d failed to write page %d: %v", workerID, i, err)
				}
			}
			done <- true
		}(w)
	}

	// Wait for all workers
	for w := 0; w < workers; w++ {
		<-done
	}

	expectedCount := workers * pagesPerWorker
	if count := builder.PageCount(); count != expectedCount {
		t.Errorf("expected %d pages, got %d", expectedCount, count)
	}
}
