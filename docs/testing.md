# Testing Guide

Complete guide for testing the quarry crawler and semantic search engine.

## Testing Philosophy

Quarry follows a rigorous testing philosophy built on the principle that "a test that cannot fail provides no value." Every test must verify meaningful behavior: edge cases that have caused bugs, invariants that must never be violated, and integrations that have broken in production. Tests are not documentation or future-proofing; they are executable specifications that catch regressions. A passing test suite proves the system works; a failing test proves something changed that should not have.

## Test Matrix

| Test File | Component | Type | What It Proves | Run Command |
|-----------|-----------|------|----------------|-------------|
| crawler/tests/robots_test.go | Crawler | Unit | robots.txt parsing and enforcement | `go test ./tests/ -run TestRobots -v` |
| crawler/tests/extractor_test.go | Crawler | Unit | HTML extraction correctness | `go test ./tests/ -run TestExtractor -v` |
| crawler/tests/dedup_test.go | Crawler | Unit | URL deduplication accuracy | `go test ./tests/ -run TestDedup -v` |
| crawler/tests/crawler_test.go | Crawler | Integration | End-to-end crawl with mock server | `go test ./tests/ -run TestCrawler -v` |
| crawler/tests/goleak_test.go | Crawler | Leak | No goroutine leaks | `go test ./tests/ -run TestGoleak -v` |
| semantic-search/tests/embedder_unit_tests.rs | Search | Unit | Tokenization and embedding logic | `cargo test embedder --lib` |
| semantic-search/tests/indexer_unit_tests.rs | Search | Unit | HNSW insertion and search | `cargo test indexer --lib` |
| semantic-search/tests/integration_tests.rs | Search | Integration | Full pipeline with GPU/CPU | `cargo test --test integration_tests` |
| tests/e2e/pipeline_test.sh | System | E2E | Cross-language data contract | `./tests/e2e/pipeline_test.sh` |

## Section A: Go Tests

### Running Tests

```bash
cd crawler

# Run all unit tests
go test ./... -v

# Run with race detector
go test ./... -race -v

# Run specific test
go test ./tests/ -run TestRobotsParsing -v

# Run with coverage
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out -o coverage.html

# Run with goroutine leak detection
go test ./... -tags=goleak -v
```

### Mock Server

The mock HTTP server provides deterministic responses for integration tests:

```go
// tests/mock_server_test.go
type MockServer struct {
    server   *httptest.Server
    mux      *http.ServeMux
    requests []MockRequest
}

type MockRequest struct {
    Path       string
    Response   string
    StatusCode int
    Headers    map[string]string
}

func NewMockServer() *MockServer {
    mux := http.NewServeMux()
    server := httptest.NewServer(mux)
    return &MockServer{server: server, mux: mux}
}

func (m *MockServer) AddHandler(path string, handler http.HandlerFunc) {
    m.mux.HandleFunc(path, handler)
}

func (m *MockServer) URL() string {
    return m.server.URL
}

func (m *MockServer) Close() {
    m.server.Close()
}
```

### Extending Mock Server

Add new fixtures for edge case testing:

```go
func TestCircularRedirect(t *testing.T) {
    mock := NewMockServer()
    defer mock.Close()
    
    // A redirects to B, B redirects to A
    mock.AddHandler("/a", func(w http.ResponseWriter, r *http.Request) {
        http.Redirect(w, r, mock.URL()+"/b", 302)
    })
    mock.AddHandler("/b", func(w http.ResponseWriter, r *http.Request) {
        http.Redirect(w, r, mock.URL()+"/a", 302)
    })
    
    crawler := NewCrawler(Config{
        Entry:   mock.URL() + "/a",
        Workers: 1,
    })
    
    // Should detect and break the redirect loop
    err := crawler.Run()
    assert.Error(t, err)
    assert.Contains(t, err.Error(), "redirect loop")
}
```

### Coverage Report

Generate HTML coverage report:

```bash
go test ./... -coverprofile=coverage.out -covermode=atomic
go tool cover -html=coverage.out -o coverage.html
open coverage.html
```

Coverage targets:
- Internal packages: 80%+ statement coverage
- Integration tests: 60%+ (focus on critical paths)

## Section B: Rust Tests

### Running Tests

```bash
cd semantic-search

# Run all unit tests
cargo test --workspace --lib

# Run specific test module
cargo test --lib -p quarry-embedder

# Run with verbose output
cargo test --workspace --lib -- --nocapture

# Run integration tests
cargo test --test integration_tests

# Run with specific test name
cargo test test_embedding_pipeline -- --nocapture

# Run with address sanitizer (requires nightly)
RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --workspace --target x86_64-unknown-linux-gnu
```

### GPU Tests vs CPU Fallback

Tests automatically detect GPU availability:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    fn skip_if_no_gpu() -> Option<()> {
        if std::env::var("QUARRY_SKIP_GPU_TESTS").is_ok() {
            return None;
        }
        if !gpu_available() {
            println!("Skipping GPU test: no GPU available");
            return None;
        }
        Some(())
    }
    
    #[test]
    fn test_gpu_inference() {
        skip_if_no_gpu().unwrap_or_else(|| return);
        
        let embedder = Embedder::new("models/model.onnx").unwrap();
        let result = embedder.embed(vec!["test sentence".to_string()]).unwrap();
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);
    }
    
    #[test]
    fn test_cpu_inference() {
        std::env::set_var("ORT_USE_CUDA", "0");
        
        let embedder = Embedder::new("models/model.onnx").unwrap();
        let result = embedder.embed(vec!["test sentence".to_string()]).unwrap();
        
        assert_eq!(result.len(), 1);
    }
}
```

### Memory Leak Detection

Use heaptrack for Linux memory profiling:

```bash
# Install heaptrack
sudo apt install heaptrack

# Run indexer with leak detection
heaptrack ./target/release/indexer --input ../crawler/output/pages/

# View results
heaptrack_print heaptrack.*.gz
```

Use valgrind for detailed leak checking:

```bash
# Run with valgrind
valgrind --leak-check=full --show-leak-kinds=all \
    ./target/release/search --query "test"

# Expected output:
# LEAK SUMMARY:
#    definitely lost: 0 bytes in 0 blocks
#    indirectly lost: 0 bytes in 0 blocks
```

## Section C: E2E Pipeline Test

### What pipeline_test.sh Does

The end-to-end pipeline test verifies the complete data flow from crawling through search:

1. **Setup**: Create temporary directories, start mock HTTP server
2. **Crawl**: Run Go crawler against mock site
3. **Verify JSON**: Check that page JSON files are created with correct schema
4. **Index**: Run Rust indexer on crawler output
5. **Verify Index**: Check that HNSW index files are created
6. **Query**: Run search queries and verify results
7. **Assert Rankings**: Verify that relevant documents rank above threshold
8. **Cleanup**: Remove temporary files

### Fixtures

| File | Content | Purpose |
|------|---------|---------|
| site/index.html | Homepage with links to all pages | Entry point for crawl |
| site/installation.html | Installation guide content | Primary search target |
| site/launcher.html | Launcher options documentation | Secondary search target |
| site/troubleshooting.html | Common issues and fixes | Edge case content |
| robots.txt | Allow all paths | Basic compliance test |

### fixture: site/index.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>Minecraft Linux Guide</title>
</head>
<body>
    <h1>Minecraft Linux Guide</h1>
    <p>Welcome to the comprehensive guide for running Minecraft on Linux.</p>
    <nav>
        <a href="/installation.html">Installation Guide</a>
        <a href="/launcher.html">Launcher Options</a>
        <a href="/troubleshooting.html">Troubleshooting</a>
    </nav>
</body>
</html>
```

### fixture: site/installation.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>Installation Guide - Minecraft Linux</title>
</head>
<body>
    <h1>Installing Minecraft on Linux</h1>
    <p>To install Minecraft on Linux, you will need a Java runtime environment.
    The recommended approach is to use the official launcher from Mojang.
    Alternatively, open-source launchers like HMCL or Prism Launcher provide
    additional features and mod support.</p>
    
    <h2>Prerequisites</h2>
    <ul>
        <li>Java 17 or later (for Minecraft 1.18+)</li>
        <li>At least 4GB RAM allocated to Minecraft</li>
        <li>OpenGL 4.4 compatible graphics driver</li>
    </ul>
    
    <h2>Steps</h2>
    <ol>
        <li>Download the launcher from the official website</li>
        <li>Make the file executable: chmod +x Minecraft.jar</li>
        <li>Run the launcher: java -jar Minecraft.jar</li>
        <li>Login with your Mojang or Microsoft account</li>
    </ol>
</body>
</html>
```

### Expected stdout for Full-Pass Scenario

```
=== Quarry E2E Pipeline Test ===
[1/6] Setting up mock HTTP server...
      Mock server listening at http://127.0.0.1:8080

[2/6] Running Go crawler...
      Entry URL: http://127.0.0.1:8080/
      Workers: 5
      Output: /tmp/quarry-e2e-XXXXX/pages/
      Crawled 5 pages in 0.3s
      ✓ Crawl completed successfully

[3/6] Verifying JSON output...
      ✓ Found 5 page JSON files
      ✓ All files have required fields: url, title, body, links, crawled_at
      ✓ URL normalization verified
      ✓ Deduplication verified (5 unique pages)

[4/6] Running Rust indexer...
      Input: /tmp/quarry-e2e-XXXXX/pages/
      Model: models/model.onnx
      Batch size: 8
      Indexed 5 documents in 0.8s
      ✓ Indexing completed successfully

[5/6] Verifying vector store...
      ✓ index.hnsw exists
      ✓ index.vecs exists
      ✓ index.meta exists
      ✓ Checksum verification passed
      ✓ Document count: 5

[6/6] Running search queries...
      Query: "how to install minecraft on linux"
        Result 1: /installation.html (score: 0.89)
        Result 2: / (score: 0.71)
        Result 3: /launcher.html (score: 0.65)
      ✓ Installation page ranked #1
      ✓ Score > 0.85 threshold met (0.89)

      Query: "java runtime requirements"
        Result 1: /installation.html (score: 0.82)
        Result 2: /troubleshooting.html (score: 0.58)
      ✓ Relevant page in top 3

      Query: "launcher options"
        Result 1: /launcher.html (score: 0.91)
      ✓ Exact match ranked #1

=== All tests passed ===
Pipeline completed in 2.1 seconds
Exit code: 0
```

## Edge Cases Documented

### Crawler Edge Cases

| Test Name | What It Tests | Failure Mode |
|-----------|---------------|--------------|
| TestEmptySite | Crawl site with no pages | Returns empty output without error |
| TestCircularRedirect | A→B→A redirect loop | Detects loop, returns error |
| TestRobotsDisallowAll | robots.txt blocks everything | Exits with code 2 |
| TestTenThousandLinks | Page with 10K outbound links | Processes all links without OOM |
| TestNonUTF8Content | Latin-1 encoded page | Transcodes to UTF-8 correctly |
| TestAllRequestsTimeout | Every request times out | Completes with 0 pages, exit 0 |
| TestRateLimit429 | Server returns 429 | Backs off and retries |
| TestMaxBodySize | Response exceeds limit | Skips page, logs warning |
| TestInvalidSSL | Self-signed certificate | Skips URL, logs warning |

### Search Edge Cases

| Test Name | What It Tests | Failure Mode |
|------------|---------------|--------------|
| TestEmptyIndexQuery | Query against empty index | Returns empty results without error |
| TestZeroSemanticMatch | Query with no similar documents | Returns empty results |
| TestQueryTooLong | Query exceeds max sequence length | Truncates and processes |
| TestGPUOOSimulation | VRAM exhaustion with large batch | Halves batch size dynamically |
| TestCorruptVectorStore | Invalid checksum in .meta | Returns Error::CorruptionDetected |
| TestConcurrentQueries | 100 parallel search requests | Handles via async runtime |
| TestUnicodeQuery | Chinese, emoji, RTL text queries | Tokenizes and embeds correctly |

## CI Integration

### Example GitHub Actions Workflow

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  go-test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      
      - name: Go vet
        working-directory: crawler
        run: go vet ./...
      
      - name: Run Go tests
        working-directory: crawler
        run: go test ./... -race -v -timeout 10m

  rust-test:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      
      - name: Cache cargo
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            semantic-search/target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Rustfmt check
        working-directory: semantic-search
        run: cargo fmt -- --check
      
      - name: Clippy
        working-directory: semantic-search
        run: cargo clippy -- -D warnings
      
      - name: Run Rust tests
        working-directory: semantic-search
        run: cargo test --workspace --verbose
        env:
          QUARRY_SKIP_GPU_TESTS: 1  # No GPU in CI
```

## Writing New Tests

### Go Test Guidelines

1. **File naming**: `*_test.go` in the same package or `tests/` directory
2. **Function naming**: `Test<FunctionName>_<Scenario>`, e.g., `TestParse_RobotsDisallowsPath`
3. **Use testify**: `assert` and `require` for readable assertions
4. **Table-driven tests**: For multiple scenarios testing the same function
5. **Cleanup**: Use `t.Cleanup()` for resource cleanup

```go
func TestNormalizeURL(t *testing.T) {
    tests := []struct {
        name     string
        input    string
        expected string
    }{
        {
            name:     "adds trailing slash to root",
            input:    "https://example.com",
            expected: "https://example.com/",
        },
        {
            name:     "removes trailing slash from path",
            input:    "https://example.com/path/",
            expected: "https://example.com/path",
        },
        {
            name:     "lowercases scheme",
            input:    "HTTPS://example.com/",
            expected: "https://example.com/",
        },
        {
            name:     "removes fragment",
            input:    "https://example.com/page#section",
            expected: "https://example.com/page",
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := normalizeURL(tt.input)
            assert.Equal(t, tt.expected, result)
        })
    }
}
```

### Rust Test Guidelines

1. **Unit tests**: In `#[cfg(test)] mod tests` within the source file
2. **Integration tests**: In `tests/` directory at crate root
3. **Use assert_matches**: For Result type assertions
4. **Use tempfile crate**: For temporary file handling
5. **Avoid unwrap in tests**: Use `expect()` with descriptive messages

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_normalize_vector() {
        let input = vec![3.0, 4.0];
        let result = normalize(&input);
        
        let magnitude: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6, "Vector should be unit length");
    }
    
    #[test]
    fn test_vector_store_roundtrip() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path();
        
        let mut store = VectorStore::new(384);
        store.insert(vec![0, 1, 2], vec![0.1; 384]);
        
        store.save(path).expect("Failed to save");
        let loaded = VectorStore::load(path).expect("Failed to load");
        
        assert_eq!(store.doc_count(), loaded.doc_count());
    }
}
```

### Required Test Structure

Every test file must have:

1. **Package/build declaration**: `package crawler_test` or `mod tests`
2. **Imports**: Standard library, then external, then internal
3. **Test functions**: One behavior per test
4. **Cleanup**: Ensure no resources leak after test completes

```go
// Good: Clear structure, cleanup handled
func TestCrawler_CompletesCrawl(t *testing.T) {
    // Setup
    mock := NewMockServer()
    defer mock.Close()
    
    tmpDir := t.TempDir()  // Auto-cleaned
    
    // Execute
    crawler := NewCrawler(Config{
        Entry:     mock.URL(),
        Workers:   2,
        OutputDir: tmpDir,
    })
    err := crawler.Run()
    
    // Assert
    require.NoError(t, err)
    assert.FileExists(t, filepath.Join(tmpDir, "inverted-index.json"))
}
```

```rust
// Good: Clear structure, cleanup via tempfile
#[test]
fn test_indexer_processes_documents() {
    let dir = tempfile::tempdir().unwrap();
    
    // Create test documents
    create_test_document(dir.path(), "doc1.json", "Test content");
    
    // Execute
    let indexer = Indexer::new(IndexerConfig {
        input_path: dir.path().to_path_buf(),
        batch_size: 1,
    });
    let result = indexer.run().unwrap();
    
    // Assert
    assert_eq!(result.documents_indexed, 1);
}
```
