# Engineering Specification: Minecraft-Linux Documentation Crawler & Semantic Search System

**Version:** 1.0.0  
**Date:** 2026-03-22  
**Author:** Distributed Systems Architect  
**Status:** Committed Specification — No Implementation Code

---

## 1. Component Breakdown

### 1.1 Component A: Go Concurrent Crawler + Structured Indexer

**Primary Responsibility:** Crawl all reachable pages within the `minecraft-linux.github.io` domain, extract structured content, and produce machine-readable index artifacts for downstream semantic search consumption.

**Sub-Responsibilities:**

| Sub-component | Responsibility |
|---------------|----------------|
| `robots` | Fetch and parse robots.txt; enforce crawl-delay and disallow rules |
| `scheduler` | Manage URL frontier with priority ordering and deduplication |
| `fetcher` | Execute HTTP requests with retry logic, timeout enforcement, and rate limiting |
| `parser` | HTML parsing, content extraction, link discovery, and metadata extraction |
| `indexer` | Build per-page JSON documents and inverted index for fast grep-based lookup |
| `persister` | Atomic write of index artifacts to disk with checksum verification |

**Constraints Applied:**
- Pure Go (no cgo) to ensure cross-compilation simplicity and deterministic memory behavior
- Stdlib-first approach; only `colly` for crawling framework and `goquery` for HTML parsing permitted
- Zero unbounded allocations; all buffers, maps, and channels have defined capacities

---

### 1.2 Component B: Rust Semantic Search Engine with GPU-Accelerated ONNX Inference

**Primary Responsibility:** Load crawler output, generate embeddings via ONNX Runtime on NVIDIA MX250 GPU, and serve semantic similarity queries over the indexed content.

**Sub-Responsibilities:**

| Sub-component | Responsibility |
|---------------|----------------|
| `loader` | Parse crawler JSON output, validate schema, load into memory-mapped structures |
| `tokenizer` | Tokenize query and document text using model-specific tokenizer |
| `embedder` | Execute ONNX inference on CUDA device with batching and memory management |
| `vector_store` | Store embeddings with associated metadata; perform ANN or flat similarity search |
| `query_engine` | Accept queries, compute query embedding, retrieve top-K similar documents |
| `api` | Expose query interface (CLI or HTTP) with structured JSON I/O |

**Constraints Applied:**
- Rust stable 2021 edition; no nightly features
- ONNX Runtime with CUDA execution provider only; no PyTorch at inference
- Hard VRAM ceiling of 1.5 GiB (leaving 512 MiB system headroom on 2 GiB MX250)

---

## 2. Algorithm Selection Table

### 2.1 Component A: Go Crawler Algorithms

| Algorithm | Purpose | Choice | Justification | Time Complexity | Space Complexity |
|-----------|---------|--------|---------------|-----------------|------------------|
| **URL Deduplication** | Track visited URLs | `sync.Map` with URL normalization | For 50-200 pages, a visited map is ~20 KiB with zero false positives; Bloom filter adds complexity without benefit at this scale | O(1) amortized lookup | O(n) where n = pages |
| **Link Extraction** | Find hrefs in HTML | `goquery` CSS selector `a[href]` | Industry standard, handles malformed HTML, provides clean attribute extraction | O(DOM nodes) | O(links found) |
| **Content Extraction** | Extract readable text | Custom selector for `main`, `article`, fallback to `body` with script/style removal | Static documentation site has predictable structure; custom selector yields cleaner output than generic libraries | O(DOM nodes) | O(text length) |
| **Rate Limiting** | Respect server resources | Token bucket with 100ms minimum inter-request delay | Simple, deterministic; prevents burst requests on GitHub Pages | O(1) per request | O(1) |
| **Retry Logic** | Handle transient failures | Exponential backoff: 1s, 2s, 4s with jitter ±20% | Standard practice; jitter prevents thundering herd | O(1) | O(1) |
| **Inverted Index** | Fast keyword lookup | Hash map: term → list of (doc_id, positions) | Small corpus makes hash map ideal; no need for compressed indexes | O(1) term lookup | O(total terms) |

**Deduplication Deep Dive:**

For a corpus of 50-200 pages, the visited URL set requires approximately:
- Average URL length: 80 bytes
- 200 URLs × 80 bytes = 16 KiB (URLs only)
- With map overhead: ~32-48 KiB total

A Bloom filter with 1% false positive rate for 200 items would require:
- ~240 bytes of bit array
- But: false positives cause re-fetches; for 200 pages, the "savings" of 16 KiB is negligible

**Decision:** Use `sync.Map` (or `map[string]struct{}` with `sync.RWMutex`) for exact deduplication. The memory savings from Bloom filter do not justify the false positive risk, even at this small scale.

---

### 2.2 Component B: Rust Semantic Search Algorithms

| Algorithm | Purpose | Choice | Justification | Time Complexity | Space Complexity |
|-----------|---------|--------|---------------|-----------------|------------------|
| **Embedding Model** | Text → vector | `all-MiniLM-L6-v2` (ONNX) | 22M parameters, 80 MiB model size, 384-dim output; proven quality on MTEB benchmarks; fits MX250 VRAM with 1.4 GiB headroom | O(sequence length) | O(1) model weights |
| **Tokenization** | Text → token IDs | WordPiece tokenizer (exported with model) | Native to MiniLM; deterministic; no external dependencies | O(text length) | O(tokens) |
| **Vector Similarity** | Compare embeddings | Cosine similarity | Standard for sentence embeddings; MiniLM embeddings are normalized during export | O(dimension) per comparison | O(1) |
| **Vector Store** | Store and search embeddings | Flat index with precomputed norms | 200 docs × 384 dim × 4 bytes = 307 KiB; flat search is instantaneous; HNSW overhead not justified | O(n×d) for search | O(n×d) storage |
| **Batch Processing** | GPU memory efficiency | Dynamic batch sizing with max batch=16 | MX250 has limited VRAM; batching amortizes model load overhead | O(batch_size × seq_len) | O(batch_size × hidden_dim) |
| **Memory Management** | Prevent VRAM OOM | Arena allocation with explicit release | ONNX Runtime CUDA provider requires explicit memory management; arenas prevent fragmentation | — | O(allocated chunks) |

**Vector Store Deep Dive:**

For 200 documents with 384-dimensional float32 embeddings:
- Embedding storage: 200 × 384 × 4 bytes = 307,200 bytes ≈ 300 KiB
- Metadata storage: ~50 KiB (doc IDs, offsets)
- Total: ~350 KiB

HNSW index overhead:
- Graph structure: ~2-4× embedding size = 600-1200 KiB additional
- Build time: non-trivial for what is essentially an instant flat search

**Decision:** Use flat cosine similarity search. For 200 documents, computing 200 cosine similarities (200 × 384 × 4 = 307,200 FLOPs per query) is trivial—sub-millisecond on CPU. The overhead of HNSW (both memory and build time) provides no benefit at this scale.

**Future-proofing note:** If corpus grows beyond 10,000 documents, migrate to `hnsw` crate with M=16, ef_construction=200. Design the `VectorStore` trait to allow drop-in replacement.

---

## 3. Data Schemas

### 3.1 Crawler Output: Per-Page Document

**File:** `index/pages/{url_hash}.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PageDocument",
  "type": "object",
  "required": ["id", "url", "title", "content", "metadata", "links", "indexed_at"],
  "properties": {
    "id": {
      "type": "string",
      "description": "SHA-256 hash of normalized URL (first 16 hex chars)",
      "pattern": "^[a-f0-9]{16}$"
    },
    "url": {
      "type": "string",
      "format": "uri",
      "description": "Canonical URL (normalized, absolute)"
    },
    "title": {
      "type": "string",
      "description": "Extracted from <title> tag, stripped of whitespace"
    },
    "content": {
      "type": "object",
      "required": ["raw", "text"],
      "properties": {
        "raw": {
          "type": "string",
          "description": "Original HTML (UTF-8, max 1 MiB per page)"
        },
        "text": {
          "type": "string",
          "description": "Extracted plain text, normalized whitespace"
        },
        "sections": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["heading", "level", "content"],
            "properties": {
              "heading": { "type": "string" },
              "level": { "type": "integer", "minimum": 1, "maximum": 6 },
              "content": { "type": "string" }
            }
          }
        }
      }
    },
    "metadata": {
      "type": "object",
      "required": ["crawl_status", "http_status", "content_type"],
      "properties": {
        "crawl_status": {
          "type": "string",
          "enum": ["success", "failed", "skipped"]
        },
        "http_status": {
          "type": "integer",
          "description": "Final HTTP status code after redirects"
        },
        "content_type": {
          "type": "string",
          "description": "Content-Type header value"
        },
        "last_modified": {
          "type": ["string", "null"],
          "format": "date-time"
        },
        "redirect_chain": {
          "type": "array",
          "items": { "type": "string", "format": "uri" },
          "description": "Sequence of redirect URLs if any"
        },
        "error": {
          "type": ["string", "null"],
          "description": "Error message if crawl_status is 'failed'"
        }
      }
    },
    "links": {
      "type": "object",
      "required": ["internal", "external"],
      "properties": {
        "internal": {
          "type": "array",
          "items": { "type": "string", "format": "uri" },
          "description": "Links to same domain"
        },
        "external": {
          "type": "array",
          "items": { "type": "string", "format": "uri" },
          "description": "Links to external domains (for reference only)"
        }
      }
    },
    "indexed_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

**Example:**
```json
{
  "id": "a3f2c8e1d4b5a6f7",
  "url": "https://minecraft-linux.github.io/source_build/launcher.html",
  "title": "Building the Launcher | Minecraft Linux",
  "content": {
    "raw": "<!DOCTYPE html>...",
    "text": "Building the Launcher\n\nThis guide covers...",
    "sections": [
      {"heading": "Prerequisites", "level": 2, "content": "Before building..."},
      {"heading": "Building from Source", "level": 2, "content": "Run the following..."}
    ]
  },
  "metadata": {
    "crawl_status": "success",
    "http_status": 200,
    "content_type": "text/html; charset=utf-8",
    "last_modified": null,
    "redirect_chain": []
  },
  "links": {
    "internal": [
      "https://minecraft-linux.github.io/",
      "https://minecraft-linux.github.io/source_build/"
    ],
    "external": ["https://github.com/minecraft-linux"]
  },
  "indexed_at": "2026-03-22T10:30:00Z"
}
```

---

### 3.2 Crawler Output: Inverted Index

**File:** `index/inverted.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InvertedIndex",
  "type": "object",
  "required": ["version", "built_at", "stats", "index"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "built_at": { "type": "string", "format": "date-time" },
    "stats": {
      "type": "object",
      "required": ["total_documents", "total_terms", "avg_doc_length"],
      "properties": {
        "total_documents": { "type": "integer" },
        "total_terms": { "type": "integer" },
        "avg_doc_length": { "type": "number" }
      }
    },
    "index": {
      "type": "object",
      "additionalProperties": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["doc_id", "count", "positions"],
          "properties": {
            "doc_id": { "type": "string" },
            "count": { "type": "integer", "minimum": 1 },
            "positions": {
              "type": "array",
              "items": { "type": "integer", "minimum": 0 },
              "description": "Byte offsets of term occurrences in text"
            }
          }
        }
      }
    }
  }
}
```

**Design rationale:** The inverted index enables fast `grep`-like lookup by AI agents. Position information allows context extraction. This is a secondary access path; semantic search via embeddings is the primary.

---

### 3.3 Crawler Output: Manifest

**File:** `index/manifest.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CrawlManifest",
  "type": "object",
  "required": ["version", "crawl_id", "started_at", "completed_at", "config", "summary"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "crawl_id": {
      "type": "string",
      "description": "UUID v4 for this crawl run"
    },
    "started_at": { "type": "string", "format": "date-time" },
    "completed_at": { "type": "string", "format": "date-time" },
    "config": {
      "type": "object",
      "required": ["entry_point", "max_depth", "rate_limit_ms", "user_agent"],
      "properties": {
        "entry_point": { "type": "string", "format": "uri" },
        "max_depth": { "type": "integer" },
        "rate_limit_ms": { "type": "integer" },
        "user_agent": { "type": "string" }
      }
    },
    "summary": {
      "type": "object",
      "required": ["pages_crawled", "pages_failed", "pages_skipped", "bytes_downloaded"],
      "properties": {
        "pages_crawled": { "type": "integer" },
        "pages_failed": { "type": "integer" },
        "pages_skipped": { "type": "integer" },
        "bytes_downloaded": { "type": "integer" }
      }
    }
  }
}
```

---

### 3.4 Go → Rust Boundary: Embedding Input

**File:** `index/embeddings_input.jsonl`

Line-delimited JSON for streaming compatibility:

```json
{"id": "a3f2c8e1d4b5a6f7", "url": "https://minecraft-linux.github.io/source_build/launcher.html", "title": "Building the Launcher", "text": "Building the Launcher\n\nThis guide covers...", "section_count": 3}
{"id": "b4d5e6f7a8b9c0d1", "url": "https://minecraft-linux.github.io/", "title": "Minecraft Linux", "text": "Welcome to Minecraft Linux...", "section_count": 5}
```

**Schema per line:**
```json
{
  "type": "object",
  "required": ["id", "url", "title", "text"],
  "properties": {
    "id": { "type": "string" },
    "url": { "type": "string" },
    "title": { "type": "string" },
    "text": { "type": "string" },
    "section_count": { "type": "integer" }
  }
}
```

---

### 3.5 Rust Output: Embedding Store

**File:** `index/embeddings.bin`

Binary format for efficient loading:

```
┌─────────────────────────────────────────────────────────┐
│ Header (32 bytes)                                       │
│ ├─ Magic: "SEMB" (4 bytes)                              │
│ ├─ Version: u32 (4 bytes)                               │
│ ├─ Embedding dimension: u32 (4 bytes)                   │
│ ├─ Document count: u32 (4 bytes)                        │
│ ├─ Reserved: 16 bytes                                   │
├─────────────────────────────────────────────────────────┤
│ Document Index (variable)                               │
│ ├─ For each document:                                   │
│ │   ├─ ID length: u16                                   │
│ │   ├─ ID: [u8; ID length]                              │
│ │   ├─ URL length: u16                                  │
│ │   ├─ URL: [u8; URL length]                            │
│ │   ├─ Title length: u16                                │
│ │   ├─ Title: [u8; Title length]                        │
│ │   └─ Embedding offset: u64 (in embeddings section)    │
├─────────────────────────────────────────────────────────┤
│ Embeddings (aligned to 4 bytes)                         │
│ ├─ For each document:                                   │
│ │   └─ [f32; dimension] (embedding vector)              │
├─────────────────────────────────────────────────────────┤
│ Footer (8 bytes)                                        │
│ └─ CRC32 checksum of all preceding bytes                │
└─────────────────────────────────────────────────────────┘
```

**Why binary vs JSON:** Embeddings are float arrays; JSON encoding would bloat 4× (each float becomes ~10-15 chars). Binary format enables memory mapping (`mmap`) for instant load without parsing.

---

### 3.6 Query API: Request

**Input format (JSON):**

```json
{
  "query": "how to build the launcher from source",
  "top_k": 5,
  "min_score": 0.3,
  "filters": {
    "url_pattern": "*/source_build/*"
  }
}
```

**Schema:**
```json
{
  "type": "object",
  "required": ["query"],
  "properties": {
    "query": { "type": "string", "minLength": 1, "maxLength": 1000 },
    "top_k": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
    "min_score": { "type": "number", "minimum": 0, "maximum": 1, "default": 0 },
    "filters": {
      "type": "object",
      "properties": {
        "url_pattern": { "type": "string", "description": "Glob pattern for URL filtering" }
      }
    }
  }
}
```

---

### 3.7 Query API: Response

**Output format (JSON):**

```json
{
  "query": "how to build the launcher from source",
  "processing_time_ms": 23,
  "results": [
    {
      "rank": 1,
      "score": 0.8923,
      "document": {
        "id": "a3f2c8e1d4b5a6f7",
        "url": "https://minecraft-linux.github.io/source_build/launcher.html",
        "title": "Building the Launcher",
        "snippet": "Building the Launcher\n\nThis guide covers compiling the launcher from source code..."
      }
    },
    {
      "rank": 2,
      "score": 0.7841,
      "document": {
        "id": "b4d5e6f7a8b9c0d1",
        "url": "https://minecraft-linux.github.io/source_build/",
        "title": "Source Build Overview",
        "snippet": "Source Build Overview\n\nAll source build guides..."
      }
    }
  ]
}
```

**Schema:**
```json
{
  "type": "object",
  "required": ["query", "processing_time_ms", "results"],
  "properties": {
    "query": { "type": "string" },
    "processing_time_ms": { "type": "integer" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["rank", "score", "document"],
        "properties": {
          "rank": { "type": "integer" },
          "score": { "type": "number", "minimum": 0, "maximum": 1 },
          "document": {
            "type": "object",
            "required": ["id", "url", "title", "snippet"],
            "properties": {
              "id": { "type": "string" },
              "url": { "type": "string" },
              "title": { "type": "string" },
              "snippet": { "type": "string", "description": "First 300 chars of text" }
            }
          }
        }
      }
    }
  }
}
```

---

## 4. File/Package Layout

### 4.1 Component A: Go Crawler

```
crawler/
├── cmd/
│   └── crawl/
│       └── main.go                 # Entry point, CLI flags, orchestration
├── internal/
│   ├── config/
│   │   └── config.go              # Configuration struct, validation, defaults
│   ├── robots/
│   │   ├── parser.go              # robots.txt parsing (RFC 9309)
│   │   └── checker.go             # URL allow/disallow checking
│   ├── frontier/
│   │   ├── queue.go               # URL queue with depth tracking
│   │   └── dedup.go               # URL normalization and deduplication
│   ├── fetcher/
│   │   ├── client.go              # HTTP client with retry, timeout, rate limiting
│   │   └── response.go            # Response wrapper, status handling
│   ├── parser/
│   │   ├── html.go                # HTML parsing via goquery
│   │   ├── content.go             # Text extraction, section detection
│   │   └── links.go               # Link extraction and normalization
│   ├── indexer/
│   │   ├── document.go            # PageDocument construction
│   │   ├── inverted.go            # Inverted index building
│   │   └── manifest.go            # Manifest generation
│   └── persister/
│       ├── writer.go              # Atomic file writes with temp+rename
│       └── checksum.go            # SHA-256 checksums for integrity
├── pkg/
│   └── normalize/
│       └── url.go                 # URL normalization utilities
├── output/
│   └── index/                     # Default output directory (gitignored)
│       ├── manifest.json
│       ├── inverted.json
│       └── pages/
│           ├── a3f2c8e1d4b5a6f7.json
│           └── ...
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

**Package responsibilities:**

| Package | Responsibility | Export Surface |
|---------|---------------|----------------|
| `config` | Parse CLI flags, env vars, config file | `Config` struct, `Load() error` |
| `robots` | Respect robots.txt directives | `Checker.FetchAndParse()`, `Checker.IsAllowed(url)` |
| `frontier` | Manage crawl queue | `Queue.Push()`, `Queue.Pop()`, `Queue.Seen(url) bool` |
| `fetcher` | Execute HTTP requests | `Client.Fetch(url) (*Response, error)` |
| `parser` | Extract content from HTML | `ParseHTML(raw) (*ParsedPage, error)` |
| `indexer` | Build index structures | `BuildDocument()`, `BuildInvertedIndex()` |
| `persister` | Write to disk atomically | `WriteJSON(path, data) error` |
| `normalize` | URL canonicalization | `NormalizeURL(raw) (string, error)` |

---

### 4.2 Component B: Rust Semantic Search

```
semantic-search/
├── src/
│   ├── main.rs                    # Entry point, CLI argument parsing
│   ├── lib.rs                     # Library root, public API
│   ├── config.rs                  # Configuration struct, defaults
│   ├── loader/
│   │   ├── mod.rs
│   │   ├── json.rs                # JSONL parsing for crawler output
│   │   └── binary.rs              # Binary embedding store I/O
│   ├── tokenizer/
│   │   ├── mod.rs
│   │   └── wordpiece.rs           # WordPiece tokenizer implementation
│   ├── embedder/
│   │   ├── mod.rs
│   │   ├── onnx.rs                # ONNX Runtime session management
│   │   ├── cuda.rs                # CUDA device configuration
│   │   └── batch.rs               # Dynamic batching logic
│   ├── vector/
│   │   ├── mod.rs
│   │   ├── store.rs               # VectorStore trait
│   │   ├── flat.rs                # Flat index implementation
│   │   └── similarity.rs          # Cosine similarity computation
│   ├── query/
│   │   ├── mod.rs
│   │   ├── engine.rs              # Query processing pipeline
│   │   └── filters.rs             # URL pattern filtering
│   └── api/
│       ├── mod.rs
│       └── json.rs                # JSON request/response handling
├── models/
│   ├── minilm-l6-v2.onnx          # Embedding model (80 MiB)
│   ├── tokenizer.json             # WordPiece vocab and config
│   └── config.json                # Model metadata
├── data/
│   └── index/                     # Symlink to crawler output
├── tests/
│   ├── integration/
│   │   └── e2e_test.rs
│   └── fixtures/
│       ├── sample_pages.jsonl
│       └── sample_queries.json
├── Cargo.toml
├── Cargo.lock
├── build.rs                       # Optional: verify CUDA/ONNX at build time
├── Makefile
└── README.md
```

**Module responsibilities:**

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `config` | Runtime configuration | `Config::from_args()` |
| `loader` | Parse crawler output | `load_jsonl(path) -> Vec<Document>` |
| `tokenizer` | Tokenize text | `Tokenizer::encode(text) -> Encoding` |
| `embedder` | Generate embeddings | `Embedder::embed_batch(texts) -> Vec<Vec<f32>>` |
| `vector` | Store and search vectors | `VectorStore::search(query, k) -> Vec<Hit>` |
| `query` | Orchestrate search | `Engine::search(query) -> Response` |
| `api` | CLI/HTTP interface | `run_cli()`, `serve_http()` |

---

### 4.3 Shared Artifacts

```
shared/
├── schemas/
│   ├── page_document.json         # JSON Schema for page documents
│   ├── inverted_index.json        # JSON Schema for inverted index
│   ├── manifest.json              # JSON Schema for manifest
│   ├── embeddings_input.json      # JSON Schema for input JSONL
│   ├── query_request.json         # JSON Schema for query request
│   └── query_response.json        # JSON Schema for query response
└── scripts/
    ├── validate_schemas.sh        # Validate all outputs against schemas
    └── generate_docs.sh           # Generate markdown from schemas
```

---

## 5. Build Order

### 5.1 Dependency Graph

```
┌─────────────────┐
│  Schemas (1)    │ ← Define contracts first
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Go Crawler (2) │ ← Produce data conforming to schemas
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ONNX Export (3)│ ← Export model after crawler defines corpus
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rust Search(4) │ ← Consume crawler output, use ONNX model
└─────────────────┘
```

### 5.2 Detailed Build Sequence

| Phase | Component | Action | Rationale |
|-------|-----------|--------|-----------|
| 1 | Schemas | Define and validate all JSON schemas | Contracts enable parallel development |
| 2a | Go Crawler | Implement core crawling (robots, fetch, parse) | Data generation is prerequisite for Rust |
| 2b | ONNX Export | Export `all-MiniLM-L6-v2` to ONNX (see Section 8) | Can run in parallel with crawler |
| 3 | Go Crawler | Implement indexer, persister | Produces final output format |
| 4 | Rust Search | Implement loader, tokenizer | Consumes crawler output |
| 5 | Rust Search | Implement embedder, vector store | Requires ONNX model |
| 6 | Rust Search | Implement query engine, API | Integrates all components |
| 7 | Integration | End-to-end validation | Full pipeline test |

### 5.3 Makefile Targets

```makefile
# Root Makefile
.PHONY: all schemas crawl export search validate clean

all: schemas crawl export search validate

schemas:
	cd shared/schemas && ./validate_schemas.sh

crawl:
	cd crawler && go build -o bin/crawl ./cmd/crawl
	./crawler/bin/crawl --config crawler/config.yaml

export:
	python scripts/export_onnx.py  # See Section 8

search:
	cd semantic-search && cargo build --release

validate:
	cd shared/scripts && ./validate_schemas.sh

clean:
	rm -rf crawler/output/*
	rm -rf semantic-search/data/index
```

---

## 6. Risk Register

### 6.1 Risk Matrix

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|-------------|--------|----------|------------|
| R1 | ONNX Runtime CUDA provider incompatibility with MX250 | Medium | High | **Critical** | Test ONNX export early; fallback to CPU inference if CUDA fails; see Section 8 gotchas |
| R2 | VRAM exhaustion during batch embedding | Medium | High | **Critical** | Conservative batch sizing (max 8); monitor VRAM via `nvidia-smi`; implement graceful degradation to CPU |
| R3 | Crawler encounters infinite redirect loops | Low | Medium | **Medium** | Track redirect depth (max 5); detect and break cycles; log and skip problematic URLs |
| R4 | Rate limiting triggers 429 from GitHub Pages | Low | Medium | **Medium** | Implement exponential backoff; respect Retry-After header; conservative default rate (100ms) |
| R5 | Embedding model produces poor quality for technical docs | Low | Medium | **Medium** | `all-MiniLM-L6-v2` has proven performance on technical content; if needed, evaluate `all-MiniLM-L12-v2` as alternative |
| R6 | Memory leak in long-running Rust process | Low | High | **High** | Use `valgrind` and `sanitizers` in testing; implement periodic memory profiling; explicit drop of ONNX session |
| R7 | Unicode handling issues in Go → Rust boundary | Medium | Low | **Low** | Enforce UTF-8 everywhere; validate at schema level; use `rust-peg` for parsing |
| R8 | Inverted index grows unbounded | Very Low | Low | **Very Low** | Stopword removal; term frequency thresholds; corpus is small and static |

### 6.2 Risk Deep Dives

**R1: ONNX Runtime CUDA Provider Incompatibility**

The MX250 (Pascal architecture, compute capability 6.1) with CUDA 13.0 and ONNX Runtime has known friction points:
- ONNX Runtime 1.17+ may require specific CUDA/cuDNN versions
- CUDA execution provider must be built from source if pre-built binaries don't match

*Mitigation strategy:*
1. Test ONNX Runtime with CUDA EP early in Phase 2b
2. If CUDA EP fails, fallback to CPU inference (still fast for batch=8 on small corpus)
3. Consider `ort` crate with `load-dynamic` feature for runtime provider selection

**R2: VRAM Exhaustion**

MX250 has 2048 MiB VRAM with 512 MiB reserved for system, leaving 1536 MiB for inference.

Memory budget:
- Model weights: ~80 MiB (all-MiniLM-L6-v2)
- ONNX Runtime overhead: ~100-200 MiB
- Per-sequence activation: ~5-10 MiB per 256-token sequence

*Conservative calculation:*
- Available for batches: 1536 - 80 - 200 = 1256 MiB
- Per-sequence cost: ~10 MiB (conservative)
- Max batch size: 1256 / 10 = 125 (theoretical)
- **Safe batch size: 16** (8x safety margin)

*Mitigation strategy:*
- Start with batch_size=8, increase only if VRAM monitoring shows headroom
- Implement dynamic batch sizing based on sequence lengths
- Use `CUDA_LAUNCH_BLOCKING=1` for debugging OOM issues

---

## 7. Test Strategy

### 7.1 Component A: Go Crawler Tests

#### Unit Tests

| Package | Test | Input | Expected Output |
|---------|------|-------|-----------------|
| `normalize` | `TestNormalizeURL` | Various URL forms (relative, absolute, with/without trailing slash) | Normalized absolute URLs |
| `robots` | `TestParseRobots` | robots.txt content | Parsed rules with allow/disallow |
| `robots` | `TestIsAllowed` | URL + robots rules | Boolean + matching rule |
| `frontier` | `TestDedup` | Duplicate URLs | Second push returns false |
| `fetcher` | `TestRetry` | Mock server returning 500, then 200 | Successful response after retry |
| `parser` | `TestExtractContent` | HTML with various structures | Clean text, no script/style content |
| `parser` | `TestExtractLinks` | HTML with internal/external links | Correctly categorized links |
| `indexer` | `TestBuildInverted` | Sample documents | Correct term → doc mappings |

#### Integration Tests

| Test | Scenario | Validation |
|------|----------|------------|
| `TestCrawlStaticSite` | Crawl local static site fixture | All pages captured, manifest correct |
| `TestRobotsCompliance` | Crawl site with restrictive robots.txt | Disallowed paths not visited |
| `TestRedirectHandling` | Crawl site with redirect chains | Final URLs recorded, chain in metadata |
| `TestErrorHandling` | Crawl site with simulated 404/500/429 | Errors logged, crawl continues |

#### Edge Cases

| Case | Input | Expected Behavior |
|------|-------|-------------------|
| Empty page | `<html><body></body></html>` | Document with empty content, no crash |
| Malformed HTML | Missing closing tags, invalid entities | `goquery` handles gracefully, extract what's possible |
| Binary content | PDF, image links | Skip, log as skipped |
| Unicode heavy | Chinese, emoji, RTL text | UTF-8 preserved throughout |
| Very long page | >1 MiB HTML | Truncate with warning in metadata |

---

### 7.2 Component B: Rust Semantic Search Tests

#### Unit Tests

| Module | Test | Input | Expected Output |
|--------|------|-------|-----------------|
| `loader` | `test_parse_jsonl` | Valid JSONL file | `Vec<Document>` with correct fields |
| `loader` | `test_parse_jsonl_invalid` | Malformed JSONL | Error with line number |
| `tokenizer` | `test_encode` | Sample text | Token IDs match expected |
| `tokenizer` | `test_truncation` | Text > max_length | Truncated to max_length |
| `embedder` | `test_embed_batch` | List of texts | Embeddings of correct dimension |
| `embedder` | `test_embed_empty` | Empty string | Zero embedding or error (defined behavior) |
| `vector` | `test_cosine_similarity` | Two vectors | Correct similarity in [-1, 1] |
| `vector` | `test_search_top_k` | Query + documents | Top-K sorted by similarity |
| `query` | `test_filter_url_pattern` | Query with filter | Only matching URLs returned |

#### Integration Tests

| Test | Scenario | Validation |
|------|----------|------------|
| `test_load_crawler_output` | Load real crawler JSONL | All documents loaded |
| `test_embedding_roundtrip` | Generate and search embeddings | Query finds semantically similar docs |
| `test_query_api` | Full query pipeline | Valid JSON response with scores |
| `test_vram_usage` | Monitor GPU memory during embedding | Stays under 1.5 GiB limit |
| `test_cpu_fallback` | Disable CUDA, run on CPU | Successful but slower |

#### Edge Cases

| Case | Input | Expected Behavior |
|------|-------|-------------------|
| Empty query | `{"query": ""}` | Error: query too short |
| Very long query | 2000+ char query | Truncate to max sequence length |
| No results above threshold | Obscure query | Empty results array, not error |
| Unicode query | Chinese, emoji | Tokenized correctly, embeddings generated |
| Duplicate documents | Same text multiple times | Both indexed, both returned |

---

### 7.3 End-to-End Tests

| Test | Steps | Validation |
|------|-------|------------|
| `E2E_CrawlAndSearch` | 1. Run crawler 2. Load to Rust 3. Query | Query returns relevant results |
| `E2E_SchemaValidation` | 1. Run crawler 2. Validate all outputs | All outputs pass JSON Schema |
| `E2E_Performance` | 1. Crawl full site 2. Build embeddings 3. Query | Crawl < 5 min, Embed < 30 sec, Query < 100ms |

---

## 8. ONNX + CUDA Gotchas

### 8.1 Known Issues with ONNX Runtime + CUDA 13.0 on MX250

#### Issue 1: CUDA Execution Provider Compatibility

**Problem:** ONNX Runtime pre-built binaries are often compiled against specific CUDA versions. CUDA 13.0 is recent, and matching pre-built binaries may not exist.

**Solution:**
```bash
# Option A: Use compatible ONNX Runtime version
# Check: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
pip install onnxruntime-gpu==1.17.0  # Adjust based on compatibility matrix

# Option B: Build from source
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel --use_cuda \
  --cuda_home /usr/local/cuda-13.0 --cudnn_home /usr/local/cuda-13.0
```

**Rust-side handling:**
```rust
// Use ort crate with dynamic loading
use ort::{Environment, SessionBuilder, GraphOptimizationLevel};

let environment = Environment::builder()
    .with_name("semantic-search")
    .build()?;

// Try CUDA first, fallback to CPU
let session = SessionBuilder::new(&environment)?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("models/minilm-l6-v2.onnx")
    .or_else(|_| {
        // Fallback: CPU-only session
        SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file("models/minilm-l6-v2.onnx")
    })?;
```

---

#### Issue 2: cuDNN Version Mismatch

**Problem:** CUDA 13.0 may require a specific cuDNN version. ONNX Runtime CUDA EP is sensitive to cuDNN version mismatches.

**Solution:**
```bash
# Verify versions
nvidia-smi  # CUDA version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Install matching cuDNN
# Check NVIDIA documentation for CUDA 13.0 + cuDNN compatibility
```

---

#### Issue 3: Memory Fragmentation on Small VRAM

**Problem:** MX250's 2 GiB VRAM is prone to fragmentation. Multiple inference sessions or large batches can cause OOM even if theoretical memory is sufficient.

**Solution:**
1. **Single session:** Keep one ONNX session alive for the lifetime of the application
2. **Arena allocator:** Use ONNX Runtime's memory arena
3. **Batch before session:** Pre-allocate batch buffers before creating session

```rust
// Good: Single session, reused
lazy_static! {
    static ref EMBEDDER: Embedder = Embedder::new("models/minilm-l6-v2.onnx").unwrap();
}

// Bad: New session per request
fn embed(text: &str) -> Vec<f32> {
    let session = Session::new("model.onnx").unwrap(); // WRONG: Memory leak + slow
    // ...
}
```

---

#### Issue 4: CUDA Kernel Launch Failure

**Problem:** MX250 (Pascal, SM_61) may not support all CUDA kernels required by newer ONNX operators.

**Solution:**
- Export model with **opset version 14** (compatible with Pascal)
- Avoid operators that require newer architectures

```python
# ONNX export with opset 14
from optimum.exporters.onnx import main_export
from transformers import AutoModel, AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
main_export(
    model_name_or_path=model_id,
    output="models/",
    opset=14,  # Safe for Pascal
    task="feature-extraction"
)
```

---

#### Issue 5: Batch Size Tuning for Small VRAM

**Problem:** Default batch sizes in examples are often too large for MX250.

**Empirical testing:**
```bash
# Test with increasing batch sizes
CUDA_VISIBLE_DEVICES=0 python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/minilm-l6-v2.onnx', providers=['CUDAExecutionProvider'])
for batch_size in [1, 2, 4, 8, 16, 32]:
    try:
        inputs = {'input_ids': np.zeros((batch_size, 256), dtype=np.int64),
                  'attention_mask': np.ones((batch_size, 256), dtype=np.int64)}
        session.run(None, inputs)
        print(f'batch_size={batch_size}: OK')
    except Exception as e:
        print(f'batch_size={batch_size}: FAILED - {e}')
"
```

**Recommended starting point:** batch_size = 8

---

### 8.2 ONNX Export Strategy

#### Step-by-Step Export Process

```python
#!/usr/bin/env python3
"""
export_onnx.py: Export all-MiniLM-L6-v2 to ONNX for MX250

Requirements:
  pip install transformers optimum[onnx] torch
"""

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
import shutil
import os

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "semantic-search/models/"

def export():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export model to ONNX
    main_export(
        model_name_or_path=MODEL_ID,
        output=OUTPUT_DIR,
        opset=14,  # Compatible with Pascal (MX250)
        task="feature-extraction",
        # Optimize for inference
        optimize="O2",
    )
    
    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Verify export
    import onnx
    model = onnx.load(os.path.join(OUTPUT_DIR, "model.onnx"))
    onnx.checker.check_model(model)
    print(f"Export complete: {OUTPUT_DIR}")
    print(f"Model inputs: {[i.name for i in model.graph.input]}")
    print(f"Model outputs: {[o.name for o in model.graph.output]}")

if __name__ == "__main__":
    export()
```

**Output files:**
- `model.onnx` — Main model (rename to `minilm-l6-v2.onnx`)
- `tokenizer.json` — WordPiece tokenizer config
- `tokenizer_config.json` — Tokenizer metadata
- `vocab.txt` — Vocabulary file
- `special_tokens_map.json` — Special tokens

---

### 8.3 CUDA Environment Checklist

```bash
# 1. Verify CUDA installation
nvidia-smi
# Expected: CUDA 13.0, Driver 580.95.05, GPU: MX250 2048 MiB

# 2. Verify CUDA toolkit
nvcc --version
# Expected: release 13.0

# 3. Set environment variables
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. For Rust ort crate, ensure dynamic loading works
export ORT_USE_CUDA=1  # Enable CUDA provider

# 5. Debug CUDA issues
export CUDA_LAUNCH_BLOCKING=1  # Synchronous launches for debugging
export ONNXRUNTIME_LOGGING_LEVEL=0  # Verbose logging
```

---

## Appendix A: Model Selection Justification

### A.1 all-MiniLM-L6-v2 Specifications

| Property | Value |
|----------|-------|
| Parameter count | 22,684,032 |
| Model size (PyTorch) | 90.9 MiB |
| Model size (ONNX) | ~80 MiB |
| Architecture | BERT (6 layers, 384 hidden dim) |
| Max sequence length | 256 tokens |
| Embedding dimension | 384 |
| Training data | 1B+ sentence pairs |
| MTEB rank (as of 2024) | #8 for semantic similarity |

### A.2 VRAM Budget Calculation

```
Total VRAM:              2048 MiB
Reserved for system:      512 MiB
─────────────────────────────────
Available for inference: 1536 MiB

Model weights (ONNX):      80 MiB
ONNX Runtime overhead:    150 MiB  (estimated)
CUDA context:             100 MiB  (estimated)
─────────────────────────────────
Available for activations: 1206 MiB

Per-sequence activation (256 tokens):
  - Query projection:     ~2 MiB
  - Key projection:       ~2 MiB
  - Value projection:     ~2 MiB
  - Attention scores:     ~2 MiB
  - FFN intermediate:     ~4 MiB
  - Layer outputs:        ~4 MiB
  ─────────────────────────────────
  Total per sequence:     ~16 MiB (conservative)

Max batch size (theoretical): 1206 / 16 = 75
Safe batch size (with headroom): 8-16
```

### A.3 Alternative Models Considered

| Model | Params | Size | Max Seq | Reason Rejected |
|-------|--------|------|---------|-----------------|
| all-MiniLM-L12-v2 | 33M | 120 MiB | 512 | Larger, less headroom |
| paraphrase-mpnet-base-v2 | 109M | 420 MiB | 514 | Too large for MX250 |
| all-roberta-large-v1 | 355M | 1.3 GiB | 512 | Exceeds VRAM |
| multilingual-e5-small | 33M | 470 MiB | 512 | Larger, not needed (English docs) |

---

## Appendix B: Concurrency Model Details

### B.1 Go Crawler Worker Pool

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Scheduler  │─────▶│  Job Queue  │─────▶│   Workers   │
│ (goroutine) │      │ (chan, 100) │      │ (N workers) │
└─────────────┘      └─────────────┘      └──────┬──────┘
       │                                         │
       │         ┌─────────────┐                 │
       └────────▶│  Results    │◀────────────────┘
                 │ (chan, 50)  │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Indexer    │
                 │ (goroutine) │
                 └─────────────┘
```

**Configuration:**
- Worker count: 5 (conservative for static site, prevents rate limiting)
- Job queue buffer: 100 (accommodates burst of discovered links)
- Results buffer: 50 (decouples fetch from index)
- Rate limiter: 100ms minimum between requests per worker

**Backpressure handling:**
1. Job queue full → scheduler blocks → link extraction pauses
2. Results queue full → workers block → natural throttling
3. Explicit shutdown via context cancellation

---

## Appendix C: Error Code Handling

### C.1 HTTP Status Code Handling

| Code | Category | Action | Retry | Notes |
|------|----------|--------|-------|-------|
| 200 | Success | Parse and index | N/A | Normal case |
| 301 | Redirect | Follow to new URL | N/A | Record redirect chain |
| 302 | Redirect | Follow to new URL | N/A | Record redirect chain |
| 304 | Not Modified | Skip, use cached | N/A | If ETag/Last-Modified available |
| 400 | Client Error | Log and skip | No | Malformed request |
| 401 | Auth Required | Log and skip | No | No auth expected |
| 403 | Forbidden | Log and skip | No | Check robots.txt |
| 404 | Not Found | Log and skip | No | Dead link |
| 429 | Rate Limited | Backoff and retry | Yes | Respect Retry-After header |
| 500 | Server Error | Backoff and retry | Yes (3×) | Transient failure |
| 502 | Bad Gateway | Backoff and retry | Yes (3×) | Transient failure |
| 503 | Unavailable | Backoff and retry | Yes (3×) | Transient failure |
| Timeout | Network | Backoff and retry | Yes (3×) | 30s timeout per request |

### C.2 Retry Logic Pseudocode

```go
func (c *Client) Fetch(url string) (*Response, error) {
    const maxRetries = 3
    baseDelay := 1 * time.Second
    
    for attempt := 0; attempt <= maxRetries; attempt++ {
        resp, err := c.doRequest(url)
        
        if err == nil && resp.StatusCode < 500 && resp.StatusCode != 429 {
            return resp, nil
        }
        
        if attempt < maxRetries {
            delay := baseDelay * time.Duration(1 << attempt)
            jitter := time.Duration(rand.Float64() * 0.4 * float64(delay))
            time.Sleep(delay + jitter)
        }
    }
    
    return nil, fmt.Errorf("max retries exceeded for %s", url)
}
```

---

## Appendix D: Monitoring and Observability

### D.1 Metrics to Collect

**Component A (Go Crawler):**
- `crawl_pages_total{status=success|failed|skipped}`
- `crawl_bytes_total`
- `crawl_duration_seconds`
- `crawl_active_workers`
- `crawl_queue_depth`

**Component B (Rust Search):**
- `search_queries_total`
- `search_latency_seconds`
- `search_results_count_avg`
- `embed_batch_size`
- `vram_used_bytes`
- `embed_inference_seconds`

### D.2 Logging Format

Structured JSON logging:

```json
{
  "timestamp": "2026-03-22T10:30:00.123Z",
  "level": "info",
  "component": "crawler",
  "message": "Page crawled successfully",
  "url": "https://minecraft-linux.github.io/source_build/launcher.html",
  "http_status": 200,
  "bytes": 15234,
  "duration_ms": 234
}
```

---

## Appendix E: Future Considerations

### E.1 Scalability Path

| Current Scale | Future Scale | Migration Path |
|---------------|--------------|----------------|
| 200 pages, flat search | 10,000 pages | Switch to HNSW index (`hnsw` crate) |
| Single batch embedding | Continuous indexing | Implement embedding queue with persistence |
| CLI only | HTTP API | Add `axum` server to Rust component |
| File-based I/O | Database | Migrate to SQLite or PostgreSQL with `pgvector` |
| Single GPU | Multi-GPU | Add device selection logic |

### E.2 Potential Enhancements

1. **Incremental crawling:** Track ETag/Last-Modified, only re-fetch changed pages
2. **Query caching:** Cache frequent query embeddings and results
3. **Hybrid search:** Combine keyword (inverted index) and semantic search
4. **Reranking:** Add cross-encoder for final ranking of top-K results
5. **Multi-language support:** Use multilingual model if content expands

---

**End of Specification Document**
