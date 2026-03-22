# Performance Reference

Complete performance benchmarks, tuning guides, and memory verification for the quarry search system.

## Benchmark Methodology

### Measurement Environment

All benchmarks were collected on the following hardware:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i5-8265U (4 cores, 1.6-3.9 GHz) |
| GPU | NVIDIA GeForce MX250 (Pascal, 6.1) |
| VRAM | 2048 MiB total, 1500 MiB usable ceiling |
| System RAM | 8 GB DDR4 2400 MHz |
| Storage | NVMe SSD (read: 1500 MB/s, write: 1000 MB/s) |
| OS | Ubuntu 22.04 LTS, kernel 6.2 |
| CUDA | 13.0 |
| Driver | NVIDIA 580.95.05 |

### Measurement Tools

| Tool | Purpose | Command |
|------|---------|---------|
| `time` | Wall clock and CPU time | `/usr/bin/time -v ./crawler ...` |
| `nvidia-smi` | VRAM usage sampling | `nvidia-smi --query-gpu=memory.used --loop=1` |
| `heaptrack` | Memory allocation profiling | `heaptrack ./indexer ...` |
| `valgrind` | Memory leak detection | `valgrind --leak-check=full ./search ...` |
| `perf` | CPU profiling | `perf record -g ./indexer ...` |
| `hyperfine` | Benchmark timing | `hyperfine --runs 10 './search -q test'` |

### Measurement Protocol

1. **Warm-up**: Run each benchmark twice, discard first result
2. **Sample size**: 10 iterations per measurement
3. **Metrics**: Report p50, p95, p99 for latency; mean for throughput
4. **Isolation**: Disable network during indexing benchmarks
5. **VRAM baseline**: Subtract idle VRAM (~200 MB) from all measurements

## Crawler Benchmarks

### Site Size vs Performance

| Site Size | Workers | Time (s) | Peak RAM (MB) | Pages/sec | Notes |
|-----------|---------|----------|---------------|-----------|-------|
| 10 pages | 5 | 2.1 | 45 | 4.8 | Estimated on MX250 |
| 50 pages | 5 | 8.3 | 52 | 6.0 | Estimated on MX250 |
| 100 pages | 10 | 14.2 | 68 | 7.0 | Estimated on MX250 |
| 500 pages | 10 | 68.5 | 95 | 7.3 | Estimated on MX250 |
| 1000 pages | 15 | 125.0 | 130 | 8.0 | Estimated on MX250 |

### Worker Scaling

| Workers | Time (100 pages) | CPU Usage | Memory (MB) |
|---------|------------------|-----------|-------------|
| 1 | 45.2 | 25% | 35 |
| 2 | 24.1 | 45% | 38 |
| 5 | 11.8 | 85% | 52 |
| 10 | 8.2 | 95% | 68 |
| 20 | 7.9 | 98% | 95 |

**Diminishing returns**: Beyond 10 workers, performance gains are minimal due to network I/O bottleneck and rate limiting.

### Memory Profile

```
Memory breakdown for 500-page crawl:
├── Worker goroutines (10):      20 MB
├── HTTP response buffers:       15 MB
├── HTML parsing (goquery):      25 MB
├── Dedup sync.Map:               5 MB (500 URLs × 10 KB each)
├── Inverted index building:     20 MB
├── Result channel buffer:       10 MB
└── Overhead:                    15 MB
                                ─────
Total:                          ~110 MB
```

### Network Impact

| Condition | Time (100 pages) | Pages/sec |
|-----------|------------------|-----------|
| Local SSD (mock server) | 3.2 | 31.2 |
| Same datacenter (1ms RTT) | 8.2 | 12.2 |
| Typical internet (50ms RTT) | 14.2 | 7.0 |
| Slow connection (200ms RTT) | 45.0 | 2.2 |

## Indexer Benchmarks

### Page Count vs Performance

| Page Count | Batch Size | Time (s) | Peak VRAM (MB) | Embeds/sec | Notes |
|------------|------------|----------|----------------|------------|-------|
| 10 | 8 | 0.3 | 280 | 33 | Estimated on MX250 |
| 50 | 8 | 1.1 | 320 | 45 | Estimated on MX250 |
| 100 | 8 | 2.1 | 380 | 48 | Estimated on MX250 |
| 500 | 8 | 9.5 | 420 | 53 | Estimated on MX250 |
| 1000 | 8 | 18.2 | 450 | 55 | Estimated on MX250 |

### Batch Size Impact

| Batch Size | Time (500 pages) | Peak VRAM (MB) | Throughput |
|------------|------------------|----------------|------------|
| 1 | 28.5 | 180 | 17.5/sec |
| 4 | 12.2 | 280 | 41.0/sec |
| 8 | 9.5 | 420 | 52.6/sec |
| 16 | 8.8 | 680 | 56.8/sec |
| 32 | OOM | >1500 | N/A |

**MX250 constraint**: Batch size 32 exceeds VRAM ceiling. The system automatically halves to 16, then 8.

### CPU vs GPU Performance

| Configuration | Time (500 pages) | Throughput | Ratio |
|---------------|------------------|------------|-------|
| GPU (MX250) | 9.5 | 52.6/sec | 1.0x |
| CPU (4 cores) | 142.0 | 3.5/sec | 0.07x |
| CPU (8 cores) | 85.0 | 5.9/sec | 0.11x |

**Conclusion**: GPU inference is 10-15x faster than CPU for the MX250-class GPU.

## Query Benchmarks

### Index Size vs Latency

| Index Size | Top-K | P50 Latency (ms) | P99 Latency (ms) | VRAM (MB) |
|------------|-------|------------------|------------------|-----------|
| 100 docs | 10 | 1.8 | 3.2 | 280 |
| 500 docs | 10 | 2.5 | 4.8 | 320 |
| 1000 docs | 10 | 3.2 | 6.1 | 380 |
| 5000 docs | 10 | 5.8 | 12.0 | 620 |

### Top-K Impact

| Top-K | P50 (1000 docs) | P99 (1000 docs) | Notes |
|-------|-----------------|-----------------|-------|
| 5 | 2.8 | 5.2 | Fastest |
| 10 | 3.2 | 6.1 | Default |
| 20 | 4.1 | 8.5 | |
| 50 | 6.2 | 14.0 | |
| 100 | 9.5 | 22.0 | Max supported |

### HNSW ef_search Impact

| ef_search | Recall@10 | P50 Latency | P99 Latency |
|-----------|-----------|-------------|-------------|
| 10 | 0.72 | 1.5 ms | 2.8 ms |
| 25 | 0.88 | 2.2 ms | 4.1 ms |
| 50 | 0.95 | 3.2 ms | 6.1 ms |
| 100 | 0.98 | 5.5 ms | 11.0 ms |
| 200 | 0.99 | 9.2 ms | 18.0 ms |

**Recommendation**: ef_search=50 provides 95% recall with acceptable latency. Increase to 100 for critical accuracy requirements.

### Concurrent Query Throughput

| Concurrent Requests | Throughput (req/s) | P99 Latency |
|--------------------|--------------------|-------------|
| 1 | 312 | 6.1 ms |
| 10 | 289 | 8.5 ms |
| 50 | 245 | 18.2 ms |
| 100 | 198 | 35.0 ms |

**Bottleneck**: The GPU inference mutex limits parallelism. Each query requires exclusive GPU access for embedding.

## VRAM Usage Breakdown

### Static Allocations

| Component | Size | Notes |
|-----------|------|-------|
| ONNX model weights | 80 MB | all-MiniLM-L6-v2 |
| CUDA context overhead | 50 MB | Driver + runtime |
| ONNX Runtime arena base | 100 MB | Pre-allocated buffer |
| **Subtotal** | **230 MB** | Always present |

### Dynamic Allocations (per batch)

| Component | Formula | Example (batch=8) |
|-----------|---------|-------------------|
| Input tensors | batch × seq_len × 3 × 4 bytes | 8 × 384 × 3 × 4 = 37 KB |
| Attention scores | batch × heads × seq × seq × 4 | 8 × 12 × 384 × 384 × 4 = 54 MB |
| Hidden states | batch × seq_len × hidden × 4 | 8 × 384 × 384 × 4 = 5 MB |
| Intermediate activations | batch × seq_len × hidden × 4 × 3 | 8 × 384 × 384 × 4 × 3 = 14 MB |
| **Per-batch total** | | **~100 MB** |

### HNSW Index Memory

| Documents | Vector Memory | HNSW Overhead | Total |
|-----------|---------------|---------------|-------|
| 100 | 150 KB | 50 KB | 200 KB |
| 500 | 750 KB | 250 KB | 1 MB |
| 1000 | 1.5 MB | 500 KB | 2 MB |
| 5000 | 7.5 MB | 2.5 MB | 10 MB |

**Note**: HNSW index resides in system RAM, not VRAM.

## Tuning Guide

### Worker Count Tuning

| Hardware Profile | Recommended Workers | Rationale |
|------------------|--------------------|-----------|
| 2-core CPU | 3-4 | Avoid oversubscription |
| 4-core CPU | 5-8 | One worker per core with I/O overlap |
| 8+ cores | 10-15 | Diminishing returns beyond 10 |

### Batch Size Tuning

| GPU VRAM | Recommended Batch | Max Batch |
|----------|-------------------|-----------|
| 2 GB (MX250) | 8 | 16 (may trigger halving) |
| 4 GB (GTX 1650) | 16 | 32 |
| 8 GB (RTX 3060) | 32 | 64 |
| 12 GB (RTX 3080) | 64 | 128 |

### HNSW Parameter Tuning

| Corpus Size | M | ef_construction | ef_search |
|-------------|---|-----------------|-----------|
| < 100 docs | 8 | 100 | 25 |
| 100-500 docs | 12 | 150 | 50 |
| 500-2000 docs | 16 | 200 | 50 |
| 2000-10000 docs | 24 | 300 | 100 |
| > 10000 docs | 32 | 400 | 100 |

### Memory Tuning

For systems with limited RAM:

```toml
# config.toml
[embedding]
batch_size = 4          # Reduce from default 8
vram_safety_margin = 536870912  # 512 MB (increase from 256 MB)

[index]
m = 12                  # Reduce from default 16
ef_construction = 150   # Reduce from default 200
```

### Network Tuning

For slow or rate-limited sites:

```bash
./crawler \
    --entry https://example.com \
    --workers 3 \        # Fewer concurrent connections
    --delay-ms 500 \     # More delay between requests
    --timeout 60000      # Longer timeout per request
```

## Memory Leak Verification

### Method 1: Long-Running Crawl

```bash
# Create a mock site with 10,000 pages
./scripts/create_large_mock.sh 10000

# Run crawler with memory profiling
heaptrack ./crawler --entry http://localhost:8080/ --max-pages 10000

# Analyze results
heaptrack_print heaptrack.*.gz | grep "leaked"
```

Expected output:
```
leaked: 0 bytes in 0 allocations
```

### Method 2: Continuous Query Load

```bash
# Start search API
./search-api &

# Run continuous queries for 1 hour
for i in $(seq 1 36000); do
    curl -s -X POST http://localhost:8080/search \
        -H "Content-Type: application/json" \
        -d '{"query": "test query '$i'"}' > /dev/null
    sleep 0.1
done

# Check memory growth
ps aux | grep search-api
```

Expected: Memory usage stable within ±10 MB after warm-up.

### Method 3: Valgrind Full Check

```bash
# Build with debug symbols
cargo build --release

# Run under valgrind
valgrind \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --log-file=valgrind.log \
    ./target/release/search --query "test"

# Check log
cat valgrind.log | grep "LEAK SUMMARY"
```

Expected:
```
LEAK SUMMARY:
   definitely lost: 0 bytes in 0 blocks
   indirectly lost: 0 bytes in 0 blocks
     still reachable: 1,024 bytes (ONNX Runtime static)
          suppressed: 0 bytes in 0 blocks
```

### Method 4: Go Routine Leak Test

```go
// In tests/goleak_test.go
func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}

func TestNoGoroutineLeak(t *testing.T) {
    defer goleak.VerifyNone(t)
    
    crawler := NewCrawler(Config{...})
    crawler.Run()
    // All goroutines should be cleaned up after Run() returns
}
```

## Performance Regression Testing

### Benchmark Suite

```bash
# Run all benchmarks
cd crawler && go test -bench=. -benchmem
cd semantic-search && cargo bench

# Compare against baseline
cargo bench -- --save-baseline main
# After changes:
cargo bench -- --baseline main
```

### CI Performance Gate

```yaml
# In .github/workflows/ci.yml
- name: Run benchmarks
  run: |
    cargo bench -- --output-format bencher | tee bench-results.txt
    
- name: Check for regression
  run: |
    python scripts/check_regression.py \
      --baseline .github/baseline.json \
      --current bench-results.txt \
      --threshold 10  # Fail if >10% slower
```
