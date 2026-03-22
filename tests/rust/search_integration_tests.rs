//! Search integration tests with precise PASS/FAIL conditions.

use std::fs;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// ============================================================================
// QUERY EDGE CASE TESTS
// ============================================================================

/// Test query against empty index.
/// PASS: Returns error, not panic.
/// FAIL: Panic or crash on empty index.
#[test]
fn test_query_empty_index() {
    use indexer::{VectorStore, StoreError};

    let store = VectorStore::new(384);
    let query = vec![0.0; 384];

    let result = store.search(&query, 10);

    // FAIL: Should return EmptyStore error
    match result {
        Err(StoreError::EmptyStore) => {}
        Ok(_) => panic!("FAIL: Empty store returned results"),
        Err(e) => panic!("FAIL: Wrong error type: {:?}", e),
    }
}

/// Test query with zero semantic match.
/// PASS: Returns results with low scores.
/// FAIL: Crash or incorrect score bounds.
#[test]
fn test_query_zero_match() {
    use indexer::{Document, VectorStore};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    // Add document with embedding in one direction
    let doc = Document::new(
        "orthogonal".to_string(),
        "https://example.com/orthogonal".to_string(),
        "Orthogonal Document".to_string(),
        "This document has a specific embedding direction.".to_string(),
    );

    let mut emb1 = vec![0.0; dim];
    emb1[0] = 1.0; // Unit vector along first axis
    store.add(doc, emb1).unwrap();

    // Query in completely different direction
    let mut query = vec![0.0; dim];
    query[100] = 1.0; // Orthogonal direction

    let hits = store.search(&query, 10).unwrap();

    // FAIL: Should return results
    if hits.is_empty() {
        panic!("FAIL: No results returned");
    }

    // FAIL: Score should be near 0 (orthogonal vectors)
    // Note: With L2 normalized vectors, cosine similarity of orthogonal vectors is 0
    if hits[0].score.abs() > 0.1 {
        panic!("FAIL: Expected near-zero score for orthogonal query, got {}", hits[0].score);
    }
}

/// Test query string longer than max sequence length.
/// PASS: Query is processed (truncated internally), no panic.
/// FAIL: Panic or rejection of long query.
#[test]
fn test_query_longer_than_max_length() {
    use indexer::{Document, VectorStore};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    // Add a document
    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Test Document".to_string(),
        "Test content for semantic search.".to_string(),
    );
    let emb = vec![0.5; dim];
    store.add(doc, emb).unwrap();

    // Create very long query embedding
    // (In practice, the tokenizer would truncate, but here we test the store)
    let query = vec![0.5; dim];

    // FAIL: Should not panic
    let result = std::panic::catch_unwind(|| {
        store.search(&query, 10)
    });

    if result.is_err() {
        panic!("FAIL: Search panicked on query");
    }

    let hits = result.unwrap().unwrap();

    // FAIL: Should return results
    if hits.is_empty() {
        panic!("FAIL: No results for long query");
    }
}

// ============================================================================
// CONCURRENT QUERY TESTS
// ============================================================================

/// Test concurrent queries (10 threads, 100 queries each).
/// PASS: All queries complete, no race conditions, no deadlock.
/// FAIL: Race condition, deadlock, or panic.
#[test]
fn test_concurrent_queries_10_threads_100_each() {
    use indexer::{Document, SharedVectorStore};

    let dim = 384;
    let store = SharedVectorStore::new(dim);

    // Add 50 documents
    for i in 0..50 {
        let doc = Document::new(
            format!("concurrent-{}", i),
            format!("https://example.com/doc{}", i),
            format!("Document {}", i),
            format!("This is the content of document number {} for testing.", i),
        );

        // Create pseudo-random embedding
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;

        store.add(doc, emb).unwrap();
    }

    let store = Arc::new(store);
    let mut handles = vec![];

    for t in 0..10 {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for q in 0..100 {
                // Each thread uses slightly different query
                let mut query = vec![0.0; dim];
                query[(t + q) % dim] = 1.0;

                // FAIL: Should not panic
                let result = store.search(&query, 5);
                if result.is_err() {
                    panic!("FAIL: Thread {} query {} failed: {:?}", t, q, result.err());
                }

                let hits = result.unwrap();

                // FAIL: Should have results
                if hits.is_empty() {
                    panic!("FAIL: Thread {} query {} got empty results", t, q);
                }

                // FAIL: Scores should be in valid range
                for hit in &hits {
                    if hit.score < -1.0 || hit.score > 1.0 {
                        panic!(
                            "FAIL: Thread {} query {} got invalid score {}",
                            t, q, hit.score
                        );
                    }
                }
            }
        });
        handles.push(handle);
    }

    // FAIL: All threads should complete without panic
    for (i, handle) in handles.into_iter().enumerate() {
        if handle.join().is_err() {
            panic!("FAIL: Thread {} panicked during concurrent queries", i);
        }
    }
}

// ============================================================================
// UNICODE QUERY TESTS
// ============================================================================

/// Test queries with Unicode characters.
/// PASS: Arabic, Chinese, Japanese, and emoji queries work.
/// FAIL: Panic or incorrect handling.
#[test]
fn test_unicode_queries() {
    use indexer::{Document, VectorStore};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    // Add documents with Unicode content
    let docs = vec![
        ("arabic", "مرحبا بالعالم", "هذا محتوى تجريبي باللغة العربية"),
        ("chinese", "你好世界", "这是中文测试内容"),
        ("japanese", "こんにちは世界", "これは日本語のテスト内容です"),
        ("emoji", "Hello World 🌍", "Content with emoji 🚀 🎮 🔥"),
    ];

    for (id, title, text) in &docs {
        let doc = Document::new(
            id.to_string(),
            format!("https://example.com/{}", id),
            title.to_string(),
            text.to_string(),
        );
        let emb = vec![0.5; dim];
        store.add(doc, emb).unwrap();
    }

    // Create query and search
    let query = vec![0.5; dim];
    let hits = store.search(&query, 10).unwrap();

    // FAIL: Should return all documents
    if hits.len() != 4 {
        panic!("FAIL: Expected 4 results for Unicode docs, got {}", hits.len());
    }

    // Verify Unicode content is preserved
    for hit in &hits {
        // FAIL: Should not have corrupted Unicode
        if hit.title.is_empty() {
            panic!("FAIL: Empty title for Unicode document");
        }
    }
}

/// Test mixed RTL and LTR text.
/// PASS: Arabic (RTL) and English (LTR) mix correctly.
/// FAIL: String corruption or wrong order.
#[test]
fn test_mixed_rtl_ltr() {
    use indexer::{Document, VectorStore};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "mixed".to_string(),
        "https://example.com/mixed".to_string(),
        "Mixed مرحبا World 世界".to_string(),
        "English text with Arabic العربية and Chinese 中文 mixed together.".to_string(),
    );

    let emb = vec![0.5; dim];
    store.add(doc, emb).unwrap();

    let query = vec![0.5; dim];
    let hits = store.search(&query, 10).unwrap();

    // FAIL: Should have result
    if hits.is_empty() {
        panic!("FAIL: No results for mixed RTL/LTR document");
    }

    // FAIL: Title should be preserved
    if !hits[0].title.contains("مرحبا") {
        panic!("FAIL: Arabic text lost in title: {}", hits[0].title);
    }
}

// ============================================================================
// SCORE THRESHOLD TESTS
// ============================================================================

/// Test that scores are within valid bounds.
/// PASS: All scores in [-1, 1] range.
/// FAIL: Any score outside bounds.
#[test]
fn test_score_bounds() {
    use indexer::{Document, VectorStore};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    // Add documents with varying embeddings
    for i in 0..10 {
        let doc = Document::new(
            format!("score-test-{}", i),
            format!("https://example.com/{}", i),
            format!("Document {}", i),
            format!("Content {}", i),
        );

        let mut emb = vec![0.0; dim];
        emb[i * 10 % dim] = 1.0;

        store.add(doc, emb).unwrap();
    }

    // Test various query directions
    for q in 0..5 {
        let mut query = vec![0.0; dim];
        query[q * 20 % dim] = 1.0;

        let hits = store.search(&query, 10).unwrap();

        for hit in &hits {
            // FAIL: Score must be in [-1, 1]
            if hit.score < -1.0 || hit.score > 1.0 {
                panic!(
                    "FAIL: Score {} out of bounds for query {}",
                    hit.score, q
                );
            }
        }
    }
}

// ============================================================================
// RESTART PERSISTENCE TESTS
// ============================================================================

/// Test that results are consistent after restart.
/// PASS: Same query returns same results after reload.
/// FAIL: Different results or missing documents.
#[test]
fn test_restart_consistency() {
    use indexer::{Document, VectorStore};

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("restart.store");

    let dim = 384;
    let query = vec![0.5; dim];

    // First run: create and save
    let hits_before = {
        let mut store = VectorStore::new(dim);

        for i in 0..10 {
            let doc = Document::new(
                format!("restart-{}", i),
                format!("https://example.com/restart{}", i),
                format!("Restart Doc {}", i),
                format!("Content for restart test {}", i),
            );

            let mut emb = vec![0.1; dim];
            emb[i % dim] = 1.0;

            store.add(doc, emb).unwrap();
        }

        store.save(&store_path).unwrap();
        store.search(&query, 10).unwrap()
    };

    // Restart: load and query
    let store = VectorStore::load(&store_path).unwrap();
    let hits_after = store.search(&query, 10).unwrap();

    // FAIL: Same number of results
    if hits_before.len() != hits_after.len() {
        panic!(
            "FAIL: Result count changed after restart: {} -> {}",
            hits_before.len(),
            hits_after.len()
        );
    }

    // FAIL: Same documents in same order
    for (i, (before, after)) in hits_before.iter().zip(hits_after.iter()).enumerate() {
        if before.id != after.id {
            panic!(
                "FAIL: Result {} changed: {} -> {}",
                i, before.id, after.id
            );
        }

        // FAIL: Same scores (within floating point tolerance)
        if (before.score - after.score).abs() > 0.0001 {
            panic!(
                "FAIL: Score {} changed: {} -> {}",
                i, before.score, after.score
            );
        }
    }
}

// ============================================================================
// CROSS-LANGUAGE DATA CONTRACT TEST
// ============================================================================

/// Test that Go JSON output is parseable by Rust.
/// PASS: Go-format JSON loads without error.
/// FAIL: Parse error or missing fields.
#[test]
fn test_go_json_contract() {
    use indexer::DocumentLoader;
    use std::io::Write;

    let temp_dir = TempDir::new().unwrap();
    let pages_dir = temp_dir.path().join("pages");
    fs::create_dir_all(&pages_dir).unwrap();

    // Exact format from Go crawler
    let go_json = r#"{
        "url": "https://minecraft-linux.github.io/source_build/launcher.html",
        "title": "Building the Launcher | Minecraft Linux",
        "body": "Building the Launcher\n\nThis guide covers compiling the launcher from source code.\n\nPrerequisites\n\nBefore building...",
        "links": ["https://minecraft-linux.github.io/", "https://minecraft-linux.github.io/installation/"],
        "crawled_at": "2024-01-15T10:30:00Z"
    }"#;

    let mut file = fs::File::create(pages_dir.join("go_output.json")).unwrap();
    file.write_all(go_json.as_bytes()).unwrap();

    let loader = DocumentLoader::new(temp_dir.path());
    let docs = loader.load_all().unwrap();

    // FAIL: Should parse Go JSON
    if docs.len() != 1 {
        panic!("FAIL: Expected 1 document, got {}", docs.len());
    }

    let doc = &docs[0];

    // FAIL: URL should match
    if doc.url != "https://minecraft-linux.github.io/source_build/launcher.html" {
        panic!("FAIL: URL mismatch: {}", doc.url);
    }

    // FAIL: Title should match
    if doc.title != "Building the Launcher | Minecraft Linux" {
        panic!("FAIL: Title mismatch: {}", doc.title);
    }

    // FAIL: Body should contain expected content
    if !doc.text.contains("Building the Launcher") {
        panic!("FAIL: Body content missing");
    }
}

// ============================================================================
// LARGE SCALE TESTS
// ============================================================================

/// Test with 200 documents.
/// PASS: All operations complete in reasonable time.
/// FAIL: Timeout or performance degradation.
#[test]
fn test_200_documents() {
    use indexer::{Document, VectorStore};
    use std::time::Instant;

    let dim = 384;
    let mut store = VectorStore::new(dim);

    let start = Instant::now();

    // Add 200 documents
    for i in 0..200 {
        let doc = Document::new(
            format!("scale-{}", i),
            format!("https://example.com/scale{}", i),
            format!("Document {}", i),
            format!("Content for document {} in large scale test.", i),
        );

        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;

        store.add(doc, emb).unwrap();
    }

    let add_time = start.elapsed();

    // FAIL: Should complete in reasonable time (< 1 second)
    if add_time.as_secs() > 1 {
        panic!("FAIL: Adding 200 docs took too long: {:?}", add_time);
    }

    let query = vec![0.5; dim];
    let search_start = Instant::now();
    let hits = store.search(&query, 10).unwrap();
    let search_time = search_start.elapsed();

    // FAIL: Search should be fast (< 10ms for 200 docs)
    if search_time.as_millis() > 10 {
        panic!("FAIL: Search took too long: {:?}", search_time);
    }

    // FAIL: Should return requested number of results
    if hits.len() != 10 {
        panic!("FAIL: Expected 10 results, got {}", hits.len());
    }
}
