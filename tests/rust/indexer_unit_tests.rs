//! Indexer unit tests with precise PASS/FAIL conditions.

use std::fs;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// ============================================================================
// VECTOR STORE CRUD TESTS
// ============================================================================

/// Test adding a single document.
/// PASS: Document added, count is 1, can retrieve by ID.
/// FAIL: Count != 1 or retrieval fails.
#[test]
fn test_store_add_single() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "test-1".to_string(),
        "https://example.com/test".to_string(),
        "Test Document".to_string(),
        "This is test content.".to_string(),
    );

    let embedding = vec![1.0, 0.0, 0.0, 0.0];

    // FAIL: Add should succeed
    if let Err(e) = store.add(doc.clone(), embedding.clone()) {
        panic!("FAIL: Failed to add document: {:?}", e);
    }

    // FAIL: Count should be 1
    if store.len() != 1 {
        panic!("FAIL: Expected count 1, got {}", store.len());
    }

    // FAIL: Should be able to retrieve document
    let retrieved = store.get("test-1");
    if retrieved.is_none() {
        panic!("FAIL: Could not retrieve added document");
    }

    let retrieved = retrieved.unwrap();
    if retrieved.title != "Test Document" {
        panic!("FAIL: Retrieved document has wrong title: {}", retrieved.title);
    }
}

/// Test adding multiple documents.
/// PASS: All documents added, count matches.
/// FAIL: Wrong count or missing documents.
#[test]
fn test_store_add_multiple() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    for i in 0..10 {
        let doc = Document::new(
            format!("doc-{}", i),
            format!("https://example.com/doc{}", i),
            format!("Document {}", i),
            format!("Content for document {}", i),
        );

        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;

        // FAIL: Each add should succeed
        if let Err(e) = store.add(doc, emb) {
            panic!("FAIL: Failed to add document {}: {:?}", i, e);
        }
    }

    // FAIL: Count should be 10
    if store.len() != 10 {
        panic!("FAIL: Expected count 10, got {}", store.len());
    }

    // FAIL: All documents should be retrievable
    for i in 0..10 {
        if store.get(&format!("doc-{}", i)).is_none() {
            panic!("FAIL: Could not retrieve document {}", i);
        }
    }
}

/// Test search returns correct order.
/// PASS: Results sorted by similarity descending.
/// FAIL: Wrong order or wrong documents.
#[test]
fn test_store_search_order() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    // Add documents with known embeddings
    let docs = vec![
        ("doc-1", vec![1.0, 0.0, 0.0, 0.0]),  // Most similar to query
        ("doc-2", vec![0.7, 0.3, 0.0, 0.0]),  // Second
        ("doc-3", vec![0.0, 1.0, 0.0, 0.0]),  // Third
        ("doc-4", vec![0.0, 0.0, 1.0, 0.0]),  // Fourth
    ];

    for (id, emb) in &docs {
        let doc = Document::new(
            id.to_string(),
            format!("https://example.com/{}", id),
            format!("Document {}", id),
            format!("Content for {}", id),
        );
        store.add(doc, emb.clone()).unwrap();
    }

    // Query most similar to doc-1
    let query = vec![0.9, 0.1, 0.0, 0.0];
    let hits = store.search(&query, 10).unwrap();

    // FAIL: Should return all 4 documents
    if hits.len() != 4 {
        panic!("FAIL: Expected 4 results, got {}", hits.len());
    }

    // FAIL: First result should be doc-1 (most similar)
    if hits[0].id != "doc-1" {
        panic!("FAIL: First result should be doc-1, got {}", hits[0].id);
    }

    // FAIL: Results should be sorted by score descending
    for i in 0..hits.len() - 1 {
        if hits[i].score < hits[i + 1].score {
            panic!(
                "FAIL: Results not sorted: {} ({}) < {} ({})",
                hits[i].id, hits[i].score,
                hits[i + 1].id, hits[i + 1].score
            );
        }
    }
}

/// Test search with top_k limit.
/// PASS: Exactly top_k results returned.
/// FAIL: Wrong number of results.
#[test]
fn test_store_search_top_k() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    // Add 20 documents
    for i in 0..20 {
        let doc = Document::new(
            format!("doc-{}", i),
            format!("https://example.com/doc{}", i),
            format!("Document {}", i),
            format!("Content {}", i),
        );
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        store.add(doc, emb).unwrap();
    }

    let query = vec![1.0, 0.0, 0.0, 0.0];
    
    for k in [1, 5, 10] {
        let hits = store.search(&query, k).unwrap();

        // FAIL: Should return exactly k results
        if hits.len() != k {
            panic!("FAIL: Expected {} results, got {}", k, hits.len());
        }
    }
}

/// Test search on empty store.
/// PASS: Returns EmptyStore error, not panic.
/// FAIL: Panic or wrong error type.
#[test]
fn test_store_search_empty() {
    use indexer::{VectorStore, StoreError};

    let store = VectorStore::new(4);
    let query = vec![1.0, 0.0, 0.0, 0.0];

    let result = store.search(&query, 10);

    // FAIL: Should return EmptyStore error
    match result {
        Err(StoreError::EmptyStore) => {}
        Ok(_) => panic!("FAIL: Empty store returned results"),
        Err(e) => panic!("FAIL: Wrong error type: {:?}", e),
    }
}

// ============================================================================
// PERSISTENCE TESTS
// ============================================================================

/// Test saving and loading store.
/// PASS: Loaded store has same documents and embeddings.
/// FAIL: Data loss or corruption.
#[test]
fn test_store_persistence() {
    use indexer::{VectorStore, Document};

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("test.store");

    let dim = 8;
    let docs_to_add = 5;

    // Create and save
    {
        let mut store = VectorStore::new(dim);

        for i in 0..docs_to_add {
            let doc = Document::new(
                format!("persist-{}", i),
                format!("https://example.com/persist{}", i),
                format!("Persisted Doc {}", i),
                format!("Content {}", i),
            );
            let emb: Vec<f32> = (0..dim).map(|j| (i * j) as f32).collect();
            store.add(doc, emb).unwrap();
        }

        // FAIL: Save should succeed
        if let Err(e) = store.save(&store_path) {
            panic!("FAIL: Failed to save store: {:?}", e);
        }
    }

    // FAIL: File should exist
    if !store_path.exists() {
        panic!("FAIL: Store file not created at {:?}", store_path);
    }

    // Load
    let loaded = VectorStore::load(&store_path);

    // FAIL: Load should succeed
    if let Err(e) = loaded {
        panic!("FAIL: Failed to load store: {:?}", e);
    }

    let loaded = loaded.unwrap();

    // FAIL: Count should match
    if loaded.len() != docs_to_add {
        panic!("FAIL: Expected {} docs, got {}", docs_to_add, loaded.len());
    }

    // FAIL: All documents should be retrievable
    for i in 0..docs_to_add {
        let id = format!("persist-{}", i);
        if loaded.get(&id).is_none() {
            panic!("FAIL: Document {} not found after reload", id);
        }

        // Verify embedding
        let emb = loaded.get_embedding(&id);
        if emb.is_none() {
            panic!("FAIL: Embedding for {} not found after reload", id);
        }

        let emb = emb.unwrap();
        for j in 0..dim {
            let expected = (i * j) as f32;
            if (emb[j] - expected).abs() > 0.0001 {
                panic!(
                    "FAIL: Embedding mismatch at [{},{}]: expected {}, got {}",
                    i, j, expected, emb[j]
                );
            }
        }
    }
}

/// Test corruption detection.
/// PASS: Corrupted file is detected, error returned.
/// FAIL: Corrupted file loads without error.
#[test]
fn test_store_corruption_detection() {
    use indexer::{VectorStore, StoreError};

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("corrupt.store");

    // Write garbage data
    fs::write(&store_path, b"this is not a valid store file").unwrap();

    let result = VectorStore::load(&store_path);

    // FAIL: Should return error, not succeed
    if result.is_ok() {
        panic!("FAIL: Corrupted store loaded successfully");
    }

    let err = result.unwrap_err();

    // FAIL: Should be specific error type (not generic IO error)
    match err {
        StoreError::Corrupted(_) | StoreError::ChecksumMismatch => {}
        _ => panic!("FAIL: Wrong error type for corruption: {:?}", err),
    }
}

/// Test version mismatch handling.
/// PASS: Store with wrong version is rejected.
/// FAIL: Wrong version accepted.
#[test]
fn test_store_version_check() {
    use indexer::StoreError;

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("version.store");

    // Write file with wrong version
    let content = b"VST1\x02\x00\x00\x00"; // Version 2 instead of 1
    fs::write(&store_path, content).unwrap();

    let result = indexer::VectorStore::load(&store_path);

    // Should fail (either version mismatch or corrupted)
    if result.is_ok() {
        panic!("FAIL: Wrong version was accepted");
    }
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

/// Test concurrent read access.
/// PASS: Multiple threads can read simultaneously.
/// FAIL: Deadlock or race condition.
#[test]
fn test_store_concurrent_read() {
    use indexer::{Document, SharedVectorStore};

    let dim = 4;
    let store = SharedVectorStore::new(dim);

    // Add some documents
    for i in 0..10 {
        let doc = Document::new(
            format!("concurrent-{}", i),
            format!("https://example.com/{}", i),
            format!("Doc {}", i),
            format!("Content {}", i),
        );
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        store.add(doc, emb).unwrap();
    }

    let store = Arc::new(store);
    let mut handles = vec![];

    // Spawn 10 threads, each doing 100 searches
    for t in 0..10 {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for q in 0..100 {
                let mut query = vec![0.0; dim];
                query[(t + q) % dim] = 1.0;

                let result = store.search(&query, 5);

                // FAIL: Search should not panic
                if result.is_err() {
                    panic!("FAIL: Search failed in thread {}: {:?}", t, result.err());
                }

                let hits = result.unwrap();
                
                // FAIL: Should get results
                if hits.is_empty() {
                    panic!("FAIL: Thread {} got empty results", t);
                }
            }
        });
        handles.push(handle);
    }

    // FAIL: All threads should complete
    for (i, handle) in handles.into_iter().enumerate() {
        if handle.join().is_err() {
            panic!("FAIL: Thread {} panicked", i);
        }
    }
}

/// Test concurrent write access.
/// PASS: All writes succeed, no data loss.
/// FAIL: Race condition or lost updates.
#[test]
fn test_store_concurrent_write() {
    use indexer::{Document, SharedVectorStore};

    let dim = 4;
    let store = Arc::new(SharedVectorStore::new(dim));

    let mut handles = vec![];

    // Spawn 10 threads, each adding 10 documents
    for t in 0..10 {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..10 {
                let id = format!("thread-{}-doc-{}", t, i);
                let doc = Document::new(
                    id.clone(),
                    format!("https://example.com/{}/{}", t, i),
                    format!("Thread {} Doc {}", t, i),
                    format!("Content from thread {}", t),
                );
                let emb = vec![t as f32 / 10.0; dim];
                
                // FAIL: Add should succeed
                if let Err(e) = store.add(doc, emb) {
                    panic!("FAIL: Thread {} failed to add document {}: {:?}", t, i, e);
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // FAIL: Should have exactly 100 documents
    let count = store.len();
    if count != 100 {
        panic!("FAIL: Expected 100 documents after concurrent writes, got {}", count);
    }
}

// ============================================================================
// DOCUMENT LOADER TESTS
// ============================================================================

/// Test loading documents from crawler output.
/// PASS: All valid JSON files are loaded.
/// FAIL: Missing files or parse errors.
#[test]
fn test_document_loader() {
    use indexer::DocumentLoader;
    use std::io::Write;

    let temp_dir = TempDir::new().unwrap();
    let pages_dir = temp_dir.path().join("pages");
    fs::create_dir_all(&pages_dir).unwrap();

    // Create test documents in crawler format
    let docs = vec![
        (r#"{"url":"https://example.com/1","title":"Doc 1","body":"Content 1","links":[],"crawled_at":"2024-01-01T00:00:00Z"}"#, "doc1.json"),
        (r#"{"url":"https://example.com/2","title":"Doc 2","body":"Content 2","links":[],"crawled_at":"2024-01-01T00:00:00Z"}"#, "doc2.json"),
        (r#"{"url":"https://example.com/3","title":"Doc 3","body":"Content 3","links":[],"crawled_at":"2024-01-01T00:00:00Z"}"#, "doc3.json"),
    ];

    for (content, filename) in &docs {
        let mut file = fs::File::create(pages_dir.join(filename)).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    let loader = DocumentLoader::new(temp_dir.path());
    let loaded = loader.load_all().unwrap();

    // FAIL: Should load all 3 documents
    if loaded.len() != 3 {
        panic!("FAIL: Expected 3 documents, got {}", loaded.len());
    }

    // FAIL: Documents should have correct content
    for doc in &loaded {
        if doc.url.is_empty() {
            panic!("FAIL: Loaded document has empty URL");
        }
        if doc.text.is_empty() {
            panic!("FAIL: Loaded document has empty text");
        }
    }
}

/// Test handling of malformed JSON.
/// PASS: Malformed files are skipped, others are loaded.
/// FAIL: Loader crashes or fails entirely.
#[test]
fn test_document_loader_malformed() {
    use indexer::DocumentLoader;
    use std::io::Write;

    let temp_dir = TempDir::new().unwrap();
    let pages_dir = temp_dir.path().join("pages");
    fs::create_dir_all(&pages_dir).unwrap();

    // Mix of valid and invalid files
    let files = vec![
        (r#"{"url":"https://example.com/valid","title":"Valid","body":"Content","links":[],"crawled_at":"2024-01-01T00:00:00Z"}"#, "valid.json"),
        (r#"{"url":"https://example.com/incomplete"#, "incomplete.json"), // Truncated
        (r#"not json at all"#, "notjson.json"),
        (r#"{"url":"https://example.com/valid2","title":"Valid2","body":"Content2","links":[],"crawled_at":"2024-01-01T00:00:00Z"}"#, "valid2.json"),
    ];

    for (content, filename) in &files {
        let mut file = fs::File::create(pages_dir.join(filename)).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }

    let loader = DocumentLoader::new(temp_dir.path());
    let loaded = loader.load_all().unwrap();

    // FAIL: Should load only valid files (2 out of 4)
    if loaded.len() != 2 {
        panic!("FAIL: Expected 2 valid documents, got {}", loaded.len());
    }
}

/// Test embedding text composition.
/// PASS: Title + body combined correctly.
/// FAIL: Missing title or body in output.
#[test]
fn test_document_embedding_text() {
    use indexer::Document;

    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Important Title".to_string(),
        "Body content here.".to_string(),
    );

    let text = doc.embedding_text();

    // FAIL: Should contain title
    if !text.contains("Important Title") {
        panic!("FAIL: Embedding text missing title");
    }

    // FAIL: Should contain body
    if !text.contains("Body content") {
        panic!("FAIL: Embedding text missing body");
    }
}
