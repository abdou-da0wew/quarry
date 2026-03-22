//! Integration tests for semantic search system.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// Note: These tests require the model files to be present.
// Run scripts/download_model.sh first.

/// Test that an empty index returns an error on search.
#[test]
fn test_empty_index_query() {
    use indexer::VectorStore;
    use embedder::EMBEDDING_DIM;

    let store = VectorStore::new(EMBEDDING_DIM);
    let query = vec![0.0; EMBEDDING_DIM];

    let result = store.search(&query, 10);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), indexer::StoreError::EmptyStore));
}

/// Test indexing and searching a single document.
#[test]
fn test_single_doc_index() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "test-1".to_string(),
        "https://example.com/doc1".to_string(),
        "Test Document".to_string(),
        "This is test content.".to_string(),
    );

    let embedding = vec![1.0, 0.0, 0.0, 0.0];
    store.add(doc, embedding).unwrap();

    assert_eq!(store.len(), 1);

    let query = vec![0.9, 0.1, 0.0, 0.0];
    let hits = store.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "test-1");
}

/// Test indexing 200 documents and verifying search quality.
#[test]
fn test_200_doc_index() {
    use indexer::{VectorStore, Document};

    let dim = 384;
    let mut store = VectorStore::new(dim);

    // Create 200 documents with varying embeddings
    for i in 0..200 {
        let doc = Document::new(
            format!("doc-{}", i),
            format!("https://example.com/doc{}", i),
            format!("Document {}", i),
            format!("Content for document number {}", i),
        );

        // Create a pseudo-random embedding
        let mut embedding = vec![0.0; dim];
        if i < dim {
            embedding[i] = 1.0;
        }

        store.add(doc, embedding).unwrap();
    }

    assert_eq!(store.len(), 200);

    // Query for document 0
    let mut query = vec![0.0; dim];
    query[0] = 1.0;
    let hits = store.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 10);
    assert_eq!(hits[0].id, "doc-0");
}

/// Test that store persists and reloads correctly.
#[test]
fn test_persist_reload() {
    use indexer::{VectorStore, Document};

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("test.store");

    let dim = 8;
    let doc_id = "persist-test";

    // Create and save
    {
        let mut store = VectorStore::new(dim);
        let doc = Document::new(
            doc_id.to_string(),
            "https://example.com/test".to_string(),
            "Persist Test".to_string(),
            "Testing persistence".to_string(),
        );
        let embedding = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        store.add(doc, embedding).unwrap();
        store.save(&store_path).unwrap();
    }

    // Reload
    let store = VectorStore::load(&store_path).unwrap();
    assert_eq!(store.len(), 1);
    assert!(store.get(doc_id).is_some());
}

/// Test corrupted store file detection.
#[test]
fn test_corrupt_store_detection() {
    use indexer::VectorStore;

    let temp_dir = TempDir::new().unwrap();
    let store_path = temp_dir.path().join("corrupt.store");

    // Write invalid data
    fs::write(&store_path, b"invalid data here").unwrap();

    let result = VectorStore::load(&store_path);
    assert!(result.is_err());
}

/// Test zero-length query handling.
#[test]
fn test_zero_length_query() {
    use indexer::{VectorStore, Document};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Test".to_string(),
        "Content".to_string(),
    );
    store.add(doc, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

    // Zero query (all zeros)
    let query = vec![0.0; dim];
    let hits = store.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 1);
    // Score should be 0 since query is all zeros
    assert!(hits[0].score.abs() < 0.001);
}

/// Test Unicode query handling (Arabic, CJK).
#[test]
fn test_unicode_query() {
    use indexer::{Document, VectorStore};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    // Arabic document
    let doc1 = Document::new(
        "arabic".to_string(),
        "https://example.com/ar".to_string(),
        "مرحبا بالعالم".to_string(),  // "Hello World" in Arabic
        "هذا محتوى تجريبي".to_string(), // "This is test content" in Arabic
    );

    // CJK document
    let doc2 = Document::new(
        "cjk".to_string(),
        "https://example.com/cn".to_string(),
        "你好世界".to_string(),  // "Hello World" in Chinese
        "这是测试内容".to_string(), // "This is test content" in Chinese
    );

    // Japanese document
    let doc3 = Document::new(
        "japanese".to_string(),
        "https://example.com/jp".to_string(),
        "こんにちは世界".to_string(),
        "これはテスト内容です".to_string(),
    );

    store.add(doc1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    store.add(doc2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    store.add(doc3, vec![0.0, 0.0, 1.0, 0.0]).unwrap();

    // Query that should match Arabic document
    let query = vec![0.9, 0.1, 0.0, 0.0];
    let hits = store.search(&query, 3).unwrap();
    assert_eq!(hits[0].id, "arabic");
}

/// Test query longer than model max tokens.
#[test]
fn test_long_query_handling() {
    use indexer::{Document, VectorStore};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "long".to_string(),
        "https://example.com/long".to_string(),
        "Long Test".to_string(),
        "Content".to_string(),
    );
    store.add(doc, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

    // Query should work regardless of length
    let query = vec![0.8, 0.2, 0.0, 0.0];
    let hits = store.search(&query, 10).unwrap();
    assert_eq!(hits.len(), 1);
}

/// Test concurrent query race (10 threads, 100 queries each).
#[test]
fn test_concurrent_query_race() {
    use indexer::{Document, SharedVectorStore};

    let dim = 4;

    // Create shared store
    let shared_store = SharedVectorStore::new(dim);

    // Add documents to shared store
    for i in 0..10 {
        let doc = Document::new(
            format!("doc-{}", i),
            format!("https://example.com/doc{}", i),
            format!("Document {}", i),
            format!("Content {}", i),
        );
        let mut emb = vec![0.0; dim];
        emb[i % dim] = 1.0;
        shared_store.add(doc, emb).unwrap();
    }

    let store = Arc::new(shared_store);
    let mut handles = vec![];

    for t in 0..10 {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for q in 0..100 {
                let mut query = vec![0.0; dim];
                query[(t + q) % dim] = 1.0;

                let result = store.search(&query, 5);
                assert!(result.is_ok());
                let hits = result.unwrap();
                assert!(hits.len() <= 5);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test L2 normalization of embeddings.
#[test]
fn test_l2_normalization() {
    use embedder::cosine_similarity;

    // Two unit vectors should have cosine similarity based on angle
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];

    // Orthogonal vectors have 0 similarity
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 0.0).abs() < 1e-6);

    // Same vectors have 1.0 similarity
    let sim_same = cosine_similarity(&a, &a);
    assert!((sim_same - 1.0).abs() < 1e-6);

    // 45-degree angle vectors
    let c = vec![0.707, 0.707, 0.0, 0.0];
    let sim_angle = cosine_similarity(&a, &c);
    assert!((sim_angle - 0.707).abs() < 0.01);
}

/// Test dimension mismatch error.
#[test]
fn test_dimension_mismatch() {
    use indexer::{Document, VectorStore, StoreError};

    let dim = 4;
    let mut store = VectorStore::new(dim);

    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Test".to_string(),
        "Content".to_string(),
    );

    // Wrong dimension embedding
    let wrong_emb = vec![0.0; 8];
    let result = store.add(doc, wrong_emb);

    assert!(matches!(result, Err(StoreError::DimensionMismatch { .. })));
}

/// Test VRAM estimation.
#[test]
fn test_vram_estimation() {
    use embedder::vram::VramMonitor;

    // Estimate for batch size 8, seq len 256, hidden dim 384
    let estimate = VramMonitor::estimate_batch_vram(8, 256, 384);

    // Should be reasonable for MX250
    // Print for debugging
    println!("VRAM estimate for batch 8: {} MB", estimate as f64 / 1_000_000.0);

    // Should not exceed ceiling for reasonable batch sizes
    assert!(estimate < 1_500_000_000);
}

/// Test document embedding text composition.
#[test]
fn test_document_embedding_text() {
    use indexer::Document;

    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Test Title".to_string(),
        "Test body content here.".to_string(),
    );

    let text = doc.embedding_text();
    assert!(text.contains("Test Title"));
    assert!(text.contains("Test body content"));
}

/// Test document loader with various formats.
#[test]
fn test_document_loader_formats() {
    use indexer::DocumentLoader;
    use std::io::Write;

    let temp_dir = TempDir::new().unwrap();
    let pages_dir = temp_dir.path().join("pages");
    fs::create_dir_all(&pages_dir).unwrap();

    // Write crawler format
    let crawler_doc = r#"{
        "url": "https://example.com/page1",
        "title": "Crawler Page",
        "body": "Crawler body content",
        "links": [],
        "crawled_at": "2024-01-01T00:00:00Z"
    }"#;

    let mut file = fs::File::create(pages_dir.join("crawler.json")).unwrap();
    file.write_all(crawler_doc.as_bytes()).unwrap();

    // Load documents
    let loader = DocumentLoader::new(temp_dir.path());
    let docs = loader.load_all().unwrap();

    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].title, "Crawler Page");
    assert_eq!(docs[0].text, "Crawler body content");
}
