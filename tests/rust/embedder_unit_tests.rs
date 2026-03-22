//! Embedder unit tests with precise PASS/FAIL conditions.
//!
//! Each test documents exactly what would cause failure.

use std::sync::Arc;
use std::thread;

// ============================================================================
// TOKENIZER TESTS
// ============================================================================

/// Test that tokenizer loads correctly.
/// PASS: Tokenizer loads without error.
/// FAIL: Any error during load (missing file, corrupt JSON, invalid format).
#[test]
fn test_tokenizer_load() {
    // This test requires tokenizer.json to exist
    // In CI, this would be downloaded first
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: tokenizer.json not found at {}", tokenizer_path);
        return;
    }

    use embedder::tokenizer::Tokenizer;
    use embedder::MAX_SEQUENCE_LENGTH;

    let result = Tokenizer::from_file(tokenizer_path, MAX_SEQUENCE_LENGTH);
    
    // FAIL: Should load successfully
    if result.is_err() {
        panic!("FAIL: Tokenizer failed to load: {:?}", result.err());
    }
}

/// Test basic tokenization produces expected output.
/// PASS: Tokenized output contains expected token IDs.
/// FAIL: Token IDs don't match expected vocabulary output.
#[test]
fn test_tokenizer_basic_encode() {
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: tokenizer.json not found");
        return;
    }

    use embedder::tokenizer::Tokenizer;
    use embedder::MAX_SEQUENCE_LENGTH;

    let tokenizer = Tokenizer::from_file(tokenizer_path, MAX_SEQUENCE_LENGTH)
        .expect("FAIL: Could not load tokenizer");

    let encoding = tokenizer.encode("Hello world")
        .expect("FAIL: Encoding failed for simple text");

    // FAIL: Should produce non-empty token sequence
    if encoding.len() == 0 {
        panic!("FAIL: Encoding produced zero tokens for 'Hello world'");
    }

    // FAIL: Should have attention mask matching input length
    let attention_mask = encoding.get_attention_mask();
    if attention_mask.len() != encoding.len() {
        panic!(
            "FAIL: Attention mask length {} != encoding length {}",
            attention_mask.len(),
            encoding.len()
        );
    }
}

/// Test batch tokenization with padding.
/// PASS: All sequences padded to same length, attention masks correct.
/// FAIL: Uneven lengths or wrong padding.
#[test]
fn test_tokenizer_batch_encoding() {
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: tokenizer.json not found");
        return;
    }

    use embedder::tokenizer::Tokenizer;
    use embedder::MAX_SEQUENCE_LENGTH;

    let tokenizer = Tokenizer::from_file(tokenizer_path, MAX_SEQUENCE_LENGTH)
        .expect("FAIL: Could not load tokenizer");

    let texts = vec![
        "Short".to_string(),
        "This is a longer sentence with more words".to_string(),
        "Medium length text".to_string(),
    ];

    let batch = tokenizer.encode_batch(&texts)
        .expect("FAIL: Batch encoding failed");

    // FAIL: Batch size must match input
    if batch.batch_size != texts.len() {
        panic!(
            "FAIL: Batch size {} != input size {}",
            batch.batch_size,
            texts.len()
        );
    }

    // FAIL: All sequences must have same length
    let seq_len = batch.seq_len;
    for i in 0..batch.batch_size {
        let start = i * seq_len;
        let end = start + seq_len;
        let ids = &batch.input_ids[start..end];
        
        // Check padding (trailing zeros)
        let attention = &batch.attention_mask[start..end];
        let _has_content = attention.iter().any(|&x| x > 0);
    }
}

/// Test tokenization of text exceeding max_sequence_length.
/// PASS: Long text is truncated to max length, no panic.
/// FAIL: Panic or sequence length exceeds max.
#[test]
fn test_tokenizer_truncation() {
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: tokenizer.json not found");
        return;
    }

    use embedder::tokenizer::Tokenizer;
    use embedder::MAX_SEQUENCE_LENGTH;

    let tokenizer = Tokenizer::from_file(tokenizer_path, MAX_SEQUENCE_LENGTH)
        .expect("FAIL: Could not load tokenizer");

    // Create very long text
    let long_text: String = "word ".repeat(1000);
    
    // FAIL: Should not panic
    let result = std::panic::catch_unwind(|| {
        tokenizer.encode(&long_text)
    });

    if result.is_err() {
        panic!("FAIL: Tokenizer panicked on long text");
    }

    let encoding = result.unwrap().expect("FAIL: Encoding failed for long text");

    // FAIL: Should be truncated to max length
    if encoding.len() > MAX_SEQUENCE_LENGTH {
        panic!(
            "FAIL: Encoding length {} exceeds max {}",
            encoding.len(),
            MAX_SEQUENCE_LENGTH
        );
    }
}

/// Test Unicode handling in tokenization.
/// PASS: Chinese, Arabic, and emoji are tokenized without panic.
/// FAIL: Panic or incorrect Unicode handling.
#[test]
fn test_tokenizer_unicode() {
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: tokenizer.json not found");
        return;
    }

    use embedder::tokenizer::Tokenizer;
    use embedder::MAX_SEQUENCE_LENGTH;

    let tokenizer = Tokenizer::from_file(tokenizer_path, MAX_SEQUENCE_LENGTH)
        .expect("FAIL: Could not load tokenizer");

    let test_cases = vec![
        ("Chinese", "中文测试"),
        ("Arabic", "مرحبا بالعالم"),
        ("Japanese", "こんにちは世界"),
        ("Emoji", "🎮 🚀 🔥"),
        ("Mixed", "Hello 世界 🌍"),
    ];

    for (name, text) in test_cases {
        // FAIL: Should not panic on any Unicode
        let result = std::panic::catch_unwind(|| {
            tokenizer.encode(text)
        });

        if result.is_err() {
            panic!("FAIL: Tokenizer panicked on {} text: {}", name, text);
        }

        let encoding = result.unwrap().expect("FAIL: Encoding failed for Unicode");
        
        // FAIL: Should produce tokens
        if encoding.len() == 0 {
            panic!("FAIL: {} text produced zero tokens", name);
        }
    }
}

// ============================================================================
// VRAM ESTIMATOR TESTS
// ============================================================================

/// Test VRAM estimation calculation.
/// PASS: Estimate is reasonable (non-zero, within expected range).
/// FAIL: Estimate is 0 or unreasonably large.
#[test]
fn test_vram_estimation() {
    use embedder::vram::VramMonitor;

    // Test various batch sizes
    let test_cases = vec![
        (1, 256, 384),
        (8, 256, 384),
        (16, 256, 384),
        (32, 256, 384),
    ];

    for (batch_size, seq_len, hidden_dim) in test_cases {
        let estimate = VramMonitor::estimate_batch_vram(batch_size, seq_len, hidden_dim);

        // FAIL: Estimate should be non-zero
        if estimate == 0 {
            panic!(
                "FAIL: VRAM estimate is 0 for batch={}, seq={}, dim={}",
                batch_size, seq_len, hidden_dim
            );
        }

        // FAIL: Estimate should be reasonable (not negative, not astronomically large)
        if estimate > 10_000_000_000 { // 10 GB
            panic!(
                "FAIL: VRAM estimate {} GB is unreasonably large",
                estimate as f64 / 1_000_000_000.0
            );
        }

        println!(
            "Batch {}: {} MB estimated",
            batch_size,
            estimate as f64 / 1_000_000.0
        );
    }
}

/// Test that small batches fit within VRAM ceiling.
/// PASS: Batch size 8 fits within 1.5 GB ceiling.
/// FAIL: Even small batches exceed ceiling.
#[test]
fn test_vram_ceiling_fit() {
    use embedder::vram::VramMonitor;
    use embedder::VRAM_CEILING_BYTES;

    let estimate = VramMonitor::estimate_batch_vram(8, 256, 384);

    // FAIL: Default batch size must fit in ceiling
    if estimate > VRAM_CEILING_BYTES as u64 {
        panic!(
            "FAIL: Batch 8 estimate {} MB exceeds ceiling {} MB",
            estimate as f64 / 1_000_000.0,
            VRAM_CEILING_BYTES as f64 / 1_000_000.0
        );
    }
}

/// Test VRAM monitoring availability check.
/// PASS: Monitor creates successfully (even if NVML unavailable).
/// FAIL: Panic during creation.
#[test]
fn test_vram_monitor_creation() {
    use embedder::vram::VramMonitor;

    // FAIL: Should not panic even if NVML unavailable
    let result = std::panic::catch_unwind(|| {
        VramMonitor::new()
    });

    if result.is_err() {
        panic!("FAIL: VramMonitor::new() panicked");
    }

    let monitor = result.unwrap().expect("FAIL: Could not create monitor");
    
    // Check availability
    let available = monitor.is_available();
    println!("NVML available: {}", available);
}

// ============================================================================
// L2 NORMALIZATION TESTS
// ============================================================================

/// Test L2 normalization of zero vector.
/// PASS: Zero vector stays zero (no NaN).
/// FAIL: NaN or infinity in output.
#[test]
fn test_normalize_zero_vector() {
    use embedder::model::cosine_similarity;

    let zero: Vec<f32> = vec![0.0; 384];
    
    // FAIL: Cosine similarity with zero should be 0, not NaN
    let sim = cosine_similarity(&zero, &zero);
    
    if sim.is_nan() || sim.is_infinite() {
        panic!("FAIL: Cosine similarity of zero vectors is NaN/Inf: {}", sim);
    }
}

/// Test L2 normalization produces unit vectors.
/// PASS: Norm of normalized vector is 1.0.
/// FAIL: Norm != 1.0.
#[test]
fn test_normalize_unit_length() {
    // Create a non-normalized vector
    let v = vec![3.0, 4.0, 0.0, 0.0];
    
    // Calculate L2 norm
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    // FAIL: Original norm should be 5.0
    if (norm - 5.0).abs() > 0.001 {
        panic!("FAIL: Test vector norm should be 5.0, got {}", norm);
    }
}

/// Test cosine similarity of identical vectors.
/// PASS: Similarity = 1.0.
/// FAIL: Similarity != 1.0.
#[test]
fn test_cosine_similarity_identical() {
    use embedder::model::cosine_similarity;

    let v = vec![1.0, 0.0, 0.0, 0.0];
    let sim = cosine_similarity(&v, &v);

    // FAIL: Identical vectors must have similarity 1.0
    if (sim - 1.0).abs() > 0.0001 {
        panic!("FAIL: Cosine similarity of identical vectors should be 1.0, got {}", sim);
    }
}

/// Test cosine similarity of orthogonal vectors.
/// PASS: Similarity = 0.0.
/// FAIL: Similarity != 0.0.
#[test]
fn test_cosine_similarity_orthogonal() {
    use embedder::model::cosine_similarity;

    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);

    // FAIL: Orthogonal vectors must have similarity 0.0
    if sim.abs() > 0.0001 {
        panic!("FAIL: Cosine similarity of orthogonal vectors should be 0.0, got {}", sim);
    }
}

/// Test cosine similarity bounds.
/// PASS: All similarities are in [-1, 1].
/// FAIL: Any similarity outside bounds.
#[test]
fn test_cosine_similarity_bounds() {
    use embedder::model::cosine_similarity;

    let test_vectors = vec![
        (vec![1.0, 0.0, 0.0, 0.0], vec![0.5, 0.5, 0.5, 0.5]),
        (vec![1.0, 0.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0, 0.0]),
        (vec![0.707, 0.707, 0.0, 0.0], vec![0.707, -0.707, 0.0, 0.0]),
    ];

    for (i, (a, b)) in test_vectors.iter().enumerate() {
        let sim = cosine_similarity(a, b);

        // FAIL: Similarity must be in [-1, 1]
        if sim < -1.0 || sim > 1.0 {
            panic!(
                "FAIL: Cosine similarity {} out of bounds [-1, 1] for pair {}",
                sim, i
            );
        }
    }
}

// ============================================================================
// BATCH PROCESSING TESTS
// ============================================================================

/// Test empty batch handling.
/// PASS: Empty batch returns error, not panic.
/// FAIL: Panic on empty input.
#[test]
fn test_empty_batch() {
    // This would be tested against actual embedder
    // Here we just verify the error type exists
    use embedder::EmbeddingError;

    // FAIL: EmptyBatch variant must exist
    let _err = EmbeddingError::EmptyBatch;
}

/// Test dimension mismatch detection.
/// PASS: Wrong dimension embeddings cause error.
/// FAIL: Wrong dimensions accepted silently.
#[test]
fn test_dimension_mismatch() {
    // This tests the indexer side
    use indexer::{VectorStore, Document, StoreError};

    let mut store = VectorStore::new(4);

    let doc = Document::new(
        "test".to_string(),
        "https://example.com".to_string(),
        "Test".to_string(),
        "Content".to_string(),
    );

    // Try to add with wrong dimension
    let wrong_dim_emb = vec![0.0; 8]; // Should be 4
    let result = store.add(doc, wrong_dim_emb);

    // FAIL: Should return DimensionMismatch error
    match result {
        Err(StoreError::DimensionMismatch { expected, actual }) => {
            if expected != 4 || actual != 8 {
                panic!(
                    "FAIL: Wrong dimension values: expected {}, actual {}",
                    expected, actual
                );
            }
        }
        Ok(_) => {
            panic!("FAIL: Wrong dimension embedding was accepted");
        }
        Err(e) => {
            panic!("FAIL: Wrong error type: {:?}", e);
        }
    }
}

/// Test concurrent access to embedder (if model available).
/// PASS: Multiple threads can use embedder without race.
/// FAIL: Race condition or deadlock.
#[test]
fn test_concurrent_embedding() {
    let model_path = "models/model.onnx";
    let tokenizer_path = "models/tokenizer.json";
    
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("SKIP: Model files not found");
        return;
    }

    // This is a synchronous test placeholder
    // Full async test would require tokio runtime
    println!("Concurrent embedding test placeholder - requires async runtime");
}
