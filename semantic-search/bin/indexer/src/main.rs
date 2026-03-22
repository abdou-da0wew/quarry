//! Indexer binary: Reads Go crawler output, generates embeddings, builds vector store.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::signal;
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

use embedder::{Embedder, EmbedderConfig, EMBEDDING_DIM};
use indexer::{Document, DocumentLoader, VectorStore};

#[derive(Debug, Deserialize)]
struct Config {
    /// Path to ONNX model
    model_path: PathBuf,
    /// Path to tokenizer
    tokenizer_path: PathBuf,
    /// Path to Go crawler output directory
    crawler_output: PathBuf,
    /// Path to save vector store
    store_output: PathBuf,
    /// Maximum batch size
    batch_size: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// VRAM ceiling in MB
    vram_ceiling_mb: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/model.onnx"),
            tokenizer_path: PathBuf::from("models/tokenizer.json"),
            crawler_output: PathBuf::from("crawler/output"),
            store_output: PathBuf::from("data/store.bin"),
            batch_size: 8,
            max_seq_len: 256,
            vram_ceiling_mb: 1500,
        }
    }
}

fn load_config() -> Result<Config> {
    let config_path = std::env::var("SEMANTIC_SEARCH_CONFIG")
        .unwrap_or_else(|_| "config.toml".to_string());

    match std::fs::read_to_string(&config_path) {
        Ok(content) => {
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse config from {}", config_path))
        }
        Err(_) => {
            warn!("Config file not found, using defaults");
            Ok(Config::default())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .json()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    info!("Starting indexer");

    // Load config
    let config = load_config()?;
    info!(
        model = %config.model_path.display(),
        tokenizer = %config.tokenizer_path.display(),
        input = %config.crawler_output.display(),
        output = %config.store_output.display(),
        batch_size = config.batch_size,
        "Configuration loaded"
    );

    // Setup signal handler for graceful shutdown
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        info!("Received shutdown signal");
        shutdown_clone.store(true, Ordering::SeqCst);
    });

    // Initialize embedder
    let embedder_config = EmbedderConfig {
        model_path: config.model_path,
        tokenizer_path: config.tokenizer_path,
        max_batch_size: config.batch_size,
        max_sequence_length: config.max_seq_len,
        vram_ceiling_bytes: config.vram_ceiling_mb * 1_000_000,
        use_cuda: true,
    };

    info!("Initializing embedder");
    let embedder = Embedder::new(embedder_config)
        .await
        .context("Failed to initialize embedder")?;

    // Warm up
    embedder.warmup()
        .await
        .context("Failed to warm up model")?;

    // Load documents
    info!(path = %config.crawler_output.display(), "Loading documents from crawler output");
    let loader = DocumentLoader::new(&config.crawler_output);
    let documents = loader.load_all()
        .context("Failed to load documents")?;

    if documents.is_empty() {
        error!("No documents found in crawler output");
        return Ok(());
    }

    info!(count = documents.len(), "Documents loaded");

    // Create vector store
    let mut store = VectorStore::new(EMBEDDING_DIM);

    // Process documents in batches
    let total_docs = documents.len();
    let mut processed = 0usize;
    let mut errors = 0usize;
    let start_time = Instant::now();

    for chunk in documents.chunks(config.batch_size) {
        // Check for shutdown signal
        if shutdown.load(Ordering::SeqCst) {
            warn!("Shutdown requested, saving progress");
            break;
        }

        // Prepare texts for embedding
        let texts: Vec<String> = chunk.iter()
            .map(|d| d.embedding_text())
            .collect();

        // Generate embeddings
        match embedder.embed_batch(&texts).await {
            Ok(embeddings) => {
                for (doc, embedding) in chunk.iter().zip(embeddings.into_iter()) {
                    if let Err(e) = store.add(doc.clone(), embedding) {
                        warn!(doc_id = %doc.id, error = %e, "Failed to add document to store");
                        errors += 1;
                    } else {
                        processed += 1;
                    }
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to generate embeddings for batch");
                errors += chunk.len();
            }
        }

        // Log progress
        if processed % 50 == 0 {
            let elapsed = start_time.elapsed();
            let docs_per_sec = processed as f64 / elapsed.as_secs_f64().max(0.001);
            info!(
                processed = processed,
                total = total_docs,
                errors = errors,
                docs_per_sec = format!("{:.1}", docs_per_sec),
                "Indexing progress"
            );
        }
    }

    let elapsed = start_time.elapsed();

    // Save vector store
    info!(
        path = %config.store_output.display(),
        docs = store.len(),
        "Saving vector store"
    );

    // Create parent directory if needed
    if let Some(parent) = config.store_output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    store.save(&config.store_output)
        .context("Failed to save vector store")?;

    // Log final statistics
    info!(
        processed = processed,
        errors = errors,
        total = total_docs,
        elapsed_secs = elapsed.as_secs(),
        docs_per_sec = format!("{:.1}", processed as f64 / elapsed.as_secs_f64().max(0.001)),
        "Indexing complete"
    );

    // Log VRAM usage
    if let Ok(usage) = embedder.vram_usage() {
        info!(
            vram_used_mb = usage.used_bytes as f64 / 1_000_000.0,
            vram_total_mb = usage.total_bytes as f64 / 1_000_000.0,
            "Final VRAM usage"
        );
    }

    Ok(())
}
