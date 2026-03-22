//! Embedder crate: ONNX embedding inference with CUDA GPU execution.
//!
//! Provides GPU-accelerated text embedding generation using ONNX Runtime.
//! Designed for NVIDIA MX250 (2GB VRAM) with 1.5GB hard ceiling.
//!
//! # Example
//! ```no_run
//! use embedder::{Embedder, EmbedderConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = EmbedderConfig::default();
//!     let embedder = Embedder::new(config).await?;
//!     
//!     let texts = vec!["Hello world".to_string(), "Test document".to_string()];
//!     let embeddings = embedder.embed_batch(&texts).await?;
//!     
//!     println!("Generated {} embeddings of dimension {}", embeddings.len(), embeddings[0].len());
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod tokenizer;
pub mod vram;

pub use model::{Embedder, EmbedderConfig, EmbeddingError};
pub use tokenizer::Tokenizer;
pub use vram::{VramMonitor, VramError};

/// Embedding dimension for all-MiniLM-L6-v2
pub const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length for the model
pub const MAX_SEQUENCE_LENGTH: usize = 256;

/// Default batch size (safe for MX250)
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// VRAM ceiling in bytes (1.5 GB for MX250 with 512 MB headroom)
pub const VRAM_CEILING_BYTES: usize = 1_500_000_000;

/// Result type for embedder operations
pub type Result<T> = std::result::Result<T, EmbeddingError>;
