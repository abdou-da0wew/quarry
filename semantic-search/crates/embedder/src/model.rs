//! ONNX model wrapper for embedding inference.
//!
//! Provides GPU-accelerated embedding generation using ONNX Runtime.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use tokio::sync::{RwLock, Mutex};

use crate::tokenizer::{BatchEncoding, Tokenizer, TokenizerError};
use crate::vram::{VramError, VramMonitor};
use crate::{EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, VRAM_CEILING_BYTES};

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Failed to load ONNX model from {path}: {source}")]
    ModelLoadError {
        path: String,
        #[source]
        source: ort::Error,
    },

    #[error("Failed to initialize ONNX environment: {0}")]
    EnvironmentError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] TokenizerError),

    #[error("VRAM error: {0}")]
    VramError(#[from] VramError),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Empty batch")]
    EmptyBatch,

    #[error("CUDA not available, falling back to CPU")]
    CudaFallback,

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Configuration for the embedder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Path to ONNX model file
    pub model_path: PathBuf,
    /// Path to tokenizer.json file
    pub tokenizer_path: PathBuf,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// VRAM ceiling in bytes
    pub vram_ceiling_bytes: u64,
    /// Use CUDA if available
    pub use_cuda: bool,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/model.onnx"),
            tokenizer_path: PathBuf::from("models/tokenizer.json"),
            max_batch_size: 8,
            max_sequence_length: MAX_SEQUENCE_LENGTH,
            vram_ceiling_bytes: VRAM_CEILING_BYTES as u64,
            use_cuda: true,
        }
    }
}

impl EmbedderConfig {
    /// Create config from paths.
    pub fn new<P: AsRef<Path>>(model_path: P, tokenizer_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: tokenizer_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Set maximum batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.max_batch_size = batch_size;
        self
    }

    /// Set VRAM ceiling.
    pub fn with_vram_ceiling(mut self, bytes: u64) -> Self {
        self.vram_ceiling_bytes = bytes;
        self
    }

    /// Disable CUDA.
    pub fn with_cpu_only(mut self) -> Self {
        self.use_cuda = false;
        self
    }
}

/// Embedder generates text embeddings using ONNX Runtime.
pub struct Embedder {
    session: Arc<Mutex<Session>>,
    tokenizer: Tokenizer,
    vram_monitor: VramMonitor,
    config: EmbedderConfig,
    /// Name of the input_ids input
    input_ids_name: String,
    /// Name of the attention_mask input
    attention_mask_name: String,
    /// Name of the output tensor
    output_name: String,
    /// Pre-allocated output buffer for single queries
    query_buffer: RwLock<Vec<f32>>,
}

impl Embedder {
    /// Create a new embedder with the given configuration.
    pub async fn new(config: EmbedderConfig) -> Result<Self, EmbeddingError> {
        info!(
            model_path = %config.model_path.display(),
            tokenizer_path = %config.tokenizer_path.display(),
            max_batch_size = config.max_batch_size,
            "Initializing embedder"
        );

        // Initialize tokenizer
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path, config.max_sequence_length)?;

        // Initialize VRAM monitor
        let vram_monitor = VramMonitor::new()?;

        // Build ONNX session
        let (session, input_ids_name, attention_mask_name, output_name) = Self::build_session(&config)?;

        // Log initial VRAM usage
        vram_monitor.log_usage("after_model_load");

        // Pre-allocate query buffer
        let query_buffer = RwLock::new(vec![0.0f32; EMBEDDING_DIM]);

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            vram_monitor,
            config,
            input_ids_name,
            attention_mask_name,
            output_name,
            query_buffer,
        })
    }

    /// Build ONNX session with appropriate providers.
    fn build_session(config: &EmbedderConfig) -> Result<(Session, String, String, String), EmbeddingError> {
        let path_str = config.model_path.display().to_string();

        let mut builder = Session::builder()
            .map_err(|e| EmbeddingError::EnvironmentError(e.to_string()))?;
        
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbeddingError::EnvironmentError(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| EmbeddingError::EnvironmentError(e.to_string()))?;

        let session = builder
            .commit_from_file(&config.model_path)
            .map_err(|e| EmbeddingError::ModelLoadError {
                path: path_str,
                source: e,
            })?;

        // Get input/output names
        let inputs = session.inputs();
        let outputs = session.outputs();

        debug!(
            num_inputs = inputs.len(),
            num_outputs = outputs.len(),
            "ONNX session created"
        );

        // Extract input names - typically "input_ids" and "attention_mask"
        let input_ids_name = inputs
            .iter()
            .find(|i| i.name().contains("input") || i.name().contains("ids"))
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| inputs[0].name().to_string());

        let attention_mask_name = inputs
            .iter()
            .find(|i| i.name().contains("attention") || i.name().contains("mask"))
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| {
                if inputs.len() > 1 {
                    inputs[1].name().to_string()
                } else {
                    "attention_mask".to_string()
                }
            });

        // Get output name
        let output_name = outputs
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "last_hidden_state".to_string());

        for (i, input) in inputs.iter().enumerate() {
            debug!(
                input_index = i,
                name = input.name(),
                "Model input"
            );
        }

        for (i, output) in outputs.iter().enumerate() {
            debug!(
                output_index = i,
                name = output.name(),
                "Model output"
            );
        }

        Ok((session, input_ids_name, attention_mask_name, output_name))
    }

    /// Warm up the model with a dummy inference.
    pub async fn warmup(&self) -> Result<(), EmbeddingError> {
        info!("Warming up model");
        let dummy_texts = vec!["warmup".to_string()];
        self.embed_batch(&dummy_texts).await?;
        self.vram_monitor.log_usage("after_warmup");
        info!("Model warmup complete");
        Ok(())
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// Automatically adjusts batch size if VRAM is insufficient.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Err(EmbeddingError::EmptyBatch);
        }

        // Tokenize all texts
        let encoding = self.tokenizer.encode_batch(texts)?;

        // Check VRAM and adjust batch size if needed
        let batch_size = self.find_safe_batch_size(encoding.batch_size, encoding.seq_len)?;

        // Process in sub-batches if needed
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk_start in (0..texts.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(texts.len());
            let chunk_texts: Vec<String> = texts[chunk_start..chunk_end].to_vec();

            let chunk_encoding = self.tokenizer.encode_batch(&chunk_texts)?;
            let embeddings = self.run_inference(&chunk_encoding).await?;

            all_embeddings.extend(embeddings);
        }

        self.vram_monitor.log_usage("after_batch");

        Ok(all_embeddings)
    }

    /// Generate embedding for a single text.
    ///
    /// Uses pre-allocated buffer for zero-allocation hot path.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let texts = vec![text.to_string()];
        let embeddings = self.embed_batch(&texts).await?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    /// Run ONNX inference on tokenized input.
    async fn run_inference(&self, encoding: &BatchEncoding) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let batch_size = encoding.batch_size;
        let seq_len = encoding.seq_len;

        // Create input tensors using (shape, data) format for ort 2.0
        let input_ids_shape = vec![batch_size, seq_len];
        let attention_mask_shape = vec![batch_size, seq_len];

        // Create ONNX values
        let input_ids_value = Value::from_array((input_ids_shape, encoding.input_ids.clone()))
            .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create input_ids value: {}", e)))?;
        let attention_mask_value = Value::from_array((attention_mask_shape, encoding.attention_mask.clone()))
            .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create attention_mask value: {}", e)))?;

        // Run inference using the extracted names
        let mut session = self.session.lock().await;
        let outputs = session
            .run(vec![
                (self.input_ids_name.as_str(), input_ids_value),
                (self.attention_mask_name.as_str(), attention_mask_value),
            ])
            .map_err(|e| EmbeddingError::InferenceError(format!("Inference failed: {}", e)))?;

        // Extract output tensor by name
        let output_tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| EmbeddingError::InferenceError(format!("Output '{}' not found", self.output_name)))?;

        // Extract the tensor data
        // try_extract_tensor returns a tuple (&Shape, &[f32])
        let (shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::InferenceError(format!("Failed to extract output: {}", e)))?;
        
        // Get the shape and convert to vec
        let output_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        
        debug!(
            shape = ?output_shape,
            "Inference output shape"
        );

        // Extract embeddings
        let mut embeddings = Vec::with_capacity(batch_size);

        // Shape is typically [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        match output_shape.as_slice() {
            [bs, seq, dim] if *bs == batch_size && *dim == EMBEDDING_DIM => {
                // Shape: [batch_size, seq_len, hidden_dim]
                // Mean pool over sequence dimension
                for i in 0..batch_size {
                    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
                    let mut count = 0.0f32;

                    for j in 0..*seq {
                        if encoding.attention_mask[i * seq_len + j] > 0 {
                            for k in 0..EMBEDDING_DIM {
                                let idx = i * seq * EMBEDDING_DIM + j * EMBEDDING_DIM + k;
                                embedding[k] += data[idx];
                            }
                            count += 1.0;
                        }
                    }

                    // Normalize and L2-normalize
                    if count > 0.0 {
                        for k in 0..EMBEDDING_DIM {
                            embedding[k] /= count;
                        }
                    }
                    l2_normalize(&mut embedding);
                    embeddings.push(embedding);
                }
            }
            [bs, dim] if *bs == batch_size && *dim == EMBEDDING_DIM => {
                // Shape: [batch_size, hidden_dim] - already pooled
                for i in 0..batch_size {
                    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
                    for k in 0..EMBEDDING_DIM {
                        let idx = i * EMBEDDING_DIM + k;
                        embedding[k] = data[idx];
                    }
                    l2_normalize(&mut embedding);
                    embeddings.push(embedding);
                }
            }
            _ => {
                return Err(EmbeddingError::InvalidShape {
                    expected: vec![batch_size, EMBEDDING_DIM],
                    actual: output_shape,
                });
            }
        }

        Ok(embeddings)
    }

    /// Find a safe batch size that fits in VRAM.
    fn find_safe_batch_size(&self, requested: usize, seq_len: usize) -> Result<usize, EmbeddingError> {
        let max_batch = requested.min(self.config.max_batch_size);

        let safe_batch = self.vram_monitor.find_safe_batch_size(
            max_batch,
            seq_len,
            EMBEDDING_DIM,
            self.config.vram_ceiling_bytes,
        )?;

        if safe_batch < requested {
            warn!(
                requested = requested,
                safe = safe_batch,
                "Reduced batch size due to VRAM constraints"
            );
        }

        Ok(safe_batch)
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Get current VRAM usage.
    pub fn vram_usage(&self) -> Result<crate::vram::VramUsage, EmbeddingError> {
        self.vram_monitor.get_usage().map_err(EmbeddingError::VramError)
    }
}

/// L2 normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Assuming vectors are already L2 normalized, dot product = cosine similarity
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        let d = vec![1.0, 0.0];
        let sim2 = cosine_similarity(&c, &d);
        assert!((sim2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_default() {
        let config = EmbedderConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_sequence_length, 256);
    }
}
