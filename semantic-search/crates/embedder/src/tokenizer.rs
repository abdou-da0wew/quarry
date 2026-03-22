//! Tokenizer wrapper using HuggingFace tokenizers crate.
//!
//! Provides text tokenization for ONNX embedding models.

use std::path::Path;
use thiserror::Error;
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::Encoding;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer from {path}: {source}")]
    LoadError {
        path: String,
        #[source]
        source: tokenizers::Error,
    },

    #[error("Failed to tokenize text: {0}")]
    TokenizationError(#[from] tokenizers::Error),

    #[error("Tokenizer not initialized")]
    NotInitialized,
}

/// Tokenizer wraps HuggingFace tokenizers for ONNX model input preparation.
pub struct Tokenizer {
    inner: HfTokenizer,
    max_length: usize,
}

impl Tokenizer {
    /// Load tokenizer from a file path.
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    /// * `max_length` - Maximum sequence length (will truncate longer sequences)
    pub fn from_file<P: AsRef<Path>>(path: P, max_length: usize) -> Result<Self, TokenizerError> {
        let path_str = path.as_ref().display().to_string();
        let inner = HfTokenizer::from_file(&path)
            .map_err(|e| TokenizerError::LoadError {
                path: path_str,
                source: e,
            })?;

        Ok(Self { inner, max_length })
    }

    /// Load tokenizer from a HuggingFace model directory.
    ///
    /// # Arguments
    /// * `model_dir` - Path to directory containing tokenizer.json
    /// * `max_length` - Maximum sequence length
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P, max_length: usize) -> Result<Self, TokenizerError> {
        let tokenizer_path = model_dir.as_ref().join("tokenizer.json");
        Self::from_file(&tokenizer_path, max_length)
    }

    /// Tokenize a single text string.
    ///
    /// Returns encoded input IDs and attention mask.
    pub fn encode(&self, text: &str) -> Result<Encoding, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(TokenizerError::TokenizationError)?;

        Ok(encoding)
    }

    /// Tokenize a batch of texts with padding to uniform length.
    ///
    /// # Arguments
    /// * `texts` - Slice of text strings to tokenize
    ///
    /// # Returns
    /// * `input_ids` - Token IDs tensor [batch_size, seq_len]
    /// * `attention_mask` - Attention mask tensor [batch_size, seq_len]
    pub fn encode_batch(&self, texts: &[String]) -> Result<BatchEncoding, TokenizerError> {
        let batch_size = texts.len();
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(TokenizerError::TokenizationError)?;

        // Find max length in batch
        let max_len = encodings
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let seq_len = max_len.max(1); // At least 1 token

        // Pre-allocate buffers
        let mut input_ids = vec![0i64; batch_size * seq_len];
        let mut attention_mask = vec![0i64; batch_size * seq_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(seq_len);

            let offset = i * seq_len;
            for j in 0..len {
                input_ids[offset + j] = ids[j] as i64;
                attention_mask[offset + j] = mask[j] as i64;
            }
            // Remaining positions are already 0 (padded)
        }

        Ok(BatchEncoding {
            input_ids,
            attention_mask,
            batch_size,
            seq_len,
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the model's maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Batch encoding result containing input IDs and attention mask.
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Input token IDs [batch_size, seq_len]
    pub input_ids: Vec<i64>,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Vec<i64>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
}

impl BatchEncoding {
    /// Get input IDs as a 2D array view.
    pub fn input_ids_2d(&self) -> Vec<Vec<i64>> {
        let mut result = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            let start = i * self.seq_len;
            let end = start + self.seq_len;
            result.push(self.input_ids[start..end].to_vec());
        }
        result
    }

    /// Get attention mask as a 2D array view.
    pub fn attention_mask_2d(&self) -> Vec<Vec<i64>> {
        let mut result = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            let start = i * self.seq_len;
            let end = start + self.seq_len;
            result.push(self.attention_mask[start..end].to_vec());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a tokenizer file to be present
    // Run download_model.sh first

    #[test]
    fn test_batch_encoding() {
        let encoding = BatchEncoding {
            input_ids: vec![1, 2, 3, 0, 4, 5, 0, 0],
            attention_mask: vec![1, 1, 1, 0, 1, 1, 0, 0],
            batch_size: 2,
            seq_len: 4,
        };

        let ids_2d = encoding.input_ids_2d();
        assert_eq!(ids_2d.len(), 2);
        assert_eq!(ids_2d[0], vec![1, 2, 3, 0]);
        assert_eq!(ids_2d[1], vec![4, 5, 0, 0]);
    }
}
