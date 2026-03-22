//! Vector store for embedding storage and similarity search.
//!
//! Uses flat cosine similarity for small corpora (< 10,000 documents).
//! Persisted to disk in a binary format with checksum verification.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use thiserror::Error;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use bincode;

use crate::document::Document;

const STORE_MAGIC: &[u8; 4] = b"VST1";
const STORE_VERSION: u32 = 1;

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("Store file corrupted: {0}")]
    Corrupted(String),

    #[error("Invalid store format: expected version {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("Checksum mismatch: file may be corrupted")]
    ChecksumMismatch,

    #[error("Document not found: {0}")]
    NotFound(String),

    #[error("Empty store")]
    EmptyStore,

    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Search hit result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Document ID
    pub id: String,
    /// Document URL
    pub url: String,
    /// Document title
    pub title: String,
    /// Similarity score (0.0 to 1.0 for cosine)
    pub score: f32,
    /// Text snippet
    pub snippet: String,
}

/// Document metadata stored alongside embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub id: String,
    pub url: String,
    pub title: String,
    pub text: String,
}

/// Vector store for embeddings with flat similarity search.
pub struct VectorStore {
    /// Document metadata
    docs: HashMap<String, DocMeta>,
    /// Embeddings (doc_id -> embedding)
    embeddings: HashMap<String, Vec<f32>>,
    /// Embedding dimension
    dimension: usize,
    /// File path for persistence
    path: Option<PathBuf>,
    /// Dirty flag for tracking changes
    dirty: bool,
}

impl VectorStore {
    /// Create a new empty vector store.
    pub fn new(dimension: usize) -> Self {
        Self {
            docs: HashMap::new(),
            embeddings: HashMap::new(),
            dimension,
            path: None,
            dirty: false,
        }
    }

    /// Load a vector store from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading vector store");

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != STORE_MAGIC {
            return Err(StoreError::Corrupted(
                "Invalid magic number".to_string(),
            ));
        }

        // Read version
        let version = read_u32(&mut reader)?;
        if version != STORE_VERSION {
            return Err(StoreError::VersionMismatch {
                expected: STORE_VERSION,
                actual: version,
            });
        }

        // Read dimension
        let dimension = read_u64(&mut reader)? as usize;

        // Read stored checksum
        let stored_checksum = read_u64(&mut reader)?;

        // Read the rest of the file for checksum verification
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Compute checksum of the header + data
        let mut hasher = Sha256::new();
        hasher.update(&magic);
        hasher.update(&version.to_le_bytes());
        hasher.update(&(dimension as u64).to_le_bytes());
        hasher.update(&data);
        let computed_checksum = u64::from_be_bytes(
            hasher.finalize()[..8].try_into().map_err(|_| StoreError::ChecksumMismatch)?
        );

        if stored_checksum != computed_checksum {
            return Err(StoreError::ChecksumMismatch);
        }

        // Deserialize the store data
        let store_data: StoreData = bincode::deserialize(&data)?;

        info!(
            docs_loaded = store_data.docs.len(),
            dimension = dimension,
            "Vector store loaded"
        );

        Ok(Self {
            docs: store_data.docs,
            embeddings: store_data.embeddings,
            dimension,
            path: Some(path.to_path_buf()),
            dirty: false,
        })
    }

    /// Save the vector store to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), StoreError> {
        let path = path.as_ref();
        info!(path = %path.display(), docs = self.docs.len(), "Saving vector store");

        let store_data = StoreData {
            docs: self.docs.clone(),
            embeddings: self.embeddings.clone(),
        };

        let data = bincode::serialize(&store_data)?;

        // Build file content
        let mut content = Vec::new();
        content.extend_from_slice(STORE_MAGIC);
        content.extend_from_slice(&STORE_VERSION.to_le_bytes());
        content.extend_from_slice(&(self.dimension as u64).to_le_bytes());

        // Compute checksum placeholder
        let checksum_offset = content.len();
        content.extend_from_slice(&0u64.to_le_bytes());

        // Add data
        content.extend_from_slice(&data);

        // Compute final checksum
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let checksum = u64::from_be_bytes(
            hasher.finalize()[..8].try_into().map_err(|_| StoreError::ChecksumMismatch)?
        );

        // Write checksum
        content[checksum_offset..checksum_offset + 8].copy_from_slice(&checksum.to_le_bytes());

        // Write to temp file first, then rename for atomicity
        let temp_path = path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&content)?;
        writer.flush()?;

        fs::rename(&temp_path, path)?;

        info!(path = %path.display(), "Vector store saved");
        Ok(())
    }

    /// Add a document with its embedding.
    pub fn add(&mut self, doc: Document, embedding: Vec<f32>) -> Result<(), StoreError> {
        if embedding.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        let id = doc.id.clone();
        let meta = DocMeta {
            id: id.clone(),
            url: doc.url,
            title: doc.title,
            text: doc.text,
        };

        self.docs.insert(id.clone(), meta);
        self.embeddings.insert(id, embedding);
        self.dirty = true;

        Ok(())
    }

    /// Search for similar documents.
    ///
    /// Uses flat cosine similarity search (O(n) where n = number of documents).
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchHit>, StoreError> {
        if self.embeddings.is_empty() {
            return Err(StoreError::EmptyStore);
        }

        if query_embedding.len() != self.dimension {
            return Err(StoreError::DimensionMismatch {
                expected: self.dimension,
                actual: query_embedding.len(),
            });
        }

        // Compute similarities
        let mut scores: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .map(|(id, emb)| {
                let score = cosine_similarity(query_embedding, emb);
                (id.clone(), score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        let hits: Vec<SearchHit> = scores
            .into_iter()
            .take(top_k)
            .filter_map(|(id, score)| {
                let doc = self.docs.get(&id)?;
                let snippet = if doc.text.len() > 300 {
                    format!("{}...", &doc.text[..300])
                } else {
                    doc.text.clone()
                };

                Some(SearchHit {
                    id: doc.id.clone(),
                    url: doc.url.clone(),
                    title: doc.title.clone(),
                    score,
                    snippet,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Get document by ID.
    pub fn get(&self, id: &str) -> Option<&DocMeta> {
        self.docs.get(id)
    }

    /// Get embedding by ID.
    pub fn get_embedding(&self, id: &str) -> Option<&Vec<f32>> {
        self.embeddings.get(id)
    }

    /// Get number of documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Get embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if store has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get all document IDs.
    pub fn doc_ids(&self) -> Vec<String> {
        self.docs.keys().cloned().collect()
    }

    /// Persist if dirty.
    pub fn persist(&mut self) -> Result<(), StoreError> {
        if let Some(ref path) = self.path.clone() {
            if self.dirty {
                self.save(path)?;
                self.dirty = false;
            }
        }
        Ok(())
    }
}

/// Internal serialization format.
#[derive(Serialize, Deserialize)]
struct StoreData {
    docs: HashMap<String, DocMeta>,
    embeddings: HashMap<String, Vec<f32>>,
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Assuming vectors are already L2 normalized
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Read a u32 from a reader in little-endian.
fn read_u32<R: Read>(reader: &mut R) -> Result<u32, StoreError> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a u64 from a reader in little-endian.
fn read_u64<R: Read>(reader: &mut R) -> Result<u64, StoreError> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Thread-safe wrapper for VectorStore.
#[derive(Clone)]
pub struct SharedVectorStore {
    inner: Arc<RwLock<VectorStore>>,
}

impl SharedVectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VectorStore::new(dimension))),
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        let store = VectorStore::load(path)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(store)),
        })
    }

    pub fn add(&self, doc: Document, embedding: Vec<f32>) -> Result<(), StoreError> {
        self.inner.write().map_err(|e| StoreError::Corrupted(e.to_string()))?.add(doc, embedding)
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, StoreError> {
        self.inner.read().map_err(|e| StoreError::Corrupted(e.to_string()))?.search(query, top_k)
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|s| s.len()).unwrap_or(0)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), StoreError> {
        self.inner.read().map_err(|e| StoreError::Corrupted(e.to_string()))?.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_embedding(dim: usize, val: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        if v.len() > 0 {
            v[0] = val;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        v
    }

    fn make_doc(id: &str, title: &str) -> Document {
        Document::new(
            id.to_string(),
            format!("https://example.com/{}", id),
            title.to_string(),
            format!("Content for {}", title),
        )
    }

    #[test]
    fn test_add_and_search() {
        let mut store = VectorStore::new(4);

        let doc1 = make_doc("doc1", "First Document");
        let doc2 = make_doc("doc2", "Second Document");

        store.add(doc1, make_embedding(4, 1.0)).unwrap();
        store.add(doc2, make_embedding(4, 0.5)).unwrap();

        assert_eq!(store.len(), 2);

        let query = make_embedding(4, 1.0);
        let hits = store.search(&query, 10).unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "doc1"); // Should be most similar
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.store");

        // Create and save
        {
            let mut store = VectorStore::new(4);
            store.add(make_doc("doc1", "Test"), make_embedding(4, 1.0)).unwrap();
            store.save(&path).unwrap();
        }

        // Load
        let store = VectorStore::load(&path).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.get("doc1").is_some());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = VectorStore::new(4);
        let result = store.add(make_doc("doc1", "Test"), vec![0.0; 8]);
        assert!(matches!(result, Err(StoreError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_empty_search() {
        let store = VectorStore::new(4);
        let query = vec![0.0; 4];
        let result = store.search(&query, 10);
        assert!(matches!(result, Err(StoreError::EmptyStore)));
    }

    #[test]
    fn test_checksum_verification() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.store");

        // Create and save
        {
            let mut store = VectorStore::new(4);
            store.add(make_doc("doc1", "Test"), make_embedding(4, 1.0)).unwrap();
            store.save(&path).unwrap();
        }

        // Corrupt the file
        let content = fs::read(&path).unwrap();
        let mut corrupted = content.clone();
        corrupted[50] ^= 0xFF; // Flip some bits
        fs::write(&path, &corrupted).unwrap();

        // Try to load - should fail
        let result = VectorStore::load(&path);
        assert!(matches!(result, Err(StoreError::ChecksumMismatch)));
    }
}
