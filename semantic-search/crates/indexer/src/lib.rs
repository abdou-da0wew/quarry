//! Indexer crate: Vector storage and document indexing.
//!
//! Provides efficient vector storage with persistence and similarity search.
//! Uses flat cosine similarity for small corpora (< 10,000 documents).

pub mod store;
pub mod document;

pub use store::{VectorStore, SearchHit, StoreError};
pub use document::{Document, DocumentLoader, DocumentError};

/// Result type for indexer operations
pub type Result<T> = std::result::Result<T, StoreError>;
