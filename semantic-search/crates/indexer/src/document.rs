//! Document loading from Go crawler output.

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use thiserror::Error;
use tracing::{debug, info, warn};
use walkdir::WalkDir;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Error, Debug)]
pub enum DocumentError {
    #[error("Failed to read file {path}: {source}")]
    ReadError {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse JSON from {path}: {source}")]
    ParseError {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("Invalid document structure in {path}: {reason}")]
    InvalidDocument { path: String, reason: String },

    #[error("No documents found in {path}")]
    NoDocuments { path: String },
}

/// Page document from Go crawler output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlerPage {
    /// Page URL
    pub url: String,
    /// Page title
    pub title: String,
    /// Page body text
    pub body: String,
    /// Links found on page
    pub links: Vec<String>,
    /// Crawl timestamp
    pub crawled_at: String,
}

/// Internal document representation for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document ID
    pub id: String,
    /// Document URL
    pub url: String,
    /// Document title
    pub title: String,
    /// Document text content
    pub text: String,
    /// Index timestamp
    pub indexed_at: DateTime<Utc>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: String, url: String, title: String, text: String) -> Self {
        Self {
            id,
            url,
            title,
            text,
            indexed_at: Utc::now(),
        }
    }

    /// Create from crawler page format.
    pub fn from_crawler_page(page: CrawlerPage) -> Self {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(page.url.as_bytes());
        let hash = hasher.finalize();
        
        // Convert first 8 bytes to hex string
        let id = hash[..8]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        Self {
            id,
            url: page.url,
            title: page.title,
            text: page.body,
            indexed_at: Utc::now(),
        }
    }

    /// Get text suitable for embedding (title + body).
    pub fn embedding_text(&self) -> String {
        if self.title.is_empty() {
            self.text.clone()
        } else {
            format!("{}\n\n{}", self.title, self.text)
        }
    }
}

/// Document loader reads documents from Go crawler output.
pub struct DocumentLoader {
    /// Root directory containing page JSON files
    root_dir: PathBuf,
}

impl DocumentLoader {
    /// Create a new document loader.
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
        }
    }

    /// Load all documents from the crawler output directory.
    ///
    /// Looks for JSON files in `root_dir/pages/` subdirectory.
    pub fn load_all(&self) -> Result<Vec<Document>, DocumentError> {
        let pages_dir = self.root_dir.join("pages");

        if !pages_dir.exists() {
            return Err(DocumentError::NoDocuments {
                path: pages_dir.display().to_string(),
            });
        }

        let mut documents = Vec::new();
        let mut file_count = 0;
        let mut error_count = 0;

        for entry in WalkDir::new(&pages_dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.extension().map_or(false, |ext| ext == "json") {
                continue;
            }

            file_count += 1;

            match self.load_file(path) {
                Ok(doc) => documents.push(doc),
                Err(e) => {
                    error_count += 1;
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to load document"
                    );
                }
            }
        }

        info!(
            files_found = file_count,
            documents_loaded = documents.len(),
            errors = error_count,
            "Document loading complete"
        );

        if documents.is_empty() && file_count > 0 {
            return Err(DocumentError::NoDocuments {
                path: pages_dir.display().to_string(),
            });
        }

        Ok(documents)
    }

    /// Load a single document file.
    fn load_file(&self, path: &Path) -> Result<Document, DocumentError> {
        let path_str = path.display().to_string();

        let content = fs::read_to_string(path).map_err(|e| DocumentError::ReadError {
            path: path_str.clone(),
            source: e,
        })?;

        // Try to parse as crawler page format first
        if let Ok(page) = serde_json::from_str::<CrawlerPage>(&content) {
            return Ok(Document::from_crawler_page(page));
        }

        // Try alternative formats
        #[derive(Deserialize)]
        struct AltPage {
            url: String,
            title: Option<String>,
            body: Option<String>,
            text: Option<String>,
        }

        if let Ok(alt) = serde_json::from_str::<AltPage>(&content) {
            let id = format!("{:x}", {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(alt.url.as_bytes());
                hasher.finalize()
            });
            let id = &id[..16.min(id.len())];

            return Ok(Document::new(
                id.to_string(),
                alt.url,
                alt.title.unwrap_or_default(),
                alt.body.or(alt.text).unwrap_or_default(),
            ));
        }

        Err(DocumentError::InvalidDocument {
            path: path_str,
            reason: "Unrecognized document format".to_string(),
        })
    }

    /// Load documents from a JSONL file.
    pub fn load_jsonl(&self, path: &Path) -> Result<Vec<Document>, DocumentError> {
        let path_str = path.display().to_string();

        let file = File::open(path).map_err(|e| DocumentError::ReadError {
            path: path_str.clone(),
            source: e,
        })?;

        let reader = BufReader::new(file);
        let mut documents = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| DocumentError::ReadError {
                path: path_str.clone(),
                source: e,
            })?;

            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<CrawlerPage>(&line) {
                Ok(page) => documents.push(Document::from_crawler_page(page)),
                Err(e) => {
                    warn!(
                        line = line_num + 1,
                        error = %e,
                        "Failed to parse JSONL line"
                    );
                }
            }
        }

        Ok(documents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::io::Write;

    #[test]
    fn test_document_embedding_text() {
        let doc = Document::new(
            "test-id".to_string(),
            "https://example.com".to_string(),
            "Test Title".to_string(),
            "Test body content".to_string(),
        );

        let text = doc.embedding_text();
        assert!(text.contains("Test Title"));
        assert!(text.contains("Test body content"));
    }

    #[test]
    fn test_document_from_crawler_page() {
        let page = CrawlerPage {
            url: "https://example.com/page".to_string(),
            title: "Test Page".to_string(),
            body: "Content here".to_string(),
            links: vec![],
            crawled_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let doc = Document::from_crawler_page(page);
        assert_eq!(doc.url, "https://example.com/page");
        assert_eq!(doc.title, "Test Page");
        assert_eq!(doc.text, "Content here");
    }

    #[test]
    fn test_document_loader() {
        let temp_dir = TempDir::new().unwrap();
        let pages_dir = temp_dir.path().join("pages");
        fs::create_dir_all(&pages_dir).unwrap();

        let doc_path = pages_dir.join("test.json");
        let content = r#"{
            "url": "https://example.com/test",
            "title": "Test Document",
            "body": "Test body",
            "links": [],
            "crawled_at": "2024-01-01T00:00:00Z"
        }"#;

        let mut file = File::create(&doc_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let loader = DocumentLoader::new(temp_dir.path());
        let docs = loader.load_all().unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title, "Test Document");
    }
}
