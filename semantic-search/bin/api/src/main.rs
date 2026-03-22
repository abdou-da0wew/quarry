//! HTTP API for semantic search.
//!
//! Provides a REST API for querying the vector store.
//!
//! ## Endpoints
//!
//! - `POST /search` - Search for similar documents
//! - `GET /health` - Health check
//! - `GET /stats` - Store statistics

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

use embedder::{Embedder, EmbedderConfig, EMBEDDING_DIM};
use indexer::{SharedVectorStore, VectorStore};

/// Application state shared across handlers.
struct AppState {
    store: SharedVectorStore,
    embedder: Embedder,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    /// Search query text
    query: String,
    /// Number of results to return
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Minimum similarity score (0.0 to 1.0)
    #[serde(default)]
    min_score: f32,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    /// Original query
    query: String,
    /// Processing time in milliseconds
    processing_time_ms: u64,
    /// Search results
    results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    rank: usize,
    score: f32,
    document: DocumentInfo,
}

#[derive(Debug, Serialize)]
struct DocumentInfo {
    id: String,
    url: String,
    title: String,
    snippet: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    document_count: usize,
    embedding_dimension: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .json()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    // Parse arguments
    let store_path = std::env::var("STORE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/store.bin"));
    let model_path = std::env::var("MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/model.onnx"));
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/tokenizer.json"));
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    info!(
        store = %store_path.display(),
        model = %model_path.display(),
        port = port,
        "Starting API server"
    );

    // Load vector store
    let store = if store_path.exists() {
        match SharedVectorStore::load(&store_path) {
            Ok(s) => {
                info!(docs = s.len(), "Vector store loaded from file");
                s
            }
            Err(e) => {
                warn!(error = %e, "Failed to load vector store, starting with empty store");
                SharedVectorStore::new(EMBEDDING_DIM)
            }
        }
    } else {
        warn!(path = %store_path.display(), "Store file not found, starting with empty store");
        SharedVectorStore::new(EMBEDDING_DIM)
    };

    info!(docs = store.len(), "Vector store ready");

    // Initialize embedder
    let config = EmbedderConfig::new(&model_path, &tokenizer_path);
    let embedder = Embedder::new(config).await
        .context("Failed to initialize embedder")?;

    embedder.warmup().await
        .context("Failed to warm up model")?;

    info!("Embedder initialized");

    // Create app state
    let state = Arc::new(AppState { store, embedder });

    // Build router
    let app = Router::new()
        .route("/search", post(search))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(addr = %addr, "Server starting");

    let listener = tokio::net::TcpListener::bind(addr).await
        .context("Failed to bind server address")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("Server error")?;

    info!("Server stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received");
}

async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    // Validate query
    if req.query.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Query cannot be empty".to_string(),
            }),
        );
    }

    // Generate query embedding
    let query_embedding = match state.embedder.embed(&req.query).await {
        Ok(e) => e,
        Err(e) => {
            error!(error = %e, "Failed to embed query");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Embedding error: {}", e),
                }),
            );
        }
    };

    // Search
    let hits = match state.store.search(&query_embedding, req.top_k) {
        Ok(h) => h,
        Err(e) => {
            error!(error = %e, "Search failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Search error: {}", e),
                }),
            );
        }
    };

    // Filter by min_score
    let results: Vec<SearchResult> = hits
        .into_iter()
        .filter(|h| h.score >= req.min_score)
        .enumerate()
        .map(|(i, h)| SearchResult {
            rank: i + 1,
            score: h.score,
            document: DocumentInfo {
                id: h.id,
                url: h.url,
                title: h.title,
                snippet: h.snippet,
            },
        })
        .collect();

    let response = SearchResponse {
        query: req.query,
        processing_time_ms: start.elapsed().as_millis() as u64,
        results,
    };

    (StatusCode::OK, Json(response))
}

async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(StatsResponse {
        document_count: state.store.len(),
        embedding_dimension: EMBEDDING_DIM,
    })
}
