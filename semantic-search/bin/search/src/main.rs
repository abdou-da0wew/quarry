//! Search binary: Loads vector store and performs semantic search queries.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use embedder::{Embedder, EmbedderConfig};
use indexer::VectorStore;

fn print_usage() {
    println!("Usage: search [OPTIONS] <query>");
    println!();
    println!("Options:");
    println!("  --store <PATH>      Path to vector store file (default: data/store.bin)");
    println!("  --model <PATH>      Path to ONNX model (default: models/model.onnx)");
    println!("  --tokenizer <PATH>  Path to tokenizer (default: models/tokenizer.json)");
    println!("  --top-k <N>         Number of results (default: 10)");
    println!("  --json              Output as JSON");
    println!("  --help              Show this help");
}

#[derive(Debug)]
struct Args {
    query: String,
    store_path: PathBuf,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    top_k: usize,
    json_output: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1).peekable();

    let mut query = None;
    let mut store_path = PathBuf::from("data/store.bin");
    let mut model_path = PathBuf::from("models/model.onnx");
    let mut tokenizer_path = PathBuf::from("models/tokenizer.json");
    let mut top_k = 10;
    let mut json_output = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--store" => {
                store_path = args.next()
                    .context("--store requires a path argument")?
                    .into();
            }
            "--model" => {
                model_path = args.next()
                    .context("--model requires a path argument")?
                    .into();
            }
            "--tokenizer" => {
                tokenizer_path = args.next()
                    .context("--tokenizer requires a path argument")?
                    .into();
            }
            "--top-k" => {
                top_k = args.next()
                    .context("--top-k requires a number argument")?
                    .parse()
                    .context("Invalid top-k value")?;
            }
            "--json" => {
                json_output = true;
            }
            other if other.starts_with("--") => {
                anyhow::bail!("Unknown option: {}", other);
            }
            _ => {
                query = Some(arg);
            }
        }
    }

    let query = query.context("No query provided")?;

    Ok(Args {
        query,
        store_path,
        model_path,
        tokenizer_path,
        top_k,
        json_output,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = parse_args()?;

    info!(
        store = %args.store_path.display(),
        query = %args.query,
        top_k = args.top_k,
        "Loading vector store"
    );

    // Load vector store
    let start = Instant::now();
    let store = VectorStore::load(&args.store_path)
        .with_context(|| format!("Failed to load vector store from {}", args.store_path.display()))?;

    info!(
        docs = store.len(),
        load_time_ms = start.elapsed().as_millis(),
        "Vector store loaded"
    );

    // Initialize embedder
    let config = EmbedderConfig::new(&args.model_path, &args.tokenizer_path);
    let embedder = Embedder::new(config).await
        .context("Failed to initialize embedder")?;

    // Generate query embedding
    let start = Instant::now();
    let query_embedding = embedder.embed(&args.query).await
        .context("Failed to embed query")?;

    info!(
        embed_time_ms = start.elapsed().as_millis(),
        "Query embedded"
    );

    // Search
    let start = Instant::now();
    let hits = store.search(&query_embedding, args.top_k)
        .context("Search failed")?;

    info!(
        results = hits.len(),
        search_time_ms = start.elapsed().as_millis(),
        "Search complete"
    );

    // Output results
    if args.json_output {
        let output = serde_json::json!({
            "query": args.query,
            "results": hits.iter().map(|h| serde_json::json!({
                "id": h.id,
                "url": h.url,
                "title": h.title,
                "score": h.score,
                "snippet": h.snippet,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nQuery: {}\n", args.query);
        println!("Results ({} found):\n", hits.len());

        for (i, hit) in hits.iter().enumerate() {
            println!("{}. {} (score: {:.4})", i + 1, hit.title, hit.score);
            println!("   URL: {}", hit.url);
            println!("   {}", hit.snippet);
            println!();
        }
    }

    Ok(())
}
