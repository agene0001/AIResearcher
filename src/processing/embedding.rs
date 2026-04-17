use anyhow::Result;
use crate::processing::harrier;

/// Generate an embedding for a query (applies retrieval instruction prefix).
/// Uses harrier-oss-v1-0.6b locally — free, no API key needed.
pub async fn generate_embedding(text: &str) -> Result<Vec<f32>> {
    // Run the embedding on a blocking thread since candle is synchronous
    let text = text.to_string();
    tokio::task::spawn_blocking(move || harrier::embed_query(&text))
        .await?
}

/// Generate an embedding for a document (no instruction prefix).
pub async fn generate_document_embedding(text: &str) -> Result<Vec<f32>> {
    let text = text.to_string();
    tokio::task::spawn_blocking(move || harrier::embed_document(&text))
        .await?
}

/// Generate embeddings for a batch of documents in a single forward pass.
pub async fn generate_document_embeddings_batch(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    tokio::task::spawn_blocking(move || harrier::embed_document_batch(&texts))
        .await?
}
