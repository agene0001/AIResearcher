use anyhow::Result;
use reqwest::Client;
use crate::models::paper::{Paper, PaperSource};

/// Search Semantic Scholar's public API.
/// Free without a key (rate-limited to ~100 req/5 min). Set SEMANTIC_SCHOLAR_API_KEY for higher limits.
pub async fn search_semantic_scholar(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = Client::new();
    let limit = max_results.min(100); // API max is 100

    let mut req = client
        .get("https://api.semanticscholar.org/graph/v1/paper/search")
        .query(&[
            ("query", query),
            ("limit", &limit.to_string()),
            ("fields", &"paperId,title,abstract,year,authors,externalIds,url".to_string()),
        ]);

    // Use API key if available for higher rate limits
    if let Ok(key) = std::env::var("SEMANTIC_SCHOLAR_API_KEY") {
        req = req.header("x-api-key", key);
    }

    let res = req.send().await?.error_for_status()?;
    let body: serde_json::Value = res.json().await?;

    let mut papers = Vec::new();

    if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
        for item in data {
            let paper_id = item["paperId"].as_str().unwrap_or("").to_string();
            let title = item["title"].as_str().unwrap_or("").to_string();
            let abstract_text = item["abstract"].as_str().unwrap_or("").to_string();
            let year = item["year"].as_u64().map(|y| y as u32);
            let url = item["url"].as_str().map(|s| s.to_string());

            let doi = item.get("externalIds")
                .and_then(|ids| ids.get("DOI"))
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let authors: Vec<String> = item.get("authors")
                .and_then(|a| a.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|a| a.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default();

            if !title.is_empty() && !abstract_text.is_empty() {
                papers.push(Paper {
                    id: format!("s2:{}", paper_id),
                    title,
                    abstract_text,
                    content: None,
                    source: PaperSource::SemanticScholar,
                    year,
                    doi,
                    url,
                    authors,
                });
            }
        }
    }

    Ok(papers)
}
