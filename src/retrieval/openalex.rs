use anyhow::Result;
use crate::retrieval::HTTP;
use serde::Deserialize;
use crate::models::paper::Paper;
use crate::pipelines::ingest::{paper_from_work, Work};

#[derive(Default, Deserialize)]
#[serde(default)]
struct SearchResponse {
    results: Vec<Work>,
}

/// Search OpenAlex — free, open scholarly metadata covering 250M+ works.
/// Optionally set OPENALEX_EMAIL for the polite pool (faster rate limits).
pub async fn search_openalex(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = &*HTTP;
    let per_page = max_results.min(50);

    let mut url = format!(
        "https://api.openalex.org/works?search={}&per_page={}&sort=relevance_score:desc",
        urlencoded(query),
        per_page,
    );

    if let Ok(email) = std::env::var("OPENALEX_EMAIL") {
        if !email.is_empty() {
            url.push_str(&format!("&mailto={}", email));
        }
    }
    if let Ok(key) = std::env::var("OPENALEX_API_KEY") {
        if !key.is_empty() {
            url.push_str(&format!("&api_key={}", key));
        }
    }

    let res: SearchResponse = client
        .get(&url)
        .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    // Retrieval keeps papers even when they have no abstract, so this passes
    // `require_abstract = false` (ingest passes `true`). Parsing/field extraction
    // is shared with the ingest path via `paper_from_work`.
    let papers = res.results.iter()
        .filter_map(|w| paper_from_work(w, false))
        .collect();

    Ok(papers)
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}
