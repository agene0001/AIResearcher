use anyhow::Result;
use reqwest::Client;
use crate::models::paper::{Paper, PaperSource};

/// Search OpenAlex — free, open scholarly metadata covering 250M+ works.
/// Optionally set OPENALEX_EMAIL for the polite pool (faster rate limits).
pub async fn search_openalex(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = Client::new();
    let per_page = max_results.min(50);

    let mut url = format!(
        "https://api.openalex.org/works?search={}&per_page={}&sort=relevance_score:desc",
        urlencoded(query),
        per_page,
    );

    // Polite pool: add mailto for better rate limits
    if let Ok(email) = std::env::var("OPENALEX_EMAIL") {
        url.push_str(&format!("&mailto={}", email));
    }

    let res: serde_json::Value = client
        .get(&url)
        .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut papers = Vec::new();

    if let Some(results) = res.get("results").and_then(|r| r.as_array()) {
        for item in results {
            let openalex_id = item["id"].as_str().unwrap_or("").to_string();
            let title = item["title"].as_str().unwrap_or("").to_string();

            // OpenAlex stores abstract as an inverted index — reconstruct it
            let abstract_text = reconstruct_abstract(item.get("abstract_inverted_index"));

            let year = item["publication_year"].as_u64().map(|y| y as u32);
            let doi = item.get("doi").and_then(|d| d.as_str())
                .map(|s| s.replace("https://doi.org/", ""));

            let authors: Vec<String> = item.get("authorships")
                .and_then(|a| a.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|a| a.get("author")
                        .and_then(|au| au.get("display_name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default();

            // Extract the short ID from the URL
            let short_id = openalex_id.split('/').last().unwrap_or(&openalex_id).to_string();

            if !title.is_empty() {
                let pdf_url = ["best_oa_location", "primary_location"].iter().find_map(|key| {
                    item.get(*key)
                        .filter(|v| !v.is_null())
                        .and_then(|loc| loc.get("pdf_url"))
                        .and_then(|u| u.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                });
                papers.push(Paper {
                    id: format!("openalex:{}", short_id),
                    title,
                    abstract_text,
                    content: None,
                    source: PaperSource::OpenAlex,
                    year,
                    doi,
                    url: Some(openalex_id),
                    pdf_url,
                    authors,
                });
            }
        }
    }

    Ok(papers)
}

/// Reconstruct abstract from OpenAlex's inverted index format.
/// The inverted index maps words to position arrays, e.g. {"the": [0, 5], "cat": [1]}
fn reconstruct_abstract(inverted_index: Option<&serde_json::Value>) -> String {
    let index = match inverted_index.and_then(|v| v.as_object()) {
        Some(obj) => obj,
        None => return String::new(),
    };

    let mut positions: Vec<(usize, &str)> = Vec::new();

    for (word, pos_array) in index {
        if let Some(arr) = pos_array.as_array() {
            for pos in arr {
                if let Some(p) = pos.as_u64() {
                    positions.push((p as usize, word.as_str()));
                }
            }
        }
    }

    positions.sort_by_key(|(pos, _)| *pos);
    positions.iter().map(|(_, word)| *word).collect::<Vec<_>>().join(" ")
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}
