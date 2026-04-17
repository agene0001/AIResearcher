use anyhow::Result;
use reqwest::Client;
use crate::models::paper::{Paper, PaperSource};

/// Search Crossref — open metadata for 150M+ DOI-registered works.
/// Optionally set CROSSREF_MAILTO for the polite pool.
pub async fn search_crossref(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = Client::new();
    let rows = max_results.min(50);

    let mut url = format!(
        "https://api.crossref.org/works?query={}&rows={}&sort=relevance&order=desc",
        urlencoded(query),
        rows,
    );

    if let Ok(email) = std::env::var("CROSSREF_MAILTO") {
        url.push_str(&format!("&mailto={}", email));
    }

    let res: serde_json::Value = client
        .get(&url)
        .header("User-Agent", "autoresearch-lab/0.1 (https://github.com/autoresearch-lab; mailto:research@example.com)")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut papers = Vec::new();

    if let Some(items) = res.get("message")
        .and_then(|m| m.get("items"))
        .and_then(|i| i.as_array())
    {
        for item in items {
            let doi = item.get("DOI").and_then(|d| d.as_str()).unwrap_or("").to_string();
            let title = item.get("title")
                .and_then(|t| t.as_array())
                .and_then(|arr| arr.first())
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();

            // Crossref abstract may or may not be present
            let abstract_text = item.get("abstract")
                .and_then(|a| a.as_str())
                .unwrap_or("")
                // Strip JATS XML tags that Crossref sometimes includes
                .replace("<jats:p>", "")
                .replace("</jats:p>", "")
                .replace("<jats:italic>", "")
                .replace("</jats:italic>", "")
                .trim()
                .to_string();

            let year = item.get("published-print")
                .or_else(|| item.get("published-online"))
                .or_else(|| item.get("created"))
                .and_then(|d| d.get("date-parts"))
                .and_then(|dp| dp.as_array())
                .and_then(|arr| arr.first())
                .and_then(|inner| inner.as_array())
                .and_then(|inner| inner.first())
                .and_then(|y| y.as_u64())
                .map(|y| y as u32);

            let authors: Vec<String> = item.get("author")
                .and_then(|a| a.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|a| {
                        let given = a.get("given").and_then(|g| g.as_str()).unwrap_or("");
                        let family = a.get("family").and_then(|f| f.as_str()).unwrap_or("");
                        if family.is_empty() { None } else { Some(format!("{} {}", given, family).trim().to_string()) }
                    })
                    .collect())
                .unwrap_or_default();

            let url = if !doi.is_empty() {
                Some(format!("https://doi.org/{}", doi))
            } else {
                None
            };

            if !title.is_empty() && !doi.is_empty() {
                papers.push(Paper {
                    id: format!("doi:{}", doi),
                    title,
                    abstract_text,
                    content: None,
                    source: PaperSource::Crossref,
                    year,
                    doi: Some(doi),
                    url,
                    authors,
                });
            }
        }
    }

    Ok(papers)
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}
