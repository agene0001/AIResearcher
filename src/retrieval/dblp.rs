use anyhow::Result;
use reqwest::Client;
use crate::models::paper::{Paper, PaperSource};

/// Search DBLP — the standard CS bibliography database. Free, reliable, no key needed.
pub async fn search_dblp(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = Client::new();
    let limit = max_results.min(50);

    let url = format!(
        "https://dblp.org/search/publ/api?q={}&h={}&format=json",
        urlencoded(query),
        limit,
    );

    let res: serde_json::Value = client
        .get(&url)
        .header("User-Agent", "autoresearch-lab/0.1")
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut papers = Vec::new();

    let hits = res.get("result")
        .and_then(|r| r.get("hits"))
        .and_then(|h| h.get("hit"))
        .and_then(|h| h.as_array());

    if let Some(hits) = hits {
        for hit in hits {
            let info = match hit.get("info") {
                Some(i) => i,
                None => continue,
            };

            let title = info.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string();
            let dblp_url = info.get("url").and_then(|u| u.as_str()).unwrap_or("").to_string();
            let doi = info.get("doi").and_then(|d| d.as_str()).map(|s| s.to_string());
            let year = info.get("year").and_then(|y| y.as_str()).and_then(|y| y.parse::<u32>().ok());

            // DBLP authors can be a single object or an array
            let authors: Vec<String> = match info.get("authors").and_then(|a| a.get("author")) {
                Some(serde_json::Value::Array(arr)) => arr.iter()
                    .filter_map(|a| {
                        // Each author entry can be a string or an object with "text" field
                        a.as_str().map(|s| s.to_string())
                            .or_else(|| a.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
                    })
                    .collect(),
                Some(serde_json::Value::Object(obj)) => {
                    obj.get("text").and_then(|t| t.as_str()).map(|s| vec![s.to_string()]).unwrap_or_default()
                }
                Some(serde_json::Value::String(s)) => vec![s.clone()],
                _ => vec![],
            };

            // Generate a stable ID from the DBLP key
            let key = hit.get("@id").and_then(|k| k.as_str()).unwrap_or(&dblp_url).to_string();

            if !title.is_empty() {
                papers.push(Paper {
                    id: format!("dblp:{}", key),
                    title,
                    abstract_text: String::new(), // DBLP doesn't provide abstracts
                    content: None,
                    source: PaperSource::Dblp,
                    year,
                    doi,
                    url: if dblp_url.is_empty() { None } else { Some(dblp_url) },
                    pdf_url: None,
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
