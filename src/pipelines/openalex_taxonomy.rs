//! OpenAlex field/subfield name resolver.
//!
//! OpenAlex exposes its topic hierarchy at `/fields` (26 entries) and
//! `/subfields` (252 entries). Rather than make the user hunt for numeric IDs,
//! we fetch both lists once at startup and let them type human-readable names
//! like `"Computer Science"` or `"Artificial Intelligence"` on the CLI.
//! Match is case-insensitive; on miss we suggest close matches via substring.

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct Page {
    results: Vec<Entry>,
    meta: Meta,
}

#[derive(Deserialize)]
struct Meta {
    next_cursor: Option<String>,
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    display_name: String,
}

pub struct Taxonomy {
    fields: Vec<(u32, String)>,
    subfields: Vec<(u32, String)>,
}

impl Taxonomy {
    pub async fn fetch(client: &reqwest::Client) -> Result<Self> {
        let fields = fetch_all(client, "https://api.openalex.org/fields")
            .await
            .context("fetching OpenAlex /fields")?;
        let subfields = fetch_all(client, "https://api.openalex.org/subfields")
            .await
            .context("fetching OpenAlex /subfields")?;
        tracing::info!(
            fields = fields.len(),
            subfields = subfields.len(),
            "OpenAlex taxonomy loaded"
        );
        Ok(Self { fields, subfields })
    }

    pub fn resolve_field(&self, name: &str) -> Result<u32> {
        resolve(&self.fields, name, "field")
    }

    pub fn resolve_subfields(&self, names: &[String]) -> Result<Vec<u32>> {
        names.iter().map(|n| resolve(&self.subfields, n, "subfield")).collect()
    }
}

async fn fetch_all(client: &reqwest::Client, base_url: &str) -> Result<Vec<(u32, String)>> {
    let mut out = Vec::new();
    let mut cursor = "*".to_string();
    let mailto_qs = std::env::var("OPENALEX_EMAIL")
        .ok()
        .filter(|e| !e.is_empty())
        .map(|e| format!("&mailto={}", e))
        .unwrap_or_default();

    loop {
        let url = format!(
            "{}?per_page=200&cursor={}{}",
            base_url, cursor, mailto_qs,
        );
        let page: Page = fetch_page_with_retry(client, &url).await?;
        for e in page.results {
            // e.id looks like "https://openalex.org/subfields/1702"
            if let Some(id) = e.id.rsplit('/').next().and_then(|s| s.parse::<u32>().ok()) {
                out.push((id, e.display_name));
            }
        }
        match page.meta.next_cursor {
            Some(c) if !c.is_empty() => cursor = c,
            _ => break,
        }
    }
    Ok(out)
}

/// Retry-aware fetch for taxonomy pages. Honors `Retry-After` on 429, and
/// exponential-backoffs on transient network errors. Capped at 8 attempts —
/// taxonomy is small, no point spinning forever.
async fn fetch_page_with_retry(client: &reqwest::Client, url: &str) -> Result<Page> {
    let mut attempt: u32 = 0;
    loop {
        let res = client
            .get(url)
            .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
            .send()
            .await;

        let resp = match res {
            Ok(r) => r,
            Err(e) => {
                attempt += 1;
                if attempt > 8 {
                    return Err(e.into());
                }
                let delay = retry_delay(attempt);
                tracing::warn!(attempt = attempt, delay_s = delay, error = ?e, "taxonomy fetch error, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }
        };

        if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            attempt += 1;
            if attempt > 8 {
                return Err(anyhow::anyhow!("taxonomy fetch: 429 after 8 retries"));
            }
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let delay = retry_after.unwrap_or_else(|| retry_delay(attempt));
            tracing::warn!(attempt = attempt, delay_s = delay, retry_after = ?retry_after, "taxonomy 429, retrying");
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
            continue;
        }

        let resp = resp.error_for_status()?;
        let page: Page = resp.json().await?;
        return Ok(page);
    }
}

/// Same exponential schedule used by ingest: 5, 10, 20, 40, 80, 160, 300, 300...
fn retry_delay(attempt: u32) -> u64 {
    let shift = attempt.min(6);
    (5u64.saturating_mul(1u64 << shift)).min(300)
}

fn resolve(entries: &[(u32, String)], name: &str, kind: &str) -> Result<u32> {
    let norm = name.trim().to_lowercase();
    if let Some((id, _)) = entries.iter().find(|(_, n)| n.to_lowercase() == norm) {
        return Ok(*id);
    }
    let suggestions: Vec<String> = entries
        .iter()
        .filter(|(_, n)| {
            let low = n.to_lowercase();
            low.contains(&norm) || norm.contains(&low)
        })
        .take(10)
        .map(|(id, n)| format!("  - \"{}\" (id {})", n, id))
        .collect();
    if suggestions.is_empty() {
        anyhow::bail!(
            "Unknown OpenAlex {} '{}'. Browse the full list at https://api.openalex.org/{}s",
            kind, name, kind
        );
    }
    anyhow::bail!(
        "Unknown OpenAlex {} '{}'. Did you mean:\n{}",
        kind, name, suggestions.join("\n")
    );
}
