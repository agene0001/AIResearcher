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
    loop {
        let url = format!("{}?per_page=200&cursor={}", base_url, cursor);
        let page: Page = client
            .get(&url)
            .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
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
