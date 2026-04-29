//! Tier-2 PDF URL resolver. Runs at read time when `papers.pdf_url` is NULL,
//! consulting external services that may know about a free copy that OpenAlex
//! didn't have at ingest time.
//!
//! Resolution order, ordered by likelihood × cost:
//!   1. arXiv DOI pattern: `10.48550/arXiv.X` → arxiv.org/pdf/X.pdf  (free, regex)
//!   2. Unpaywall API by DOI                                          (free, 1 HTTP)
//!   3. arXiv title search + author/title similarity match            (free, 1 HTTP)
//!   4. Semantic Scholar API by DOI                                   (free, 1 HTTP)
//!
//! On a hit, the caller updates `papers.pdf_url` so future reads skip this entirely.

use anyhow::Result;
use std::collections::HashSet;
use std::time::Duration;

use crate::storage::postgres::ResolverMeta;

const HTTP_TIMEOUT: Duration = Duration::from_secs(15);
const TITLE_SIM_THRESHOLD: f64 = 0.7;

/// RFC-3986 percent-encoding for arbitrary URL components. We avoid pulling in
/// a dedicated crate just for this — only `unreserved` chars pass through.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Resolve a PDF URL for a paper using external services. Returns None if no
/// step found anything. Each step's failure is logged but doesn't abort the chain.
pub async fn resolve_pdf_url(meta: &ResolverMeta) -> Result<Option<String>> {
    if let Some(url) = step_arxiv_doi(meta) {
        tracing::info!(source = "arxiv_doi", url = %url, "resolver hit");
        return Ok(Some(url));
    }

    if let Some(doi) = meta.doi.as_deref() {
        match step_unpaywall(doi).await {
            Ok(Some(url)) => {
                tracing::info!(source = "unpaywall", url = %url, "resolver hit");
                return Ok(Some(url));
            }
            Ok(None) => tracing::debug!("unpaywall: no OA copy for {}", doi),
            Err(e) => tracing::warn!(error = ?e, "unpaywall lookup failed"),
        }
    }

    if !meta.title.trim().is_empty() {
        match step_arxiv_title_search(&meta.title, meta.first_author.as_deref()).await {
            Ok(Some(url)) => {
                tracing::info!(source = "arxiv_title", url = %url, "resolver hit");
                return Ok(Some(url));
            }
            Ok(None) => tracing::debug!("arxiv title search: no match"),
            Err(e) => tracing::warn!(error = ?e, "arxiv title search failed"),
        }
    }

    if let Some(doi) = meta.doi.as_deref() {
        match step_semantic_scholar(doi).await {
            Ok(Some(url)) => {
                tracing::info!(source = "semantic_scholar", url = %url, "resolver hit");
                return Ok(Some(url));
            }
            Ok(None) => tracing::debug!("semantic scholar: no openAccessPdf for {}", doi),
            Err(e) => tracing::warn!(error = ?e, "semantic scholar lookup failed"),
        }
    }

    Ok(None)
}

/// Step 1: arXiv-issued DOIs encode the arXiv ID directly.
/// e.g. doi `10.48550/arXiv.2401.12345` → `https://arxiv.org/pdf/2401.12345.pdf`
fn step_arxiv_doi(meta: &ResolverMeta) -> Option<String> {
    let doi = meta.doi.as_deref()?;
    let lower = doi.to_ascii_lowercase();
    let prefix = "10.48550/arxiv.";
    let id = lower.strip_prefix(prefix)?;
    if id.is_empty() {
        return None;
    }
    Some(format!("https://arxiv.org/pdf/{}.pdf", id))
}

fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .user_agent("autoresearch-lab/0.1 (research tool)")
        .build()?)
}

fn polite_email() -> String {
    std::env::var("OPENALEX_EMAIL").unwrap_or_else(|_| "anonymous@example.com".to_string())
}

/// Step 2: Unpaywall aggregates "legitimately free PDF" locations across arXiv,
/// institutional repos, journal-published OA, and author homepages.
/// Free, requires email param. Docs: https://unpaywall.org/products/api
async fn step_unpaywall(doi: &str) -> Result<Option<String>> {
    let url = format!(
        "https://api.unpaywall.org/v2/{}?email={}",
        urlencode(doi),
        urlencode(&polite_email()),
    );
    let client = http_client()?;
    let resp = client.get(&url).send().await?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let body: serde_json::Value = resp.error_for_status()?.json().await?;

    // Try best_oa_location.url_for_pdf, then url, then walk oa_locations[].
    let pick = |loc: &serde_json::Value| -> Option<String> {
        for key in ["url_for_pdf", "url"] {
            if let Some(s) = loc
                .get(key)
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                return Some(s.to_string());
            }
        }
        None
    };

    if let Some(loc) = body.get("best_oa_location").filter(|v| !v.is_null()) {
        if let Some(url) = pick(loc) {
            return Ok(Some(url));
        }
    }
    if let Some(arr) = body.get("oa_locations").and_then(|v| v.as_array()) {
        for loc in arr {
            if let Some(url) = pick(loc) {
                return Ok(Some(url));
            }
        }
    }
    Ok(None)
}

/// Step 3: search arXiv's API by title and pick the top match if the title
/// similarity (Jaccard over content tokens) is high and, if available, the
/// first-author last name matches. Catches CS papers with arXiv preprints
/// that OpenAlex didn't link to.
async fn step_arxiv_title_search(
    title: &str,
    first_author: Option<&str>,
) -> Result<Option<String>> {
    // arXiv API treats double-quotes as exact-phrase. Encode the title verbatim.
    let query = format!("ti:{}", title.replace('"', " "));
    let url = format!(
        "https://export.arxiv.org/api/query?search_query={}&max_results=5",
        urlencode(&query)
    );
    let client = http_client()?;
    let xml = client.get(&url).send().await?.error_for_status()?.text().await?;

    let doc = roxmltree::Document::parse(&xml)?;
    let want_authors_lc = first_author.map(last_name);

    let mut best: Option<(f64, String)> = None;
    for entry in doc.descendants().filter(|n| n.has_tag_name("entry")) {
        let arxiv_title = entry
            .children()
            .find(|n| n.has_tag_name("title"))
            .and_then(|n| n.text())
            .unwrap_or("")
            .to_string();
        let arxiv_id = entry
            .children()
            .find(|n| n.has_tag_name("id"))
            .and_then(|n| n.text())
            .unwrap_or("");
        let arxiv_authors: Vec<String> = entry
            .children()
            .filter(|n| n.has_tag_name("author"))
            .filter_map(|a| a.children().find(|n| n.has_tag_name("name")))
            .filter_map(|n| n.text().map(|s| s.to_string()))
            .collect();

        let sim = title_similarity(title, &arxiv_title);
        if sim < TITLE_SIM_THRESHOLD {
            continue;
        }
        // Author check: if we know a first author, require any arXiv author
        // last-name to match. If we don't know, accept on title alone.
        if let Some(want) = &want_authors_lc {
            let any_match = arxiv_authors
                .iter()
                .any(|a| last_name(a) == *want);
            if !any_match {
                continue;
            }
        }
        // Extract bare arxiv id from the entry's id URL: http://arxiv.org/abs/XXXX.XXXX[vN]
        let bare = arxiv_id
            .rsplit('/')
            .next()
            .map(|s| s.split('v').next().unwrap_or(s))
            .unwrap_or("");
        if bare.is_empty() {
            continue;
        }
        let pdf = format!("https://arxiv.org/pdf/{}.pdf", bare);
        let candidate = (sim, pdf);
        if best.as_ref().map(|b| candidate.0 > b.0).unwrap_or(true) {
            best = Some(candidate);
        }
    }
    Ok(best.map(|(_, url)| url))
}

/// Step 4: Semantic Scholar exposes `openAccessPdf.url` for many DOIs that
/// OpenAlex/Unpaywall miss. Free, optional API key for higher rate limits.
async fn step_semantic_scholar(doi: &str) -> Result<Option<String>> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/DOI:{}?fields=openAccessPdf",
        urlencode(doi)
    );
    let mut req = http_client()?.get(&url);
    if let Ok(key) = std::env::var("SEMANTIC_SCHOLAR_API_KEY") {
        if !key.is_empty() {
            req = req.header("x-api-key", key);
        }
    }
    let resp = req.send().await?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let body: serde_json::Value = resp.error_for_status()?.json().await?;
    let url = body
        .get("openAccessPdf")
        .and_then(|v| v.get("url"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    Ok(url)
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "by",
    "is", "are", "from", "via", "using", "based",
];

fn tokenize(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !STOPWORDS.contains(t))
        .map(|t| t.to_string())
        .collect()
}

fn title_similarity(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    let inter = ta.intersection(&tb).count() as f64;
    let union = ta.union(&tb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Extract a normalized last name from a full author string.
/// Handles "Doe, John" and "John Doe" forms.
fn last_name(full: &str) -> String {
    let s = full.trim();
    if let Some(comma_idx) = s.find(',') {
        s[..comma_idx].trim().to_lowercase()
    } else {
        s.split_whitespace()
            .last()
            .unwrap_or("")
            .to_lowercase()
    }
}

