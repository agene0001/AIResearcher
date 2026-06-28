use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::models::paper::{Paper, PaperSource};
use crate::processing::embedding::generate_document_embeddings_batch;
use crate::storage::postgres::DbClient;

/// Exponential backoff for API retries: 5, 10, 20, 40, 80, 160, 300, 300, ...
/// Unbounded in retry count, capped at 300s per sleep.
fn retry_delay(attempt: u32) -> u64 {
    let shift = attempt.min(6);
    (5u64.saturating_mul(1u64 << shift)).min(300)
}

/// Topic filter for ingest, resolved from CLI `--field`/`--subfield`.
/// Used by both the snapshot path (local JSONL match) and the API path
/// (converted to an OpenAlex `filter=` clause).
#[derive(Debug, Clone)]
pub enum TopicFilter {
    Field(u32),
    Subfields(Vec<u32>),
}

impl TopicFilter {
    /// Build the OpenAlex API filter clause, e.g.
    /// `"topics.field.id:17"` or `"topics.subfield.id:1702|1707"`.
    pub fn to_api_clause(&self) -> String {
        match self {
            TopicFilter::Field(id) => format!("topics.field.id:{}", id),
            TopicFilter::Subfields(ids) => format!(
                "topics.subfield.id:{}",
                ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("|")
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Typed OpenAlex work schema (only the fields we use).
//
// Deserializing straight into these structs skips building a `serde_json::Value`
// DOM for every record — faster and far less allocation on the snapshot hot path
// (2M+ records, most filtered out). Every field is optional so missing/null
// values degrade gracefully instead of failing the whole record.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Work {
    id: Option<String>,
    title: Option<String>,
    doi: Option<String>,
    publication_year: Option<u32>,
    abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,
    authorships: Option<Vec<Authorship>>,
    topics: Option<Vec<Topic>>,
    best_oa_location: Option<Location>,
    primary_location: Option<Location>,
    locations: Option<Vec<Location>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Authorship {
    author: Option<Author>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Author {
    display_name: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Topic {
    field: Option<OaEntity>,
    subfield: Option<OaEntity>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct OaEntity {
    id: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Location {
    pdf_url: Option<String>,
    landing_page_url: Option<String>,
    is_oa: Option<bool>,
}

/// One page of the OpenAlex `/works` API response.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ApiResponse {
    meta: Option<Meta>,
    /// Kept as raw values, not `Vec<Work>`, so a single malformed record can't
    /// fail the whole-page deserialize (which would hang the retry loop). Each
    /// is decoded into `Work` independently where it's consumed.
    results: Vec<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Meta {
    count: Option<u64>,
    next_cursor: Option<String>,
}

/// Parse the trailing numeric id out of an OpenAlex entity URL
/// (`https://openalex.org/subfields/1702` -> `1702`).
fn entity_numeric_id(e: &OaEntity) -> Option<u32> {
    e.id.as_deref()
        .and_then(|s| s.rsplit('/').next())
        .and_then(|s| s.parse::<u32>().ok())
}

/// Derive a direct PDF URL from a `landing_page_url` for known OA hosts that
/// publish at predictable paths but where OpenAlex frequently leaves `pdf_url`
/// itself null. Big coverage wins for CS specifically:
///   - arXiv:        /abs/{id}       → /pdf/{id}.pdf
///   - ACL Anthology: /{id}          → /{id}.pdf
///   - OpenReview:   /forum?id={id}  → /pdf?id={id}
///   - bioRxiv/medRxiv: /content/{id} → /content/{id}.full.pdf
///   - already-pdf landing pages (e.g. ".pdf" suffix): used as-is
fn derive_pdf_from_landing(landing: &str) -> Option<String> {
    let l = landing.trim();
    if l.is_empty() {
        return None;
    }

    // arXiv abstract page → derived PDF
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/"] {
        if let Some(rest) = l.strip_prefix(prefix) {
            let id = rest.trim_end_matches('/');
            // Strip a trailing version suffix like /v2 since arxiv.org/pdf accepts both
            return Some(format!("https://arxiv.org/pdf/{}.pdf", id));
        }
    }

    // ACL Anthology
    for prefix in ["https://aclanthology.org/", "http://aclanthology.org/"] {
        if let Some(rest) = l.strip_prefix(prefix) {
            let id = rest.trim_end_matches('/');
            if !id.is_empty() && !id.ends_with(".pdf") {
                return Some(format!("https://aclanthology.org/{}.pdf", id));
            }
        }
    }

    // OpenReview forum → PDF
    if let Some(idx) = l.find("openreview.net/forum?id=") {
        let id = &l[idx + "openreview.net/forum?id=".len()..];
        let id = id.split('&').next().unwrap_or(id);
        if !id.is_empty() {
            return Some(format!("https://openreview.net/pdf?id={}", id));
        }
    }

    // bioRxiv / medRxiv: content/<path> → content/<path>.full.pdf
    if (l.contains("biorxiv.org/content/") || l.contains("medrxiv.org/content/")) && !l.ends_with(".pdf") {
        let trimmed = l.trim_end_matches('/');
        let trimmed = trimmed.trim_end_matches(".full");
        return Some(format!("{}.full.pdf", trimmed));
    }

    // Already a direct PDF link
    if l.to_ascii_lowercase().ends_with(".pdf") {
        return Some(l.to_string());
    }

    None
}

/// Extract a direct PDF URL from an OpenAlex work.
///
/// Resolution order, widest-coverage first:
///   1. `best_oa_location.pdf_url` — OpenAlex's curated pick.
///   2. `primary_location.pdf_url` — publisher-of-record when it's open.
///   3. `locations[]` entries with `is_oa = true` and a non-empty `pdf_url`.
///   4. `locations[]` entries with a non-empty `pdf_url`, OA flag aside.
///   5. Steps 1-4 again, but trying to *derive* the PDF URL from the location's
///      `landing_page_url` for arXiv / ACL / OpenReview / bioRxiv when OpenAlex
///      left `pdf_url` itself null — they very often do this for arXiv abstracts
///      even though the PDF is a one-line URL transform away.
fn extract_pdf_url(work: &Work) -> Option<String> {
    let pick_direct = |loc: &Location| -> Option<String> {
        loc.pdf_url.as_deref().filter(|s| !s.is_empty()).map(|s| s.to_string())
    };
    let pick_derived = |loc: &Location| -> Option<String> {
        loc.landing_page_url.as_deref().and_then(derive_pdf_from_landing)
    };

    let curated = [work.best_oa_location.as_ref(), work.primary_location.as_ref()];
    let locations = work.locations.as_deref().unwrap_or(&[]);

    // Pass 1: direct pdf_url across curated picks + locations[]
    for loc in curated.into_iter().flatten() {
        if let Some(url) = pick_direct(loc) {
            return Some(url);
        }
    }
    for loc in locations.iter().filter(|l| l.is_oa.unwrap_or(false)) {
        if let Some(url) = pick_direct(loc) {
            return Some(url);
        }
    }
    for loc in locations {
        if let Some(url) = pick_direct(loc) {
            return Some(url);
        }
    }

    // Pass 2: derive from landing_page_url for known patterns
    for loc in curated.into_iter().flatten() {
        if let Some(url) = pick_derived(loc) {
            return Some(url);
        }
    }
    for loc in locations.iter().filter(|l| l.is_oa.unwrap_or(false)) {
        if let Some(url) = pick_derived(loc) {
            return Some(url);
        }
    }
    for loc in locations {
        if let Some(url) = pick_derived(loc) {
            return Some(url);
        }
    }

    None
}

/// Check whether a work's `topics[]` contains a topic matching the filter.
fn matches_topic_filter(topics: &[Topic], filter: &TopicFilter) -> bool {
    topics.iter().any(|t| match filter {
        TopicFilter::Field(id) => t.field.as_ref().and_then(entity_numeric_id) == Some(*id),
        TopicFilter::Subfields(ids) => t
            .subfield
            .as_ref()
            .and_then(entity_numeric_id)
            .map_or(false, |sid| ids.contains(&sid)),
    })
}

/// Build a `Paper` from a parsed OpenAlex work. Returns `None` (skip) when the
/// work has no title, or — when `require_abstract` — no usable abstract.
/// Shared by the snapshot ingest, API ingest, and the `search_openalex`
/// retrieval path.
pub(crate) fn paper_from_work(work: &Work, require_abstract: bool) -> Option<Paper> {
    let title = work.title.as_deref().filter(|t| !t.is_empty())?;

    let abstract_text = work
        .abstract_inverted_index
        .as_ref()
        .map(reconstruct_abstract)
        .unwrap_or_default();
    if require_abstract && abstract_text.is_empty() {
        return None;
    }

    let openalex_id = work.id.as_deref().unwrap_or("");
    let short_id = openalex_id.rsplit('/').next().unwrap_or(openalex_id);
    let doi = work.doi.as_deref().map(|s| s.replace("https://doi.org/", ""));

    let authors: Vec<String> = work
        .authorships
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|a| a.author.as_ref()?.display_name.clone())
        .collect();

    let pdf_url = extract_pdf_url(work);

    Some(Paper {
        id: format!("openalex:{}", short_id),
        title: title.to_string(),
        abstract_text,
        content: None,
        source: PaperSource::OpenAlex,
        year: work.publication_year,
        doi,
        url: Some(openalex_id.to_string()),
        pdf_url,
        authors,
    })
}

/// Read, decompress and parse one OpenAlex snapshot `.gz` part file, applying
/// the snapshot-only filters (abstract present, year, topic). Returns the
/// papers that passed plus the number of records skipped. Pure/blocking — meant
/// to run on a `spawn_blocking` thread so many files parse in parallel.
fn parse_gz_file(path: &Path, min_year: u32, topic_filter: &TopicFilter) -> Result<(Vec<Paper>, usize)> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(GzDecoder::new(file));

    let mut papers = Vec::new();
    let mut skipped = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let work: Work = match serde_json::from_str(&line) {
            Ok(w) => w,
            Err(_) => continue,
        };

        // Filter: must have an abstract.
        if work.abstract_inverted_index.is_none() {
            skipped += 1;
            continue;
        }
        // Filter: publication year.
        if work.publication_year.map_or(true, |y| y < min_year) {
            skipped += 1;
            continue;
        }
        // Filter: must match the selected field/subfield(s).
        if !matches_topic_filter(work.topics.as_deref().unwrap_or(&[]), topic_filter) {
            skipped += 1;
            continue;
        }

        match paper_from_work(&work, true) {
            Some(p) => papers.push(p),
            None => skipped += 1,
        }
    }

    Ok((papers, skipped))
}

/// Ingest papers from an OpenAlex snapshot directory.
/// Expects the directory structure: <snapshot_dir>/data/works/updated_date=*/part_*.gz
///
/// Pipelined: a pool of blocking workers decompress+parse `.gz` files in
/// parallel (CPU), the main task batches and embeds (GPU), and a writer task
/// inserts into Postgres (DB) — so all three stages overlap instead of running
/// strictly one-after-another.
pub async fn ingest_snapshot(
    snapshot_dir: &str,
    min_year: u32,
    batch_size: usize,
    max_papers: Option<usize>,
    topic_filter: &TopicFilter,
) -> Result<()> {
    let works_dir = Path::new(snapshot_dir).join("data").join("works");
    if !works_dir.exists() {
        anyhow::bail!(
            "Works directory not found at {}. Download with:\n  aws s3 sync \"s3://openalex/data/works\" \"{}/data/works\" --no-sign-request",
            works_dir.display(),
            snapshot_dir
        );
    }

    let db = DbClient::new().await.context("Database connection required for ingestion")?;

    // Collect all .gz files
    let mut gz_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&works_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // updated_date=YYYY-MM-DD directories
            for inner in fs::read_dir(&path)? {
                let inner = inner?;
                let inner_path = inner.path();
                if inner_path.extension().map(|e| e == "gz").unwrap_or(false) {
                    gz_files.push(inner_path);
                }
            }
        }
    }

    gz_files.sort();
    let num_files = gz_files.len();
    eprintln!("Found {} compressed files to process", num_files);

    let total_ingested = Arc::new(AtomicUsize::new(0));
    let total_errors = Arc::new(AtomicUsize::new(0));
    let mut total_skipped: usize = 0;
    let mut dispatched: usize = 0;
    let start_time = Instant::now();

    let pb = ProgressBar::new(num_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} files | {msg}")
            .unwrap()
    );

    // Writer task: inserts arrive over a bounded channel so DB writes overlap
    // with the next batch's embedding. Bound is small so we apply backpressure
    // instead of buffering unbounded papers in memory.
    let (ins_tx, mut ins_rx) =
        tokio::sync::mpsc::channel::<(Vec<Paper>, Vec<Vec<f32>>)>(2);
    let writer = {
        let db = db.clone();
        let ingested = total_ingested.clone();
        let errors = total_errors.clone();
        tokio::spawn(async move {
            while let Some((papers, embeddings)) = ins_rx.recv().await {
                match db.insert_papers_batch(&papers, &embeddings).await {
                    Ok(_) => {
                        ingested.fetch_add(papers.len(), Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!("Batch DB insert error: {}", e);
                        errors.fetch_add(papers.len(), Ordering::Relaxed);
                    }
                }
            }
        })
    };

    // Parse stage: up to `parse_concurrency` files decompressed+parsed at once.
    let parse_concurrency = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let topic_filter = Arc::new(topic_filter.clone());
    let mut parsed = stream::iter(gz_files)
        .map(|path| {
            let tf = topic_filter.clone();
            tokio::task::spawn_blocking(move || parse_gz_file(&path, min_year, &tf))
        })
        .buffer_unordered(parse_concurrency);

    let mut buffer: Vec<Paper> = Vec::new();
    let mut reached_cap = false;

    'outer: while let Some(joined) = parsed.next().await {
        match joined {
            Ok(Ok((papers, skipped))) => {
                total_skipped += skipped;
                buffer.extend(papers);
            }
            Ok(Err(e)) => {
                tracing::warn!("file parse error: {:#}", e);
            }
            Err(e) => {
                tracing::warn!("parse task join error: {}", e);
            }
        }
        pb.inc(1);

        while buffer.len() >= batch_size {
            if let Some(max) = max_papers {
                if dispatched >= max {
                    reached_cap = true;
                    break 'outer;
                }
            }
            let batch: Vec<Paper> = buffer.drain(..batch_size).collect();
            let texts: Vec<String> = batch.iter()
                .map(|p| format!("{} {}", p.title, p.abstract_text))
                .collect();

            match generate_document_embeddings_batch(texts).await {
                Ok(embeddings) => {
                    dispatched += batch.len();
                    if ins_tx.send((batch, embeddings)).await.is_err() {
                        // Writer task is gone — nothing more we can do.
                        break 'outer;
                    }
                }
                Err(e) => {
                    tracing::warn!("Batch embedding error: {}", e);
                    total_errors.fetch_add(batch.len(), Ordering::Relaxed);
                }
            }

            let ingested = total_ingested.load(Ordering::Relaxed);
            let errors = total_errors.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { ingested as f64 / elapsed } else { 0.0 };
            pb.set_message(format!(
                "{} ingested, {} skipped, {} errors | {:.1}/s | ETA: {:.1}h for 2M",
                ingested, total_skipped, errors, rate,
                if rate > 0.0 { (2_000_000.0 - ingested as f64) / rate / 3600.0 } else { 0.0 }
            ));
        }
    }

    // Flush the final partial batch (respecting the cap).
    if !reached_cap && !buffer.is_empty() {
        let remaining = max_papers.map_or(usize::MAX, |max| max.saturating_sub(dispatched));
        let take = buffer.len().min(remaining);
        if take > 0 {
            let batch: Vec<Paper> = buffer.drain(..take).collect();
            let texts: Vec<String> = batch.iter()
                .map(|p| format!("{} {}", p.title, p.abstract_text))
                .collect();
            match generate_document_embeddings_batch(texts).await {
                Ok(embeddings) => {
                    let _ = ins_tx.send((batch, embeddings)).await;
                }
                Err(e) => {
                    tracing::warn!("Batch embedding error: {}", e);
                    total_errors.fetch_add(batch.len(), Ordering::Relaxed);
                }
            }
        }
    }

    // Signal the writer there's no more work and wait for outstanding inserts.
    drop(ins_tx);
    let _ = writer.await;

    pb.finish_with_message("Done!");

    let ingested = total_ingested.load(Ordering::Relaxed);
    let errors = total_errors.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  INGESTION COMPLETE");
    eprintln!("{}", "=".repeat(60));
    eprintln!("  Papers ingested:  {}", ingested);
    eprintln!("  Papers skipped:   {}", total_skipped);
    eprintln!("  Errors:           {}", errors);
    eprintln!("  Total time:       {:.1?}", elapsed);
    eprintln!("  Rate:             {:.1} papers/sec", ingested as f64 / elapsed.as_secs_f64());
    eprintln!("{}", "=".repeat(60));

    Ok(())
}

/// Fetch one page of the OpenAlex `/works` API, retrying forever with backoff on
/// transport errors, 429s (honoring `Retry-After`) and decode failures. Returns
/// the parsed page once it succeeds.
async fn fetch_page(
    client: &reqwest::Client,
    base_url: &str,
    cursor: &str,
    pb: &ProgressBar,
) -> Result<ApiResponse> {
    let url = format!("{}&cursor={}", base_url, cursor);
    let mut backoff: u32 = 0;

    loop {
        let res = match client
            .get(&url)
            .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let delay = retry_delay(backoff);
                backoff = backoff.saturating_add(1);
                pb.println(format!(
                    "API request failed (attempt {}): {:#}. Retrying in {}s...",
                    backoff, e, delay
                ));
                tracing::warn!(attempt = backoff, delay_s = delay, error = ?e, "API request failed, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }
        };

        if res.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = res.headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let delay = retry_after.unwrap_or_else(|| retry_delay(backoff));
            backoff = backoff.saturating_add(1);
            let suffix = if retry_after.is_some() { " (from Retry-After)" } else { "" };
            pb.println(format!(
                "Rate limited (429, attempt {}). Waiting {}s{}...",
                backoff, delay, suffix
            ));
            tracing::warn!(attempt = backoff, delay_s = delay, retry_after = ?retry_after, "429 rate-limited");
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
            continue;
        }

        match res.error_for_status() {
            Ok(r) => match r.json::<ApiResponse>().await {
                Ok(body) => return Ok(body),
                Err(e) => {
                    let delay = retry_delay(backoff);
                    backoff = backoff.saturating_add(1);
                    pb.println(format!(
                        "API JSON decode failed (attempt {}): {:#}. Retrying in {}s...",
                        backoff, e, delay
                    ));
                    tracing::warn!(attempt = backoff, delay_s = delay, error = ?e, "JSON decode failed");
                    tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                    continue;
                }
            },
            Err(e) => {
                let delay = retry_delay(backoff);
                backoff = backoff.saturating_add(1);
                pb.println(format!(
                    "API HTTP error (attempt {}): {:#}. Retrying in {}s...",
                    backoff, e, delay
                ));
                tracing::warn!(attempt = backoff, delay_s = delay, error = ?e, "HTTP error");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }
        }
    }
}

/// Ingest papers via the OpenAlex API with cursor-based pagination.
/// `topics_filter` is a fully-formed OpenAlex filter clause like
/// `"topics.field.id:17"` or `"topics.subfield.id:1702|1707"` — built by the caller
/// (see `openalex_taxonomy::Taxonomy` for name-to-ID resolution).
///
/// The next page is prefetched concurrently while the current page is embedded
/// and inserted, so network latency overlaps GPU + DB work.
pub async fn ingest_api(min_year: u32, batch_size: usize, max_papers: Option<usize>, topics_filter: &str) -> Result<()> {
    let db = DbClient::new().await.context("Database connection required for ingestion")?;
    let client = reqwest::Client::new();

    let per_page = 200; // OpenAlex max per page

    tracing::info!(
        topics_filter = %topics_filter,
        min_year = min_year,
        batch_size = batch_size,
        max_papers = ?max_papers,
        "Starting OpenAlex API ingest"
    );

    let mut total_ingested: usize = 0;
    let mut total_skipped: usize = 0;
    let mut total_errors: usize = 0;
    let mut total_backfilled: usize = 0;
    let start_time = Instant::now();

    // Progress bar length is set after the first response, once we know OpenAlex's meta.count.
    let pb = ProgressBar::new(max_papers.map(|m| m as u64).unwrap_or(0));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} papers | {msg}")
            .unwrap()
    );
    let mut meta_count_logged = false;

    let mut base_url = format!(
        "https://api.openalex.org/works?filter={},has_abstract:true,publication_year:>{}&per_page={}&sort=publication_year:desc",
        topics_filter,
        min_year - 1,
        per_page,
    );

    // Polite pool (mailto) + API key ($1/day budget vs $0.10/day without key).
    if let Ok(email) = std::env::var("OPENALEX_EMAIL") {
        if !email.is_empty() {
            base_url.push_str(&format!("&mailto={}", email));
            tracing::info!(email = %email, "Using OpenAlex polite pool");
        }
    } else {
        tracing::info!("OPENALEX_EMAIL not set — using public pool. Set it for priority queue + higher effective throughput.");
    }
    if let Ok(key) = std::env::var("OPENALEX_API_KEY") {
        if !key.is_empty() {
            base_url.push_str(&format!("&api_key={}", key));
            tracing::info!("Using OpenAlex API key");
        }
    }

    // Fetch the first page up front; subsequent pages are prefetched below.
    let mut current = fetch_page(&client, &base_url, "*", &pb).await?;

    loop {
        if let Some(cap) = max_papers {
            if total_ingested >= cap {
                break;
            }
        }

        // On the first successful response, report OpenAlex's actual total matching count
        // and size the progress bar to it (or to the user-imposed max_papers cap, whichever is smaller).
        if !meta_count_logged {
            if let Some(count) = current.meta.as_ref().and_then(|m| m.count) {
                let target = match max_papers {
                    Some(cap) => (cap as u64).min(count),
                    None => count,
                };
                pb.set_length(target);
                pb.println(format!(
                    "OpenAlex reports {} matching works for filter `{}` (target this run: {})",
                    count, topics_filter, target
                ));
                tracing::info!(openalex_count = count, target = target, "OpenAlex total matching works");
            }
            meta_count_logged = true;
        }

        if current.results.is_empty() {
            break;
        }

        let next_cursor = current.meta.as_ref().and_then(|m| m.next_cursor.clone());

        // Kick off the next page fetch now so the network round-trip overlaps the
        // embed + insert work below. Skip if we're at/over the cap or out of pages.
        let under_cap = max_papers.map_or(true, |cap| total_ingested < cap);
        let prefetch = match &next_cursor {
            Some(c) if !c.is_empty() && under_cap => {
                let client = client.clone();
                let base_url = base_url.clone();
                let cursor = c.clone();
                let pb = pb.clone();
                Some(tokio::spawn(async move { fetch_page(&client, &base_url, &cursor, &pb).await }))
            }
            _ => None,
        };

        // Parse papers from this page. Each record is decoded into `Work`
        // independently so one malformed work is skipped, not fatal.
        let results = std::mem::take(&mut current.results);
        let mut page_papers: Vec<Paper> = Vec::with_capacity(results.len());
        for item in results {
            match serde_json::from_value::<Work>(item) {
                Ok(work) => match paper_from_work(&work, true) {
                    Some(p) => page_papers.push(p),
                    None => total_skipped += 1,
                },
                Err(_) => total_skipped += 1,
            }
        }

        // Embed and insert in batches.
        for batch in page_papers.chunks(batch_size) {
            if let Some(cap) = max_papers {
                if total_ingested >= cap {
                    break;
                }
            }

            // Skip papers already in DB so reruns don't re-embed ingested rows.
            // Cheap SELECT (one round-trip, indexed lookup) vs. forward pass + GPU work.
            let batch_ids: Vec<String> = batch.iter().map(|p| p.id.clone()).collect();
            let existing = match db.existing_paper_ids(&batch_ids).await {
                Ok(set) => set,
                Err(e) => {
                    pb.println(format!(
                        "existing_paper_ids lookup failed: {:#}. Proceeding without resume-dedup for this batch.", e
                    ));
                    std::collections::HashSet::new()
                }
            };

            // Opportunistic backfill: for papers already in the DB, fill in pdf_url
            // (and other newly-tracked fields in the future) when the existing row
            // has them NULL but our new fetch has them populated. No re-embedding.
            let backfill_data: Vec<(String, String)> = batch.iter()
                .filter(|p| existing.contains(&p.id))
                .filter_map(|p| p.pdf_url.as_ref().map(|u| (p.id.clone(), u.clone())))
                .collect();
            if !backfill_data.is_empty() {
                match db.backfill_pdf_urls(&backfill_data).await {
                    Ok(n) => total_backfilled += n,
                    Err(e) => pb.println(format!("backfill_pdf_urls error: {:#}", e)),
                }
            }

            let new_papers: Vec<Paper> = batch.iter()
                .filter(|p| !existing.contains(&p.id))
                .cloned()
                .collect();
            total_skipped += batch.len() - new_papers.len();

            if new_papers.is_empty() {
                pb.set_position((total_ingested + total_skipped) as u64);
                pb.set_message(format!(
                    "{} ingested, {} skipped (+{} pdf_url backfilled), {} errors | catching up...",
                    total_ingested, total_skipped, total_backfilled, total_errors,
                ));
                continue;
            }

            let texts: Vec<String> = new_papers.iter()
                .map(|p| format!("{} {}", p.title, p.abstract_text))
                .collect();

            match generate_document_embeddings_batch(texts).await {
                Ok(embeddings) => {
                    match db.insert_papers_batch(&new_papers, &embeddings).await {
                        Ok(_) => {
                            total_ingested += new_papers.len();
                        }
                        Err(e) => {
                            if total_errors < 5 * new_papers.len() {
                                pb.println(format!("Batch DB insert error: {:#}", e));
                            }
                            total_errors += new_papers.len();
                        }
                    }
                }
                Err(e) => {
                    if total_errors < 5 * new_papers.len() {
                        pb.println(format!("Batch embedding error: {:#}", e));
                    }
                    total_errors += new_papers.len();
                }
            }

            let elapsed = start_time.elapsed().as_secs_f64();
            let walked = total_ingested + total_skipped;
            let rate = if elapsed > 0.0 { walked as f64 / elapsed } else { 0.0 };
            pb.set_position(walked as u64);
            pb.set_message(format!(
                "{} ingested, {} skipped (+{} pdf_url backfilled), {} errors | {:.1}/s",
                total_ingested, total_skipped, total_backfilled, total_errors, rate,
            ));
        }

        // Advance to the prefetched page, or stop if there wasn't one.
        match prefetch {
            Some(handle) => {
                current = match handle.await {
                    Ok(Ok(body)) => body,
                    Ok(Err(e)) => return Err(e),
                    Err(join_err) => anyhow::bail!("page prefetch task failed: {}", join_err),
                };
            }
            None => break,
        }
    }

    pb.finish_with_message("Done!");

    let elapsed = start_time.elapsed();
    let rate = total_ingested as f64 / elapsed.as_secs_f64();
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  API INGESTION COMPLETE");
    eprintln!("{}", "=".repeat(60));
    eprintln!("  Papers ingested:  {}", total_ingested);
    eprintln!("  Papers skipped:   {}", total_skipped);
    eprintln!("  pdf_url backfilled: {}", total_backfilled);
    eprintln!("  Errors:           {}", total_errors);
    eprintln!("  Total time:       {:.1?}", elapsed);
    eprintln!("  Rate:             {:.1} papers/sec", rate);
    eprintln!("{}", "=".repeat(60));

    tracing::info!(
        ingested = total_ingested,
        skipped = total_skipped,
        backfilled = total_backfilled,
        errors = total_errors,
        elapsed_s = elapsed.as_secs_f64(),
        rate_per_s = rate,
        "API ingest complete"
    );

    Ok(())
}

/// Reconstruct an abstract from OpenAlex's inverted-index format
/// (`{word: [positions...]}`) into plain text.
fn reconstruct_abstract(index: &HashMap<String, Vec<u32>>) -> String {
    let mut positions: Vec<(u32, &str)> = Vec::new();
    for (word, posns) in index {
        for &p in posns {
            positions.push((p, word.as_str()));
        }
    }

    positions.sort_by_key(|(pos, _)| *pos);

    // Build the string directly instead of collecting an intermediate Vec<&str>.
    let mut out = String::with_capacity(positions.iter().map(|(_, w)| w.len() + 1).sum());
    for (i, (_, word)) in positions.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}
