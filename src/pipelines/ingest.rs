use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
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

fn extract_openalex_id(obj: &serde_json::Value) -> Option<u32> {
    obj.get("id")
        .and_then(|i| i.as_str())
        .and_then(|s| s.rsplit('/').next())
        .and_then(|s| s.parse::<u32>().ok())
}

/// Extract a direct PDF URL from an OpenAlex work, preferring `best_oa_location`
/// (OpenAlex's curated best open-access copy) and falling back to `primary_location`.
/// Returns None if no PDF link is published — a tier-2 reader can fall back to
/// `url`/DOI in that case.
fn extract_pdf_url(work: &serde_json::Value) -> Option<String> {
    for key in ["best_oa_location", "primary_location"] {
        let url = work
            .get(key)
            .filter(|v| !v.is_null())
            .and_then(|loc| loc.get("pdf_url"))
            .and_then(|u| u.as_str())
            .filter(|s| !s.is_empty());
        if let Some(s) = url {
            return Some(s.to_string());
        }
    }
    None
}

/// Check if a snapshot work's `topics[]` array contains a topic whose
/// field/subfield matches the filter.
fn matches_topic_filter(work: &serde_json::Value, filter: &TopicFilter) -> bool {
    let topics = match work.get("topics").and_then(|t| t.as_array()) {
        Some(t) => t,
        None => return false,
    };
    for topic in topics {
        match filter {
            TopicFilter::Field(id) => {
                if let Some(obj) = topic.get("field") {
                    if extract_openalex_id(obj) == Some(*id) {
                        return true;
                    }
                }
            }
            TopicFilter::Subfields(ids) => {
                if let Some(obj) = topic.get("subfield") {
                    if let Some(sid) = extract_openalex_id(obj) {
                        if ids.contains(&sid) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Ingest papers from an OpenAlex snapshot directory.
/// Expects the directory structure: <snapshot_dir>/data/works/updated_date=*/part_*.gz
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
    let mut gz_files: Vec<_> = Vec::new();
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
    eprintln!("Found {} compressed files to process", gz_files.len());

    let total_ingested = Arc::new(AtomicUsize::new(0));
    let total_skipped = Arc::new(AtomicUsize::new(0));
    let total_errors = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();

    let pb = ProgressBar::new(gz_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} files | {msg}")
            .unwrap()
    );

    for gz_path in &gz_files {
        let file_start = Instant::now();
        let mut file_papers: Vec<Paper> = Vec::new();

        // Read and parse the gzipped JSONL file
        let file = fs::File::open(gz_path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            let work: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Filter: must have abstract
            let abstract_inverted_index = work.get("abstract_inverted_index");
            if abstract_inverted_index.is_none() || abstract_inverted_index == Some(&serde_json::Value::Null) {
                total_skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Filter: publication year
            let year = work.get("publication_year")
                .and_then(|y| y.as_u64())
                .map(|y| y as u32);
            if year.map(|y| y < min_year).unwrap_or(true) {
                total_skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Filter: must match the selected field/subfield(s)
            if !matches_topic_filter(&work, topic_filter) {
                total_skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Filter: must have a title
            let title = match work.get("title").and_then(|t| t.as_str()) {
                Some(t) if !t.is_empty() => t.to_string(),
                _ => {
                    total_skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            // Reconstruct abstract
            let abstract_text = reconstruct_abstract(abstract_inverted_index);
            if abstract_text.is_empty() {
                total_skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Extract metadata
            let openalex_id = work.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
            let short_id = openalex_id.split('/').last().unwrap_or(&openalex_id).to_string();

            let doi = work.get("doi")
                .and_then(|d| d.as_str())
                .map(|s| s.replace("https://doi.org/", ""));

            let authors: Vec<String> = work.get("authorships")
                .and_then(|a| a.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|a| a.get("author")
                        .and_then(|au| au.get("display_name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string()))
                    .collect())
                .unwrap_or_default();

            let pdf_url = extract_pdf_url(&work);

            file_papers.push(Paper {
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

        // Embed and insert in batches
        for batch in file_papers.chunks(batch_size) {
            // Check if we've hit the max papers limit
            if let Some(max) = max_papers {
                if total_ingested.load(Ordering::Relaxed) >= max {
                    pb.finish_with_message("Reached max papers limit");
                    break;
                }
            }
            // Prepare texts for batch embedding
            let texts: Vec<String> = batch.iter()
                .map(|p| format!("{} {}", p.title, p.abstract_text))
                .collect();

            // Batch embed
            match generate_document_embeddings_batch(texts).await {
                Ok(embeddings) => {
                    // Batch insert into DB
                    match db.insert_papers_batch(batch, &embeddings).await {
                        Ok(_) => {
                            total_ingested.fetch_add(batch.len(), Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::warn!("Batch DB insert error: {}", e);
                            total_errors.fetch_add(batch.len(), Ordering::Relaxed);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Batch embedding error: {}", e);
                    total_errors.fetch_add(batch.len(), Ordering::Relaxed);
                }
            }

            let ingested = total_ingested.load(Ordering::Relaxed);
            let skipped = total_skipped.load(Ordering::Relaxed);
            let errors = total_errors.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { ingested as f64 / elapsed } else { 0.0 };

            pb.set_message(format!(
                "{} ingested, {} skipped, {} errors | {:.1}/s | ETA: {:.1}h for 2M",
                ingested, skipped, errors, rate,
                if rate > 0.0 { (2_000_000.0 - ingested as f64) / rate / 3600.0 } else { 0.0 }
            ));
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done!");

    let ingested = total_ingested.load(Ordering::Relaxed);
    let skipped = total_skipped.load(Ordering::Relaxed);
    let errors = total_errors.load(Ordering::Relaxed);
    let elapsed = start_time.elapsed();

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  INGESTION COMPLETE");
    eprintln!("{}", "=".repeat(60));
    eprintln!("  Papers ingested:  {}", ingested);
    eprintln!("  Papers skipped:   {}", skipped);
    eprintln!("  Errors:           {}", errors);
    eprintln!("  Total time:       {:.1?}", elapsed);
    eprintln!("  Rate:             {:.1} papers/sec", ingested as f64 / elapsed.as_secs_f64());
    eprintln!("{}", "=".repeat(60));

    Ok(())
}

/// Ingest papers via the OpenAlex API with cursor-based pagination.
/// `topics_filter` is a fully-formed OpenAlex filter clause like
/// `"topics.field.id:17"` or `"topics.subfield.id:1702|1707"` — built by the caller
/// (see `openalex_taxonomy::Taxonomy` for name-to-ID resolution).
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

    let mut cursor = "*".to_string();
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

    // Use polite pool if email is set
    if let Ok(email) = std::env::var("OPENALEX_EMAIL") {
        base_url.push_str(&format!("&mailto={}", email));
        tracing::info!(email = %email, "Using OpenAlex polite pool");
    } else {
        tracing::info!("OPENALEX_EMAIL not set — using public pool. Set it for priority queue + higher effective throughput.");
    }

    let mut backoff_attempts: u32 = 0;

    loop {
        if let Some(cap) = max_papers {
            if total_ingested >= cap {
                break;
            }
        }

        let url = format!("{}&cursor={}", base_url, cursor);

        let res = match client
            .get(&url)
            .header("User-Agent", "autoresearch-lab/0.1 (research tool)")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let delay = retry_delay(backoff_attempts);
                backoff_attempts = backoff_attempts.saturating_add(1);
                let msg = format!(
                    "API request failed (attempt {}): {:#}. Retrying in {}s...",
                    backoff_attempts, e, delay
                );
                pb.println(&msg);
                tracing::warn!(attempt = backoff_attempts, delay_s = delay, error = ?e, "API request failed, retrying");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }
        };

        if res.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = res.headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let delay = retry_after.unwrap_or_else(|| retry_delay(backoff_attempts));
            backoff_attempts = backoff_attempts.saturating_add(1);
            let suffix = if retry_after.is_some() { " (from Retry-After)" } else { "" };
            pb.println(format!(
                "Rate limited (429, attempt {}). Waiting {}s{}...",
                backoff_attempts, delay, suffix
            ));
            tracing::warn!(attempt = backoff_attempts, delay_s = delay, retry_after = ?retry_after, "429 rate-limited");
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
            continue;
        }

        let body: serde_json::Value = match res.error_for_status() {
            Ok(r) => match r.json().await {
                Ok(v) => v,
                Err(e) => {
                    let delay = retry_delay(backoff_attempts);
                    backoff_attempts = backoff_attempts.saturating_add(1);
                    pb.println(format!(
                        "API JSON decode failed (attempt {}): {:#}. Retrying in {}s...",
                        backoff_attempts, e, delay
                    ));
                    tracing::warn!(attempt = backoff_attempts, delay_s = delay, error = ?e, "JSON decode failed");
                    tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                    continue;
                }
            },
            Err(e) => {
                let delay = retry_delay(backoff_attempts);
                backoff_attempts = backoff_attempts.saturating_add(1);
                pb.println(format!(
                    "API HTTP error (attempt {}): {:#}. Retrying in {}s...",
                    backoff_attempts, e, delay
                ));
                tracing::warn!(attempt = backoff_attempts, delay_s = delay, error = ?e, "HTTP error");
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }
        };

        // Request succeeded — reset backoff.
        backoff_attempts = 0;

        // On the first successful response, report OpenAlex's actual total matching count
        // and size the progress bar to it (or to the user-imposed max_papers cap, whichever is smaller).
        if !meta_count_logged {
            if let Some(count) = body
                .get("meta")
                .and_then(|m| m.get("count"))
                .and_then(|c| c.as_u64())
            {
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

        // Get next cursor
        let next_cursor = body.get("meta")
            .and_then(|m| m.get("next_cursor"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        let results = match body.get("results").and_then(|r| r.as_array()) {
            Some(r) => r,
            None => break,
        };

        if results.is_empty() {
            break;
        }

        // Parse papers from this page
        let mut page_papers: Vec<Paper> = Vec::new();
        for item in results {
            let title = match item.get("title").and_then(|t| t.as_str()) {
                Some(t) if !t.is_empty() => t.to_string(),
                _ => { total_skipped += 1; continue; }
            };

            let abstract_text = reconstruct_abstract(item.get("abstract_inverted_index"));
            if abstract_text.is_empty() {
                total_skipped += 1;
                continue;
            }

            let openalex_id = item.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
            let short_id = openalex_id.split('/').last().unwrap_or(&openalex_id).to_string();
            let year = item.get("publication_year").and_then(|y| y.as_u64()).map(|y| y as u32);
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

            let pdf_url = extract_pdf_url(item);

            page_papers.push(Paper {
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

        // Embed and insert in batches
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

        // Advance cursor
        match next_cursor {
            Some(c) => cursor = c,
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

/// Reconstruct abstract from OpenAlex's inverted index format.
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
