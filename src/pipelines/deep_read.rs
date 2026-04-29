//! Tier 2: full-paper deep-read pipeline.
//!
//! `read_paper(id)` is the only public entry point. It:
//!   1. Returns cached markdown from `paper_full_text` if present.
//!   2. Otherwise looks up `papers.pdf_url`, downloads the PDF to a temp dir,
//!      shells out to a configured parser, caches the result, and returns it.
//!
//! Parser auto-detection: prefers `marker_single` (markdown with tables/equations
//! preserved, ML-based) when on PATH, falls back to `pdftotext` (plain text via
//! poppler). Override with `PDF_PARSER=marker|pdftotext` env var.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use crate::storage::postgres::DbClient;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Parser {
    Marker,
    Pdftotext,
}

impl Parser {
    fn name(&self) -> &'static str {
        match self {
            Parser::Marker => "marker_single",
            Parser::Pdftotext => "pdftotext",
        }
    }
}

static PARSER: OnceLock<Parser> = OnceLock::new();

fn detect_parser() -> Parser {
    *PARSER.get_or_init(|| {
        if let Ok(name) = std::env::var("PDF_PARSER") {
            return match name.to_ascii_lowercase().as_str() {
                "marker" | "marker_single" | "marker-pdf" => Parser::Marker,
                "pdftotext" => Parser::Pdftotext,
                other => {
                    tracing::warn!(parser = %other, "Unknown PDF_PARSER value; auto-detecting");
                    auto_detect()
                }
            };
        }
        auto_detect()
    })
}

fn parser_available(cmd: &str, version_arg: &str) -> bool {
    Command::new(cmd)
        .arg(version_arg)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn auto_detect() -> Parser {
    if parser_available("marker_single", "--help") {
        tracing::info!("PDF parser auto-detected: marker_single");
        return Parser::Marker;
    }
    if parser_available("pdftotext", "-v") {
        tracing::info!("PDF parser auto-detected: pdftotext");
        return Parser::Pdftotext;
    }
    tracing::warn!(
        "Neither marker_single nor pdftotext on PATH. Tier-2 reads will fail until one is installed.\n\
         Install pdftotext: brew/apt install poppler / poppler-utils, or winget install Anaconda.Miniconda3 then conda install -c conda-forge poppler.\n\
         Install marker (better quality): pip install marker-pdf"
    );
    Parser::Pdftotext
}

/// Make a paper id safe for use as a filesystem path component.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Stream a PDF to disk. Times out at 60s, follows redirects.
async fn download_pdf(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .user_agent("autoresearch-lab/0.1 (research tool)")
        .build()?;
    let bytes = client.get(url).send().await?.error_for_status()?.bytes().await?;
    if bytes.is_empty() {
        anyhow::bail!("downloaded 0 bytes from {}", url);
    }
    tokio::fs::write(dest, &bytes).await?;
    Ok(())
}

/// Run `pdftotext -layout PDF -` and return stdout.
fn run_pdftotext(pdf_path: &Path) -> Result<String> {
    let output = Command::new("pdftotext")
        .arg("-layout")
        .arg(pdf_path)
        .arg("-") // stdout
        .output()
        .context("invoking pdftotext (install via poppler / poppler-utils)")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pdftotext exited with {}: {}", output.status, stderr.trim());
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        anyhow::bail!("pdftotext produced empty output");
    }
    Ok(text)
}

/// Run `marker_single PDF --output_dir OUT --output_format markdown` and read the resulting .md.
fn run_marker(pdf_path: &Path, out_dir: &Path) -> Result<String> {
    let output = Command::new("marker_single")
        .arg(pdf_path)
        .arg("--output_dir")
        .arg(out_dir)
        .arg("--output_format")
        .arg("markdown")
        .output()
        .context("invoking marker_single (install: pip install marker-pdf)")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("marker_single exited with {}: {}", output.status, stderr.trim());
    }

    // marker writes <out_dir>/<pdf_stem>/<pdf_stem>.md
    let stem = pdf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("paper");
    let nested = out_dir.join(stem).join(format!("{}.md", stem));
    if nested.exists() {
        return std::fs::read_to_string(&nested)
            .with_context(|| format!("reading marker markdown at {}", nested.display()));
    }
    let flat = out_dir.join(format!("{}.md", stem));
    if flat.exists() {
        return std::fs::read_to_string(&flat)
            .with_context(|| format!("reading marker markdown at {}", flat.display()));
    }
    // Last resort: scan the output dir for any .md file
    if let Ok(entries) = std::fs::read_dir(out_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "md").unwrap_or(false) {
                return std::fs::read_to_string(&p)
                    .with_context(|| format!("reading marker markdown at {}", p.display()));
            }
        }
    }
    anyhow::bail!(
        "marker_single completed but no .md output found under {}",
        out_dir.display()
    );
}

/// Fetch the full text of a paper. Cache hit returns instantly; cache miss
/// downloads + parses + caches. Cached errors short-circuit; delete the
/// `paper_full_text` row to force a retry.
pub async fn read_paper(paper_id: &str) -> Result<String> {
    let db = DbClient::new()
        .await
        .context("DB connection required for tier-2 read_paper")?;

    if let Some((markdown, error)) = db.get_full_text(paper_id).await? {
        if let Some(md) = markdown {
            return Ok(md);
        }
        if let Some(err) = error {
            anyhow::bail!(
                "cached extraction failure for {}: {}\n(delete from paper_full_text to retry)",
                paper_id,
                err
            );
        }
    }

    // Fetch full resolver-relevant metadata (DOI, title, first author) so we
    // can fall back to external services when papers.pdf_url is NULL.
    let meta = db
        .get_paper_resolver_meta(paper_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("paper {} not found in DB", paper_id))?;

    let pdf_url = match meta.pdf_url.clone() {
        Some(url) => url,
        None => {
            tracing::info!(paper_id = %paper_id, "pdf_url is NULL; running resolver chain");
            match crate::pipelines::pdf_resolver::resolve_pdf_url(&meta).await? {
                Some(url) => {
                    // Cache the discovery so future reads skip the resolver entirely.
                    if let Err(e) = db.update_pdf_url(paper_id, &url).await {
                        tracing::warn!(error = ?e, "failed to cache resolved pdf_url");
                    }
                    url
                }
                None => anyhow::bail!(
                    "no PDF URL found for paper {}. Tried OpenAlex (NULL), arXiv DOI \
                     pattern, Unpaywall, arXiv title search, Semantic Scholar — none had \
                     a free copy. The paper may be paywall-only.",
                    paper_id
                ),
            }
        }
    };

    let temp_dir: PathBuf =
        std::env::temp_dir().join(format!("autoresearch-pdf-{}", sanitize_id(paper_id)));
    std::fs::create_dir_all(&temp_dir)?;
    let pdf_path = temp_dir.join("paper.pdf");

    let parser = detect_parser();

    let parse_result: Result<String> = (async {
        download_pdf(&pdf_url, &pdf_path)
            .await
            .with_context(|| format!("downloading {}", pdf_url))?;
        match parser {
            Parser::Pdftotext => run_pdftotext(&pdf_path),
            Parser::Marker => run_marker(&pdf_path, &temp_dir),
        }
    })
    .await;

    let _ = std::fs::remove_dir_all(&temp_dir);

    match parse_result {
        Ok(markdown) => {
            db.cache_full_text(paper_id, &markdown, parser.name()).await?;
            Ok(markdown)
        }
        Err(e) => {
            let err_str = format!("{:#}", e);
            let _ = db.cache_full_text_error(paper_id, &err_str).await;
            Err(e)
        }
    }
}
