pub mod arxiv;
pub mod semantic;
pub mod pubmed;
pub mod openalex;
pub mod crossref;
pub mod dblp;
pub mod vector;

use std::sync::LazyLock;
use std::time::Duration;

/// Shared HTTP client for all retrieval providers. Reusing one client keeps
/// connection pools warm and avoids rebuilding TLS config on every request; the
/// timeout also stops a single hung provider from stalling the whole fan-out.
pub static HTTP: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .user_agent("autoresearch-lab/0.1 (research tool)")
        .timeout(Duration::from_secs(30))
        .build()
        .expect("failed to build shared reqwest client")
});
