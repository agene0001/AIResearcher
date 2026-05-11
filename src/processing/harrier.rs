//! Harrier embedder client.
//!
//! The real model lives in a child process (see `harrier_worker.rs`). This
//! module is just the parent-side facade: it lazily spawns a worker, ships
//! JSON-encoded requests over its stdin, reads JSON responses from its
//! stdout, and — crucially — *respawns* the child whenever it dies or
//! returns a `fatal` error (i.e. CUDA context poisoned).
//!
//! Why a subprocess? `CUDA_ERROR_ILLEGAL_ADDRESS` permanently poisons the
//! per-process CUDA primary context. Restarting the model in-process
//! silently falls back to CPU. Restarting the *process* gives us a fresh
//! CUDA context with zero shared state. Only the embedder restarts — the
//! main pipeline keeps its DB cursor, rate-limit tokens, OAI-PMH pagination,
//! etc.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

/// Argv marker used to dispatch the same binary as the embedder worker.
/// Must match the dispatch in `main.rs`. Intentionally underscored so it
/// can't be confused with a user-facing CLI subcommand.
pub const WORKER_ARG: &str = "__harrier_worker__";

// ---------- Wire types (must match `harrier_worker.rs`) ----------

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request<'a> {
    EmbedQuery { text: &'a str },
    EmbedDocument { text: &'a str },
    EmbedDocumentBatch { texts: &'a [String] },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Response {
    Embedding { vector: Vec<f32> },
    Embeddings { vectors: Vec<Vec<f32>> },
    Error { message: String, fatal: bool },
}

// ---------- Client ----------

struct HarrierClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl HarrierClient {
    fn spawn() -> Result<Self> {
        let exe = std::env::current_exe().context("locating current_exe for harrier worker")?;
        let mut child = Command::new(&exe)
            .arg(WORKER_ARG)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so the worker's model-loading messages and OOM
            // warnings appear directly in the parent's terminal, just like
            // when the embedder ran in-process.
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning harrier worker: {}", exe.display()))?;
        let stdin = child.stdin.take().context("worker stdin not captured")?;
        let stdout_pipe = child.stdout.take().context("worker stdout not captured")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout_pipe),
        })
    }

    /// Send one request and read one response. Returns Err on any I/O failure
    /// (including EOF), which the caller interprets as "child died".
    fn call(&mut self, req: &Request) -> Result<Response> {
        let payload = serde_json::to_string(req).context("encode request")?;
        self.stdin
            .write_all(payload.as_bytes())
            .context("write request to worker stdin")?;
        self.stdin
            .write_all(b"\n")
            .context("write request newline to worker stdin")?;
        self.stdin.flush().context("flush worker stdin")?;

        let mut line = String::new();
        let n = self
            .stdout
            .read_line(&mut line)
            .context("read response from worker stdout")?;
        if n == 0 {
            anyhow::bail!("harrier worker closed stdout (EOF) — child likely exited");
        }
        let resp: Response =
            serde_json::from_str(line.trim_end()).context("decode response from worker")?;
        Ok(resp)
    }
}

impl Drop for HarrierClient {
    fn drop(&mut self) {
        // Closing stdin causes the worker's read_line to return Ok(0) and
        // exit cleanly. We still call kill() as a backstop in case the
        // worker is mid-forward-pass on the GPU and isn't reading stdin.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

static CLIENT: Mutex<Option<HarrierClient>> = Mutex::new(None);

/// Try one request against the current (or freshly spawned) client. If the
/// worker is dead or replies with `fatal: true`, kill it, spawn a new one,
/// and retry exactly once.
fn call_with_respawn(req: &Request) -> Result<Response> {
    let mut guard = CLIENT
        .lock()
        .map_err(|e| anyhow::anyhow!("harrier client mutex poisoned: {}", e))?;

    // Lazy spawn on first use.
    if guard.is_none() {
        *guard = Some(HarrierClient::spawn()?);
    }

    // First attempt.
    let first = {
        let client = guard.as_mut().expect("just initialized");
        client.call(req)
    };

    match first {
        Ok(Response::Error { fatal: true, message }) => {
            eprintln!(
                "Harrier worker reported fatal error ({}). Respawning and retrying once...",
                message
            );
            tracing::warn!(error = %message, "harrier worker fatal; respawning");
        }
        Ok(other) => return Ok(other),
        Err(e) => {
            eprintln!(
                "Harrier worker I/O failed ({}). Respawning and retrying once...",
                e
            );
            tracing::warn!(error = %e, "harrier worker io error; respawning");
        }
    }

    // Tear down the dead client (Drop kills+waits) and spawn fresh.
    *guard = None;
    *guard = Some(HarrierClient::spawn()?);
    eprintln!("Harrier worker respawned; retrying request.");
    tracing::info!("harrier worker respawned");

    let client = guard.as_mut().expect("just initialized");
    let second = client.call(req)?;
    match second {
        Response::Error { fatal: true, message } => {
            anyhow::bail!("harrier worker fatal error on retry: {}", message)
        }
        other => Ok(other),
    }
}

// ---------- Public API (unchanged signatures vs. the old in-process impl) ----------

/// Generate an embedding for a query (with retrieval instruction).
pub fn embed_query(query: &str) -> Result<Vec<f32>> {
    let resp = call_with_respawn(&Request::EmbedQuery { text: query })?;
    match resp {
        Response::Embedding { vector } => Ok(vector),
        Response::Error { message, .. } => anyhow::bail!("embed_query: {}", message),
        Response::Embeddings { .. } => {
            anyhow::bail!("embed_query: unexpected batch response from worker")
        }
    }
}

/// Generate an embedding for a document (paper title + abstract, no instruction prefix).
pub fn embed_document(text: &str) -> Result<Vec<f32>> {
    let resp = call_with_respawn(&Request::EmbedDocument { text })?;
    match resp {
        Response::Embedding { vector } => Ok(vector),
        Response::Error { message, .. } => anyhow::bail!("embed_document: {}", message),
        Response::Embeddings { .. } => {
            anyhow::bail!("embed_document: unexpected batch response from worker")
        }
    }
}

/// Embed a batch of documents in a single forward pass.
pub fn embed_document_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let resp = call_with_respawn(&Request::EmbedDocumentBatch { texts })?;
    match resp {
        Response::Embeddings { vectors } => Ok(vectors),
        Response::Error { message, .. } => anyhow::bail!("embed_document_batch: {}", message),
        Response::Embedding { .. } => {
            anyhow::bail!("embed_document_batch: unexpected single-embedding response from worker")
        }
    }
}
