//! Harrier embedder worker.
//!
//! This module is the *server side* of the embedder. It is intended to run in
//! its own OS process, spawned by the parent pipeline via `current_exe()` with
//! a special argv marker (see `harrier.rs` and `main.rs`).
//!
//! Why a subprocess? CUDA's `CUDA_ERROR_ILLEGAL_ADDRESS` permanently poisons
//! the per-process CUDA primary context — no amount of dropping handles or
//! resetting devices within the same process reliably brings the GPU back.
//! Putting the model in a child process means recovery is just "kill the
//! child, spawn a new one": the new CUDA context is guaranteed clean. The
//! parent keeps its DB connections, OAI-PMH cursor, rate-limiter state, etc.
//!
//! Wire protocol: newline-delimited JSON on stdin/stdout. Anything written to
//! stderr is inherited from the parent's stderr (so the user still sees model
//! loading messages and OOM warnings in the live terminal).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "microsoft/harrier-oss-v1-0.6b";
const EMBEDDING_DIM: usize = 1024;
/// Max padded tokens (sub_batch_size × max_seq_len_in_sub_batch) per forward pass.
/// At fp32 with 28 layers × 1024 hidden, ~700 KB of activation+KV memory per token,
/// so 12288 tokens ≈ 8.4 GB — fits a 12 GB card with weights (2.4 GB) and headroom.
const MAX_BATCH_TOKENS: usize = 12288;
/// Last-resort truncation length if a single paper OOMs at its full token count.
const OOM_TRUNCATION_TOKENS: usize = 4096;
/// Exit code on a poisoned CUDA context. Distinct from generic failure (1)
/// so the parent's logs/supervisor can distinguish "recoverable" from "broken".
pub const EXIT_CUDA_POISONED: i32 = 75;

const RETRIEVAL_INSTRUCTION: &str =
    "Instruct: Given a research query, retrieve relevant academic papers that address the query\nQuery: ";

fn is_oom_error(err: &anyhow::Error) -> bool {
    let s = format!("{:#}", err);
    s.contains("out of memory") || s.contains("OUT_OF_MEMORY")
}

fn is_illegal_address_error(err: &anyhow::Error) -> bool {
    let s = format!("{:#}", err);
    s.contains("ILLEGAL_ADDRESS") || s.contains("illegal memory access")
}

// ---------- Wire types (kept in sync with `harrier.rs` client) ----------

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    EmbedQuery { text: String },
    EmbedDocument { text: String },
    EmbedDocumentBatch { texts: Vec<String> },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response {
    Embedding { vector: Vec<f32> },
    Embeddings { vectors: Vec<Vec<f32>> },
    /// `fatal = true` means the worker is about to exit (e.g. CUDA poisoned).
    /// The parent should not retry on this same child — respawn instead.
    Error { message: String, fatal: bool },
}

// ---------- Model ----------

struct Embedder {
    /// Stored tensors (with "model." prefix) for recreating the model with fresh KV cache.
    tensors: HashMap<String, Tensor>,
    config: Qwen3Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl Embedder {
    fn load() -> Result<Self> {
        eprintln!("Loading harrier-oss-v1-0.6b embedding model (first run downloads ~1.2GB)...");

        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).context("CUDA init failed in worker")?;
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        let device = Device::new_metal(0).context("Metal init failed in worker")?;
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = Device::Cpu;
        let dtype = DType::F32;

        let api = Api::new().context("Failed to initialize HuggingFace Hub API")?;
        let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").context("Failed to download config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;
        let weights_path = repo.get("model.safetensors").context("Failed to download model.safetensors")?;

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Qwen3Config = serde_json::from_str(&config_str)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load safetensors and add "model." prefix to match candle's expected tensor names
        let raw_tensors = candle_core::safetensors::load(&weights_path, &device)?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, tensor) in raw_tensors {
            let tensor = tensor.to_dtype(dtype)?;
            tensors.insert(format!("model.{}", name), tensor);
        }

        // Verify it works by creating one model instance
        let vb = VarBuilder::from_tensors(tensors.clone(), dtype, &device);
        let _test = Qwen3Model::new(&config, vb)?;

        eprintln!("Harrier embedding model loaded on {:?}", device);

        Ok(Self {
            tensors,
            config,
            tokenizer,
            device,
            dtype,
        })
    }

    /// Create a fresh Model instance with empty KV cache.
    fn fresh_model(&self) -> Result<Qwen3Model> {
        let vb = VarBuilder::from_tensors(self.tensors.clone(), self.dtype, &self.device);
        Ok(Qwen3Model::new(&self.config, vb)?)
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let token_ids = encoding.get_ids();

        let token_ids = if token_ids.len() > 32768 {
            &token_ids[..32768]
        } else {
            token_ids
        };

        let seq_len = token_ids.len();
        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;

        let mut model = self.fresh_model()?;
        let hidden_states = model.forward(&input_ids, 0)?;

        // Pool then normalize in fp32 — bf16 underflows in the L2 norm sum.
        let last_token = hidden_states
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        let norm = last_token.sqr()?.sum_all()?.sqrt()?;
        let normalized = last_token.broadcast_div(&norm)?;

        let embedding: Vec<f32> = normalized.to_vec1()?;
        debug_assert_eq!(embedding.len(), EMBEDDING_DIM, "Expected {} dims, got {}", EMBEDDING_DIM, embedding.len());
        Ok(embedding)
    }

    /// Embed a batch of texts. Tokenizes everything, sorts by length, and packs
    /// sub-batches up to MAX_BATCH_TOKENS of padded tokens so one long outlier
    /// can't blow up GPU memory. Returns embeddings in the original input order.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            return Ok(vec![self.embed_single(&texts[0])?]);
        }

        let mut items: Vec<(usize, Vec<u32>)> = texts.iter()
            .enumerate()
            .map(|(i, text)| {
                let encoding = self.tokenizer.encode(text.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                let ids = encoding.get_ids();
                let ids = if ids.len() > 32768 { ids[..32768].to_vec() } else { ids.to_vec() };
                Ok((i, ids))
            })
            .collect::<Result<Vec<_>>>()?;

        items.sort_by_key(|(_, ids)| std::cmp::Reverse(ids.len()));

        let mut embeddings: Vec<Option<Vec<f32>>> = (0..texts.len()).map(|_| None).collect();
        let mut cursor = 0;
        while cursor < items.len() {
            let max_len = items[cursor].1.len().max(1);
            let mut take = (MAX_BATCH_TOKENS / max_len).max(1);
            take = take.min(items.len() - cursor);

            let sub_batch = &items[cursor..cursor + take];
            let sub_embeddings = self.embed_sub_batch_recoverable(sub_batch)?;

            let nan_positions: Vec<usize> = sub_embeddings.iter()
                .enumerate()
                .filter(|(_, e)| e.iter().any(|v| !v.is_finite()))
                .map(|(i, _)| i)
                .collect();

            if !nan_positions.is_empty() {
                eprintln!(
                    "\n=== Sub-batch NaN summary ===\n\
                     sub_batch_size: {}, max_len: {}, NaN items: {:?} (positions 0-indexed)",
                    sub_batch.len(), max_len, nan_positions
                );

                let first_nan = nan_positions[0];
                let (orig_idx, ids) = &sub_batch[first_nan];
                let text = &texts[*orig_idx];
                let preview: String = text.chars().take(150).collect();
                eprintln!("Re-running item {} (orig_idx {}, seq_len {}) in isolation...",
                    first_nan, orig_idx, ids.len());
                match self.embed_single(text) {
                    Ok(single_emb) => {
                        let has_nan = single_emb.iter().any(|v| !v.is_finite());
                        eprintln!("  embed_single result: {} (batched was NaN)",
                            if has_nan { "ALSO NaN — content issue" } else { "CLEAN — batching bug" });
                    }
                    Err(e) => eprintln!("  embed_single errored: {:#}", e),
                }

                let dump_path = std::env::temp_dir().join(format!("harrier_nan_{}.txt", orig_idx));
                let _ = std::fs::write(&dump_path, text.as_bytes());
                eprintln!("text preview: {}\nfirst 10 ids: {:?}\nfull text: {}\n============================\n",
                    preview,
                    ids.iter().take(10).copied().collect::<Vec<_>>(),
                    dump_path.display());
                anyhow::bail!("NaN in sub-batch — see diagnostic above");
            }

            for ((orig_idx, _), emb) in sub_batch.iter().zip(sub_embeddings) {
                embeddings[*orig_idx] = Some(emb);
            }
            cursor += take;
        }

        Ok(embeddings.into_iter().map(|e| e.expect("every index filled")).collect())
    }

    /// Wrapper around `embed_sub_batch` that catches CUDA OOM and recursively halves
    /// the sub-batch until the forward pass fits in VRAM. At single-item granularity,
    /// falls back to truncating the input to OOM_TRUNCATION_TOKENS.
    /// Items are assumed to be sorted by length descending (so items[0] dictates max_len).
    fn embed_sub_batch_recoverable(&self, items: &[(usize, Vec<u32>)]) -> Result<Vec<Vec<f32>>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let max_len = items[0].1.len().max(1);

        match self.embed_sub_batch(items, max_len) {
            Ok(embs) => Ok(embs),
            Err(e) if is_oom_error(&e) && items.len() > 1 => {
                eprintln!("GPU OOM in sub-batch (size={}, max_len={}) — halving and retrying",
                    items.len(), max_len);
                let mid = items.len() / 2;
                let mut out = self.embed_sub_batch_recoverable(&items[..mid])?;
                let right = self.embed_sub_batch_recoverable(&items[mid..])?;
                out.extend(right);
                Ok(out)
            }
            Err(e) if is_oom_error(&e) && items.len() == 1 => {
                let (orig_idx, ids) = &items[0];
                if ids.len() > OOM_TRUNCATION_TOKENS {
                    eprintln!("single-paper OOM (orig_idx={}, tokens={}) — truncating to {} and retrying",
                        orig_idx, ids.len(), OOM_TRUNCATION_TOKENS);
                    let truncated = vec![(*orig_idx, ids[..OOM_TRUNCATION_TOKENS].to_vec())];
                    self.embed_sub_batch(&truncated, OOM_TRUNCATION_TOKENS)
                } else {
                    Err(e)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Run a single forward pass on a uniform sub-batch (already length-sorted, padded to max_len).
    fn embed_sub_batch(&self, items: &[(usize, Vec<u32>)], max_len: usize) -> Result<Vec<Vec<f32>>> {
        let batch_size = items.len();
        let pad_token_id: u32 = 0;

        let mut batch_data: Vec<u32> = Vec::with_capacity(batch_size * max_len);
        let mut seq_lengths: Vec<usize> = Vec::with_capacity(batch_size);
        for (_, ids) in items {
            seq_lengths.push(ids.len());
            batch_data.extend_from_slice(ids);
            for _ in ids.len()..max_len {
                batch_data.push(pad_token_id);
            }
        }

        let input_ids = Tensor::from_vec(batch_data, (batch_size, max_len), &self.device)?;
        let mut model = self.fresh_model()?;
        let hidden_states = model.forward(&input_ids, 0)?;

        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let last_pos = seq_lengths[i] - 1;
            let last_token = hidden_states
                .narrow(0, i, 1)?
                .narrow(1, last_pos, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            let norm = last_token.sqr()?.sum_all()?.sqrt()?;
            let normalized = last_token.broadcast_div(&norm)?;
            let embedding: Vec<f32> = normalized.to_vec1()?;
            debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

// ---------- I/O loop ----------

fn write_response<W: Write>(w: &mut W, resp: &Response) -> Result<()> {
    let s = serde_json::to_string(resp).context("encode response")?;
    w.write_all(s.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

/// Entry point for the worker subprocess.
///
/// Reads newline-delimited JSON requests from stdin, writes newline-delimited
/// JSON responses to stdout. On a CUDA illegal-address, emits a `fatal: true`
/// error response and exits with `EXIT_CUDA_POISONED` so the parent's
/// respawn-on-failure logic kicks in with a fresh process (and thus a fresh
/// CUDA context).
pub fn run() -> Result<()> {
    let embedder = match Embedder::load() {
        Ok(e) => e,
        Err(e) => {
            // We can't even load — tell the parent and bail. Parent will see
            // EOF on the next request and surface the error.
            let stdout = io::stdout();
            let mut w = stdout.lock();
            let _ = write_response(
                &mut w,
                &Response::Error {
                    message: format!("worker init failed: {:#}", e),
                    fatal: true,
                },
            );
            return Err(e);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).context("read request line")?;
        if n == 0 {
            // Parent closed stdin → clean shutdown.
            return Ok(());
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        let req: Request = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::Error {
                    message: format!("bad request: {}", e),
                    fatal: false,
                };
                write_response(&mut writer, &resp)?;
                continue;
            }
        };

        let result: Result<Response> = match req {
            Request::EmbedQuery { text } => {
                let full = format!("{}{}", RETRIEVAL_INSTRUCTION, text);
                embedder.embed_single(&full).map(|v| Response::Embedding { vector: v })
            }
            Request::EmbedDocument { text } => {
                embedder.embed_single(&text).map(|v| Response::Embedding { vector: v })
            }
            Request::EmbedDocumentBatch { texts } => {
                embedder.embed_batch(&texts).map(|v| Response::Embeddings { vectors: v })
            }
        };

        match result {
            Ok(resp) => write_response(&mut writer, &resp)?,
            Err(e) if is_illegal_address_error(&e) => {
                eprintln!(
                    "CUDA context poisoned in worker ({}). Exiting; parent will respawn.",
                    e
                );
                let resp = Response::Error {
                    message: format!("{:#}", e),
                    fatal: true,
                };
                // Best-effort write; if the pipe is already dead the parent
                // will see EOF and respawn anyway.
                let _ = write_response(&mut writer, &resp);
                std::process::exit(EXIT_CUDA_POISONED);
            }
            Err(e) => {
                let resp = Response::Error {
                    message: format!("{:#}", e),
                    fatal: false,
                };
                write_response(&mut writer, &resp)?;
            }
        }
    }
}
