use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "microsoft/harrier-oss-v1-0.6b";
const EMBEDDING_DIM: usize = 1024;
/// Max padded tokens (sub_batch_size × max_seq_len_in_sub_batch) per forward pass.
/// At fp32 with 28 layers × 1024 hidden, ~700 KB of activation+KV memory per token,
/// so 12288 tokens ≈ 8.4 GB — fits a 12 GB card with weights (2.4 GB) and headroom.
const MAX_BATCH_TOKENS: usize = 12288;

const RETRIEVAL_INSTRUCTION: &str =
    "Instruct: Given a research query, retrieve relevant academic papers that address the query\nQuery: ";

static EMBEDDER: std::sync::OnceLock<Mutex<HarrierEmbedder>> = std::sync::OnceLock::new();

pub struct HarrierEmbedder {
    /// Stored tensors (with "model." prefix) for recreating the model with fresh KV cache.
    tensors: HashMap<String, Tensor>,
    config: Qwen3Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

unsafe impl Send for HarrierEmbedder {}

impl HarrierEmbedder {
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

        // Create a fresh model with empty KV cache for each embedding
        let mut model = self.fresh_model()?;
        let hidden_states = model.forward(&input_ids, 0)?; // [1, seq_len, hidden_size]

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

        // Tokenize once; track original index so we can reorder outputs.
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

        // Sort by length descending so each greedy sub-batch is filled by a known max_len.
        items.sort_by_key(|(_, ids)| std::cmp::Reverse(ids.len()));

        let mut embeddings: Vec<Option<Vec<f32>>> = (0..texts.len()).map(|_| None).collect();

        let mut cursor = 0;
        while cursor < items.len() {
            let max_len = items[cursor].1.len().max(1);
            // How many items fit at this max_len under the token budget?
            let mut take = (MAX_BATCH_TOKENS / max_len).max(1);
            take = take.min(items.len() - cursor);

            let sub_batch = &items[cursor..cursor + take];
            let sub_embeddings = self.embed_sub_batch(sub_batch, max_len)?;

            // First pass: collect results and find NaN count across the sub-batch.
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

                // Retry the first NaN'd item in isolation via embed_single.
                // If that succeeds, the bug is in the batched forward path.
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

fn init_embedder() -> Result<Mutex<HarrierEmbedder>> {
    eprintln!("Loading harrier-oss-v1-0.6b embedding model (first run downloads ~1.2GB)...");

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
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

    Ok(Mutex::new(HarrierEmbedder {
        tensors,
        config,
        tokenizer,
        device,
        dtype,
    }))
}

fn get_embedder() -> Result<&'static Mutex<HarrierEmbedder>> {
    let embedder = EMBEDDER.get_or_init(|| {
        init_embedder().expect("Failed to load harrier embedding model")
    });
    Ok(embedder)
}

/// Generate an embedding for a query (with retrieval instruction).
pub fn embed_query(query: &str) -> Result<Vec<f32>> {
    let text = format!("{}{}", RETRIEVAL_INSTRUCTION, query);
    let embedder = get_embedder()?;
    let lock = embedder.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
    lock.embed_single(&text)
}

/// Generate an embedding for a document (paper title + abstract, no instruction prefix).
pub fn embed_document(text: &str) -> Result<Vec<f32>> {
    let embedder = get_embedder()?;
    let lock = embedder.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
    lock.embed_single(text)
}

/// Embed a batch of documents in a single forward pass. Much faster than calling embed_document in a loop.
pub fn embed_document_batch(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let embedder = get_embedder()?;
    let lock = embedder.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
    lock.embed_batch(texts)
}
