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

        // Last-token pooling
        let last_token = hidden_states
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;

        // L2 normalize
        let norm = last_token.sqr()?.sum_all()?.sqrt()?;
        let normalized = last_token.broadcast_div(&norm)?;

        let embedding: Vec<f32> = normalized.to_dtype(DType::F32)?.to_vec1()?;

        debug_assert_eq!(embedding.len(), EMBEDDING_DIM, "Expected {} dims, got {}", EMBEDDING_DIM, embedding.len());

        Ok(embedding)
    }
}

fn init_embedder() -> Result<Mutex<HarrierEmbedder>> {
    eprintln!("Loading harrier-oss-v1-0.6b embedding model (first run downloads ~1.2GB)...");

    // Try Metal with F32 — BF16 rms-norm is unsupported but F32 may work
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
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
