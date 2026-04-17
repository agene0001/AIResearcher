use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks the last request timestamp for rate limiting (millis since epoch)
static LAST_REQUEST_MS: AtomicU64 = AtomicU64::new(0);
/// Minimum interval between requests in milliseconds
/// Set to 0 for local models or paid tiers with high RPM
const MIN_REQUEST_INTERVAL_MS: u64 = 0;

/// LLM backend configuration. Resolved from environment variables.
pub struct LlmConfig {
    pub api_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub supports_json_mode: bool,
}

/// Resolve the cheap (high-volume) LLM config.
/// Defaults to Gemma 4 4B via Ollama (fast, free, good for structured extraction).
pub fn cheap_config() -> LlmConfig {
    let ollama_host = env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = env::var("CHEAP_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    LlmConfig {
        api_url: format!("{}/v1/chat/completions", ollama_host.trim_end_matches('/')),
        api_key: None,
        model,
        supports_json_mode: false,
    }
}

/// Resolve the strong (synthesis/reasoning) LLM config.
/// Defaults to Gemma 4 27B via Ollama (free, high quality).
/// Set STRONG_MODEL to override (e.g., "gpt-4o" with OPENAI_API_KEY for cloud).
pub fn strong_config() -> LlmConfig {
    let model = env::var("STRONG_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());

    // If the model looks like it needs OpenAI/Anthropic, use the appropriate host
    let default_host = if model.starts_with("gpt") {
        "https://api.openai.com"
    } else if model.starts_with("claude") {
        "https://api.anthropic.com"
    } else {
        // Default to Ollama for everything else
        &env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string())
    };

    let host = env::var("STRONG_MODEL_HOST").unwrap_or_else(|_| default_host.to_string());
    let api_key = env::var("OPENAI_API_KEY").ok().or_else(|| env::var("ANTHROPIC_API_KEY").ok());

    // If host already contains a full path (e.g. Gemini's /v1beta/openai), append only /chat/completions
    let api_url = if host.contains("/openai") {
        format!("{}/chat/completions", host.trim_end_matches('/'))
    } else {
        format!("{}/v1/chat/completions", host.trim_end_matches('/'))
    };

    LlmConfig {
        api_url,
        api_key,
        model,
        supports_json_mode: !host.contains("localhost"), // Cloud APIs support JSON mode
    }
}

/// Rate-limit: wait until enough time has passed since the last request.
async fn rate_limit_wait(is_cloud: bool) {
    if !is_cloud {
        return; // No rate limiting for local Ollama
    }
    loop {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let last = LAST_REQUEST_MS.load(Ordering::Relaxed);
        let elapsed = now.saturating_sub(last);
        if elapsed >= MIN_REQUEST_INTERVAL_MS {
            // Try to claim this slot
            if LAST_REQUEST_MS.compare_exchange(last, now, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                return;
            }
            // Another task claimed it, retry
            continue;
        }
        let wait = MIN_REQUEST_INTERVAL_MS - elapsed;
        tokio::time::sleep(std::time::Duration::from_millis(wait)).await;
    }
}

/// Send a chat completion request to any OpenAI-compatible endpoint.
/// Includes rate limiting for cloud APIs and retry with backoff on 429.
pub async fn chat_completion(
    config: &LlmConfig,
    system_prompt: &str,
    user_prompt: &str,
    json_mode: bool,
) -> Result<String> {
    let client = Client::new();
    let is_cloud = !config.api_url.contains("localhost");

    let mut body = json!({
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    });

    if json_mode && config.supports_json_mode {
        body["response_format"] = json!({"type": "json_object"});
    }

    let max_retries = if is_cloud { 5 } else { 1 };

    for attempt in 0..max_retries {
        rate_limit_wait(is_cloud).await;

        let mut req = client.post(&config.api_url).json(&body);
        if let Some(ref key) = config.api_key {
            req = req.bearer_auth(key);
        }

        let res = req.send().await?;

        if res.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let wait_secs = if attempt < 4 { 15 + attempt as u64 * 5 } else { 30 };
            eprintln!("    Rate limited, waiting {}s (attempt {}/{})...", wait_secs, attempt + 1, max_retries);
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }

        if !res.status().is_success() {
            let status = res.status();
            let body_text = res.text().await.unwrap_or_default();
            anyhow::bail!("LLM request failed ({}): {}", status, body_text);
        }

        let res_json: serde_json::Value = res.json().await?;

        let content = res_json["choices"][0]["message"]["content"]
            .as_str()
            .context("No content in LLM response")?
            .to_string();

        return Ok(content);
    }

    anyhow::bail!("LLM request failed after {} retries due to rate limiting", max_retries)
}
