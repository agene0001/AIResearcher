use anyhow::Result;
use crate::llm::client::{strong_config, chat_completion};

/// Synthesize a recommendation using the strong model (Gemma 4 27B via Ollama by default).
pub async fn synthesize(methods: &str, problem: &str) -> Result<String> {
    let config = strong_config();

    let system_prompt = "You are an expert AI research scientist.";
    let user_prompt = format!(
        "Given these methods:\n{}\n\nFor problem:\n{}\n\nRecommend the best approach. Explain why, compare the tradeoffs, and provide a step-by-step plan.",
        methods, problem
    );

    chat_completion(&config, system_prompt, &user_prompt, false).await
}
