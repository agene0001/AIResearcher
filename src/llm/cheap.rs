use anyhow::Result;
use crate::llm::client::{cheap_config, chat_completion};
use crate::models::summary::PaperSummary;

/// Extract a structured summary from paper text using the cheap model (Gemma 4 4B via Ollama).
pub async fn extract_summary(paper_text: &str) -> Result<PaperSummary> {
    let system_prompt = "You are a helpful research assistant. Extract key information from academic papers. Always output valid JSON matching the exact schema requested. Do not include any text outside the JSON object.";

    let user_prompt = format!(
        "Extract method, strengths, weaknesses, and use cases from this paper. Also extract implementation-ready guidance such as hyperparameters, failure modes, and implementation details. Output strictly as JSON matching this schema: {{ \"method\": \"...\", \"core_idea\": \"...\", \"strengths\": [\"...\"], \"weaknesses\": [\"...\"], \"use_cases\": [\"...\"], \"hyperparameters\": \"...\", \"failure_modes\": [\"...\"], \"implementation_details\": \"...\", \"data_requirements\": \"...\", \"compute_cost\": \"...\" }}\n\nPaper:\n{}",
        paper_text
    );

    let config = cheap_config();
    let content = chat_completion(&config, system_prompt, &user_prompt, config.supports_json_mode).await?;

    // Extract JSON from the response (handle models that wrap JSON in markdown code blocks)
    let json_str = extract_json_from_response(&content);
    let summary: PaperSummary = serde_json::from_str(json_str)?;

    Ok(summary)
}

/// Extract JSON from a model response that might wrap it in ```json ... ``` blocks.
fn extract_json_from_response(content: &str) -> &str {
    let trimmed = content.trim();

    if let Some(start) = trimmed.find("```json") {
        let after_marker = &trimmed[start + 7..];
        if let Some(end) = after_marker.find("```") {
            return after_marker[..end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let after_marker = &trimmed[start + 3..];
        if let Some(end) = after_marker.find("```") {
            let inner = after_marker[..end].trim();
            if inner.starts_with('{') {
                return inner;
            }
        }
    }

    trimmed
}
