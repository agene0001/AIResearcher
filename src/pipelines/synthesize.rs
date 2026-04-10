use anyhow::Result;
use crate::llm::strong::synthesize;
use crate::models::summary::PaperSummary;

pub async fn synthesize_methods(
    summaries: Vec<PaperSummary>,
    problem: &str,
) -> Result<String> {
    // Convert our structured JSON summaries into a readable format for the strong LLM
    let mut combined = String::new();
    for (i, s) in summaries.iter().enumerate() {
        combined.push_str(&format!(
            "Method {}: {}\nCore Idea: {}\nStrengths: {:?}\nWeaknesses: {:?}\nHyperparameters: {:?}\nFailure Modes: {:?}\nImplementation Details: {:?}\n\n",
            i + 1, s.method, s.core_idea, s.strengths, s.weaknesses, s.hyperparameters, s.failure_modes, s.implementation_details
        ));
    }

    let result = synthesize(&combined, problem).await?;

    Ok(result)
}
