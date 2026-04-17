use anyhow::Result;
use crate::models::summary::PaperSummary;
use crate::llm::strong::synthesize;

/// Tool 3: Deep Dive
/// Drills into one specific paper to extract exact implementation details, hyperparameters, and failure modes.
pub async fn deep_dive(paper_summary: &PaperSummary, specific_context: &str) -> Result<String> {
    println!("🤿 [deep_dive] Deep diving into method: '{}'", paper_summary.method);
    
    let prompt = format!(
        "We are implementing the following method:\nMethod: {}\nCore Idea: {}\n\nExtracted Hyperparameters: {:?}\nExtracted Failure Modes: {:?}\nExtracted Implementation Details: {:?}\n\nContext for our implementation:\n{}\n\nProvide implementation-ready guidance. What exact hyperparameters should we use? What are the specific failure modes we should watch out for? Give a step-by-step implementation plan.",
        paper_summary.method,
        paper_summary.core_idea,
        paper_summary.hyperparameters,
        paper_summary.failure_modes,
        paper_summary.implementation_details,
        specific_context
    );

    let result = synthesize(&prompt, specific_context).await?;
    
    Ok(result)
}
