use anyhow::Result;
use crate::llm::strong::synthesize;
use crate::models::summary::PaperSummary;

/// Tool 2: Compare Approaches
/// Evaluates multiple candidates side by side to find the best fit for a specific architecture/problem.
pub async fn compare_approaches(problem: &str, candidates: &[PaperSummary]) -> Result<String> {
    println!("⚖️ [compare_approaches] Comparing {} candidates for problem: '{}'", candidates.len(), problem);
    
    let mut combined = String::new();
    for (i, s) in candidates.iter().enumerate() {
        combined.push_str(&format!(
            "Candidate {}: {}\nCore Idea: {}\nStrengths: {:?}\nWeaknesses: {:?}\n\n",
            i + 1, s.method, s.core_idea, s.strengths, s.weaknesses
        ));
    }

    let prompt = format!(
        "Given these candidate methods:\n{}\n\nFor this specific problem/architecture:\n{}\n\nCompare these approaches side-by-side. Evaluate their limitations and applicability to this specific setting. Recommend the best one.",
        combined, problem
    );

    // Re-use the strong LLM synthesis function, but with our targeted prompt
    let result = synthesize(&prompt, problem).await?;
    
    Ok(result)
}
