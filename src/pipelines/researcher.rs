use anyhow::Result;
use crate::pipelines::retrieve::retrieve_papers;
use crate::pipelines::structure::structure_papers;
use crate::llm::strong::synthesize;

/// Tool 4: Researcher (End-to-End Proposal)
/// Researches a problem and proposes concrete, implementation-ready ideas from academic papers 
/// tailored to a codebase context, so the LLM can integrate them and the user can run them manually.
pub async fn research_and_propose(problem: &str, codebase_context: &str) -> Result<String> {
    println!("🔬 [researcher] Researching problem: '{}'", problem);
    
    // 1. Retrieve papers
    let papers = retrieve_papers(problem).await?;
    
    // 2. Structure the papers
    let summaries = structure_papers(papers).await?;
    
    // 3. Combine for the strong LLM
    let mut combined = String::new();
    for (i, s) in summaries.iter().enumerate() {
        combined.push_str(&format!(
            "Paper {}: {}\nCore Idea: {}\nHyperparameters: {:?}\nImplementation Details: {:?}\nFailure Modes: {:?}\n\n",
            i + 1, s.method, s.core_idea, s.hyperparameters, s.implementation_details, s.failure_modes
        ));
    }

    let prompt = format!(
        "You are an expert AI research assistant. The user wants to solve the following problem:\n{}\n\nThey have the following codebase context or constraints:\n{}\n\nBased on the following recent research papers, propose 2-3 concrete, implementation-ready ideas that the user can integrate into their code. For each idea, provide:\n1. The method and theoretical backing.\n2. Exact hyperparameters or formulas to use.\n3. Specific code changes or architectural adjustments needed.\n4. Potential failure modes to watch out for.\n\nResearch Papers:\n{}",
        problem, codebase_context, combined
    );

    // 4. Synthesize the final proposal
    let result = synthesize(&prompt, problem).await?;
    
    Ok(result)
}
