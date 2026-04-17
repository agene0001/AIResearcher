use anyhow::Result;
use crate::pipelines::retrieve::retrieve_papers;
use crate::pipelines::structure::structure_papers;
use crate::models::summary::PaperSummary;

/// Tool 1: Explore Approaches
/// Finds multiple candidate techniques from recent papers for a specific problem.
pub async fn explore_approaches(query: &str) -> Result<Vec<PaperSummary>> {
    println!("🔍 [explore_approaches] Searching for: '{}'", query);
    
    // 1. Retrieve papers (using arXiv for now, later hybrid search)
    let papers = retrieve_papers(query).await?;
    
    // 2. Structure the papers to extract methods, strengths, weaknesses, etc.
    let summaries = structure_papers(papers).await?;
    
    Ok(summaries)
}
