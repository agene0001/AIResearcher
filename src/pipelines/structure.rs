use anyhow::Result;
use crate::llm::cheap::extract_summary;
use crate::models::summary::PaperSummary;
use crate::models::paper::Paper;

pub async fn structure_papers(papers: Vec<Paper>) -> Result<Vec<PaperSummary>> {
    let mut summaries = Vec::new();

    for paper in papers {
        // Feed the title and abstract to the LLM
        let text = format!("Title: {}\nAbstract: {}", paper.title, paper.abstract_text);
        
        match extract_summary(&text).await {
            Ok(summary) => summaries.push(summary),
            Err(e) => eprintln!("Failed to parse paper '{}': {}", paper.title, e),
        }
    }

    Ok(summaries)
}
