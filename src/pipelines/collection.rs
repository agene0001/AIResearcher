use anyhow::Result;
use crate::pipelines::retrieve::retrieve_papers;
use crate::pipelines::structure::structure_papers;
use crate::llm::strong::synthesize;

/// Generates a curated learning collection for a topic.
/// This mimics the Paper Lantern "Collections" feature, which puts papers in the best reading order
/// and teaches them in minutes.
pub async fn generate_collection(topic: &str) -> Result<String> {
    println!("📚 Building learning collection for topic: '{}'...", topic);
    
    // 1. Retrieve papers
    let papers = retrieve_papers(topic).await?;
    if papers.is_empty() {
        return Ok(format!("No papers found for topic: {}", topic));
    }
    
    // 2. Structure the papers
    println!("🧠 Structuring {} papers...", papers.len());
    let summaries = structure_papers(papers).await?;
    
    // 3. Combine for the strong LLM
    let mut combined = String::new();
    for (i, s) in summaries.iter().enumerate() {
        combined.push_str(&format!(
            "Paper {}: {}\nCore Idea: {}\nStrengths: {:?}\nWeaknesses: {:?}\n\n",
            i + 1, s.method, s.core_idea, s.strengths, s.weaknesses
        ));
    }

    let prompt = format!(
        "You are an expert professor creating a learning collection for your students. The topic is: '{}'.\n\nBased on the following papers, create a curated reading list and study guide. Put them in the best reading order to learn the topic from scratch. For each paper, explain what the student will learn from it, why it is impactful, and how it connects to the next paper in the sequence.\n\nPapers:\n{}",
        topic, combined
    );

    println!("🧾 Synthesizing the curriculum...");
    // 4. Synthesize the final collection
    let result = synthesize(&prompt, topic).await?;
    
    Ok(result)
}
