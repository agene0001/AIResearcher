use anyhow::Result;
use std::collections::HashMap;
use crate::models::paper::Paper;
use crate::retrieval::arxiv::search_arxiv;
use crate::retrieval::semantic::search_semantic_scholar;
use crate::retrieval::openalex::search_openalex;
use crate::llm::strong::synthesize;
use crate::llm::client::{cheap_config, chat_completion};

/// Detect emerging research trends in a given field.
/// This searches for very recent papers, groups them by topic, and identifies
/// which areas are seeing the most activity and novel ideas.
pub async fn detect_trends(field: &str) -> Result<String> {
    eprintln!("Detecting trends in: '{}'...", field);

    // Search for recent papers across multiple providers
    let recent_query = format!("{} 2025 2026 recent novel", field);
    let broad_query = format!("{} survey overview breakthrough", field);

    let (recent_arxiv, recent_s2, recent_openalex, broad_arxiv) = tokio::join!(
        search_arxiv(&recent_query, 20),
        search_semantic_scholar(&recent_query, 20),
        search_openalex(&recent_query, 20),
        search_arxiv(&broad_query, 10),
    );

    let mut all_papers: Vec<Paper> = Vec::new();
    let mut seen_titles = std::collections::HashSet::new();

    let mut merge = |result: Result<Vec<Paper>>| {
        if let Ok(papers) = result {
            for paper in papers {
                let norm = paper.title.to_lowercase();
                if seen_titles.insert(norm) && !paper.abstract_text.is_empty() {
                    all_papers.push(paper);
                }
            }
        }
    };

    merge(recent_arxiv);
    merge(recent_s2);
    merge(recent_openalex);
    merge(broad_arxiv);

    if all_papers.is_empty() {
        return Ok(format!("No recent papers found for field: {}", field));
    }

    // Sort by year descending to prioritize recent work
    all_papers.sort_by(|a, b| b.year.unwrap_or(0).cmp(&a.year.unwrap_or(0)));

    // Take top papers for analysis
    let top_papers: Vec<&Paper> = all_papers.iter().take(30).collect();

    eprintln!("Analyzing {} recent papers for trends...", top_papers.len());

    // Step 1: Use cheap model to cluster papers into topics
    let paper_list: String = top_papers.iter().enumerate().map(|(i, p)| {
        format!(
            "{}. [{}] \"{}\" — {}",
            i + 1,
            p.year.map(|y| y.to_string()).unwrap_or("?".to_string()),
            p.title,
            truncate(&p.abstract_text, 200),
        )
    }).collect::<Vec<_>>().join("\n");

    let cluster_prompt = format!(
        "Analyze these recent papers and group them into 4-7 emerging research topics/trends. For each topic, list which paper numbers belong to it.\n\nOutput as JSON: {{ \"topics\": [ {{ \"name\": \"...\", \"papers\": [1, 3, 5], \"description\": \"brief description\" }} ] }}\n\nPapers:\n{}",
        paper_list
    );

    let config = cheap_config();
    let clusters = match chat_completion(&config, "You are a research analyst. Output valid JSON only.", &cluster_prompt, config.supports_json_mode).await {
        Ok(c) => c,
        Err(_) => {
            // Fall back to strong model for clustering
            let strong = crate::llm::client::strong_config();
            chat_completion(&strong, "You are a research analyst. Output valid JSON only.", &cluster_prompt, true).await?
        }
    };

    // Step 2: Count papers per year to find growth signals
    let mut year_counts: HashMap<u32, usize> = HashMap::new();
    for p in &all_papers {
        if let Some(y) = p.year {
            *year_counts.entry(y).or_insert(0) += 1;
        }
    }

    let year_distribution: String = {
        let mut years: Vec<_> = year_counts.iter().collect();
        years.sort_by_key(|(y, _)| *y);
        years.iter().map(|(y, c)| format!("{}: {} papers", y, c)).collect::<Vec<_>>().join(", ")
    };

    // Step 3: Synthesize a trend report using the strong model
    let synthesis_prompt = format!(
        "You are a research trend analyst for the field of '{}'.\n\nBased on these topic clusters from recent papers:\n{}\n\nYear distribution of papers found: {}\n\nPaper details:\n{}\n\nGenerate a comprehensive trend report that includes:\n1. **Emerging Trends**: What new research directions are gaining traction?\n2. **Breakthrough Ideas**: Which specific papers or techniques represent potential breakthroughs?\n3. **Growth Areas**: Which topics show accelerating publication rates?\n4. **Declining Areas**: Any topics that seem to be losing momentum?\n5. **Predictions**: Based on current trajectories, what should researchers watch for in the next 6-12 months?\n\nBe specific — cite paper titles and explain why each trend matters.",
        field, clusters, year_distribution, paper_list
    );

    let report = synthesize(&synthesis_prompt, field).await?;

    Ok(report)
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else {
        format!("{}...", &s[..max_chars])
    }
}
