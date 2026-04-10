use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::llm::client::{strong_config, chat_completion};
use crate::pipelines::retrieve::retrieve_papers;
use crate::models::paper::Paper;
/// A single benchmark query with metadata about its type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuery {
    pub query: String,
    pub category: QueryCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryCategory {
    /// Short keyword-style queries: "vision transformers"
    FewWords,
    /// Natural language questions: "how do LLMs handle long context?"
    NaturalLanguage,
    /// Technical/niche queries: "gradient normalization for transformer pre-training stability"
    Technical,
}

impl std::fmt::Display for QueryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryCategory::FewWords => write!(f, "few_words"),
            QueryCategory::NaturalLanguage => write!(f, "natural_language"),
            QueryCategory::Technical => write!(f, "technical"),
        }
    }
}

/// Score for a single query-paper pair.
#[derive(Debug, Serialize, Deserialize)]
pub struct RelevanceScore {
    pub query: String,
    pub paper_title: String,
    pub paper_source: String,
    pub score: f64,
    pub reasoning: String,
}

/// Results for a single query.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub category: QueryCategory,
    pub num_papers_found: usize,
    pub scores: Vec<RelevanceScore>,
    pub mean_relevance: f64,
}

/// Full benchmark report.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub total_queries: usize,
    pub overall_mean_relevance: f64,
    pub mean_by_category: Vec<(String, f64)>,
    pub mean_by_rank: Vec<(usize, f64)>,
    pub query_results: Vec<QueryResult>,
}

/// Default benchmark query set — covers the three categories Paper Lantern tested.
pub fn default_queries() -> Vec<BenchmarkQuery> {
    vec![
        // Few words (keyword-style)
        bq("vision transformers", QueryCategory::FewWords),
        bq("retrieval augmented generation", QueryCategory::FewWords),
        bq("graph neural networks", QueryCategory::FewWords),
        bq("protein folding", QueryCategory::FewWords),
        bq("federated learning", QueryCategory::FewWords),
        bq("neural architecture search", QueryCategory::FewWords),
        bq("diffusion models", QueryCategory::FewWords),
        bq("causal inference", QueryCategory::FewWords),
        bq("knowledge distillation", QueryCategory::FewWords),
        bq("reinforcement learning robotics", QueryCategory::FewWords),

        // Natural language queries
        bq("how do large language models handle long context windows?", QueryCategory::NaturalLanguage),
        bq("what are the best methods for reducing hallucinations in LLMs?", QueryCategory::NaturalLanguage),
        bq("how can transformers be made more efficient for edge devices?", QueryCategory::NaturalLanguage),
        bq("what techniques improve few-shot learning in vision models?", QueryCategory::NaturalLanguage),
        bq("how do AI agents plan and reason over multiple steps?", QueryCategory::NaturalLanguage),
        bq("what are recent advances in machine translation for low-resource languages?", QueryCategory::NaturalLanguage),
        bq("how can neural networks be made robust to adversarial attacks?", QueryCategory::NaturalLanguage),
        bq("what methods exist for aligning language models with human preferences?", QueryCategory::NaturalLanguage),
        bq("how do mixture of experts models scale language model training?", QueryCategory::NaturalLanguage),
        bq("what are effective approaches for continual learning without catastrophic forgetting?", QueryCategory::NaturalLanguage),

        // Technical/niche queries
        bq("gradient normalization techniques for stabilizing transformer pre-training at scale", QueryCategory::Technical),
        bq("sparse attention mechanisms with subquadratic complexity for long document understanding", QueryCategory::Technical),
        bq("contrastive learning objectives for multimodal representation alignment", QueryCategory::Technical),
        bq("quantization-aware training methods for sub-4-bit large language models", QueryCategory::Technical),
        bq("positional encoding schemes that generalize beyond training sequence length", QueryCategory::Technical),
        bq("memory-augmented architectures for retrieval-grounded language generation", QueryCategory::Technical),
        bq("variance reduction techniques for policy gradient methods in continuous control", QueryCategory::Technical),
        bq("self-supervised pre-training objectives for graph transformers", QueryCategory::Technical),
        bq("efficient fine-tuning methods with parameter-efficient adapters for billion-scale models", QueryCategory::Technical),
        bq("tokenization strategies for multilingual models covering morphologically rich languages", QueryCategory::Technical),
    ]
}

fn bq(query: &str, category: QueryCategory) -> BenchmarkQuery {
    BenchmarkQuery {
        query: query.to_string(),
        category,
    }
}

/// Score a single paper's relevance to a query using the strong LLM.
async fn score_relevance(query: &str, paper: &Paper) -> Result<(f64, String)> {
    let config = strong_config();

    let prompt = format!(
        r#"You are evaluating the relevance of a research paper to a search query.

Query: "{}"

Paper Title: "{}"
Paper Abstract: "{}"
Paper Source: {}
Paper Year: {}

Score the relevance of this paper to the query on a scale of 0-100:
- 0-20: Completely irrelevant
- 21-40: Tangentially related but not useful
- 41-60: Somewhat relevant, addresses a related topic
- 61-80: Relevant, directly addresses the query topic
- 81-100: Highly relevant, exactly what someone searching this query would want

Output ONLY a JSON object: {{"score": <number>, "reasoning": "<one sentence>"}}"#,
        query,
        paper.title,
        truncate(&paper.abstract_text, 500),
        paper.source,
        paper.year.map(|y| y.to_string()).unwrap_or("unknown".to_string()),
    );

    let response = chat_completion(
        &config,
        "You are a relevance evaluation system. Output only valid JSON. ",
        &prompt,
        true,
    ).await?;

    // Parse the response
    let json: serde_json::Value = serde_json::from_str(response.trim())
        .unwrap_or(serde_json::json!({"score": 0, "reasoning": "Failed to parse"}));

    let score = json["score"].as_f64().unwrap_or(0.0);
    let reasoning = json["reasoning"].as_str().unwrap_or("").to_string();

    Ok((score, reasoning))
}

/// Run the full benchmark suite.
pub async fn run_benchmark(queries: Option<Vec<BenchmarkQuery>>, top_k: usize) -> Result<BenchmarkReport> {
    let queries = queries.unwrap_or_else(default_queries);
    let total = queries.len();

    eprintln!("Running benchmark with {} queries, scoring top-{} results each...\n", total, top_k);

    let mut query_results = Vec::new();

    for (i, bq) in queries.iter().enumerate() {
        eprintln!("[{}/{}] Querying: \"{}\" ({})", i + 1, total, bq.query, bq.category);

        // Retrieve papers
        let papers = match retrieve_papers(&bq.query).await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  Error retrieving papers: {}", e);
                continue;
            }
        };

        let papers_to_score: Vec<&Paper> = papers.iter()
            .filter(|p| !p.abstract_text.is_empty())
            .take(top_k)
            .collect();

        eprintln!("  Found {} papers, scoring {}...", papers.len(), papers_to_score.len());

        let mut scores = Vec::new();
        for paper in &papers_to_score {
            match score_relevance(&bq.query, paper).await {
                Ok((score, reasoning)) => {
                    scores.push(RelevanceScore {
                        query: bq.query.clone(),
                        paper_title: paper.title.clone(),
                        paper_source: paper.source.to_string(),
                        score,
                        reasoning,
                    });
                }
                Err(e) => {
                    eprintln!("  Error scoring '{}': {}", paper.title, e);
                }
            }
        }

        let mean = if scores.is_empty() {
            0.0
        } else {
            scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
        };

        eprintln!("  Mean relevance: {:.1}\n", mean);

        query_results.push(QueryResult {
            query: bq.query.clone(),
            category: bq.category.clone(),
            num_papers_found: papers.len(),
            scores,
            mean_relevance: mean,
        });
    }

    // Calculate overall stats
    let overall_mean = if query_results.is_empty() {
        0.0
    } else {
        query_results.iter().map(|q| q.mean_relevance).sum::<f64>() / query_results.len() as f64
    };

    // Mean by category
    let mut category_scores: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for qr in &query_results {
        category_scores.entry(qr.category.to_string()).or_default().push(qr.mean_relevance);
    }
    let mean_by_category: Vec<(String, f64)> = category_scores.iter()
        .map(|(cat, scores)| {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            (cat.clone(), mean)
        })
        .collect();

    // Mean by rank position (how does relevance degrade as rank increases?)
    let mut rank_scores: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
    for qr in &query_results {
        for (rank, score) in qr.scores.iter().enumerate() {
            rank_scores.entry(rank + 1).or_default().push(score.score);
        }
    }
    let mut mean_by_rank: Vec<(usize, f64)> = rank_scores.iter()
        .map(|(rank, scores)| {
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            (*rank, mean)
        })
        .collect();
    mean_by_rank.sort_by_key(|(rank, _)| *rank);

    let report = BenchmarkReport {
        total_queries: total,
        overall_mean_relevance: overall_mean,
        mean_by_category,
        mean_by_rank,
        query_results,
    };

    Ok(report)
}

/// Print a human-readable benchmark report.
pub fn print_report(report: &BenchmarkReport) {
    let sep = "=".repeat(60);
    println!("\n{}", sep);
    println!("  BENCHMARK REPORT");
    println!("{}\n", sep);

    println!("Total queries: {}", report.total_queries);
    println!("Overall mean relevance: {:.1}/100\n", report.overall_mean_relevance);

    println!("--- By Category ---");
    for (cat, mean) in &report.mean_by_category {
        println!("  {:<20} {:.1}/100", cat, mean);
    }

    println!("\n--- By Rank Position ---");
    for (rank, mean) in &report.mean_by_rank {
        println!("  Rank #{:<3}           {:.1}/100", rank, mean);
    }

    println!("\n--- Paper Lantern Comparison ---");
    println!("  Paper Lantern NL:   82.8/100");
    println!("  Our NL:             {:.1}/100",
        report.mean_by_category.iter()
            .find(|(c, _)| c == "natural_language")
            .map(|(_, m)| *m)
            .unwrap_or(0.0)
    );

    println!("\n--- Per-Query Detail ---");
    for qr in &report.query_results {
        println!("  [{:.1}] \"{}\" ({}, {} papers)", qr.mean_relevance, qr.query, qr.category, qr.num_papers_found);
    }

    println!("\n{}", "=".repeat(60));
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max]) }
}
