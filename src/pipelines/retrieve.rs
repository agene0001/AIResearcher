use anyhow::Result;
use std::collections::HashSet;
use crate::models::paper::Paper;
use crate::retrieval::arxiv::search_arxiv;
use crate::retrieval::semantic::search_semantic_scholar;
use crate::retrieval::pubmed::search_pubmed;
use crate::retrieval::openalex::search_openalex;
use crate::retrieval::crossref::search_crossref;
use crate::retrieval::dblp::search_dblp;
use crate::storage::postgres::DbClient;
use crate::processing::embedding::{generate_embedding, generate_document_embedding};
use crate::llm::client::{cheap_config, chat_completion};

const PER_PROVIDER_LIMIT: usize = 20;

/// Use the cheap LLM to reformulate a natural language query into search-optimized keywords.
/// For short keyword queries, returns them as-is.
async fn reformulate_query(query: &str) -> Vec<String> {
    // If the query is already short keywords (≤4 words), use it directly
    let word_count = query.split_whitespace().count();
    if word_count <= 4 {
        return vec![query.to_string()];
    }

    let config = cheap_config();
    let prompt = format!(
        r#"Convert this natural language research query into 2-3 different keyword search strings that academic paper databases would match well. Each should be a different angle on the same topic.

Query: "{}"

Output ONLY a JSON array of strings, e.g. ["keyword phrase 1", "keyword phrase 2", "keyword phrase 3"]
Do not include any other text."#,
        query
    );

    match chat_completion(&config, "You extract search keywords from questions. Output only valid JSON. /no_think", &prompt, false).await {
        Ok(response) => {
            let trimmed = response.trim();
            // Try to parse JSON array from response
            let json_str = if let Some(start) = trimmed.find('[') {
                if let Some(end) = trimmed.rfind(']') {
                    &trimmed[start..=end]
                } else {
                    trimmed
                }
            } else {
                trimmed
            };

            if let Ok(keywords) = serde_json::from_str::<Vec<String>>(json_str) {
                if !keywords.is_empty() {
                    eprintln!("  Reformulated \"{}\" -> {:?}", query, keywords);
                    // Always include the original query too
                    let mut all = vec![query.to_string()];
                    all.extend(keywords);
                    return all;
                }
            }
            // Fallback: use original query
            vec![query.to_string()]
        }
        Err(_) => vec![query.to_string()],
    }
}

/// Retrieve papers from all providers in parallel, deduplicate, and return merged results.
/// If a database is available, also indexes papers and supplements results with hybrid search.
pub async fn retrieve_papers(query: &str) -> Result<Vec<Paper>> {
    // Step 1: Reformulate NL queries into keyword variants
    let search_queries = reformulate_query(query).await;

    // Step 2: Fan out to all providers with all query variants concurrently
    let mut provider_futures = Vec::new();
    for sq in &search_queries {
        let sq = sq.clone();
        provider_futures.push(tokio::spawn(async move {
            let (arxiv, s2, pubmed, openalex, crossref, dblp) = tokio::join!(
                search_arxiv(&sq, PER_PROVIDER_LIMIT),
                search_semantic_scholar(&sq, PER_PROVIDER_LIMIT),
                search_pubmed(&sq, PER_PROVIDER_LIMIT),
                search_openalex(&sq, PER_PROVIDER_LIMIT),
                search_crossref(&sq, PER_PROVIDER_LIMIT),
                search_dblp(&sq, PER_PROVIDER_LIMIT),
            );
            vec![
                ("arXiv", arxiv),
                ("Semantic Scholar", s2),
                ("PubMed", pubmed),
                ("OpenAlex", openalex),
                ("Crossref", crossref),
                ("DBLP", dblp),
            ]
        }));
    }

    let all_results = futures::future::join_all(provider_futures).await;

    let mut all_papers = Vec::new();
    let mut seen_titles: HashSet<String> = HashSet::new();
    let mut seen_dois: HashSet<String> = HashSet::new();

    let mut merge = |result: Result<Vec<Paper>>, source_name: &str| {
        match result {
            Ok(papers) => {
                tracing::info!("{}: returned {} papers", source_name, papers.len());
                for paper in papers {
                    let norm_title = normalize_title(&paper.title);

                    if let Some(ref doi) = paper.doi {
                        if !seen_dois.insert(doi.to_lowercase()) {
                            continue;
                        }
                    }

                    if !seen_titles.insert(norm_title) {
                        continue;
                    }

                    all_papers.push(paper);
                }
            }
            Err(e) => {
                tracing::warn!("{} search failed: {}", source_name, e);
                eprintln!("Warning: {} search failed: {}", source_name, e);
            }
        }
    };

    for result in all_results {
        if let Ok(provider_results) = result {
            for (source_name, result) in provider_results {
                merge(result, source_name);
            }
        }
    }

    tracing::info!("Total papers after deduplication: {}", all_papers.len());

    // Try to use the database for hybrid search and indexing
    if let Some(db) = DbClient::try_new().await {
        // Index the freshly retrieved papers in the background
        let papers_to_index: Vec<Paper> = all_papers.iter()
            .filter(|p| !p.abstract_text.is_empty())
            .cloned()
            .collect();

        if !papers_to_index.is_empty() {
            let db_clone = db.clone();
            tokio::spawn(async move {
                for paper in &papers_to_index {
                    let text = format!("{} {}", paper.title, paper.abstract_text);
                    match generate_document_embedding(&text).await {
                        Ok(embedding) => {
                            if let Err(e) = db_clone.insert_paper(paper, embedding).await {
                                tracing::warn!("Failed to index paper {}: {}", paper.id, e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to embed paper {}: {}", paper.id, e);
                        }
                    }
                }
                tracing::info!("Indexed {} papers in background", papers_to_index.len());
            });
        }

        // Also do a hybrid search on existing indexed papers to supplement results
        match generate_embedding(query).await {
            Ok(query_embedding) => {
                match db.hybrid_search(query, query_embedding, 20).await {
                    Ok(db_papers) => {
                        tracing::info!("Hybrid search returned {} papers from DB", db_papers.len());
                        for paper in db_papers {
                            let norm_title = normalize_title(&paper.title);
                            if let Some(ref doi) = paper.doi {
                                if !seen_dois.insert(doi.to_lowercase()) {
                                    continue;
                                }
                            }
                            if seen_titles.insert(norm_title) {
                                all_papers.push(paper);
                            }
                        }
                    }
                    Err(e) => tracing::warn!("Hybrid search failed: {}", e),
                }
            }
            Err(e) => tracing::warn!("Failed to generate query embedding: {}", e),
        }
    }

    // Semantic reranking: score papers by embedding similarity to the original query
    all_papers = semantic_rerank(query, all_papers).await;

    Ok(all_papers)
}

/// Rerank papers by semantic similarity to the query using harrier embeddings.
/// Papers without abstracts are pushed to the end.
async fn semantic_rerank(query: &str, mut papers: Vec<Paper>) -> Vec<Paper> {
    // Only rerank papers that have abstracts
    let papers_with_abstracts: Vec<&Paper> = papers.iter()
        .filter(|p| !p.abstract_text.is_empty())
        .collect();

    if papers_with_abstracts.is_empty() {
        return papers;
    }

    // Generate query embedding
    let query_embedding = match generate_embedding(query).await {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  Reranking skipped (embedding error): {}", e);
            return papers;
        }
    };

    // Score each paper by cosine similarity
    let mut scored: Vec<(usize, f32)> = Vec::new();
    for (idx, paper) in papers.iter().enumerate() {
        if paper.abstract_text.is_empty() {
            scored.push((idx, -1.0)); // Push to end
            continue;
        }

        let text = format!("{} {}", paper.title, paper.abstract_text);
        match generate_document_embedding(&text).await {
            Ok(doc_embedding) => {
                let sim = cosine_similarity(&query_embedding, &doc_embedding);
                scored.push((idx, sim));
            }
            Err(_) => {
                scored.push((idx, -1.0));
            }
        }
    }

    // Sort by similarity descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Rebuild papers vec in ranked order
    let reranked: Vec<Paper> = scored.into_iter()
        .map(|(idx, _)| papers[idx].clone())
        .collect();

    reranked
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Normalize a title for deduplication: lowercase, strip punctuation, collapse whitespace.
fn normalize_title(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
