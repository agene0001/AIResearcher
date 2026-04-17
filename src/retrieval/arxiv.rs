use anyhow::Result;
use reqwest::Client;
use roxmltree::Document;
use crate::models::paper::{Paper, PaperSource};

pub async fn search_arxiv(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let formatted_query = query.replace(' ', "+");
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&max_results={}&sortBy=relevance&sortOrder=descending",
        formatted_query, max_results
    );

    let client = Client::new();
    let xml_res = client.get(&url).send().await?.text().await?;

    let mut papers = Vec::new();

    let doc = Document::parse(&xml_res)?;

    for node in doc.descendants().filter(|n| n.has_tag_name("entry")) {
        let id = node.children().find(|n| n.has_tag_name("id"))
            .map(|n| n.text().unwrap_or("")).unwrap_or("").to_string();
        let title = node.children().find(|n| n.has_tag_name("title"))
            .map(|n| n.text().unwrap_or("")).unwrap_or("")
            .replace('\n', " ").trim().to_string();
        let abstract_text = node.children().find(|n| n.has_tag_name("summary"))
            .map(|n| n.text().unwrap_or("")).unwrap_or("")
            .replace('\n', " ").trim().to_string();
        let published = node.children().find(|n| n.has_tag_name("published"))
            .map(|n| n.text().unwrap_or("")).unwrap_or("").to_string();
        let year = published.get(..4).and_then(|y| y.parse::<u32>().ok());

        let authors: Vec<String> = node.children()
            .filter(|n| n.has_tag_name("author"))
            .filter_map(|a| a.children().find(|n| n.has_tag_name("name")))
            .filter_map(|n| n.text().map(|s| s.to_string()))
            .collect();

        // Extract arXiv ID from the full URL for a cleaner id
        let clean_id = if id.contains("arxiv.org/abs/") {
            format!("arxiv:{}", id.split("/abs/").last().unwrap_or(&id))
        } else {
            format!("arxiv:{}", id)
        };

        if !id.is_empty() && !title.is_empty() {
            papers.push(Paper {
                id: clean_id,
                title,
                abstract_text,
                content: None,
                source: PaperSource::Arxiv,
                year,
                doi: None,
                url: Some(id),
                authors,
            });
        }
    }

    Ok(papers)
}
