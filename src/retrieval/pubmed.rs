use anyhow::Result;
use reqwest::Client;
use crate::models::paper::{Paper, PaperSource};

/// Search PubMed via the NCBI E-utilities API (free, no key required for moderate use).
pub async fn search_pubmed(query: &str, max_results: usize) -> Result<Vec<Paper>> {
    let client = Client::new();
    let limit = max_results.min(50);

    // Step 1: Search for IDs
    let search_url = format!(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax={}&retmode=json&sort=relevance",
        urlencoded(query), limit
    );
    let search_res: serde_json::Value = client.get(&search_url).send().await?.json().await?;

    let ids: Vec<&str> = search_res["esearchresult"]["idlist"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    if ids.is_empty() {
        return Ok(vec![]);
    }

    // Step 2: Fetch details for those IDs
    let id_list = ids.join(",");
    let fetch_url = format!(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}&retmode=json",
        id_list
    );
    let fetch_res: serde_json::Value = client.get(&fetch_url).send().await?.json().await?;

    // Step 3: Fetch abstracts via efetch (XML)
    let abstract_url = format!(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={}&retmode=xml",
        id_list
    );
    let abstract_xml = client.get(&abstract_url).send().await?.text().await?;
    let abstracts = parse_pubmed_abstracts(&abstract_xml);

    let mut papers = Vec::new();

    if let Some(result) = fetch_res.get("result") {
        for pmid in &ids {
            if let Some(item) = result.get(*pmid) {
                let title = item["title"].as_str().unwrap_or("").to_string();
                let year = item["pubdate"].as_str()
                    .and_then(|d| d.get(..4))
                    .and_then(|y| y.parse::<u32>().ok());
                let doi = item.get("elocationid")
                    .and_then(|e| e.as_str())
                    .and_then(|s| s.strip_prefix("doi: "))
                    .map(|s| s.to_string());
                let authors: Vec<String> = item.get("authors")
                    .and_then(|a| a.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|a| a.get("name").and_then(|n| n.as_str()).map(|s| s.to_string()))
                        .collect())
                    .unwrap_or_default();

                let abstract_text = abstracts.get(*pmid).cloned().unwrap_or_default();

                if !title.is_empty() {
                    papers.push(Paper {
                        id: format!("pmid:{}", pmid),
                        title,
                        abstract_text,
                        content: None,
                        source: PaperSource::PubMed,
                        year,
                        doi,
                        url: Some(format!("https://pubmed.ncbi.nlm.nih.gov/{}/", pmid)),
                        pdf_url: None,
                        authors,
                    });
                }
            }
        }
    }

    Ok(papers)
}

fn urlencoded(s: &str) -> String {
    s.replace(' ', "+")
}

/// Parse abstracts from PubMed efetch XML response.
fn parse_pubmed_abstracts(xml: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();

    // Simple parsing: extract PMID and AbstractText pairs
    let doc = match roxmltree::Document::parse(xml) {
        Ok(d) => d,
        Err(_) => return map,
    };

    for article in doc.descendants().filter(|n| n.has_tag_name("PubmedArticle")) {
        let pmid = article.descendants()
            .find(|n| n.has_tag_name("PMID"))
            .and_then(|n| n.text())
            .unwrap_or("")
            .to_string();

        let abstract_parts: Vec<&str> = article.descendants()
            .filter(|n| n.has_tag_name("AbstractText"))
            .filter_map(|n| n.text())
            .collect();

        if !pmid.is_empty() && !abstract_parts.is_empty() {
            map.insert(pmid, abstract_parts.join(" "));
        }
    }

    map
}
