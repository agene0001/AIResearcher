use serde::{Deserialize, Serialize};

/// The source provider a paper was retrieved from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PaperSource {
    Arxiv,
    SemanticScholar,
    PubMed,
    OpenAlex,
    Crossref,
    Dblp,
}

impl std::fmt::Display for PaperSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PaperSource::Arxiv => write!(f, "arxiv"),
            PaperSource::SemanticScholar => write!(f, "semantic_scholar"),
            PaperSource::PubMed => write!(f, "pubmed"),
            PaperSource::OpenAlex => write!(f, "openalex"),
            PaperSource::Crossref => write!(f, "crossref"),
            PaperSource::Dblp => write!(f, "dblp"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paper {
    pub id: String,
    pub title: String,
    pub abstract_text: String,
    pub content: Option<String>,
    /// Which provider this paper came from
    #[serde(default = "default_source")]
    pub source: PaperSource,
    /// Year of publication, if known
    pub year: Option<u32>,
    /// DOI, if available
    pub doi: Option<String>,
    /// URL to the paper
    pub url: Option<String>,
    /// Authors list
    #[serde(default)]
    pub authors: Vec<String>,
}

fn default_source() -> PaperSource {
    PaperSource::Arxiv
}
