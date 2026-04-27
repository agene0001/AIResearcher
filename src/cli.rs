use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "autoresearch-lab")]
#[command(about = "AI-powered research system and MCP server", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run the MCP server (used by editors)
    Mcp,
    
    /// Search for research papers matching a query
    Search {
        /// The search query
        query: String,
    },
    
    /// Generate a curated learning collection for a topic
    Collection {
        /// The topic to learn about
        topic: String,
    },
    
    /// Detect emerging research trends in a field
    Trends {
        /// The research field to analyze trends in
        field: String,
    },

    /// Run the search quality benchmark (compares against Paper Lantern's scores)
    Benchmark {
        /// Number of top results to score per query (default: 5)
        #[arg(long, default_value = "5")]
        top_k: usize,

        /// Save the full report as JSON to this path
        #[arg(long)]
        output: Option<String>,
    },

    /// Ingest papers from OpenAlex into the database
    Ingest {
        /// Ingestion source: "api" (no download, rate-limited) or "snapshot" (bulk download, faster)
        #[arg(long, default_value = "api")]
        source: String,

        /// Path to the OpenAlex snapshot directory (required for --source snapshot)
        #[arg(long)]
        snapshot_dir: Option<String>,

        /// Minimum publication year to include (default: 2015)
        #[arg(long, default_value = "2015")]
        min_year: u32,

        /// Batch size for embedding/insertion (default: 128)
        #[arg(long, default_value = "128")]
        batch_size: usize,

        /// Max papers to ingest, 0 for unlimited (default: 0)
        #[arg(long, default_value = "0")]
        max_papers: usize,

        /// OpenAlex field name (coarse filter — 26 exist, e.g. "Computer Science", "Medicine").
        /// Default when neither --field nor --subfield is set: "Computer Science".
        /// Browse all fields at https://api.openalex.org/fields. Mutually exclusive with --subfield.
        #[arg(long, conflicts_with = "subfield")]
        field: Option<String>,

        /// OpenAlex subfield name (fine filter — 252 exist). Pass multiple times for multiple subfields (OR'd).
        /// Example: --subfield "Artificial Intelligence" --subfield "Computer Vision and Pattern Recognition".
        /// Browse all subfields at https://api.openalex.org/subfields. Mutually exclusive with --field.
        #[arg(long, action = clap::ArgAction::Append)]
        subfield: Vec<String>,
    },

    /// Test embedding speed with harrier-oss-v1-0.6b
    TestEmbedding,

    /// Tier-2 deep read: download a paper's PDF, parse it (marker/pdftotext),
    /// cache the result in `paper_full_text`, and print the extracted text.
    /// The paper must already exist in the DB and have a pdf_url.
    ReadPaper {
        /// Paper id, e.g. "openalex:W12345" or "arxiv:2401.12345".
        id: String,
    },

    /// Install the MCP server to an editor (cursor, claude)
    InstallMcp {
        /// The editor to install to (e.g., 'cursor' or 'claude')
        editor: String,
    },
}
