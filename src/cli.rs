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

    /// Test embedding speed with harrier-oss-v1-0.6b
    TestEmbedding,

    /// Install the MCP server to an editor (cursor, claude)
    InstallMcp {
        /// The editor to install to (e.g., 'cursor' or 'claude')
        editor: String,
    },
}
