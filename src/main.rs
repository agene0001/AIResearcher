use anyhow::Result;
use clap::Parser;
use dotenv::dotenv;
use std::fs;
use std::path::PathBuf;

use crate::cli::{Cli, Commands};
use crate::pipelines::retrieve::retrieve_papers;
use crate::pipelines::collection::generate_collection;
use crate::pipelines::trends::detect_trends;
use crate::evaluation::benchmark::{run_benchmark, print_report};
use crate::mcp_server::run_mcp_server;

mod cli;
mod config;
mod pipelines;
mod llm;
mod retrieval;
mod models;
mod processing;
mod storage;
mod evaluation;
mod utils;
mod mcp_server;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Mcp => {
            // Run the MCP server loop
            run_mcp_server().await?;
        }
        Commands::Search { query } => {
            println!("🔍 Searching for: '{}'", query);
            let papers = retrieve_papers(&query).await?;
            if papers.is_empty() {
                println!("No papers found.");
            } else {
                println!("✅ Found {} papers:\n", papers.len());
                for (i, paper) in papers.iter().enumerate() {
                    println!("{}. {}", i + 1, paper.title);
                    println!("   ID: {}", paper.id);
                    println!("   Abstract: {}...\n", &paper.abstract_text.chars().take(200).collect::<String>());
                }
            }
        }
        Commands::Collection { topic } => {
            match generate_collection(&topic).await {
                Ok(collection) => {
                    println!("\n==================================\n");
                    println!("📚 Collection: {}\n", topic);
                    println!("{}", collection);
                    println!("\n==================================\n");
                }
                Err(e) => eprintln!("Failed to generate collection: {}", e),
            }
        }
        Commands::Trends { field } => {
            match detect_trends(&field).await {
                Ok(report) => {
                    println!("\n==================================\n");
                    println!("Trend Report: {}\n", field);
                    println!("{}", report);
                    println!("\n==================================\n");
                }
                Err(e) => eprintln!("Failed to detect trends: {}", e),
            }
        }
        Commands::Benchmark { top_k, output } => {
            match run_benchmark(None, top_k).await {
                Ok(report) => {
                    print_report(&report);

                    if let Some(path) = output {
                        let json = serde_json::to_string_pretty(&report)?;
                        std::fs::write(&path, json)?;
                        println!("\nFull report saved to: {}", path);
                    }
                }
                Err(e) => eprintln!("Benchmark failed: {}", e),
            }
        }
        Commands::TestEmbedding => {
            use std::time::Instant;
            use crate::processing::harrier;

            println!("Loading harrier-oss-v1-0.6b...");
            let load_start = Instant::now();
            // Warm up — first call downloads + loads the model
            let _ = harrier::embed_query("test");
            println!("Model loaded in {:.2?}\n", load_start.elapsed());

            let test_texts = vec![
                "transformer attention mechanisms",
                "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains.",
                "We introduce a novel approach to gradient normalization for stabilizing transformer pre-training at scale. Our method dynamically adjusts the gradient clipping threshold based on the running statistics of gradient norms across all layers, enabling more stable training of billion-parameter models without manual tuning of hyperparameters.",
            ];

            let mut times = Vec::new();
            for (i, text) in test_texts.iter().enumerate() {
                let start = Instant::now();
                let embedding = harrier::embed_query(text).unwrap();
                let elapsed = start.elapsed();
                times.push(elapsed);
                println!("Text {}: {} chars, {} tokens (est) -> {}d embedding in {:.2?}",
                    i + 1, text.len(), text.len() / 4, embedding.len(), elapsed);
            }

            let avg = times.iter().map(|t| t.as_millis()).sum::<u128>() / times.len() as u128;
            println!("\nAverage: {}ms per embedding", avg);
            println!("Estimated 1K papers: {:.1}s", avg as f64 * 1000.0 / 1000.0);
            println!("Estimated 10K papers: {:.1}m", avg as f64 * 10000.0 / 1000.0 / 60.0);
            println!("Estimated 2M papers: {:.1}h", avg as f64 * 2_000_000.0 / 1000.0 / 3600.0);
        }
        Commands::InstallMcp { editor } => {
            install_mcp(&editor)?;
        }
    }

    Ok(())
}

fn install_mcp(editor: &str) -> Result<()> {
    let current_dir = std::env::current_dir()?;
    let manifest_path = current_dir.join("Cargo.toml");
    
    let config = serde_json::json!({
        "mcpServers": {
            "autoresearch-lab": {
                "command": "cargo",
                "args": ["run", "--manifest-path", manifest_path.to_str().unwrap(), "--", "mcp"]
            }
        }
    });

    match editor.to_lowercase().as_str() {
        "cursor" => {
            let cursor_dir = current_dir.join(".cursor");
            fs::create_dir_all(&cursor_dir)?;
            let mcp_json_path = cursor_dir.join("mcp.json");
            
            fs::write(&mcp_json_path, serde_json::to_string_pretty(&config)?)?;
            println!("✅ Installed MCP config for Cursor at {}", mcp_json_path.display());
            println!("Restart Cursor or reload the window for the changes to take effect.");
        }
        "claude" => {
            let home_dir = dirs::home_dir().expect("Could not find home directory");
            
            let claude_config_path = if cfg!(target_os = "macos") {
                home_dir.join("Library/Application Support/Claude/claude_desktop_config.json")
            } else if cfg!(target_os = "windows") {
                PathBuf::from(std::env::var("APPDATA").unwrap_or_default()).join("Claude/claude_desktop_config.json")
            } else {
                home_dir.join(".config/Claude/claude_desktop_config.json") // Linux fallback
            };

            if let Some(parent) = claude_config_path.parent() {
                fs::create_dir_all(parent)?;
            }

            // Read existing config if it exists, otherwise create new
            let mut existing_config = if claude_config_path.exists() {
                let content = fs::read_to_string(&claude_config_path)?;
                serde_json::from_str(&content).unwrap_or(serde_json::json!({ "mcpServers": {} }))
            } else {
                serde_json::json!({ "mcpServers": {} })
            };

            // Merge our server
            if let Some(servers) = existing_config.get_mut("mcpServers").and_then(|s| s.as_object_mut()) {
                servers.insert(
                    "autoresearch-lab".to_string(),
                    config["mcpServers"]["autoresearch-lab"].clone()
                );
            }

            fs::write(&claude_config_path, serde_json::to_string_pretty(&existing_config)?)?;
            println!("✅ Installed MCP config for Claude Desktop at {}", claude_config_path.display());
            println!("Restart Claude Desktop for the changes to take effect.");
        }
        _ => {
            println!("❌ Unsupported editor '{}'. Please choose 'cursor' or 'claude'.", editor);
        }
    }

    Ok(())
}
