use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

use crate::pipelines::{
    explore::explore_approaches,
    compare::compare_approaches,
    deep_dive::deep_dive,
    researcher::research_and_propose,
    trends::detect_trends,
};

#[derive(Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct RpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

pub async fn run_mcp_server() -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let req: RpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to parse request: {}", e);
                continue;
            }
        };

        if req.jsonrpc != "2.0" {
            continue;
        }

        let id = req.id.unwrap_or(Value::Null);
        let mut response = RpcResponse {
            jsonrpc: "2.0".to_string(),
            id: id.clone(),
            result: None,
            error: None,
        };

        match req.method.as_str() {
            "initialize" => {
                response.result = Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "autoresearch-lab",
                        "version": "0.1.0"
                    }
                }));
            }
            "notifications/initialized" => {
                // No response needed for notifications
                continue;
            }
            "tools/list" => {
                response.result = Some(json!({
                    "tools": [
                        {
                            "name": "explore_approaches",
                            "description": "Finds multiple candidate techniques from recent papers for a specific problem.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query, e.g., 'techniques for stabilizing gradient norms'"
                                    }
                                },
                                "required": ["query"]
                            }
                        },
                        {
                            "name": "compare_approaches",
                            "description": "Evaluates multiple candidate approaches side by side to find the best fit for a specific architecture or problem.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "problem": {
                                        "type": "string",
                                        "description": "The specific problem or architecture context."
                                    },
                                    "candidates_json": {
                                        "type": "string",
                                        "description": "A JSON string containing an array of PaperSummary objects returned by explore_approaches."
                                    }
                                },
                                "required": ["problem", "candidates_json"]
                            }
                        },
                        {
                            "name": "deep_dive",
                            "description": "Drills into one specific paper to extract exact implementation details, hyperparameters, and failure modes.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "paper_summary_json": {
                                        "type": "string",
                                        "description": "A JSON string containing the PaperSummary object to deep dive into."
                                    },
                                    "specific_context": {
                                        "type": "string",
                                        "description": "The context for the implementation."
                                    }
                                },
                                "required": ["paper_summary_json", "specific_context"]
                            }
                        },
                        {
                            "name": "researcher",
                            "description": "End-to-end tool that researches a problem and proposes concrete, implementation-ready ideas from academic papers tailored to your codebase context. Use this when you want research-backed ideas to integrate into the code without running an automated experiment loop.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "problem_description": {
                                        "type": "string",
                                        "description": "The specific problem you are trying to solve or improve."
                                    },
                                    "codebase_context": {
                                        "type": "string",
                                        "description": "The current state of your code, constraints, or architecture."
                                    }
                                },
                                "required": ["problem_description", "codebase_context"]
                            }
                        },
                        {
                            "name": "detect_trends",
                            "description": "Detect emerging research trends and breakthrough ideas in a given field. Analyzes recent papers across multiple sources to identify what's gaining traction, growth areas, and predictions.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "field": {
                                        "type": "string",
                                        "description": "The research field to analyze trends in, e.g. 'large language models', 'protein folding', 'reinforcement learning'"
                                    }
                                },
                                "required": ["field"]
                            }
                        }
                    ]
                }));
            }
            "tools/call" => {
                let params = req.params.unwrap_or(json!({}));
                let name = params["name"].as_str().unwrap_or("");
                let args = params["arguments"].as_object().cloned().unwrap_or_default();

                match name {
                    "explore_approaches" => {
                        if let Some(query) = args.get("query").and_then(|v| v.as_str()) {
                            match explore_approaches(query).await {
                                Ok(summaries) => {
                                    response.result = Some(json!({
                                        "content": [{
                                            "type": "text",
                                            "text": serde_json::to_string_pretty(&summaries).unwrap_or_default()
                                        }]
                                    }));
                                }
                                Err(e) => {
                                    response.error = Some(json!({ "code": -32000, "message": e.to_string() }));
                                }
                            }
                        } else {
                            response.error = Some(json!({ "code": -32602, "message": "Missing 'query' argument" }));
                        }
                    }
                    "compare_approaches" => {
                        let problem = args.get("problem").and_then(|v| v.as_str());
                        let candidates_json = args.get("candidates_json").and_then(|v| v.as_str());
                        
                        if let (Some(p), Some(c_json)) = (problem, candidates_json) {
                            match serde_json::from_str::<Vec<crate::models::summary::PaperSummary>>(c_json) {
                                Ok(candidates) => {
                                    match compare_approaches(p, &candidates).await {
                                        Ok(res) => {
                                            response.result = Some(json!({
                                                "content": [{ "type": "text", "text": res }]
                                            }));
                                        }
                                        Err(e) => response.error = Some(json!({ "code": -32000, "message": e.to_string() })),
                                    }
                                }
                                Err(e) => response.error = Some(json!({ "code": -32602, "message": format!("Invalid candidates_json: {}", e) })),
                            }
                        } else {
                            response.error = Some(json!({ "code": -32602, "message": "Missing 'problem' or 'candidates_json' argument" }));
                        }
                    }
                    "deep_dive" => {
                        let summary_json = args.get("paper_summary_json").and_then(|v| v.as_str());
                        let context = args.get("specific_context").and_then(|v| v.as_str());
                        
                        if let (Some(s_json), Some(ctx)) = (summary_json, context) {
                            match serde_json::from_str(s_json) {
                                Ok(summary) => {
                                    match deep_dive(&summary, ctx).await {
                                        Ok(res) => {
                                            response.result = Some(json!({
                                                "content": [{ "type": "text", "text": res }]
                                            }));
                                        }
                                        Err(e) => response.error = Some(json!({ "code": -32000, "message": e.to_string() })),
                                    }
                                }
                                Err(e) => response.error = Some(json!({ "code": -32602, "message": format!("Invalid paper_summary_json: {}", e) })),
                            }
                        } else {
                            response.error = Some(json!({ "code": -32602, "message": "Missing 'paper_summary_json' or 'specific_context' argument" }));
                        }
                    }
                    "researcher" => {
                        let problem = args.get("problem_description").and_then(|v| v.as_str());
                        let context = args.get("codebase_context").and_then(|v| v.as_str());
                        
                        if let (Some(p), Some(ctx)) = (problem, context) {
                            match research_and_propose(p, ctx).await {
                                Ok(res) => {
                                    response.result = Some(json!({
                                        "content": [{ "type": "text", "text": res }]
                                    }));
                                }
                                Err(e) => response.error = Some(json!({ "code": -32000, "message": e.to_string() })),
                            }
                        } else {
                            response.error = Some(json!({ "code": -32602, "message": "Missing 'problem_description' or 'codebase_context' argument" }));
                        }
                    }
                    "detect_trends" => {
                        if let Some(field) = args.get("field").and_then(|v| v.as_str()) {
                            match detect_trends(field).await {
                                Ok(report) => {
                                    response.result = Some(json!({
                                        "content": [{ "type": "text", "text": report }]
                                    }));
                                }
                                Err(e) => response.error = Some(json!({ "code": -32000, "message": e.to_string() })),
                            }
                        } else {
                            response.error = Some(json!({ "code": -32602, "message": "Missing 'field' argument" }));
                        }
                    }
                    _ => {
                        response.error = Some(json!({ "code": -32601, "message": "Method not found" }));
                    }
                }
            }
            _ => {
                response.error = Some(json!({ "code": -32601, "message": "Method not found" }));
            }
        }

        if id != Value::Null {
            let out = serde_json::to_string(&response)?;
            println!("{}", out);
            stdout.flush()?;
        }
    }

    Ok(())
}
