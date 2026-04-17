use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PaperSummary {
    pub method: String,
    pub core_idea: String,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub use_cases: Vec<String>,
    
    // New fields aligned with Paper Lantern's "implementation-ready" focus
    pub hyperparameters: Option<String>,
    pub failure_modes: Vec<String>,
    pub implementation_details: Option<String>,
    
    pub data_requirements: Option<String>,
    pub compute_cost: Option<String>,
}
