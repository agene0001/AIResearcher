use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ResearchPlan {
    pub goal: String,
    pub steps: Vec<String>,
    pub experiments: Vec<String>,
}
