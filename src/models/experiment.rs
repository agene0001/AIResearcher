use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub method: String,
    pub parameters: serde_json::Value,
    pub score: f64,
}
