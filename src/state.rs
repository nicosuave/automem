use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub size: u64,
    pub mtime: i64,
    pub offset: u64,
    pub turn_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestState {
    pub next_doc_id: u64,
    pub files: HashMap<String, FileState>,
}

impl Default for IngestState {
    fn default() -> Self {
        Self {
            next_doc_id: 1,
            files: HashMap::new(),
        }
    }
}

impl IngestState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)?;
        let state = serde_json::from_str(&data)?;
        Ok(state)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}
