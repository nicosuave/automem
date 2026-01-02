use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileState {
    pub size: u64,
    pub mtime: i64,
    pub offset: u64,
    pub turn_id: u32,
}

/// Tracks when we last scanned for changes, allowing us to skip
/// redundant scans if called again within a short TTL.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanCache {
    /// Unix timestamp (seconds) of last successful scan
    pub last_scan_ts: u64,
    /// Number of files found in last scan
    pub file_count: usize,
    /// Total bytes across all source files
    pub total_bytes: u64,
}

impl ScanCache {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)?;
        let cache = serde_json::from_str(&data)?;
        Ok(cache)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Check if the cache is still valid (within TTL seconds)
    pub fn is_fresh(&self, ttl_seconds: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.last_scan_ts) < ttl_seconds
    }

    /// Update cache with current scan results
    pub fn update(&mut self, file_count: usize, total_bytes: u64) {
        self.last_scan_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.file_count = file_count;
        self.total_bytes = total_bytes;
    }
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
