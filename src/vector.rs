use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

pub struct VectorIndex {
    dims: usize,
    path: PathBuf,
    vectors: Vec<f32>,
    doc_ids: Vec<u64>,
    doc_id_set: HashSet<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorMeta {
    dimensions: usize,
}

impl VectorIndex {
    pub fn open_or_create(dir: &Path, dimensions: usize) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let meta_path = dir.join("meta.json");
        let vectors_path = dir.join("vectors.f32");
        let ids_path = dir.join("doc_ids.u64");

        let mut reset = false;
        if meta_path.exists() {
            let data = fs::read_to_string(&meta_path)?;
            let meta: VectorMeta = serde_json::from_str(&data)?;
            if meta.dimensions != dimensions {
                reset = true;
            }
        }

        if reset {
            let _ = fs::remove_file(&meta_path);
            let _ = fs::remove_file(&vectors_path);
            let _ = fs::remove_file(&ids_path);
        }

        if !meta_path.exists() {
            let meta = VectorMeta { dimensions };
            fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
        }

        let (doc_ids, vectors) = if vectors_path.exists() && ids_path.exists() {
            let ids_bytes = fs::read(&ids_path)?;
            let vec_bytes = fs::read(&vectors_path)?;
            let doc_ids = bytes_to_u64(&ids_bytes);
            let vectors = bytes_to_f32(&vec_bytes);
            if doc_ids.len() * dimensions != vectors.len() {
                return Err(anyhow!("vector index corrupt"));
            }
            (doc_ids, vectors)
        } else {
            (Vec::new(), Vec::new())
        };

        let doc_id_set = doc_ids.iter().copied().collect();

        Ok(Self {
            dims: dimensions,
            path: dir.to_path_buf(),
            vectors,
            doc_ids,
            doc_id_set,
        })
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let meta_path = dir.join("meta.json");
        let vectors_path = dir.join("vectors.f32");
        let ids_path = dir.join("doc_ids.u64");
        if !meta_path.exists() || !vectors_path.exists() || !ids_path.exists() {
            return Err(anyhow!("vector index not found"));
        }
        let data = fs::read_to_string(&meta_path)?;
        let meta: VectorMeta = serde_json::from_str(&data)?;

        let ids_bytes = fs::read(&ids_path)?;
        let vec_bytes = fs::read(&vectors_path)?;
        let doc_ids = bytes_to_u64(&ids_bytes);
        let vectors = bytes_to_f32(&vec_bytes);
        if doc_ids.len() * meta.dimensions != vectors.len() {
            return Err(anyhow!("vector index corrupt"));
        }
        let doc_id_set = doc_ids.iter().copied().collect();

        Ok(Self {
            dims: meta.dimensions,
            path: dir.to_path_buf(),
            vectors,
            doc_ids,
            doc_id_set,
        })
    }

    pub fn add(&mut self, doc_id: u64, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dims {
            return Err(anyhow!(
                "embedding dimensions mismatch: expected {}, got {}",
                self.dims,
                embedding.len()
            ));
        }
        if !self.doc_id_set.insert(doc_id) {
            return Ok(());
        }
        let mut vec = embedding.to_vec();
        normalize(&mut vec);
        self.doc_ids.push(doc_id);
        self.vectors.extend_from_slice(&vec);
        Ok(())
    }

    pub fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<(u64, f32)>> {
        if embedding.len() != self.dims {
            return Err(anyhow!(
                "embedding dimensions mismatch: expected {}, got {}",
                self.dims,
                embedding.len()
            ));
        }
        if self.doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut query = embedding.to_vec();
        normalize(&mut query);

        let mut heap: Vec<(u64, f32)> = Vec::new();
        for (idx, doc_id) in self.doc_ids.iter().copied().enumerate() {
            let start = idx * self.dims;
            let end = start + self.dims;
            let vec = &self.vectors[start..end];
            let dot = dot_product(&query, vec);
            let distance = 1.0 - dot;
            if heap.len() < limit {
                heap.push((doc_id, distance));
                if heap.len() == limit {
                    heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                }
            } else if let Some(top) = heap.first_mut() {
                if distance < top.1 {
                    *top = (doc_id, distance);
                    heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                }
            }
        }

        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(heap)
    }

    pub fn save(&self) -> Result<()> {
        let vectors_path = self.path.join("vectors.f32");
        let ids_path = self.path.join("doc_ids.u64");
        let tmp_vectors = self.path.join("vectors.f32.tmp");
        let tmp_ids = self.path.join("doc_ids.u64.tmp");
        fs::write(&tmp_vectors, f32_to_bytes(&self.vectors))?;
        fs::write(&tmp_ids, u64_to_bytes(&self.doc_ids))?;
        fs::rename(&tmp_vectors, &vectors_path)?;
        fs::rename(&tmp_ids, &ids_path)?;
        Ok(())
    }

    pub fn contains(&self, doc_id: u64) -> bool {
        self.doc_id_set.contains(&doc_id)
    }

    #[allow(dead_code)]
    pub fn dimensions(&self) -> usize {
        self.dims
    }
}

fn normalize(vec: &mut [f32]) {
    let mut sum = 0.0f32;
    for v in vec.iter() {
        sum += v * v;
    }
    if sum <= 0.0 {
        return;
    }
    let inv = sum.sqrt().recip();
    for v in vec.iter_mut() {
        *v *= inv;
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

fn bytes_to_u64(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn u64_to_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
