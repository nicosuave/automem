use anyhow::{Result, anyhow};
use model2vec_rs::model::StaticModel;

pub struct EmbedderHandle {
    model: StaticModel,
    pub dims: usize,
}

impl EmbedderHandle {
    pub fn new() -> Result<Self> {
        let model = StaticModel::from_pretrained("minishlab/potion-base-8M", None, None, None)?;
        let dims = model
            .encode(&[String::from("dimension_check")])
            .first()
            .map(|vec| vec.len())
            .ok_or_else(|| anyhow!("no embedding returned"))?;
        Ok(Self { model, dims })
    }

    pub fn embed_texts(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        Ok(self.model.encode_with_args(&input, Some(512), 64))
    }
}
