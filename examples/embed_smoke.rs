use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

fn main() -> Result<()> {
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::EmbeddingGemma300M).with_show_download_progress(true),
    )?;
    let input = vec!["hello world", "small embedding smoke test"];
    let embeddings = model.embed(input, None)?;
    if embeddings.is_empty() {
        anyhow::bail!("no embeddings returned");
    }
    let dims = embeddings[0].len();
    println!("embeddings: {} vectors, dims {}", embeddings.len(), dims);
    if let Some(first) = embeddings.first() {
        let preview: Vec<String> = first.iter().take(8).map(|v| format!("{v:.4}")).collect();
        println!("first: [{}]", preview.join(", "));
    }
    Ok(())
}
