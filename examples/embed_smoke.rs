use anyhow::Result;
use model2vec_rs::model::StaticModel;

fn main() -> Result<()> {
    let model = StaticModel::from_pretrained("minishlab/potion-base-8M", None, None, None)?;
    let input = vec![
        "hello world".to_string(),
        "small embedding smoke test".to_string(),
    ];
    let embeddings = model.encode_with_args(&input, Some(512), 64);
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
