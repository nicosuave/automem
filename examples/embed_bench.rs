use anyhow::Result;
use memex::embed::{EmbedderHandle, ModelChoice};
use std::time::Instant;

fn generate_texts(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "This is test sentence number {} for embedding benchmarks. \
                 We add some extra text here to make the sentences more realistic \
                 and representative of actual embedding workloads.",
                i
            )
        })
        .collect()
}

fn main() -> Result<()> {
    println!("Embedding Benchmark - Testing New Performance Options");
    println!("======================================================");
    println!("CPU cores: {}", std::thread::available_parallelism()?.get());

    let texts = generate_texts(500);
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Test 1: Baseline (single embedder, no special options)
    println!("\n--- Test 1: Baseline (default settings) ---");
    {
        let start = Instant::now();
        let mut embedder = EmbedderHandle::with_model(ModelChoice::MiniLM)?;
        println!("  Model init: {}ms", start.elapsed().as_millis());

        // Warmup
        let _ = embedder.embed_texts(&["warmup"])?;

        let start = Instant::now();
        let results = embedder.embed_texts(&text_refs)?;
        let elapsed = start.elapsed();
        println!(
            "  500 texts: {}ms ({:.0} texts/sec) - {} embeddings",
            elapsed.as_millis(),
            500.0 / elapsed.as_secs_f64(),
            results.len()
        );
    }

    // Test 2: Different compute units
    for unit in ["all", "ane", "gpu", "cpu"] {
        println!("\n--- Test 2: MEMEX_COMPUTE_UNITS={} ---", unit);
        unsafe {
            std::env::set_var("MEMEX_COMPUTE_UNITS", unit);
        }

        let start = Instant::now();
        let mut embedder = match EmbedderHandle::with_model(ModelChoice::MiniLM) {
            Ok(e) => e,
            Err(e) => {
                println!("  Failed to init: {}", e);
                continue;
            }
        };
        println!("  Model init: {}ms", start.elapsed().as_millis());

        let _ = embedder.embed_texts(&["warmup"])?;

        let start = Instant::now();
        let results = embedder.embed_texts(&text_refs)?;
        let elapsed = start.elapsed();
        println!(
            "  500 texts: {}ms ({:.0} texts/sec)",
            elapsed.as_millis(),
            500.0 / elapsed.as_secs_f64()
        );
        let _ = results;
    }

    // Test 3: Parallel embedding with multiple model instances
    println!("\n--- Test 3: Parallel embedding (multiple model instances) ---");
    for num_threads in [2, 4, 8] {
        let texts_owned: Vec<String> = texts.iter().cloned().collect();

        let start = Instant::now();
        let results = EmbedderHandle::embed_texts_parallel(
            texts_owned,
            ModelChoice::MiniLM,
            num_threads,
        )?;
        let elapsed = start.elapsed();
        println!(
            "  {} threads: {}ms ({:.0} texts/sec) - {} embeddings",
            num_threads,
            elapsed.as_millis(),
            500.0 / elapsed.as_secs_f64(),
            results.len()
        );
    }

    // Test 4: Larger batch with Gemma model
    println!("\n--- Test 4: Gemma model (larger, higher quality) ---");
    unsafe {
        std::env::set_var("MEMEX_COMPUTE_UNITS", "all");
    }
    {
        let start = Instant::now();
        let mut embedder = EmbedderHandle::with_model(ModelChoice::Gemma)?;
        println!("  Model init: {}ms", start.elapsed().as_millis());

        let _ = embedder.embed_texts(&["warmup"])?;

        let start = Instant::now();
        let results = embedder.embed_texts(&text_refs)?;
        let elapsed = start.elapsed();
        println!(
            "  500 texts: {}ms ({:.0} texts/sec)",
            elapsed.as_millis(),
            500.0 / elapsed.as_secs_f64()
        );
        let _ = results;
    }

    // Test 5: Gemma parallel
    println!("\n--- Test 5: Gemma parallel (4 threads) ---");
    {
        let texts_owned: Vec<String> = texts.iter().cloned().collect();
        let start = Instant::now();
        let results = EmbedderHandle::embed_texts_parallel(
            texts_owned,
            ModelChoice::Gemma,
            4,
        )?;
        let elapsed = start.elapsed();
        println!(
            "  4 threads: {}ms ({:.0} texts/sec) - {} embeddings",
            elapsed.as_millis(),
            500.0 / elapsed.as_secs_f64(),
            results.len()
        );
    }

    println!("\nDone!");
    Ok(())
}
