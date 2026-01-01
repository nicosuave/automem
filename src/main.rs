mod cli;
mod config;
mod embed;
mod index;
mod ingest;
mod progress;
mod state;
mod types;
mod vector;

fn main() -> anyhow::Result<()> {
    cli::run()
}
