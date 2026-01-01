# automem

Fast local history search for Claude and Codex logs.

## Build

```
cargo build
```

Binary:
```
./target/debug/automem
```

## Quickstart

Index (incremental):
```
./target/debug/automem index
```

Search (JSONL default):
```
./target/debug/automem search "your query" --limit 20
```

Notes:
- Embeddings are enabled by default.
- Searches run an incremental reindex by default (configurable).

Full transcript:
```
./target/debug/automem session <session_id>
```

Single record:
```
./target/debug/automem show <doc_id>
```

Human output:
```
./target/debug/automem search "your query" -v
```

## Search modes

| Need | Command |
| --- | --- |
| Exact terms | `search "exact term"` |
| Fuzzy concepts | `search "concept" --semantic` |
| Mixed | `search "term concept" --hybrid` |

## Common filters

- `--project <name>`
- `--role <user|assistant|tool_use|tool_result>`
- `--tool <tool_name>`
- `--session <session_id>`
- `--source claude|codex`
- `--since <iso|unix>` / `--until <iso|unix>`
- `--limit <n>`
- `--min-score <float>`
- `--sort score|ts`
- `--top-n-per-session <n>`
- `--unique-session`
- `--fields score,ts,doc_id,session_id,snippet`
- `--json-array`

## Embeddings

Disable:
```
./target/debug/automem index --no-embeddings
```

## Config (optional)

Create `~/.automem/config.toml` (or `<root>/config.toml` if you use `--root`):

```toml
embeddings = true
auto_index_on_search = true
```

SKILL.md is included as an example.
