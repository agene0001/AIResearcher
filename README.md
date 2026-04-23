# AIResearcher

A zero-cost, local-first MCP  server for academic paper research. Search, analyze, and synthesize research papers from 6 providers using free local AI models.

## Features

- **Multi-provider search** — arXiv, Semantic Scholar, PubMed, OpenAlex, Crossref, DBLP searched in parallel
- **Query reformulation** — natural language questions automatically converted to keyword variants for better retrieval
- **Hybrid search** — semantic (pgvector embeddings) + keyword (BM25) at 60/40 weighting
- **Semantic reranking** — results reranked by embedding similarity using harrier-oss-v1-0.6b
- **Local embeddings** — harrier-oss-v1-0.6b via Candle (44ms/embedding on Apple Silicon Metal, similar on CUDA)
- **Local LLMs** — qwen3.5:9b via Ollama, no API keys required
- **Trend detection** — cluster recent papers by topic and synthesize emerging trends
- **Benchmark tool** — 30-query evaluation suite to measure search quality

## Requirements

- Rust 1.75+
- PostgreSQL with pgvector extension
- [Ollama](https://ollama.com) with `qwen3.5:9b` pulled

## Setup

1. **Install dependencies**
   ```bash
   brew install postgresql@17 pgvector
   brew services start postgresql@17
   ollama pull qwen3.5:9b
   ```

2. **Set up the database**
   ```bash
   createdb autoresearch
   psql autoresearch -c "CREATE EXTENSION vector;"
   psql autoresearch -f migrations/001_init.sql
   psql autoresearch -f migrations/002_multi_provider.sql
   psql autoresearch -f migrations/003_harrier_embeddings.sql
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Build and run**
   ```bash
   # macOS (Metal GPU)
   cargo run --features metal -- search "transformer attention mechanisms"

   # Windows/Linux (CUDA GPU)
   cargo run --features cuda -- search "transformer attention mechanisms"
   ```

## Commands

```bash
# Search for papers
cargo run --features metal -- search "your query"

# Detect trends in a research area
cargo run --features metal -- trends "large language models"

# Run benchmark evaluation
cargo run --features metal -- benchmark

# Test embedding speed
cargo run --features metal -- test-embedding

# Ingest OpenAlex data (see Bulk Ingestion section)
cargo run --features cuda -- ingest --source snapshot --snapshot-dir ./openalex-snapshot --batch-size 128
cargo run --features cuda -- ingest --source api --batch-size 128
```

## Bulk Ingestion (OpenAlex)

To build a local index of ~2-3M CS/AI/ML papers for high-quality hybrid search. Two methods available:

### Option A: Snapshot (faster, needs ~250GB disk)

1. **Install AWS CLI** (no account needed):
   ```bash
   # Windows
   winget install Amazon.AWSCLI

   # macOS
   brew install awscli
   ```

2. **Download the works snapshot** (~250GB compressed):
   ```bash
   aws s3 sync "s3://openalex/data/works" "openalex-snapshot/data/works" --no-sign-request
   ```

3. **Run ingestion** (filters to papers with abstracts, 2015+, in the selected field/subfield — defaults to the whole Computer Science field; see "Choosing topics" below to narrow):
   ```bash
   # CUDA (recommended, ~8-10 hours for 2M papers)
   cargo run --features cuda -- ingest --source snapshot --snapshot-dir ./openalex-snapshot --batch-size 128

   # Metal
   cargo run --features metal -- ingest --source snapshot --snapshot-dir ./openalex-snapshot --batch-size 128
   ```

4. **Delete the snapshot** after ingestion to reclaim ~250GB:
   ```bash
   rm -rf openalex-snapshot
   ```

### Option B: API (no download, rate-limited)

No bulk download needed — fetches papers directly from the OpenAlex API. Default filter is the whole **Computer Science** field (AI + ML + CV + NLP + HCI + signal processing + ...):

```bash
cargo run --features cuda -- ingest --source api --batch-size 128
```

OpenAlex caps the free tier at **100,000 requests/day** (≈ 20M papers/day at 200 papers/page), so the daily limit is not a concern at this scale. Set `OPENALEX_EMAIL` in `.env` to enter the polite pool (priority queue, lower latency, same daily cap):

```env
OPENALEX_EMAIL=you@example.com
```

No API key is required for the free tier — `mailto` is the only authentication that matters.

**Resumable reruns.** If you Ctrl-C mid-ingest or hit an error, just rerun the same command. The pipeline checks Postgres for already-ingested IDs before embedding, so previously-completed pages fly past in milliseconds; any batches that errored previously (e.g. DOI conflicts) are re-attempted and land cleanly thanks to widened `ON CONFLICT` handling.

### Choosing topics: --field and --subfield

OpenAlex organizes works into 26 **fields** and 252 **subfields**. Ingest supports both granularities by **name** (not numeric IDs — names are resolved against OpenAlex's live taxonomy at startup). The flags apply to **both** `--source api` and `--source snapshot`.

```bash
# Coarse: ingest an entire field (mutually exclusive with --subfield)
cargo run --features cuda -- ingest --field "Medicine"

# Fine: one or more subfields (repeat the flag, OR'd together)
cargo run --features cuda -- ingest \
    --subfield "Artificial Intelligence" \
    --subfield "Computer Vision and Pattern Recognition"

# Sports analytics (note the subfield name contains commas — that's fine, each --subfield takes one quoted name):
cargo run --features cuda -- ingest \
    --subfield "Physical Therapy, Sports Therapy and Rehabilitation"
```

Names are matched case-insensitively. On a typo or miss, the resolver prints `Did you mean: ...` suggestions. Browse the full taxonomy at:
- https://api.openalex.org/fields
- https://api.openalex.org/subfields

If neither flag is given, the default is `--field "Computer Science"` (~5M+ works 2015+ with abstracts across AI, ML, CV, NLP, HCI, DB, systems, theory, etc.).

**Seeing the real count.** On `--source api`, the first response from OpenAlex includes a `meta.count` of how many works actually match your filter — this is printed once at startup and used as the progress-bar target. (Previously the bar showed a hardcoded `3M`, which was just an arbitrary ceiling.)

### Common options

```bash
--min-year 2015                    # Only include papers from this year onward (default: 2015)
--batch-size 128                   # Papers per GPU batch (reduce to 64 if OOM)
--max-papers 10000                 # Limit total papers (useful for testing)
--field "Computer Science"         # Coarse topic filter (one field, API only)
--subfield "Artificial Intelligence"  # Fine topic filter (repeatable, API only)
```

### Logs

Each run writes structured logs to `logs/autoresearch.log.YYYY-MM-DD` (rotates daily, gitignored). The console shows the progress bar; the log file captures retries, rate-limit waits, batch errors, and a summary on completion. Override the level with `RUST_LOG=debug` for per-batch traces.

**Resource requirements:**
- **Snapshot:** ~250GB temporary disk + ~25-30GB Postgres (permanent)
- **API:** ~25-30GB Postgres only
- GPU with 12GB+ VRAM recommended (batch-size 128)
- If you get OOM errors, reduce `--batch-size` to 64

## Configuration

Create a `.env` file:

```env
DATABASE_URL=postgresql://localhost/autoresearch

# LLM settings (defaults to local Ollama)
CHEAP_MODEL=qwen3.5:9b
STRONG_MODEL=qwen3.5:9b
OLLAMA_HOST=http://localhost:11434

# Optional: use cloud LLM for scoring (e.g. Gemini paid tier)
# STRONG_MODEL_HOST=https://generativelanguage.googleapis.com/v1beta/openai
# STRONG_MODEL=gemini-2.5-flash
# OPENAI_API_KEY=your_key_here

# Optional: higher Semantic Scholar rate limits
# SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

## MCP Server

Run as an MCP server for Claude Desktop or Cursor:

```bash
cargo run --features metal -- mcp
```

Add to your MCP config:
```json
{
  "mcpServers": {
    "airesearcher": {
      "command": "/path/to/target/release/autoresearch-lab",
      "args": ["mcp"]
    }
  }
}
```

## Architecture

- **Retrieval** — parallel fanout to 6 APIs, deduplication by DOI and normalized title
- **Embeddings** — harrier-oss-v1-0.6b (0.6B params, 1024 dims, 70.75 MTEB score) via Candle
- **Storage** — PostgreSQL + pgvector for hybrid search, HNSW index
- **LLMs** — any Ollama model or OpenAI-compatible endpoint
