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
```

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
