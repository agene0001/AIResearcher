-- Switch embedding dimensions from 1536 (OpenAI text-embedding-3-small) to 1024 (harrier-oss-v1-0.6b)
-- This requires dropping and recreating the embedding column and index.

DROP INDEX IF EXISTS papers_embedding_idx;
ALTER TABLE papers DROP COLUMN IF EXISTS embedding;
ALTER TABLE papers ADD COLUMN embedding vector(1024);
CREATE INDEX IF NOT EXISTS papers_embedding_idx ON papers USING hnsw (embedding vector_cosine_ops);
