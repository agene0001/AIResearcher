-- Add new columns for multi-provider support
ALTER TABLE papers ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'arxiv';
ALTER TABLE papers ADD COLUMN IF NOT EXISTS year INTEGER;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS doi TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS url TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS authors JSONB NOT NULL DEFAULT '[]';

-- Index for DOI lookups (dedup)
CREATE UNIQUE INDEX IF NOT EXISTS papers_doi_idx ON papers (doi) WHERE doi IS NOT NULL;

-- Index for year-based filtering
CREATE INDEX IF NOT EXISTS papers_year_idx ON papers (year DESC) WHERE year IS NOT NULL;
