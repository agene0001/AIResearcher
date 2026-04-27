-- Tier 2: cached full-text extractions of papers (PDF -> markdown/plain text).
-- Populated on demand by the deep_read pipeline; one row per paper that has been
-- fetched. Either `markdown` or `error` is set (never both null after a fetch).
CREATE TABLE IF NOT EXISTS paper_full_text (
    paper_id   TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    markdown   TEXT,
    parser     TEXT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error      TEXT
);

CREATE INDEX IF NOT EXISTS paper_full_text_fetched_at_idx
    ON paper_full_text (fetched_at);
