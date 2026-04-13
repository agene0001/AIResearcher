use anyhow::Result;
use pgvector::Vector;
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, Row};
use std::env;
use crate::models::paper::{Paper, PaperSource};

#[derive(Clone)]
pub struct DbClient {
    pub pool: Pool<Postgres>,
}

impl DbClient {
    pub async fn new() -> Result<Self> {
        let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/autoresearch".to_string());

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;

        // Run migrations automatically
        sqlx::migrate!("./migrations").run(&pool).await?;

        Ok(Self { pool })
    }

    /// Try to connect, returning None if the database is unavailable.
    pub async fn try_new() -> Option<Self> {
        match Self::new().await {
            Ok(client) => Some(client),
            Err(e) => {
                tracing::warn!("Database unavailable, running without persistence: {}", e);
                None
            }
        }
    }

    /// Insert a paper with its embedding into the database.
    pub async fn insert_paper(&self, paper: &Paper, embedding: Vec<f32>) -> Result<()> {
        let vector = Vector::from(embedding);
        let authors_json = serde_json::to_value(&paper.authors)?;

        sqlx::query(
            r#"
            INSERT INTO papers (id, title, abstract_text, content, source, year, doi, url, authors, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE
            SET title = EXCLUDED.title,
                abstract_text = EXCLUDED.abstract_text,
                content = EXCLUDED.content,
                source = EXCLUDED.source,
                year = EXCLUDED.year,
                doi = EXCLUDED.doi,
                url = EXCLUDED.url,
                authors = EXCLUDED.authors,
                embedding = EXCLUDED.embedding
            "#
        )
        .bind(&paper.id)
        .bind(&paper.title)
        .bind(&paper.abstract_text)
        .bind(&paper.content)
        .bind(paper.source.to_string())
        .bind(paper.year.map(|y| y as i32))
        .bind(&paper.doi)
        .bind(&paper.url)
        .bind(&authors_json)
        .bind(vector)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Hybrid Search: Combines Keyword Search (BM25) and Semantic Search (pgvector).
    /// Returns full Paper objects ranked by combined score.
    pub async fn hybrid_search(&self, query: &str, query_embedding: Vec<f32>, limit: i64) -> Result<Vec<Paper>> {
        let vector = Vector::from(query_embedding);

        let records = sqlx::query(
            r#"
            WITH semantic_search AS (
                SELECT id, title, abstract_text, content, source, year, doi, url, authors,
                       1 - (embedding <=> $1) AS semantic_score
                FROM papers
                ORDER BY embedding <=> $1
                LIMIT 50
            ),
            keyword_search AS (
                SELECT id, title, abstract_text, content, source, year, doi, url, authors,
                       ts_rank_cd(search_vector, plainto_tsquery('english', $2)) AS keyword_score
                FROM papers
                WHERE search_vector @@ plainto_tsquery('english', $2)
                ORDER BY keyword_score DESC
                LIMIT 50
            )
            SELECT
                COALESCE(s.id, k.id) as id,
                COALESCE(s.title, k.title) as title,
                COALESCE(s.abstract_text, k.abstract_text) as abstract_text,
                COALESCE(s.content, k.content) as content,
                COALESCE(s.source, k.source) as source,
                COALESCE(s.year, k.year) as year,
                COALESCE(s.doi, k.doi) as doi,
                COALESCE(s.url, k.url) as url,
                COALESCE(s.authors, k.authors) as authors,
                COALESCE(s.semantic_score, 0.0) as semantic_score,
                COALESCE(k.keyword_score, 0.0) as keyword_score,
                (COALESCE(s.semantic_score, 0.0) * 0.6 + COALESCE(k.keyword_score, 0.0) * 0.4) as final_score
            FROM semantic_search s
            FULL OUTER JOIN keyword_search k ON s.id = k.id
            ORDER BY final_score DESC
            LIMIT $3
            "#
        )
        .bind(vector)
        .bind(query)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut papers = Vec::new();
        for r in records {
            let source_str: String = r.try_get("source").unwrap_or_else(|_| "arxiv".to_string());
            let source = match source_str.as_str() {
                "semantic_scholar" => PaperSource::SemanticScholar,
                "pubmed" => PaperSource::PubMed,
                "openalex" => PaperSource::OpenAlex,
                "crossref" => PaperSource::Crossref,
                "dblp" => PaperSource::Dblp,
                _ => PaperSource::Arxiv,
            };
            let authors_json: serde_json::Value = r.try_get("authors").unwrap_or(serde_json::json!([]));
            let authors: Vec<String> = serde_json::from_value(authors_json).unwrap_or_default();

            papers.push(Paper {
                id: r.try_get("id").unwrap_or_default(),
                title: r.try_get("title").unwrap_or_default(),
                abstract_text: r.try_get("abstract_text").unwrap_or_default(),
                content: r.try_get("content").ok(),
                source,
                year: r.try_get::<Option<i32>, _>("year").ok().flatten().map(|y| y as u32),
                doi: r.try_get("doi").ok().flatten(),
                url: r.try_get("url").ok().flatten(),
                authors,
            });
        }

        Ok(papers)
    }

    /// Batch insert papers with embeddings. Much faster than individual inserts.
    pub async fn insert_papers_batch(&self, papers: &[Paper], embeddings: &[Vec<f32>]) -> Result<()> {
        if papers.is_empty() {
            return Ok(());
        }

        // Use a transaction for atomicity and speed
        let mut tx = self.pool.begin().await?;

        for (paper, embedding) in papers.iter().zip(embeddings.iter()) {
            let vector = Vector::from(embedding.clone());
            let authors_json = serde_json::to_value(&paper.authors)?;

            sqlx::query(
                r#"
                INSERT INTO papers (id, title, abstract_text, content, source, year, doi, url, authors, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO NOTHING
                "#
            )
            .bind(&paper.id)
            .bind(&paper.title)
            .bind(&paper.abstract_text)
            .bind(&paper.content)
            .bind(paper.source.to_string())
            .bind(paper.year.map(|y| y as i32))
            .bind(&paper.doi)
            .bind(&paper.url)
            .bind(&authors_json)
            .bind(vector)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Store papers from a retrieval run and generate embeddings.
    pub async fn index_papers(&self, papers: &[Paper], embed_fn: impl Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>>> + Send>>) -> Result<()> {
        for paper in papers {
            if paper.abstract_text.is_empty() {
                continue;
            }
            let text = format!("{} {}", paper.title, paper.abstract_text);
            match embed_fn(&text).await {
                Ok(embedding) => {
                    if let Err(e) = self.insert_paper(paper, embedding).await {
                        tracing::warn!("Failed to index paper {}: {}", paper.id, e);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to embed paper {}: {}", paper.id, e);
                }
            }
        }
        Ok(())
    }
}
