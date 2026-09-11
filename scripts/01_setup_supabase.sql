-- ═══════════════════ EXTENSION ═══════════════════
CREATE EXTENSION IF NOT EXISTS vector;

-- ═══════════════════ TABLES ═══════════════════
-- Vector search table (college profiles + performance summaries)
CREATE TABLE documents (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  embedding VECTOR(1024),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Branch details (SQL filtering: branch_code, NBA, intake)
CREATE TABLE branches (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tnea_code TEXT NOT NULL,
  sl_no INT,
  branch_code TEXT NOT NULL,
  approved_intake INT,
  year_of_starting FLOAT,
  nba_accredited TEXT,
  accreditation_valid_upto TEXT,
  approval_note TEXT
);

-- Performance data (SQL sorting: pass_percentage)
CREATE TABLE performance (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tnea_code TEXT NOT NULL,
  college_name TEXT,
  district TEXT,
  total_appeared INT,
  total_passed INT,
  pass_percentage FLOAT
);

-- Semantic cache
CREATE TABLE query_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  question TEXT NOT NULL,
  embedding VECTOR(1024),
  answer TEXT NOT NULL,
  sources JSONB DEFAULT '[]',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- ═══════════════════ VECTOR SEARCH FUNCTION ═══════════════════
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(1024),
  filter JSONB DEFAULT '{}',
  match_count INT DEFAULT 20,
  allowed_tnea_codes TEXT[] DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE
    (filter = '{}'::jsonb OR d.metadata @> filter)
    AND (allowed_tnea_codes IS NULL OR (d.metadata->>'tnea_code') = ANY(allowed_tnea_codes))
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- ═══════════════════ CACHE SEARCH FUNCTION ═══════════════════
CREATE OR REPLACE FUNCTION match_cache(
  query_embedding VECTOR(1024),
  match_threshold FLOAT DEFAULT 0.92,
  match_count INT DEFAULT 1
)
RETURNS TABLE (
  id UUID,
  question TEXT,
  answer TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    q.id,
    q.question,
    q.answer,
    1 - (q.embedding <=> query_embedding) AS similarity
  FROM query_cache q
  WHERE 1 - (q.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- ═══════════════════ INDEXES ═══════════════════
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);
CREATE INDEX ON documents USING gin (metadata);
CREATE INDEX ON branches (tnea_code);
CREATE INDEX ON branches (branch_code);
CREATE INDEX ON performance (tnea_code);
CREATE INDEX ON performance (pass_percentage);
CREATE INDEX ON query_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5);

-- ═══════════════════ PERMISSIONS (fixes 403) ═══════════════════
GRANT ALL PRIVILEGES ON TABLE documents TO service_role, anon, authenticated;
GRANT ALL PRIVILEGES ON TABLE branches TO service_role, anon, authenticated;
GRANT ALL PRIVILEGES ON TABLE performance TO service_role, anon, authenticated;
GRANT ALL PRIVILEGES ON TABLE query_cache TO service_role, anon, authenticated;

GRANT ALL PRIVILEGES ON SEQUENCE documents_id_seq TO service_role, anon, authenticated;
GRANT ALL PRIVILEGES ON SEQUENCE branches_id_seq TO service_role, anon, authenticated;
GRANT ALL PRIVILEGES ON SEQUENCE performance_id_seq TO service_role, anon, authenticated;

GRANT EXECUTE ON FUNCTION match_documents TO service_role, anon, authenticated;
GRANT EXECUTE ON FUNCTION match_cache TO service_role, anon, authenticated;

NOTIFY pgrst, 'reload schema';

SELECT 'Setup complete!' AS status;