PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investigations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS investigation_roots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  root_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, root_path)
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  normalized_path TEXT NOT NULL,
  stable_id TEXT,
  size INTEGER NOT NULL DEFAULT 0,
  modified_ns INTEGER NOT NULL DEFAULT 0,
  content_hash TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  source_type TEXT NOT NULL DEFAULT '',
  parent_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
  parent_attachment_name TEXT,
  extractor_version TEXT,
  extraction_status TEXT NOT NULL DEFAULT 'pending',
  extraction_error TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, normalized_path)
);

CREATE INDEX IF NOT EXISTS idx_files_investigation_status
  ON files(investigation_id, status);
CREATE INDEX IF NOT EXISTS idx_files_stable_id
  ON files(investigation_id, stable_id);
CREATE INDEX IF NOT EXISTS idx_files_content_hash
  ON files(investigation_id, content_hash);

CREATE TABLE IF NOT EXISTS scan_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  message TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS extraction_cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_hash TEXT NOT NULL,
  extractor_version TEXT NOT NULL,
  source_type TEXT NOT NULL,
  extracted_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(content_hash, extractor_version, source_type)
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  content_hash TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  display_markdown TEXT,
  chunk_text_hash TEXT NOT NULL,
  source_type TEXT NOT NULL,
  page_number INTEGER,
  email_subject TEXT,
  email_sender TEXT,
  email_recipients TEXT,
  email_date TEXT,
  attachment_name TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  path TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1,
  extractor_version TEXT NOT NULL,
  chunker_version TEXT NOT NULL,
  embedding_model TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_scope_active
  ON chunks(investigation_id, is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_file_active
  ON chunks(file_id, is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_text_hash
  ON chunks(chunk_text_hash, embedding_model);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  investigation_id UNINDEXED,
  chunk_text,
  path UNINDEXED
);

CREATE TABLE IF NOT EXISTS embedding_cache (
  chunk_text_hash TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(chunk_text_hash, embedding_model)
);

CREATE TABLE IF NOT EXISTS disappeared_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  review_notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, file_id)
);

CREATE INDEX IF NOT EXISTS idx_disappeared_pending
  ON disappeared_files(investigation_id, status);
