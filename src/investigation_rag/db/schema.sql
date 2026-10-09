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
  active_chunk_count INTEGER NOT NULL DEFAULT 0,
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

-- Current per-file ingest/index state.  A row is created by the scan pipeline
-- only after it starts processing a file.  In particular, initialization does
-- not infer or backfill state for databases created by older application
-- versions: the absence of a row means that the file has not been processed
-- by the current staged pipeline.
CREATE TABLE IF NOT EXISTS file_pipeline_state (
  file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
  source_content_hash TEXT NOT NULL DEFAULT '',
  extraction_signature TEXT NOT NULL DEFAULT '',
  chunking_signature TEXT NOT NULL DEFAULT '',
  metadata_signature TEXT NOT NULL DEFAULT '',
  embedding_input_signature TEXT NOT NULL DEFAULT '',
  embedding_model TEXT NOT NULL DEFAULT '',
  extraction_status TEXT NOT NULL DEFAULT 'pending',
  chunk_status TEXT NOT NULL DEFAULT 'pending',
  metadata_status TEXT NOT NULL DEFAULT 'pending',
  vector_status TEXT NOT NULL DEFAULT 'pending',
  active_vector_count INTEGER NOT NULL DEFAULT 0 CHECK(active_vector_count >= 0),
  updated_at TEXT NOT NULL
);

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
  fts_rowid INTEGER PRIMARY KEY,
  id TEXT NOT NULL UNIQUE,
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
  chunk_text,
  path
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

CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  merged_into_id INTEGER REFERENCES entities(id) ON DELETE RESTRICT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(investigation_id, normalized_name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_active
  ON entities(investigation_id, merged_into_id, entity_type, name);

CREATE TABLE IF NOT EXISTS entity_mentions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  mention_text TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(chunk_id, start_offset, end_offset, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity
  ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_file
  ON entity_mentions(file_id);

CREATE TABLE IF NOT EXISTS entity_extraction_cache (
  document_text_hash TEXT NOT NULL,
  extraction_signature TEXT NOT NULL,
  model_name TEXT NOT NULL,
  raw_output TEXT NOT NULL,
  parsed_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(document_text_hash, extraction_signature)
);

CREATE TABLE IF NOT EXISTS entity_graph_views (
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  state_json TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(investigation_id, name)
);

CREATE TABLE IF NOT EXISTS entity_graph_edits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Independent knowledge graph: never consumes entities or fact-ledger output.
CREATE TABLE IF NOT EXISTS kg_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  settings_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running',
  started_at TEXT NOT NULL,
  finished_at TEXT,
  summary_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS kg_nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  node_type TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Exact names are resolved within one source, never automatically across documents.
CREATE TABLE IF NOT EXISTS kg_aliases (
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  source_key TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  node_type TEXT NOT NULL,
  node_id INTEGER NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
  PRIMARY KEY(investigation_id, source_key, normalized_name, node_type)
);

CREATE TABLE IF NOT EXISTS kg_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  claim_key TEXT NOT NULL,
  subject_id INTEGER NOT NULL REFERENCES kg_nodes(id),
  object_id INTEGER NOT NULL REFERENCES kg_nodes(id),
  predicate TEXT NOT NULL,
  polarity TEXT NOT NULL CHECK(polarity IN ('affirmed', 'negated', 'uncertain')),
  attribution TEXT NOT NULL DEFAULT '',
  time_text TEXT NOT NULL DEFAULT '',
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  quote TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  model TEXT NOT NULL,
  verification TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'suggested' CHECK(review_status IN ('suggested', 'accepted', 'rejected')),
  review_note TEXT NOT NULL DEFAULT '',
  run_id INTEGER NOT NULL REFERENCES kg_runs(id),
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, claim_key)
);
CREATE INDEX IF NOT EXISTS idx_kg_claims_scope ON kg_claims(investigation_id, chunk_id);
CREATE INDEX IF NOT EXISTS idx_kg_claims_subject ON kg_claims(subject_id);
CREATE INDEX IF NOT EXISTS idx_kg_claims_object ON kg_claims(object_id);

CREATE TABLE IF NOT EXISTS kg_claim_attributes (
  claim_id INTEGER PRIMARY KEY REFERENCES kg_claims(id) ON DELETE CASCADE,
  attribute_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_batches (
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  batch_key TEXT NOT NULL,
  run_id INTEGER NOT NULL REFERENCES kg_runs(id),
  extraction_output TEXT NOT NULL,
  verification_output TEXT NOT NULL,
  diagnostics_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(investigation_id, batch_key)
);

-- Independently reusable extraction, verification and bounded recovery plans.
CREATE TABLE IF NOT EXISTS kg_stage_cache (
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  stage TEXT NOT NULL CHECK(stage IN ('extraction', 'verification')),
  cache_key TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(investigation_id, stage, cache_key)
);

-- Imported quotes stay portable even when their indexed sources are unavailable.
-- They never create synthetic files/chunks or extraction cache entries.
CREATE TABLE IF NOT EXISTS kg_import_nodes (
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  portable_key TEXT NOT NULL,
  node_id INTEGER NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
  PRIMARY KEY(investigation_id, portable_key)
);

CREATE TABLE IF NOT EXISTS kg_imported_claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id INTEGER NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
  portable_key TEXT NOT NULL,
  claim_key TEXT NOT NULL,
  subject_id INTEGER NOT NULL REFERENCES kg_nodes(id),
  object_id INTEGER NOT NULL REFERENCES kg_nodes(id),
  predicate TEXT NOT NULL,
  polarity TEXT NOT NULL CHECK(polarity IN ('affirmed', 'negated', 'uncertain')),
  attribution TEXT NOT NULL DEFAULT '',
  time_text TEXT NOT NULL DEFAULT '',
  attribute TEXT NOT NULL DEFAULT '',
  chunk_id TEXT NOT NULL,
  linked_chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
  native_claim_id INTEGER REFERENCES kg_claims(id) ON DELETE SET NULL,
  path TEXT NOT NULL,
  chunk_index INTEGER NOT NULL DEFAULT 0,
  page_number INTEGER,
  attachment_name TEXT,
  quote TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  exported_source_current INTEGER NOT NULL DEFAULT 0,
  model TEXT NOT NULL DEFAULT '',
  verification TEXT NOT NULL,
  review_status TEXT NOT NULL CHECK(review_status IN ('suggested', 'accepted', 'rejected')),
  review_note TEXT NOT NULL DEFAULT '',
  original_run_id INTEGER,
  created_at TEXT NOT NULL,
  UNIQUE(investigation_id, portable_key)
);
CREATE INDEX IF NOT EXISTS idx_kg_imported_subject ON kg_imported_claims(subject_id);
CREATE INDEX IF NOT EXISTS idx_kg_imported_object ON kg_imported_claims(object_id);
