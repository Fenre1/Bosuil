# Local Investigation RAG

Windows-first, developer-run desktop app for indexing investigation folders and asking strict local RAG questions over `.txt`, `.docx`, `.pdf`, and `.msg` files. The UI uses PyQt5. Each investigation is an isolated workspace with its own SQLite and LanceDB stores.

No cloud APIs are used. Ollama is called on the local machine for embeddings and generation.

## Setup on Windows

```powershell
cd C:\Projects\Bosuil\local_investigation_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install and start Ollama separately, then pull the local models you want to use. Most modes use the configured or selected local models; Hyde retrieval prefers `Gemma4:e4b` for fake-document generation when that model is installed, then falls back to the selected answer model.

Create a `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
INVESTIGATION_RAG_DATA_DIR=C:\InvestigationRAGData
OLLAMA_BASE_URL=http://127.0.0.1:11434
INVESTIGATION_RAG_ALLOW_REMOTE_OLLAMA=0
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_NUM_CTX=32768
OLLAMA_EMBED_NUM_CTX=16384
OLLAMA_LLM_MODEL=<your-local-chat-model>
OLLAMA_EMBED_MODEL=<your-local-embedding-model>
OLLAMA_EMBED_QUERY_INSTRUCTION=Given a user question, retrieve relevant passages from a local document collection that answer the question.
```

By default data is stored under `%LOCALAPPDATA%\InvestigationRAG`. Real environment variables still take priority over values in `.env`, so temporary PowerShell overrides continue to work. If you keep the file somewhere else, set `INVESTIGATION_RAG_ENV_FILE` to that path before running the app.

`INVESTIGATION_RAG_DATA_DIR` contains a small `catalog.sqlite3` plus UUID-named workspaces under `investigations\`. Each workspace contains `investigation.sqlite3`, `lancedb\`, and `manifest.json`. Copying a complete workspace directory into `investigations\` makes it discoverable on the next application start; source evidence is not copied.

Evidence roots may live on SMB shares or private-cloud mounted folders. Keep the active `INVESTIGATION_RAG_DATA_DIR` on a local disk: SQLite WAL and live LanceDB files are not safe or reliably fast on many network or sync-managed folders. If a workspace must be archived or copied to private cloud, close the app first and copy the complete workspace as one unit.

The app rejects non-loopback Ollama URLs by default and the Ollama client ignores proxy environment variables. Keep `INVESTIGATION_RAG_ALLOW_REMOTE_OLLAMA=0` for sensitive data. Only set it to `1` for a trusted non-sensitive setup where the Ollama server is intentionally hosted away from the same machine.

For slower local models or long evidence-ledger RAG runs, raise `OLLAMA_TIMEOUT_SECONDS` in `.env`; for example, `600` waits up to ten minutes for each local Ollama request.

Set `OLLAMA_NUM_CTX` for answering-model requests and `OLLAMA_EMBED_NUM_CTX` for embedding requests. Set either to `0` or leave it unset to use the Ollama/model default. A 16k embedding context is ample for the default chunk sizes while using less VRAM than the model's 32k default.

## Run the Desktop App

```powershell
python -m investigation_rag.app
```

or, after editable install:

```powershell
investigation-rag-gui
```

The GUI includes an investigation selector, create/edit dialog, root folder management, manual scan, scan log, keyword/semantic/hybrid search, result preview, Windows file/folder opening, contextual conversational strict RAG questions with PDF export, and disappeared-file review.

## Performance diagnostics

Workspace-open timings are written to the console and appended to the scan log. A completed,
stopped, or failed scan also writes a timing breakdown, embedding-cache hit/miss counts, and
the ten slowest files to the scan log. To diagnose a slow collection, open the investigation,
run one scan, and copy the lines beginning with `Workspace load timing`, `Timing breakdown`,
`Embedding cache`, and `Slowest files`.

## CLI

```powershell
investigation-rag init
investigation-rag create-investigation --name "Case A" --root "D:\Cases\CaseA" --notes "Initial import"
investigation-rag list-investigations
investigation-rag scan-investigation --name "Case A"
investigation-rag scan-investigation --name "Case A" --scan-mode verify
investigation-rag scan-investigation --name "Case A" --scan-mode rebuild
investigation-rag search-investigation --name "Case A" --mode keyword "invoice"
investigation-rag search-investigation --name "Case A" --mode semantic "payments after May"
investigation-rag ask --name "Case A" "What evidence mentions payments after May?"
investigation-rag ask --name "Case A" --retrieval-profile hyde "What evidence mentions payments after May?"
```

If Ollama or LanceDB is unavailable, keyword search and non-embedding scan functions still work. Embedding and RAG operations report a clear local configuration/runtime error.

## Ingest Model

Scanning is manual and has three modes:

- **Fast rescan** (the default) builds a metadata-only directory inventory, loads the complete SQLite manifest once, and compares path, stable identity, size, modification time, content hash, and stage signatures in memory. Files whose metadata and pipeline state are current are not hashed, extracted, written to SQLite, or embedded.
- **Verify content** hashes every supported file, but rebuilds only content or processing stages that are actually outdated.
- **Rebuild index** hashes every supported file, recreates current extraction, chunks, metadata/FTS, and embeddings, and reconciles stale vector records. Content-addressed extraction and embedding caches may still satisfy unchanged work.

The portable inventory scanner uses bounded concurrent `os.scandir` calls and does not require an NTFS journal, USN access, administrator rights, or a local disk. It skips directory links/junctions and remains conservative when a cloud provider reports an incomplete or unreadable subtree: disappeared-file detection is skipped for that scan. `SCAN_DISCOVERY_WORKERS` and `SCAN_HASH_WORKERS` tune bounded concurrency; the defaults of 4 and 2 are intentionally modest for private-cloud filesystems. `SCAN_VERBOSE_FILE_LIMIT` controls when per-file scan messages are replaced by throttled aggregate progress.

Fast mode deliberately trusts unchanged size, modification time, and identity metadata. Use Verify when a provider may preserve timestamps across content changes; use Rebuild when both content and every derived index stage should be checked and recreated.

SQLite writes use one scan-scoped connection, bulk manifest/diff operations, and batched chunk/FTS updates. LanceDB table handles are reused, path-only moves do not trigger re-embedding, and expensive vector optimization is no longer part of an interactive scan. Replaced chunks are marked inactive before new chunks are inserted; disappeared files are hidden from vector search while awaiting review.

Investigations ingested before the staged scan-state schema are deliberately not backfilled or inferred as current. Recreate those few investigations and ingest them as new instead of relying on a legacy upgrade path.

File identity uses the available Python stat device/inode identity on Windows where practical, then falls back to path, size, modified time, and finally content hash as the truth for content changes. This keeps network-share behavior conservative.

Disappeared files are queued for review instead of deleted. The GUI can mark them inactive, keep them active, or update their path.

## Extraction and Chunking

Supported extractors:

- `.txt`: robust encoding fallback.
- `.docx`: `python-docx`.
- `.pdf`: PyMuPDF selectable text, page-aware.
- `.msg`: `extract-msg`, including subject, sender, recipients, date, body, and supported attachments. Nested messages are depth-limited.

Chunking is deterministic and structure-aware (`structured-v2`, `pdf-layout-v5-source`). Extracted paragraphs, headings, list items and table rows retain source locations. Chunks keep related blocks together and split oversized passages at sentence or word boundaries. PDFs retain page coordinates, use column-aware reading order, and suppress recurring page furniture. Font-span counts are not treated as evidence of a table.

The stored passage contains source text. Document metadata, section headings, table-header context and bounded overlap are composed separately for embeddings. The default target is 2,800 characters; the complete embedding input is limited to 4,800 characters and an estimated 1,400 tokens. `RAG_CHUNK_MAX_TOKENS` uses a conservative UTF-8-bytes/3 estimate, not the model's tokenizer. Ollama requests disable truncation, so a model-specific context overflow is reported instead of silently dropping evidence. Embeddings require an Ollama version with `/api/embed` support.

There is no OCR or image understanding. Scanned documents should be processed with separate tools before ingestion.

### Viewing passages

The file browser, search results and answer citations share three preview modes:

- **Source** renders the original PDF page with highlighted passage regions, page navigation, zoom and a Show passage action. For DOCX and MSG it shows extracted document structure, supported inline formatting, lists and tables with the passage highlighted. TXT preserves whitespace and shows source line numbers. DOCX rendering does not reproduce Word's exact pagination or every layout feature.
- **Readable text** shows the indexed passage, including its supported formatting, without embedding-only context.
- **Embedding input** shows the complete text composed for the embedding model, including metadata and overlap.

Earlier/Later passage navigation stays within the same document or attachment. Search-result and citation navigation remain separate. Attachments, including nested messages, are addressed through their parent message and attachment indexes rather than deleted temporary extraction paths.

Source rendering runs in a separate process with bounded page/document caches. It verifies the source content hash before applying highlights. If the file changed, disappeared, or has no source references yet, the indexed text remains available and the preview explains why source highlighting is unavailable. Email rendering loads no external images, URLs or active HTML.

### Updating existing investigations

Run **Fast rescan** after updating. Stage signatures detect the new extraction and chunking versions, regenerate source references, and refresh affected embeddings. PDF chunking changes also invalidate email files containing PDF attachments. Existing chunks remain readable before the rescan; display-only viewer changes do not require new embeddings. The older pre-staged-schema investigation limitation described above still applies.

## Tests

```powershell
pytest
```

The tests cover hashing, chunking, manifest change decisions, disappeared-file review behavior, strict prompt construction, source alignment, full embedding-input budgets, attachment boundaries, and PDF highlight placement on rotated/cropped pages. PDF regressions also use files in `test_docs` when that directory is available. These tests do not require a running Ollama server.
