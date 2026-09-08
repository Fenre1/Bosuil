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

`INVESTIGATION_RAG_DATA_DIR` contains a small `catalog.sqlite3` plus UUID-named workspaces under `investigations\`. Each workspace contains `investigation.sqlite3`, `lancedb\`, and `manifest.json`. If the complete data directory is moved, workspace discovery repairs catalog paths from the manifests on the next application start. Source evidence is not copied.

Evidence roots may live on SMB shares or private-cloud mounted folders. Keep the active `INVESTIGATION_RAG_DATA_DIR` on a local disk: SQLite WAL and live LanceDB files are not safe or reliably fast on many network or sync-managed folders.

For transfers, use **Export** and **Import** in the investigation selector. A `.bosuil` archive contains the manifest and a consistent SQLite snapshot, including extracted chunk text and the embedding cache. It deliberately excludes LanceDB because some Lance manifests contain absolute paths. Import reconstructs a fresh local vector index from the cached embeddings, so neither the source files nor Ollama are needed when the cache is complete. If cached vectors are missing, keyword search still works and the app reports that the affected chunks must be re-embedded. **Rebuild vectors** performs the same repair for an investigation that was copied manually; the previous LanceDB directory is retained as a timestamped backup.

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

The GUI includes an investigation selector, create/edit dialog, root folder management, manual scan, scan log, keyword/semantic/hybrid search, result preview, Windows file/folder opening, contextual conversational strict RAG questions with PDF export, Excel-based end-to-end RAG evaluation, and disappeared-file review.

The **Evaluatie** tab loads an `.xlsx` workbook containing `Question`, `Answer`, and `Points` columns, with optional `qID` and `Comment` columns. It runs every question independently through the currently loaded investigation and selected RAG settings, then uses the local `qwen3.6:latest` model to award points. The exported workbook preserves the input, adds the generated answer, judge score, and any evaluation error, and includes an `Evaluation Summary` worksheet with the run parameters and score totals.

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
investigation-rag export-investigation --name "Case A" --output "D:\Transfers\Case-A.bosuil"
investigation-rag import-investigation "D:\Transfers\Case-A.bosuil"
investigation-rag rebuild-vector-index --name "Case A"
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

Chunking is deterministic and versioned (`deterministic-v1`). There is no OCR and no AI/smart chunking in v1.

## Tests

```powershell
pytest
```

The tests cover hashing, chunking, manifest change decisions, disappeared-file review behavior, and strict prompt construction.

# Document-wide fact ledger

The **Feitenoverzicht** tab builds a question-independent evidence ledger for
one selected indexed document. It analyses every active chunk in contiguous
batches of five. Each batch after the first repeats the preceding batch's last
chunk, providing one chunk of context overlap. A final model request reviews
the complete preliminary ledger, removes duplicates, and combines compatible
statements while retaining all contributing chunk references.
Malformed JSON is retried once with the same chunk batch. If the retry also
fails, that batch is recorded and the remaining chunks continue; the warning
and affected chunk IDs are included on the `Bron` worksheet. Selecting a ledger
row opens its source chunk text, with previous/next controls when an item has
multiple source chunks.

Every raw model response is retained for diagnostics and exported on the
`Modeluitvoer` worksheet, including successful responses, retries, and final
consolidation responses. Each row records its stage, round, attempt, chunk IDs,
JSON parse result, and parse error. Responses exceeding Excel's per-cell text
limit are split into numbered parts without dropping text.

`FACT_LEDGER_OUTPUT_TOKENS` controls only Feitenoverzicht generations and
defaults to `16384`; it does not change entity extraction or De Uil Antwoord.

Completed ledgers can be exported as `.xlsx`. The `Feitenoverzicht` worksheet
contains the original document name, complete statements, source locations,
and chunk IDs; the separate `Bron` worksheet records the document's full source
path.

# Entity extraction

The **Entiteiten** tab sends each active document to a local Ollama generation
model once, using `qwen3.6:latest` by default. The model returns a deliberately
simple `Entity [Type]` list. The app parses that list, rejects unsupported
output, and links only exact occurrences back to the active chunks in SQLite.
Entity types and the local model name can be edited before each run.

Set `ENTITY_NUM_CTX=131072` for the intended 128k whole-document context.
Documents that do not fit are split into the largest contiguous chunk windows
that fit, and their unique results are merged. Results are cached by document
text, model, prompt version, labels, and context settings. Multiple selected
rows can still be merged under an editable canonical name and type; aliases
are retained on later extraction runs.

The **Graph** subtab visualizes the entities currently visible after applying
the table filters. Entity rectangles contain the full entity name, use
square-root scaling based on their number of distinct chunks, and are colored
by entity type. Labels stay above links and chunk markers, then hide while
their rectangle is hovered. Multiword organization names use two balanced
lines, and words longer than 15 characters are split without truncation.
Rectangles grow to the rendered label and are separated before display. A
user-adjustable minimum font size reflows the drawing without rerunning the
data or semantic layout. Semantic placement is
calculated from the existing chunk embeddings with a deterministic UMAP-style
local-neighborhood manifold projection. Selecting an entity
reveals one link per shared chunk, regardless of repeated mentions inside that
chunk. Hover a chunk marker for an excerpt or click it for the persistent full
chunk preview. User-adjustable entity and chunk limits, plus explicit safety
warnings for unusually dense memberships, keep large graphs manageable.

The repository includes a curated required-entity reference set for the local
138-page `test_pdf/onveiligheid-door-extreme-regen.pdf` test document. Save a
model's simple line output to a UTF-8 text file and score required recall and
type accuracy with:

```powershell
python -m investigation_rag.entity_evaluation `
  tests\fixtures\entity_ground_truth\onveiligheid-door-extreme-regen.json `
  qwen-entity-output.txt
```

The reference set is intentionally high-confidence rather than exhaustive, so
additional extracted entities are reported but are not counted as false
positives.
