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

Use the left sidebar to switch between the nine work areas. The page header shows the active investigation. Resizable panes and scrollable pages keep controls accessible in smaller windows; search results appear above the source preview so document columns have more room.

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

# Knowledge graph

The **Kennisgraaf** tab builds directed, source-backed relationships through a
new independent pipeline. It reads indexed passages directly and does not use
Feitenoverzicht or existing entity-extraction output. In **Documents**, select
one or more files, choose your local model (or use **Refresh models**), then
choose **Build graph**. Start with a small representative sample.

English instructions, a compact relationship schema, exact source-quote checks,
and an optional second verification call produce reviewable suggestions. The
default verification pass is enabled. Negation, allegations, attribution and
explicit time expressions remain attached to claims. Select an edge or a
**Relationships** row to inspect its quote and source; accept/reject, correct
claims/nodes, or explicitly merge identities. Equal names in different sources
start as separate nodes. Rejected or stale source evidence is hidden by default.

Relationship settings are grouped into **Time**, **Location**, **Facts and
properties**, **Processes and effects**, and **People and organizations**.
Dates/times, periods, durations, quantities with units/currencies, and states
can appear as literal values. Property names distinguish facts such as a
bridge's height and width even when their numeric values match. Explicitly
described events and concepts need not have proper names. Temporal ordering
and causation require explicit evidence. Existing saved default selections
gain the broader vocabulary automatically; custom selections are preserved.

**Extraction settings** controls the relationship types, context/output budgets,
thinking mode and request duration. Defaults are an 8k context, 4,096 output
tokens, thinking off and 180 seconds per call. Compact responses cite numbered
source sentences; the app reconstructs exact quotes. Truncated or crowded
windows automatically retry as smaller overlapping sections. Extraction and
verification have separate caches: raising the token budget or retrying failed
verification preserves completed work, including compatible older windows.
**Stop** retains completed sections and cached extraction. During a stalled request,
stopping waits for a stream update or the network timeout. **Diagnostics** shows
model-loading, input-processing and generation timings, thinking-character
counts and raw responses in **Build log → Diagnostics**. **Export graph** saves JSON; ordinary investigation
archives include the graph database as well.

The model defaults to your configured answer model. Historical Llama 3 8B paper
results are a baseline, not a recommendation over current Qwen/Gemma models.
Compare models on your own sources. **Nodes shown** lets you choose a display
limit or **All** (the default); search/review filters still apply. **Connected
groups** packs related nodes together, while **Compact grid** offers a uniform
layout. **Rearrange** resets dragged positions; **Fit graph** fits the current
view. Display preferences persist per investigation. The relationship table
retains the complete filtered claim set regardless of the display limit.

**Import graph** reads existing Bosuil knowledge graph JSON exports, including
older v1 exports. It adds nodes, relationships, exact evidence quotes, properties
and review information without calling a model. Reimporting preserves local
corrections/reviews and avoids duplicates. Matching indexed passages are linked;
otherwise claims remain visible as **imported quotes**, with source opening
disabled. JSON exports contain quotes rather than full source documents. Reimport
after indexing matching sources to attach their evidence. Imported graphs also
travel with ordinary investigation archives.

See [research basis, design choices and limitations](docs/knowledge-graph-design.md).

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

The **Graph** subtab explores the complete active entity collection independently
of the Results table's name, type, and mention filters. Find an entity or alias,
select one or more Results rows and choose **Explore selected in graph**, or
choose a document scope. A new entity exploration starts with its 20 highest
ranked neighbours; **Expand all** adds every neighbour, and **Show all entities**
draws every entity in the current scope. There is no entity, passage, membership,
or edge cap in the explorer. Queries and layout run in a cancellable worker.

Each edge represents one entity pair. **Same passage** connects entities with
an exact mention in a shared active chunk; **Same document** also connects
entities mentioned in separate passages of the same source document. Attachments
are scoped separately when their source identity is available. Edges represent
co-occurrence, not a verified relationship or causation. Counts always use the
complete selected document scope, regardless of the nodes currently drawn.

Click an entity for its aliases and complete sortable connection list. Click an
edge or connection row for evidence grouped by source document, then select a
passage to load its text. **Follow** explores the other endpoint and its
neighbours. The two entities and their aliases are highlighted in
different colours in **Readable text**; **Source** opens the existing source
viewer with passage highlighting. Document-level evidence identifies which
entity is present in each passage. Previous/Next navigates the evidence set;
Earlier/Later stays within the source document.

Rank neighbours by shared documents, shared passages, or specificity. Ranking
counts discount exact file copies and identical passage text. Specificity is
shared document contents divided by the union of document contents mentioning
either entity (Jaccard similarity); it is not confidence or source independence.
All source copies remain accessible in the evidence list. Near-duplicates are
not detected, and entities mentioned only once remain available. The status
separates collection size, document/type scope, and displayed nodes/edges. It
also reports documents with entity mentions; extraction completeness is not
inferred from the presence or absence of mentions.

The default layout follows connections and keeps existing node positions during
expansion. Drag nodes, pin them, hide them from the view, and use Back/Forward
or **Exploration > Undo view change**. Selection preserves pan and zoom; **Fit
view** is explicit. Ctrl-click nodes and choose **Shared connections** to explore
their common neighbours. Labels appear as you zoom in; selected, pinned, and
hovered nodes retain labels, with full names available in the inspector/tooltips.
**Topic map** is a separate layout using stored entity-context embeddings;
proximity there means similar text. No model generation or re-extraction is
needed. Missing embeddings fall back to connection placement.

**Collection overview** groups entities by connected component and displays
representative names, with expandable type groups and individual entities.
Double-click a group or entity to explore it. These automatic
groups are navigation aids, not investigative findings. **Exploration > Save
view / Load view** stores named explorations, including scope, pins, hidden
nodes, selection, positions, layout, and viewport, in the investigation database.
The selected evidence passage and preview mode are retained as well.
Removed entities are omitted when restoring an old view.

The inspector's **Entity actions** menu and node context menu provide type
correction and alias merging. **Undo correction** restores the previous entity
and mention records transactionally; it refuses to overwrite entity data that
changed after the correction. Corrections retain aliases for subsequent
extraction runs. View undo and correction undo are separate operations.

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
