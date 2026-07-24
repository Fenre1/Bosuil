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
OLLAMA_LLM_MODEL=<your-local-chat-model>
OLLAMA_EMBED_MODEL=<your-local-embedding-model>
OLLAMA_EMBED_QUERY_INSTRUCTION=Given a user question, retrieve relevant passages from a local document collection that answer the question.
```

By default data is stored under `%LOCALAPPDATA%\InvestigationRAG`. Real environment variables still take priority over values in `.env`, so temporary PowerShell overrides continue to work. If you keep the file somewhere else, set `INVESTIGATION_RAG_ENV_FILE` to that path before running the app.

`INVESTIGATION_RAG_DATA_DIR` contains a small `catalog.sqlite3` plus UUID-named workspaces under `investigations\`. Each workspace contains `investigation.sqlite3`, `lancedb\`, and `manifest.json`. Copying a complete workspace directory into `investigations\` makes it discoverable on the next application start; source evidence is not copied.

The app rejects non-loopback Ollama URLs by default and the Ollama client ignores proxy environment variables. Keep `INVESTIGATION_RAG_ALLOW_REMOTE_OLLAMA=0` for sensitive data. Only set it to `1` for a trusted non-sensitive setup where the Ollama server is intentionally hosted away from the same machine.

For slower local models or long evidence-ledger RAG runs, raise `OLLAMA_TIMEOUT_SECONDS` in `.env`; for example, `600` waits up to ten minutes for each local Ollama request.

Set `OLLAMA_NUM_CTX` to pass Ollama's `options.num_ctx` for answering-model requests. Use `32768` for a 32k context window, or set it to `0`/leave it unset to use the Ollama/model default.

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
investigation-rag search-investigation --name "Case A" --mode keyword "invoice"
investigation-rag search-investigation --name "Case A" --mode semantic "payments after May"
investigation-rag ask --name "Case A" "What evidence mentions payments after May?"
investigation-rag ask --name "Case A" --retrieval-profile hyde "What evidence mentions payments after May?"
```

If Ollama or LanceDB is unavailable, keyword search and non-embedding scan functions still work. Embedding and RAG operations report a clear local configuration/runtime error.

## Ingest Model

The v1 scan is manual only. The pipeline scans root folders, records cheap metadata first, compares with the SQLite manifest, hashes only new or suspicious files, reuses extraction and embedding caches, and avoids re-extracting or re-embedding when content hashes are unchanged. Replaced chunks are marked inactive in SQLite before new chunks are inserted.

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
