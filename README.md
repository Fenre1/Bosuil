# Local Investigation RAG

Windows-first, developer-run desktop app for indexing investigation folders and asking strict local RAG questions over `.txt`, `.docx`, `.pdf`, and `.msg` files. The UI uses PyQt5. Storage is one global SQLite database plus one global LanceDB database, with every file and chunk scoped by `investigation_id`.

No cloud APIs are used. Ollama is called on the local machine for embeddings and generation.

## Setup on Windows

```powershell
cd C:\Projects\Bosuil\local_investigation_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install and start Ollama separately, then pull the local models you want to use. The app does not hard-code model names.

Create a `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```env
INVESTIGATION_RAG_DATA_DIR=C:\InvestigationRAGData
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_LLM_MODEL=<your-local-chat-model>
OLLAMA_EMBED_MODEL=<your-local-embedding-model>
OLLAMA_EMBED_QUERY_INSTRUCTION=Given a user question, retrieve relevant passages from a local document collection that answer the question.
```

By default data is stored under `%LOCALAPPDATA%\InvestigationRAG`. Real environment variables still take priority over values in `.env`, so temporary PowerShell overrides continue to work. If you keep the file somewhere else, set `INVESTIGATION_RAG_ENV_FILE` to that path before running the app.

For slower local models or long evidence-ledger RAG runs, raise `OLLAMA_TIMEOUT_SECONDS` in `.env`; for example, `600` waits up to ten minutes for each local Ollama request.

## Run the Desktop App

```powershell
python -m investigation_rag.app
```

or, after editable install:

```powershell
investigation-rag-gui
```

The GUI includes an investigation selector, create/edit dialog, root folder management, manual scan, scan log, keyword/semantic/hybrid search, result preview, Windows file/folder opening, contextual conversational strict RAG questions with PDF export, and disappeared-file review.

## CLI

```powershell
investigation-rag init
investigation-rag create-investigation --name "Case A" --root "D:\Cases\CaseA" --notes "Initial import"
investigation-rag list-investigations
investigation-rag scan-investigation --name "Case A"
investigation-rag search-investigation --name "Case A" --mode keyword "invoice"
investigation-rag search-investigation --name "Case A" --mode semantic "payments after May"
investigation-rag ask --name "Case A" "What evidence mentions payments after May?"
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
