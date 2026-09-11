from __future__ import annotations

import argparse
import sys
from pathlib import Path

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.embedding_cache import EmbeddingCache
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.ingest.pipeline import ScanPipeline
from investigation_rag.ingest.scan_mode import ScanMode
from investigation_rag.portability import (
    export_workspace,
    import_workspace,
    rebuild_workspace_vectors,
)
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.strict_answering import (
    ANSWER_STYLE_BOTH,
    ANSWER_STYLE_LENIENT,
    ANSWER_STYLE_STRICT,
    StrictAnswerer,
)
from investigation_rag.retrieval.hybrid_search import HybridSearcher
from investigation_rag.retrieval.hyde import (
    HYDE_FINAL_CHUNK_LIMIT,
    HYDE_LEDGER_BATCH_SIZE,
    HydeSearcher,
    choose_hyde_model,
)
from investigation_rag.retrieval.keyword_search import KeywordSearcher
from investigation_rag.retrieval.pipeline import RetrievalPipeline
from investigation_rag.retrieval.profiles import profile_from_config
from investigation_rag.retrieval.query_expansion import QueryExpander
from investigation_rag.retrieval.vector_search import VectorSearcher
from investigation_rag.workspace import InvestigationWorkspace, WorkspaceCatalog


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    config.ensure_dirs()
    catalog = WorkspaceCatalog(config.app_data_dir)
    catalog.initialize()

    try:
        return int(args.func(args, config, catalog))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="investigation-rag")
    sub = parser.add_subparsers(required=True)

    init_parser = sub.add_parser("init", help="Initialize the local app database.")
    init_parser.set_defaults(func=cmd_init)

    create = sub.add_parser("create-investigation", help="Create an investigation.")
    create.add_argument("--name", required=True)
    create.add_argument("--root", action="append", required=True, help="Root folder. May be repeated.")
    create.add_argument("--notes", default="")
    create.set_defaults(func=cmd_create)

    list_parser = sub.add_parser("list-investigations", help="List investigations.")
    list_parser.set_defaults(func=cmd_list)

    export_parser = sub.add_parser(
        "export-investigation", help="Export a portable investigation archive."
    )
    add_investigation_selector(export_parser)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(func=cmd_export)

    import_parser = sub.add_parser(
        "import-investigation", help="Import a portable investigation archive."
    )
    import_parser.add_argument("archive", type=Path)
    import_parser.add_argument(
        "--no-rebuild-vectors",
        action="store_true",
        help="Import SQLite data without reconstructing LanceDB.",
    )
    import_parser.set_defaults(func=cmd_import)

    rebuild_parser = sub.add_parser(
        "rebuild-vector-index",
        help="Reconstruct LanceDB from embeddings cached in SQLite.",
    )
    add_investigation_selector(rebuild_parser)
    rebuild_parser.set_defaults(func=cmd_rebuild_vectors)

    scan = sub.add_parser("scan-investigation", help="Manually scan an investigation.")
    add_investigation_selector(scan)
    scan.add_argument(
        "--scan-mode",
        choices=[mode.value for mode in ScanMode],
        default=ScanMode.FAST.value,
        help="Fast metadata rescan, full content verification, or index rebuild.",
    )
    scan.set_defaults(func=cmd_scan)

    search = sub.add_parser("search-investigation", help="Search one investigation.")
    add_investigation_selector(search)
    search.add_argument("query")
    search.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="keyword")
    search.add_argument("--limit", type=int, default=None)
    search.add_argument(
        "--type",
        dest="source_types",
        action="append",
        choices=["txt", "docx", "pdf", "msg"],
        help="Limit search to a file type. May be repeated.",
    )
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="Ask a RAG question against one investigation.")
    add_investigation_selector(ask)
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--retrieval-profile", choices=["legacy", "standard", "deep", "hyde"], default="standard")
    ask.add_argument(
        "--type",
        dest="source_types",
        action="append",
        choices=["txt", "docx", "pdf", "msg"],
        help="Limit retrieval to a file type. May be repeated.",
    )
    ask.add_argument(
        "--answer-style",
        choices=[ANSWER_STYLE_STRICT, ANSWER_STYLE_LENIENT, ANSWER_STYLE_BOTH],
        default=ANSWER_STYLE_STRICT,
        help="How much freedom the answering model has.",
    )
    ask.add_argument(
        "--evidence-ledger",
        action="store_true",
        help="Build an evidence ledger before generating the final answer.",
    )
    ask.set_defaults(func=cmd_ask)

    return parser


def add_investigation_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", dest="investigation_id")
    group.add_argument("--name", dest="investigation_name")


def cmd_init(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    print(f"Initialized workspace catalog: {catalog.path}")
    print(f"Investigation workspaces: {catalog.workspaces_dir}")
    return 0


def cmd_create(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    roots = [Path(root).expanduser().resolve() for root in args.root]
    workspace = catalog.create(args.name, roots, args.notes)
    print(f"Created investigation {workspace.entry.id}: {args.name}")
    print(f"Workspace: {workspace.entry.workspace_path}")
    return 0


def cmd_list(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    for entry in catalog.list():
        workspace = catalog.open(entry)
        roots = ", ".join(str(root) for root in workspace.db.get_roots(workspace.investigation_id))
        print(f"{entry.id}\t{entry.name}\t{roots}")
    return 0


def cmd_export(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    workspace = resolve_workspace(catalog, args)
    result = export_workspace(workspace.entry, args.output)
    print(f"Exported investigation {workspace.entry.id}: {result.path}")
    print(f"SQLite snapshot: {result.database_bytes} bytes")
    return 0


def cmd_import(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    result = import_workspace(
        catalog,
        args.archive,
        preferred_embedding_model=config.ollama_embed_model,
        rebuild_vectors=not args.no_rebuild_vectors,
    )
    print(f"Imported investigation {result.workspace.entry.id}: {result.workspace.entry.name}")
    print(f"Workspace: {result.workspace.entry.workspace_path}")
    if result.vector_rebuild is not None:
        rebuilt = result.vector_rebuild
        print(
            f"Restored vectors: {rebuilt.restored_vectors}/{rebuilt.active_chunks} "
            f"(model={rebuilt.embedding_model or 'none'})"
        )
    if result.vector_warning:
        print(f"Vector warning: {result.vector_warning}", file=sys.stderr)
    return 0


def cmd_rebuild_vectors(
    args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog
) -> int:
    workspace = resolve_workspace(catalog, args)
    result = rebuild_workspace_vectors(
        workspace,
        preferred_embedding_model=config.ollama_embed_model,
    )
    print(
        f"Restored vectors: {result.restored_vectors}/{result.active_chunks} "
        f"(model={result.embedding_model or 'none'})"
    )
    if result.backup_path is not None:
        print(f"Previous vector index: {result.backup_path}")
    return 0


def cmd_scan(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    workspace = resolve_workspace(catalog, args)
    pipeline = ScanPipeline(
        config.for_workspace(workspace.entry.workspace_path), workspace.db, workspace.store, log=print
    )
    pipeline.scan_investigation(workspace.investigation_id, mode=args.scan_mode)
    return 0


def cmd_search(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    workspace = resolve_workspace(catalog, args)
    workspace_config = config.for_workspace(workspace.entry.workspace_path)
    searcher = build_searcher(workspace_config, workspace.db, args.mode)
    limit = args.limit if args.limit is not None else (None if args.mode == "keyword" else 10)
    results = searcher.search(
        workspace.investigation_id, args.query, limit=limit, source_types=args.source_types
    )
    if not results:
        print("No results.")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.locator} [{result.source_type}] score={result.score:.3f}")
        preview = result.chunk_text.replace("\n", " ")
        print(f"   {preview[:300]}")
    return 0


def cmd_ask(args: argparse.Namespace, config: AppConfig, catalog: WorkspaceCatalog) -> int:
    workspace = resolve_workspace(catalog, args)
    investigation_id = workspace.investigation_id
    config = config.for_workspace(workspace.entry.workspace_path)
    llm = OllamaLLM(
        OllamaClient(
            config.ollama_base_url,
            timeout=config.ollama_timeout_seconds,
            allow_remote=config.ollama_allow_remote,
            num_ctx=config.ollama_embed_num_ctx,
        ),
        config.ollama_llm_model,
    )
    searcher = build_rag_searcher(config, workspace.db, args.retrieval_profile, llm)
    answerer = StrictAnswerer(searcher, llm)
    if args.retrieval_profile == "hyde":
        response = answerer.answer_with_evidence_ledger(
            investigation_id,
            args.question,
            source_types=args.source_types,
            answer_style=args.answer_style,
            progress_callback=lambda message: print(message, file=sys.stderr),
            chunk_limit=HYDE_FINAL_CHUNK_LIMIT,
            batch_size=HYDE_LEDGER_BATCH_SIZE,
        )
    elif args.evidence_ledger:
        response = answerer.answer_with_evidence_ledger(
            investigation_id,
            args.question,
            source_types=args.source_types,
            answer_style=args.answer_style,
            progress_callback=lambda message: print(message, file=sys.stderr),
        )
    else:
        response = answerer.answer(
            investigation_id,
            args.question,
            top_k=args.top_k,
            source_types=args.source_types,
            answer_style=args.answer_style,
        )
    print(response.answer)
    if response.citations:
        print("\nCitations:")
        print(response.citations)
    if response.process_feedback:
        print("\nProcess:")
        for item in response.process_feedback:
            print(f"- {item}")
    return 0


def resolve_workspace(catalog: WorkspaceCatalog, args: argparse.Namespace) -> InvestigationWorkspace:
    if getattr(args, "investigation_id", None):
        entry = catalog.get(args.investigation_id)
    else:
        entry = catalog.get_by_name(args.investigation_name)
    if entry is None:
        raise ValueError("Investigation not found.")
    return catalog.open(entry)


def build_searcher(config: AppConfig, db: SQLiteDB, mode: str) -> object:
    keyword = KeywordSearcher(db)
    if mode == "keyword":
        return keyword

    store = LanceDBStore(config.lancedb_dir)
    embeddings = EmbeddingCache(
        db,
        OllamaClient(
            config.ollama_base_url,
            timeout=config.ollama_timeout_seconds,
            allow_remote=config.ollama_allow_remote,
            num_ctx=config.ollama_num_ctx,
        ),
        config.ollama_embed_model,
    )
    vector = VectorSearcher(db, store, embeddings, query_instruction=config.ollama_embed_query_instruction)
    if mode == "semantic":
        return vector
    return HybridSearcher(keyword, vector if config.ollama_embed_model else None)


def build_rag_searcher(config: AppConfig, db: SQLiteDB, profile_name: str = "standard", llm: OllamaLLM | None = None) -> object:
    keyword = KeywordSearcher(db)
    normalized_profile = (profile_name or "standard").strip().lower()
    store = LanceDBStore(config.lancedb_dir)
    vector = None
    if config.ollama_embed_model:
        embeddings = EmbeddingCache(
            db,
            OllamaClient(
                config.ollama_base_url,
                timeout=config.ollama_timeout_seconds,
                allow_remote=config.ollama_allow_remote,
                num_ctx=config.ollama_embed_num_ctx,
            ),
            config.ollama_embed_model,
        )
        vector = VectorSearcher(db, store, embeddings, query_instruction=config.ollama_embed_query_instruction)
    if normalized_profile == "hyde":
        if vector is None:
            raise ValueError("Hyde retrieval requires OLLAMA_EMBED_MODEL to be configured.")
        if llm is None:
            raise ValueError("Hyde retrieval requires an Ollama answering model.")
        hyde_model = choose_hyde_model(llm.client, llm.model)
        hyde_llm = llm if hyde_model == llm.model else OllamaLLM(llm.client, hyde_model)
        return HydeSearcher(vector, hyde_llm)
    profile = profile_from_config(config, normalized_profile)
    query_expander = QueryExpander(llm) if profile.query_expansion_enabled and llm is not None else None
    return RetrievalPipeline(
        keyword=keyword,
        vector=vector,
        db=db,
        profile=profile,
        query_expander=query_expander,
    )


if __name__ == "__main__":
    raise SystemExit(main())
