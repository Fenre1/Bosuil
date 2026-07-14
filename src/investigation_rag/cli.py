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
from investigation_rag.rag.ollama_llm import OllamaLLM
from investigation_rag.rag.strict_answering import (
    ANSWER_STYLE_BOTH,
    ANSWER_STYLE_LENIENT,
    ANSWER_STYLE_STRICT,
    StrictAnswerer,
)
from investigation_rag.retrieval.hybrid_search import HybridSearcher
from investigation_rag.retrieval.keyword_search import KeywordSearcher
from investigation_rag.retrieval.pipeline import RetrievalPipeline
from investigation_rag.retrieval.profiles import profile_from_config
from investigation_rag.retrieval.query_expansion import QueryExpander
from investigation_rag.retrieval.vector_search import VectorSearcher


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    config.ensure_dirs()
    db = SQLiteDB(config.sqlite_path)
    db.initialize()

    try:
        return int(args.func(args, config, db))
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

    scan = sub.add_parser("scan-investigation", help="Manually scan an investigation.")
    add_investigation_selector(scan)
    scan.set_defaults(func=cmd_scan)

    search = sub.add_parser("search-investigation", help="Search one investigation.")
    add_investigation_selector(search)
    search.add_argument("query")
    search.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="keyword")
    search.add_argument("--limit", type=int, default=None)
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="Ask a RAG question against one investigation.")
    add_investigation_selector(ask)
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--retrieval-profile", choices=["legacy", "standard", "deep"], default="standard")
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
    group.add_argument("--id", type=int, dest="investigation_id")
    group.add_argument("--name", dest="investigation_name")


def cmd_init(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    print(f"Initialized SQLite: {config.sqlite_path}")
    print(f"LanceDB directory: {config.lancedb_dir}")
    return 0


def cmd_create(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    roots = [Path(root).expanduser().resolve() for root in args.root]
    investigation_id = db.create_investigation(args.name, roots, args.notes)
    print(f"Created investigation {investigation_id}: {args.name}")
    return 0


def cmd_list(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    for investigation in db.list_investigations():
        roots = ", ".join(str(root) for root in db.get_roots(investigation.id))
        print(f"{investigation.id}\t{investigation.name}\t{roots}")
    return 0


def cmd_scan(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    investigation_id = resolve_investigation_id(db, args)
    store = LanceDBStore(config.lancedb_dir)
    pipeline = ScanPipeline(config, db, store, log=print)
    pipeline.scan_investigation(investigation_id)
    return 0


def cmd_search(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    investigation_id = resolve_investigation_id(db, args)
    searcher = build_searcher(config, db, args.mode)
    limit = args.limit if args.limit is not None else (None if args.mode == "keyword" else 10)
    results = searcher.search(investigation_id, args.query, limit=limit)
    if not results:
        print("No results.")
        return 0
    for index, result in enumerate(results, start=1):
        print(f"{index}. {result.locator} [{result.source_type}] score={result.score:.3f}")
        preview = result.chunk_text.replace("\n", " ")
        print(f"   {preview[:300]}")
    return 0


def cmd_ask(args: argparse.Namespace, config: AppConfig, db: SQLiteDB) -> int:
    investigation_id = resolve_investigation_id(db, args)
    llm = OllamaLLM(
        OllamaClient(
            config.ollama_base_url,
            timeout=config.ollama_timeout_seconds,
            allow_remote=config.ollama_allow_remote,
        ),
        config.ollama_llm_model,
    )
    searcher = build_rag_searcher(config, db, args.retrieval_profile, llm)
    answerer = StrictAnswerer(searcher, llm)
    if args.evidence_ledger:
        response = answerer.answer_with_evidence_ledger(
            investigation_id,
            args.question,
            answer_style=args.answer_style,
        )
    else:
        response = answerer.answer(
            investigation_id,
            args.question,
            top_k=args.top_k,
            answer_style=args.answer_style,
        )
    print(response.answer)
    if response.citations:
        print("\nCitations:")
        print(response.citations)
    return 0


def resolve_investigation_id(db: SQLiteDB, args: argparse.Namespace) -> int:
    if getattr(args, "investigation_id", None):
        row = db.get_investigation(args.investigation_id)
    else:
        row = db.get_investigation_by_name(args.investigation_name)
    if row is None:
        raise ValueError("Investigation not found.")
    return int(row["id"])


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
        ),
        config.ollama_embed_model,
    )
    vector = VectorSearcher(db, store, embeddings, query_instruction=config.ollama_embed_query_instruction)
    if mode == "semantic":
        return vector
    return HybridSearcher(keyword, vector if config.ollama_embed_model else None)


def build_rag_searcher(config: AppConfig, db: SQLiteDB, profile_name: str = "standard", llm: OllamaLLM | None = None) -> object:
    keyword = KeywordSearcher(db)
    profile = profile_from_config(config, profile_name)
    store = LanceDBStore(config.lancedb_dir)
    vector = None
    if config.ollama_embed_model:
        embeddings = EmbeddingCache(
            db,
            OllamaClient(
                config.ollama_base_url,
                timeout=config.ollama_timeout_seconds,
                allow_remote=config.ollama_allow_remote,
            ),
            config.ollama_embed_model,
        )
        vector = VectorSearcher(db, store, embeddings, query_instruction=config.ollama_embed_query_instruction)
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
