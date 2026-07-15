from investigation_rag.retrieval.fusion import reciprocal_rank_fusion
from investigation_rag.retrieval.pipeline import RetrievalPipeline
from investigation_rag.retrieval.profiles import RetrievalProfile
from investigation_rag.retrieval.ranking import SearchResult


def result(chunk_id: str, text: str = "payment received") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        investigation_id=1,
        file_id=1,
        path=f"C:/case/{chunk_id}.txt",
        chunk_text=text,
        source_type="txt",
        chunk_index=0,
        page_number=None,
        score=1.0,
    )


def test_rrf_deduplicates_and_tracks_sources():
    fused = reciprocal_rank_fusion(
        [result("a"), result("b")],
        [result("b"), result("c")],
        limit=3,
    )

    by_id = {item.chunk_id: item for item in fused}

    assert len(fused) == 3
    assert by_id["b"].retrieval_source == "keyword+vector"
    assert by_id["b"].keyword_rank == 2
    assert by_id["b"].vector_rank == 1


def test_pipeline_retrieves_broadly_but_returns_compact_final_pack():
    profile = RetrievalProfile(
        name="test",
        keyword_candidates=64,
        vector_candidates=64,
        fusion_candidates=80,
        rerank_candidates=80,
        final_answer_chunks=2,
    )
    keyword = FakeSearcher([result("weak", "unrelated"), result("strong", "invoice payment date")])
    vector = FakeSearcher([result("support", "payment was received on 5 May")])
    pipeline = RetrievalPipeline(keyword, vector, db=FakeDB(), profile=profile)

    packed = pipeline.search(1, "payment date")

    assert len(packed) == 2
    assert keyword.limits == [64]
    assert vector.limits == [64]
    assert {item.chunk_id for item in packed} <= {"strong", "support", "weak"}
    assert [step.label for step in pipeline.last_timing_steps] == [
        "Keyword retrieval",
        "Vector retrieval",
        "Result fusion",
        "Rerank",
        "Retrieval total",
    ]


def test_pipeline_passes_folder_scope_to_retrievers():
    profile = RetrievalProfile(
        name="test",
        keyword_candidates=2,
        vector_candidates=3,
        fusion_candidates=4,
        rerank_candidates=4,
        final_answer_chunks=2,
    )
    keyword = ScopedFakeSearcher([result("keyword")])
    vector = ScopedFakeSearcher([result("vector")])
    pipeline = RetrievalPipeline(keyword, vector, db=FakeDB(), profile=profile)

    pipeline.search(1, "payment", folder_paths=["C:/case/a"])

    assert keyword.folder_paths == [["C:/case/a"]]
    assert vector.folder_paths == [["C:/case/a"]]


def test_pipeline_passes_file_type_scope_to_retrievers():
    profile = RetrievalProfile(
        name="test",
        keyword_candidates=2,
        vector_candidates=3,
        fusion_candidates=4,
        rerank_candidates=4,
        final_answer_chunks=2,
    )
    keyword = ScopedFakeSearcher([result("keyword")])
    vector = ScopedFakeSearcher([result("vector")])
    pipeline = RetrievalPipeline(keyword, vector, db=FakeDB(), profile=profile)

    pipeline.search(1, "payment", source_types=["pdf"])

    assert keyword.source_types == [["pdf"]]
    assert vector.source_types == [["pdf"]]


def test_pipeline_records_deep_retrieval_timing_steps():
    profile = RetrievalProfile(
        name="deep-test",
        keyword_candidates=2,
        vector_candidates=2,
        fusion_candidates=4,
        rerank_candidates=4,
        final_answer_chunks=2,
        query_expansion_enabled=True,
        context_expansion_enabled=True,
    )
    keyword = FakeSearcher([result("keyword")])
    vector = FakeSearcher([result("vector")])
    pipeline = RetrievalPipeline(keyword, vector, db=FakeDB(), profile=profile, query_expander=FakeExpander())

    pipeline.search(1, "payment")

    assert pipeline.last_process_feedback == ["Query expansion prompt estimate: ~10 tokens (40 prompt chars)."]
    assert [step.label for step in pipeline.last_timing_steps] == [
        "Query expansion",
        "Keyword retrieval (2 queries)",
        "Vector retrieval (2 queries)",
        "Result fusion",
        "Rerank",
        "Context expansion",
        "Retrieval total",
    ]


class FakeSearcher:
    def __init__(self, results):
        self.results = results
        self.limits = []

    def search(self, investigation_id, query, limit):
        self.limits.append(limit)
        return self.results


class ScopedFakeSearcher:
    def __init__(self, results):
        self.results = results
        self.folder_paths = []
        self.source_types = []

    def search(self, investigation_id, query, limit, folder_paths=None, source_types=None):
        self.folder_paths.append(folder_paths)
        self.source_types.append(source_types)
        return self.results


class FakeDB:
    def get_neighbor_chunks(self, file_id, chunk_index, before=1, after=1, source_types=None):
        return []


class FakeExpander:
    last_prompt_estimate = "Query expansion prompt estimate: ~10 tokens (40 prompt chars)."

    def expand(self, query):
        return [query, f"{query} date"]
