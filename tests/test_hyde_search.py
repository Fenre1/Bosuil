from investigation_rag.retrieval.hyde import (
    HYDE_PREFERRED_MODEL,
    HYDE_TOP_CHUNKS_PER_DOCUMENT,
    HYDE_VECTOR_CANDIDATES_PER_DOCUMENT,
    HydeSearcher,
    build_hyde_document_prompt,
    choose_hyde_model,
)
from investigation_rag.retrieval.ranking import SearchResult


def result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        investigation_id=1,
        file_id=1,
        path=f"C:/case/{chunk_id}.txt",
        chunk_text=f"{chunk_id} contains payment evidence",
        source_type="txt",
        chunk_index=0,
        page_number=None,
        score=1.0,
    )


def test_build_hyde_document_prompt_uses_register_and_question():
    prompt = build_hyde_document_prompt("government document", "Where was the package delivered?")

    assert "register: government document" in prompt
    assert "made up answer" in prompt
    assert "Uuser question: Where was the package delivered?" in prompt


def test_hyde_search_generates_four_documents_and_reranks_each_one():
    fake_documents = [
        "Fake report paragraph.",
        "Fake witness interview paragraph.",
        "Fake government paragraph.",
        "Fake log paragraph.",
    ]
    llm = FakeLLM(fake_documents)
    vector = FakeVectorSearcher()
    reranker = FakeReranker()
    searcher = HydeSearcher(vector, llm, reranker=reranker)

    results = searcher.search(1, "When did the payment arrive?", folder_paths=["C:/case"], source_types=["pdf"])

    assert [prompt.count("Uuser question: When did the payment arrive?") for prompt in llm.prompts] == [1, 1, 1, 1]
    assert vector.embedding_texts == fake_documents
    assert vector.limits == [HYDE_VECTOR_CANDIDATES_PER_DOCUMENT] * 4
    assert vector.folder_paths == [["C:/case"]] * 4
    assert vector.source_types == [["pdf"]] * 4
    assert [call[0] for call in reranker.calls] == fake_documents
    assert [call[1] for call in reranker.calls] == [HYDE_TOP_CHUNKS_PER_DOCUMENT] * 4
    assert len(results) == 80
    assert results[0].retrieval_source == "hyde-vector"
    assert results[0].retrieval_queries == ("Hyde register: ordinary language report",)
    assert searcher.last_process_feedback[0] == f"Hyde fake-document model: {HYDE_PREFERRED_MODEL}."
    assert [step.label for step in searcher.last_timing_steps] == [
        "Hyde document generation",
        "Hyde vector retrieval (4 documents)",
        "Hyde rerank (4 documents)",
        "Hyde result merge",
        "Retrieval total",
    ]


def test_choose_hyde_model_prefers_installed_model_and_falls_back():
    assert choose_hyde_model(FakeClient([{"name": "gemma4:e4b"}]), "qwen3.6:latest") == "gemma4:e4b"
    assert choose_hyde_model(FakeClient([{"name": "other:latest"}]), "qwen3.6:latest") == "qwen3.6:latest"
    assert choose_hyde_model(BrokenClient(), "qwen3.6:latest") == "qwen3.6:latest"


class FakeLLM:
    model = HYDE_PREFERRED_MODEL

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def answer(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeVectorSearcher:
    def __init__(self):
        self.embedding_texts = []
        self.limits = []
        self.folder_paths = []
        self.source_types = []

    def search_embedding_text(self, investigation_id, embedding_text, limit, folder_paths=None, source_types=None):
        self.embedding_texts.append(embedding_text)
        self.limits.append(limit)
        self.folder_paths.append(folder_paths)
        self.source_types.append(source_types)
        prefix = len(self.embedding_texts)
        return [result(f"{prefix}-{index}") for index in range(1, limit + 1)]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, results, limit):
        self.calls.append((query, limit, [item.chunk_id for item in results]))
        return results[:limit]


class FakeClient:
    def __init__(self, models):
        self.models = models

    def list_models(self):
        return self.models


class BrokenClient:
    def list_models(self):
        raise RuntimeError("Ollama unavailable")
