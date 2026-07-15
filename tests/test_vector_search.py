from investigation_rag.retrieval.vector_search import VectorSearcher, format_embedding_query


def test_format_embedding_query_adds_instruction_wrapper():
    formatted = format_embedding_query(
        "Where is the invoice?",
        "Given a user question, retrieve relevant passages from a local document collection that answer the question.",
    )

    assert formatted == (
        "Instruct: Given a user question, retrieve relevant passages from a local document collection that answer the question.\n"
        "Query: Where is the invoice?"
    )


def test_format_embedding_query_leaves_prewrapped_query_alone():
    query = "Instruct: custom\nQuery: payment"

    assert format_embedding_query(query, "default") == query


def test_vector_search_embeds_instructed_query():
    embeddings = FakeEmbeddings()
    searcher = VectorSearcher(
        FakeDB(),
        FakeStore(),
        embeddings,
        query_instruction="Given a user question, retrieve relevant passages.",
    )

    searcher.search(1, "payment date")

    assert embeddings.texts == ["Instruct: Given a user question, retrieve relevant passages.\nQuery: payment date"]


class FakeEmbeddings:
    model = "test-model"

    def __init__(self):
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [[0.1, 0.2, 0.3]]


class FakeStore:
    def search(self, vector, investigation_id, embedding_model="", limit=10, oversample=3, source_types=None):
        return []


class FakeDB:
    def get_chunks_by_ids_in_folders(self, chunk_ids, folder_paths, source_types=None, active_only=True):
        return []
