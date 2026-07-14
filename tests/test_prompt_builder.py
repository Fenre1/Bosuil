from investigation_rag.rag.prompt_builder import (
    ConversationTurn,
    build_contextual_query_prompt,
    build_strict_prompt,
)
from investigation_rag.retrieval.ranking import SearchResult


def test_strict_prompt_uses_retrieved_context_and_citations():
    result = SearchResult(
        chunk_id="c1",
        investigation_id=1,
        file_id=1,
        path="C:/case/report.pdf",
        chunk_text="The payment was received on 5 May.",
        source_type="pdf",
        chunk_index=0,
        page_number=3,
        score=1.0,
    )

    prompt = build_strict_prompt("When was the payment received?", [result])

    assert "Do not use general world knowledge" in prompt
    assert "C:/case/report.pdf (page 3)" in prompt
    assert "The payment was received on 5 May." in prompt


def test_strict_prompt_includes_history_as_non_evidence():
    result = SearchResult(
        chunk_id="c1",
        investigation_id=1,
        file_id=1,
        path="C:/case/report.pdf",
        chunk_text="The payment was received on 5 May.",
        source_type="pdf",
        chunk_index=0,
        page_number=3,
        score=1.0,
    )
    history = [ConversationTurn("Who sent the payment email?", "Alice sent the payment email [1].")]

    prompt = build_strict_prompt("When was it received?", [result], history=history)

    assert "for resolving references only" in prompt
    assert "User: Who sent the payment email?" in prompt
    assert "Assistant: Alice sent the payment email [1]." in prompt
    assert "Conversation history may be used only to understand references" in prompt


def test_contextual_query_prompt_requests_standalone_search_query():
    history = [ConversationTurn("Who sent the payment email?", "Alice sent the payment email [1].")]

    prompt = build_contextual_query_prompt("When was it received?", history)

    assert "standalone search query" in prompt.lower()
    assert "Do not answer the question." in prompt
    assert "Current question:\nWhen was it received?" in prompt
