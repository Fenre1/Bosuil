from investigation_rag.rag.prompt_builder import ConversationTurn
from investigation_rag.rag.strict_answering import StrictAnswerer, EVIDENCE_LEDGER_CHUNK_LIMIT
from investigation_rag.retrieval.ranking import SearchResult


def test_contextual_followup_rewrites_query_before_searching():
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
    searcher = FakeSearcher([result])
    llm = FakeLLM(["Alice payment email received date", "It was received on 5 May [1]."])
    answerer = StrictAnswerer(searcher, llm)

    response = answerer.answer(
        1,
        "When was it received?",
        history=[ConversationTurn("Who sent the payment email?", "Alice sent it [1].")],
    )

    assert searcher.queries == ["Alice payment email received date"]
    assert response.retrieval_query == "Alice payment email received date"
    assert response.answer == "It was received on 5 May [1]."
    assert "Current question:\nWhen was it received?" in llm.prompts[0]
    assert "Recent conversation" in llm.prompts[1]


def test_evidence_ledger_mode_batches_chunks_and_answers_from_ledger():
    results = [
        SearchResult(
            chunk_id=f"c{index}",
            investigation_id=1,
            file_id=1,
            path=f"C:/case/report-{index}.pdf",
            chunk_text=f"Chunk {index} says the payment was received on 5 May.",
            source_type="pdf",
            chunk_index=index - 1,
            page_number=index,
            score=1.0,
        )
        for index in range(1, 7)
    ]
    searcher = FakeSearcher(results)
    llm = FakeLLM(
        [
            '{"new_evidence": [{"claim": "The payment was received on 5 May.", "source_chunk_id": "c1", '
            '"support_type": "supports", "confidence": "high", "why_relevant": "It answers the date.", '
            '"quoted_support": "received on 5 May"}]}',
            '{"new_evidence": []}',
            "The payment was received on 5 May [E1].",
        ]
    )
    progress = []
    answerer = StrictAnswerer(searcher, llm)

    response = answerer.answer_with_evidence_ledger(
        1,
        "When was the payment received?",
        progress_callback=progress.append,
    )

    assert searcher.limits == [EVIDENCE_LEDGER_CHUNK_LIMIT]
    assert len(response.evidence_ledger) == 1
    assert response.evidence_ledger[0]["source_path"] == "C:/case/report-1.pdf"
    assert response.evidence_ledger[0]["source_location"] == "page 1"
    assert response.answer == "The payment was received on 5 May [E1]."
    assert "Evidence ledger round 1 of 2" in progress[1]
    assert "[E1] The payment was received on 5 May." in llm.prompts[-1]
    assert "Raw chunk text" not in llm.prompts[-1]


class FakeSearcher:
    def __init__(self, results):
        self.results = results
        self.queries = []
        self.limits = []

    def search(self, investigation_id, query, limit):
        self.queries.append(query)
        self.limits.append(limit)
        return self.results


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def answer(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)
