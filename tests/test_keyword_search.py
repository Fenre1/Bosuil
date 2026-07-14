from investigation_rag.retrieval.keyword_search import query_terms, to_fts_query


def test_query_terms_extracts_unique_keywords_case_insensitively():
    assert query_terms("Payment payment fraud-case, PAYMENT") == ["Payment", "fraud-case"]


def test_to_fts_query_uses_query_terms():
    assert to_fts_query("  invoice: overdue! invoice ") == "invoice overdue"
