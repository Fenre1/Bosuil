from __future__ import annotations

from dataclasses import dataclass
import json

from investigation_rag.rag.citations import citation_label
from investigation_rag.retrieval.ranking import SearchResult, result_location_label


STRICT_SYSTEM_INSTRUCTIONS = """You are a local investigation assistant.
Answer using only the evidence in the retrieved context below.
Do not use general world knowledge.
Conversation history may be used only to understand references in the user's question; it is not evidence.
If the context does not contain enough evidence, say: "The retrieved evidence is insufficient to answer that."
Every factual claim must be supported by a citation like [1] or [2].
"""


LENIENT_SYSTEM_INSTRUCTIONS = """You are a local investigation assistant.
Use the retrieved context as the main source of evidence, but you may also use careful reasoning, common knowledge, and relevant background knowledge to help answer the user's question.
Do not pretend that unsupported ideas are proven. Separate what the retrieved evidence says from your interpretation, inference, or suggestion.
Cite retrieved evidence with [1], [2], etc. when making claims based directly on the retrieved context. General explanation, reasoning, and clearly marked speculation do not need citations.
Conversation history may be used to understand the user's intent and references. Treat it as context, not as primary evidence.
If the retrieved context is incomplete, still try to give a useful answer. Say what seems likely, what is uncertain, and what extra evidence would help confirm it.
Prefer a direct, natural answer. You may synthesize across sources, explain implications, identify patterns, flag contradictions, and suggest next investigative steps.
"""


CONTEXTUAL_QUERY_INSTRUCTIONS = """Rewrite the user's current question as one standalone search query for a local investigation archive.
Use the recent conversation only to resolve references like "it", "that", "those dates", or "the second one".
Do not answer the question.
Do not add facts that are not present in the conversation.
Return only the standalone search query text.
"""


EVIDENCE_LEDGER_INSTRUCTIONS = """You are building an evidence ledger for a local investigation question.
Use only the current chunk batch as evidence for new ledger entries.
Use the existing ledger only to avoid duplicate entries and to notice contradictions or qualifications.
Extract concise, query-relevant claims. Ignore irrelevant or duplicate information.
Do not infer beyond the chunk text.
Return only valid JSON with this shape:
{
  "new_evidence": [
    {
      "claim": "The document says X happened on date Y.",
      "source_chunk_id": "...",
      "support_type": "supports | contradicts | qualifies | background",
      "confidence": "high | medium | low",
      "why_relevant": "...",
      "quoted_support": "Short exact supporting phrase from the chunk."
    }
  ]
}
If the batch adds no new relevant information, return {"new_evidence": []}.
"""


LEDGER_FINAL_ANSWER_INSTRUCTIONS = """You are answering a local investigation question from an evidence ledger.
Use only the evidence ledger below. Do not use retrieved chunks, conversation history, or general world knowledge as evidence.
If the ledger does not contain enough evidence, say: "The evidence ledger is insufficient to answer that."
Every factual claim must cite ledger entries like [E1] or [E2].
"""


LENIENT_LEDGER_FINAL_ANSWER_INSTRUCTIONS = """You are answering a local investigation question from an evidence ledger.
Use the evidence ledger as the factual foundation. You may reason from the ledger, explain implications, identify patterns, and suggest plausible next steps.
Cite ledger-backed claims with [E1], [E2], etc. Clearly distinguish direct evidence from inference, interpretation, or speculation.
If the ledger is incomplete, still provide the most useful answer you can. Explain what is uncertain and what additional evidence would help.
Prefer a direct, readable answer rather than refusing unless there is truly no relevant evidence.
"""

MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 5000
MAX_HISTORY_QUESTION_CHARS = 1000
MAX_HISTORY_ANSWER_CHARS = 1800


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str


def build_contextual_query_prompt(question: str, history: list[ConversationTurn]) -> str:
    history_text = format_conversation_history(history)
    if not history_text:
        history_text = "No prior conversation."
    return (
        f"{CONTEXTUAL_QUERY_INSTRUCTIONS}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Current question:\n{question}\n\n"
        "Standalone search query:"
    )


def build_strict_prompt(
    question: str,
    results: list[SearchResult],
    history: list[ConversationTurn] | None = None,
) -> str:
    return build_retrieved_context_prompt(
        question,
        results,
        history=history,
        instructions=STRICT_SYSTEM_INSTRUCTIONS,
        answer_label="Answer with citations:",
        strict_history=True,
    )


def build_lenient_prompt(
    question: str,
    results: list[SearchResult],
    history: list[ConversationTurn] | None = None,
) -> str:
    return build_retrieved_context_prompt(
        question,
        results,
        history=history,
        instructions=LENIENT_SYSTEM_INSTRUCTIONS,
        answer_label="Answer:",
        strict_history=False,
    )


def build_retrieved_context_prompt(
    question: str,
    results: list[SearchResult],
    *,
    history: list[ConversationTurn] | None,
    instructions: str,
    answer_label: str,
    strict_history: bool,
) -> str:
    context = format_retrieved_context(results)
    history_text = format_conversation_history(history or [])
    history_section = ""
    if history_text:
        if strict_history:
            history_header = "Recent conversation (for resolving references only; do not cite it as evidence):"
        else:
            history_header = "Recent conversation (for context and reference resolution; do not cite it as primary evidence):"
        history_section = f"{history_header}\n{history_text}\n\n"
    return (
        f"{instructions}\n"
        f"{history_section}"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{answer_label}"
    )


def format_retrieved_context(results: list[SearchResult]) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        metadata = [
            f"Chunk ID: {result.chunk_id}",
            f"Source document: {result.path}",
            f"Location: {location_label(result)}",
        ]
        if result.retrieval_source:
            metadata.append(f"Retrieval source: {result.retrieval_source}")
        if result.context_role and result.context_role != "primary":
            metadata.append(f"Context role: {result.context_role}")
        rank_parts = []
        if result.keyword_rank is not None:
            rank_parts.append(f"keyword={result.keyword_rank}")
        if result.vector_rank is not None:
            rank_parts.append(f"vector={result.vector_rank}")
        if result.fused_rank is not None:
            rank_parts.append(f"fusion={result.fused_rank}")
        if result.rerank_score is not None:
            rank_parts.append(f"rerank={result.rerank_score:.3f}")
        if rank_parts:
            metadata.append(f"Ranks: {', '.join(rank_parts)}")
        if result.email_subject:
            metadata.append(f"Subject: {result.email_subject}")
        if result.email_sender:
            metadata.append(f"Sender: {result.email_sender}")
        if result.email_date:
            metadata.append(f"Date: {result.email_date}")
        if result.attachment_name:
            metadata.append(f"Attachment: {result.attachment_name}")
        metadata_text = "\n".join(metadata)
        header = citation_label(index, result)
        context_blocks.append(
            f"Evidence [{index}]\n"
            f"{header}\n"
            f"{metadata_text}\n\n"
            f"Raw chunk text:\n{result.chunk_text}"
        )

    if not context_blocks:
        return "No retrieved context."
    return "\n\n---\n\n".join(context_blocks)


def build_evidence_ledger_prompt(
    question: str,
    existing_ledger: list[dict[str, object]],
    results: list[SearchResult],
    *,
    batch_number: int,
    total_batches: int,
) -> str:
    existing = json.dumps(existing_ledger, indent=2, ensure_ascii=False) if existing_ledger else "[]"
    chunks = "\n\n---\n\n".join(
        format_ledger_chunk(index, result)
        for index, result in enumerate(results, start=1)
    )
    return (
        f"{EVIDENCE_LEDGER_INSTRUCTIONS}\n\n"
        f"Question:\n{question}\n\n"
        f"Existing ledger before batch {batch_number} of {total_batches}:\n{existing}\n\n"
        f"Current chunk batch {batch_number} of {total_batches}:\n{chunks}\n\n"
        "Valid JSON:"
    )


def build_ledger_final_answer_prompt(question: str, ledger: list[dict[str, object]]) -> str:
    return build_ledger_answer_prompt(
        question,
        ledger,
        instructions=LEDGER_FINAL_ANSWER_INSTRUCTIONS,
        answer_label="Answer with ledger citations:",
    )


def build_lenient_ledger_final_answer_prompt(question: str, ledger: list[dict[str, object]]) -> str:
    return build_ledger_answer_prompt(
        question,
        ledger,
        instructions=LENIENT_LEDGER_FINAL_ANSWER_INSTRUCTIONS,
        answer_label="Answer:",
    )


def build_ledger_answer_prompt(
    question: str,
    ledger: list[dict[str, object]],
    *,
    instructions: str,
    answer_label: str,
) -> str:
    ledger_text = "\n".join(format_ledger_entry(index, entry) for index, entry in enumerate(ledger, start=1))
    if not ledger_text:
        ledger_text = "No ledger entries."
    return (
        f"{instructions}\n\n"
        f"Evidence ledger:\n{ledger_text}\n\n"
        f"Question: {question}\n\n"
        f"{answer_label}"
    )


def format_ledger_chunk(index: int, result: SearchResult) -> str:
    metadata = [
        f"Batch evidence [{index}]",
        f"Chunk ID: {result.chunk_id}",
        f"Source document: {result.path}",
        f"Location: {location_label(result)}",
    ]
    if result.email_subject:
        metadata.append(f"Subject: {result.email_subject}")
    if result.email_sender:
        metadata.append(f"Sender: {result.email_sender}")
    if result.email_date:
        metadata.append(f"Date: {result.email_date}")
    if result.attachment_name:
        metadata.append(f"Attachment: {result.attachment_name}")
    return "\n".join(metadata) + f"\n\nRaw chunk text:\n{result.chunk_text}"


def format_ledger_entry(index: int, entry: dict[str, object]) -> str:
    claim = str(entry.get("claim", "")).strip()
    chunk_id = str(entry.get("source_chunk_id", "")).strip()
    source = str(entry.get("source_path", "")).strip()
    location = str(entry.get("source_location", "")).strip()
    support_type = str(entry.get("support_type", "")).strip()
    confidence = str(entry.get("confidence", "")).strip()
    why = str(entry.get("why_relevant", "")).strip()
    quote = str(entry.get("quoted_support", "")).strip()
    lines = [f"[E{index}] {claim}"]
    details = []
    if support_type:
        details.append(f"type={support_type}")
    if confidence:
        details.append(f"confidence={confidence}")
    if chunk_id:
        details.append(f"chunk={chunk_id}")
    if source:
        details.append(f"source={source}")
    if location:
        details.append(f"location={location}")
    if details:
        lines.append("; ".join(details))
    if why:
        lines.append(f"relevance: {why}")
    if quote:
        lines.append(f"quote: {quote}")
    return "\n".join(lines)


def format_conversation_history(history: list[ConversationTurn]) -> str:
    if not history:
        return ""

    blocks = []
    total_chars = 0
    recent_history = history[-MAX_HISTORY_TURNS:]
    for turn in reversed(recent_history):
        question = truncate_text(normalize_prompt_text(turn.question), MAX_HISTORY_QUESTION_CHARS)
        answer = truncate_text(normalize_prompt_text(turn.answer), MAX_HISTORY_ANSWER_CHARS)
        block = f"User: {question}\nAssistant: {answer}"
        if blocks and total_chars + len(block) > MAX_HISTORY_CHARS:
            break
        if not blocks and len(block) > MAX_HISTORY_CHARS:
            block = truncate_text(block, MAX_HISTORY_CHARS)
        blocks.append(block)
        total_chars += len(block)
    blocks.reverse()
    return "\n\n".join(blocks)


def normalize_prompt_text(value: str) -> str:
    return " ".join(value.split())


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."


def location_label(result: SearchResult) -> str:
    return result_location_label(result)
