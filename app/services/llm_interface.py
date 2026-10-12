from langchain_ollama import OllamaLLM
from app.services.memorymanager import memory_manager
from app.services.vector_store import VectorStore


llm = OllamaLLM(model="llama3")


def generate_explanation(
    session_id: str,
    analysis_report: str,
    user_question: str = None,
    rag_query: str = None,
) -> str:
    """
    Generate a graph-grounded natural language explanation.

    FIX: rag_query parameter added.
      endpoints.py now passes graph_summary (structured indicator/category
      string) as rag_query instead of the full analysis_report. This means
      the vector store retrieves genuinely similar *past* cases rather than
      the just-stored current report (which would always be the top-1 hit).

      If rag_query is None, falls back to analysis_report for backwards
      compatibility (e.g. direct calls from tests or other callers).

    FIX: retrieved chunks are de-duplicated against the current report
      so the LLM does not see the same content twice in one prompt.
    """

    vs = VectorStore(session_id)

    query = rag_query if rag_query else analysis_report
    retrieved_chunks = vs.search(query, top_k=4)   # fetch one extra to allow dedup

    # De-duplicate: skip any retrieved chunk whose text is near-identical
    # to the current report (avoids circular self-retrieval).
    report_preview = analysis_report[:200]          # compare only the header portion
    unique_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("text", "")[:200] != report_preview
    ][:3]                                           # keep at most 3 after dedup

    context_text = "\n---\n".join(
        chunk["text"] for chunk in unique_chunks
    ) if unique_chunks else "No prior similar cases in session memory."

    history = memory_manager.get_history(session_id)

    # Trim history: only include last 6 turns to stay within LLM context window.
    # The full history is persisted in SQLite; this is a prompt budget decision.
    recent_history = history[-6:] if len(history) > 6 else history
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in recent_history
    )

    prompt = f"""You are a cybersecurity expert explaining phishing detection results.

You are given a STRUCTURED reasoning output from a phishing detection engine.

Your job is NOT to guess — only explain the reasoning provided.

Conversation History (last {len(recent_history)} turns):
{history_text}

Current Analysis Report:
{analysis_report}

Similar Past Cases from Session Memory:
{context_text}

User Question:
{user_question if user_question else "Explain why this message is or is not phishing based on the reasoning graph."}

Instructions:
- Explain each triggered indicator clearly and connect it to its risk category
- Reference the risk score and verdict explicitly
- If cross-modal edges are present (contradiction or reinforcement), explain their effect
- Keep explanation concise and factual — do not speculate beyond the graph
"""

    response = llm.invoke(prompt)

    memory_manager.add_message(session_id, "assistant", response)

    return response.strip()