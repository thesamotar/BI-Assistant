from functools import lru_cache
from typing import List, Dict, Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

from backend.config import get_settings
from backend.services.embeddings import get_embeddings
from backend.services.feedback_rl import get_supabase_client, get_feedback_scores


BI_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a business intelligence assistant.
Answer the user's query using ONLY the context provided below.
Always cite sources using [URL] notation when referencing information.
Be concise, factual, and direct. If the context does not contain enough
information to answer the question, say so clearly.

Context:
{context}""",
        ),
        ("human", "{query}"),
    ]
)


class FeedbackAwareRetriever(BaseRetriever):
    """
    Retriever that calls the match_documents Supabase RPC directly,
    then re-ranks results by combining vector similarity with UCB1 bandit
    scores derived from historical user feedback.
    """

    top_k: int = Field(default=5)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        settings = get_settings()
        client = get_supabase_client()
        emb_model = get_embeddings()

        # 1. Embed the query
        query_embedding = emb_model.embed_query(query)

        # 2. Fetch 2x candidates via match_documents RPC
        response = client.rpc(
            settings.match_function,
            {
                "query_embedding": query_embedding,
                "match_count": self.top_k * 2,
                "filter": {},
            },
        ).execute()

        # 3. Build (Document, vector_score) pairs
        docs_with_scores: List[tuple] = []
        for row in response.data:
            doc = Document(
                page_content=row["content"],
                metadata=row.get("metadata") or {},
            )
            docs_with_scores.append((doc, float(row.get("similarity", 0.0))))

        # 4. Apply UCB1 re-ranking
        urls = [doc.metadata.get("url", "") for doc, _ in docs_with_scores]
        ucb_scores = get_feedback_scores(urls)

        ranked: List[tuple] = []
        for doc, vector_score in docs_with_scores:
            url = doc.metadata.get("url", "")
            ucb1 = ucb_scores.get(url, 0.0)
            final_score = vector_score + ucb1
            doc.metadata["vector_score"] = round(vector_score, 4)
            doc.metadata["ucb1_score"] = round(ucb1, 4)
            doc.metadata["final_score"] = round(final_score, 4)
            ranked.append((doc, final_score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


@lru_cache()
def get_llm() -> ChatGoogleGenerativeAI:
    """Cached Gemini LLM instance."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )


def run_rag_pipeline(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG pipeline:
      1. Retrieve feedback-aware docs from Supabase (match_documents RPC).
      2. Build context block.
      3. Run LCEL chain (prompt | LLM | parser).
      4. Return answer, sources, scores, and model name.
    """
    settings = get_settings()
    retriever = FeedbackAwareRetriever(top_k=top_k)
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('url', 'unknown')}\nContent: {doc.page_content}"
        for doc in docs
    )

    chain = BI_ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})

    return {
        "answer": answer,
        "sources": [doc.metadata.get("url", "") for doc in docs],
        "scores": [doc.metadata.get("final_score", 0.0) for doc in docs],
        "model": settings.gemini_model,
    }
