"""
LangGraph Ingestion Pipeline
=============================
Full pipeline: fetch live articles from EventRegistry, translate non-English
content, chunk, embed, and upsert into Supabase (pgvector). Deterministic
doc_ids ensure re-runs never create duplicates.

Run:
    python -m workflows.langgraph_pipeline
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from eventregistry import EventRegistry, QueryArticlesIter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from langgraph.graph import END, StateGraph

COMPANIES = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Meta AI",
    "Microsoft AI",
    "Mistral AI",
    "Cohere",
    "Hugging Face",
]


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependency issues when running standalone
# ---------------------------------------------------------------------------


def _get_settings():
    from backend.config import get_settings
    return get_settings()


def _get_embeddings():
    from backend.services.embeddings import get_embeddings
    return get_embeddings()


def _get_supabase_client():
    from backend.services.feedback_rl import get_supabase_client
    return get_supabase_client()


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    raw_articles: List[Dict[str, Any]]
    translated_articles: List[Dict[str, Any]]
    documents: List[Document]
    embeddings: List[List[float]]
    indexed_count: int
    errors: List[str]
    settings: Optional[Any]


# ---------------------------------------------------------------------------
# Node: fetch_articles
# ---------------------------------------------------------------------------


def fetch_articles(state: PipelineState) -> PipelineState:
    """
    Fetch the latest articles from EventRegistry for each company in COMPANIES
    and save them to the configured JSON path. Downstream load_articles reads
    this file, so the rest of the pipeline is unchanged.
    """
    settings = state.get("settings") or _get_settings()
    date_end = datetime.utcnow().strftime("%Y-%m-%d")
    date_start = (datetime.utcnow() - timedelta(days=settings.news_lookback_days)).strftime("%Y-%m-%d")
    print(f"[fetch_articles] Fetching articles from {date_start} to {date_end}")

    try:
        er = EventRegistry(apiKey=settings.event_registry_api_key, allowUseOfArchive=True)
    except Exception as exc:
        msg = f"[fetch_articles] Failed to initialise EventRegistry: {exc}"
        print(msg)
        return {**state, "errors": state["errors"] + [msg], "settings": settings}

    all_articles = []
    for company in COMPANIES:
        print(f"[fetch_articles]   Fetching: {company} ...")
        try:
            query = QueryArticlesIter(
                keywords=company,
                dateStart=date_start,
                dateEnd=date_end,
                lang=["eng", "spa", "fra", "deu", "zho"],
            )
            count = 0
            for art in query.execQuery(er, maxItems=settings.news_max_items_per_company):
                all_articles.append({
                    "source": art.get("source", {}).get("title", ""),
                    "company": company,
                    "title": art.get("title"),
                    "date": art.get("dateTime"),
                    "url": art.get("url"),
                    "content": art.get("body"),
                })
                count += 1
            print(f"[fetch_articles]     -> {count} articles")
        except Exception as exc:
            msg = f"[fetch_articles] Failed for company '{company}': {exc}"
            print(msg)
            state["errors"].append(msg)

    print(f"[fetch_articles] Total fetched: {len(all_articles)} articles")

    try:
        with open(settings.articles_json_path, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        print(f"[fetch_articles] Saved to {settings.articles_json_path}")
    except Exception as exc:
        msg = f"[fetch_articles] Failed to save JSON: {exc}"
        print(msg)
        return {**state, "errors": state["errors"] + [msg], "settings": settings}

    return {**state, "settings": settings}


# ---------------------------------------------------------------------------
# Node: load_articles
# ---------------------------------------------------------------------------


def load_articles(state: PipelineState) -> PipelineState:
    settings = state.get("settings") or _get_settings()
    path = settings.articles_json_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"[load_articles] Loaded {len(articles)} articles from {path}")
        return {**state, "raw_articles": articles, "settings": settings}
    except Exception as exc:
        msg = f"[load_articles] Failed to load {path}: {exc}"
        print(msg)
        return {
            **state,
            "raw_articles": [],
            "errors": state["errors"] + [msg],
            "settings": settings,
        }


# ---------------------------------------------------------------------------
# Node: translate_non_english
# ---------------------------------------------------------------------------


def translate_non_english(state: PipelineState) -> PipelineState:
    settings = state.get("settings") or _get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_translation_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    )

    translated = []
    for article in state["raw_articles"]:
        content = article.get("body") or article.get("content", "")
        try:
            lang = detect(content[:500]) if content else "en"
        except Exception:
            lang = "en"

        if lang != "en" and content:
            try:
                msg = HumanMessage(
                    content=(
                        "Translate the following text to English. "
                        "Return only the translation:\n\n"
                        + content[:3000]
                    )
                )
                response = llm.invoke([msg])
                article = {**article, "body": response.content, "original_lang": lang}
            except Exception as exc:
                print(f"[translate] Failed for article '{article.get('title', '')}': {exc}")

        translated.append(article)

    print(f"[translate_non_english] Processed {len(translated)} articles")
    return {**state, "translated_articles": translated}


# ---------------------------------------------------------------------------
# Node: chunk_documents
# ---------------------------------------------------------------------------


def chunk_documents(state: PipelineState) -> PipelineState:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3200,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for article in state["translated_articles"]:
        content = article.get("body") or article.get("content", "")
        if not content:
            continue

        url = article.get("url", "")
        title = article.get("title", "")
        source = article.get("source", {})
        company = article.get("company") or (
            source.get("title", "") if isinstance(source, dict) else ""
        )

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{url}_{i}".encode()).hexdigest()
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "url": url,
                        "title": title,
                        "company": company,
                        "chunk_index": i,
                        "doc_id": doc_id,
                    },
                )
            )

    print(
        f"[chunk_documents] Created {len(documents)} chunks "
        f"from {len(state['translated_articles'])} articles"
    )
    return {**state, "documents": documents}


# ---------------------------------------------------------------------------
# Node: generate_embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(state: PipelineState) -> PipelineState:
    emb_model = _get_embeddings()
    texts = [doc.page_content for doc in state["documents"]]

    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = emb_model.embed_documents(batch)
        embeddings.extend(batch_embs)

    print(f"[generate_embeddings] Generated {len(embeddings)} embeddings")
    return {**state, "embeddings": embeddings}


# ---------------------------------------------------------------------------
# Node: index_to_supabase
# ---------------------------------------------------------------------------


def index_to_supabase(state: PipelineState) -> PipelineState:
    """
    Upsert document chunks + pre-generated embeddings into Supabase.
    Uses doc_id as the conflict target so re-runs are idempotent.
    """
    settings = state.get("settings") or _get_settings()
    client = _get_supabase_client()

    documents = state["documents"]
    embeddings = state["embeddings"]
    indexed_count = 0
    errors = list(state["errors"])
    batch_size = 50  # smaller batches keep request size manageable

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        rows = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": emb,
                "doc_id": doc.metadata["doc_id"],
            }
            for doc, emb in zip(batch_docs, batch_embs)
        ]

        try:
            client.table(settings.documents_table).upsert(
                rows, on_conflict="doc_id"
            ).execute()
            indexed_count += len(batch_docs)
            print(
                f"[index] Batch {i // batch_size + 1}: "
                f"upserted {len(batch_docs)} docs (total {indexed_count})"
            )
        except Exception as exc:
            msg = f"[index] Batch {i // batch_size + 1} failed: {exc}"
            print(msg)
            errors.append(msg)

    print(f"[index_to_supabase] Total indexed: {indexed_count}")
    return {**state, "indexed_count": indexed_count, "errors": errors}


# ---------------------------------------------------------------------------
# Build the compiled pipeline
# ---------------------------------------------------------------------------


def build_pipeline():
    """Compile and return the LangGraph ingestion pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node("fetch_articles", fetch_articles)
    graph.add_node("load_articles", load_articles)
    graph.add_node("translate_non_english", translate_non_english)
    graph.add_node("chunk_documents", chunk_documents)
    graph.add_node("generate_embeddings", generate_embeddings)
    graph.add_node("index_to_supabase", index_to_supabase)

    graph.set_entry_point("fetch_articles")
    graph.add_edge("fetch_articles", "load_articles")
    graph.add_edge("load_articles", "translate_non_english")
    graph.add_edge("translate_non_english", "chunk_documents")
    graph.add_edge("chunk_documents", "generate_embeddings")
    graph.add_edge("generate_embeddings", "index_to_supabase")
    graph.add_edge("index_to_supabase", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "raw_articles": [],
        "translated_articles": [],
        "documents": [],
        "embeddings": [],
        "indexed_count": 0,
        "errors": [],
        "settings": None,
    }

    print("Starting LangGraph ingestion pipeline -> Supabase")
    print("=" * 60)
    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Articles loaded   : {len(final_state['raw_articles'])}")
    print(f"  Chunks created    : {len(final_state['documents'])}")
    print(f"  Indexed to Supabase: {final_state['indexed_count']}")
    if final_state["errors"]:
        print(f"  Errors ({len(final_state['errors'])}):")
        for err in final_state["errors"]:
            print(f"    - {err}")
    else:
        print("  Errors            : none")
