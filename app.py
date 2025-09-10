from fastapi import FastAPI, Body
from datetime import datetime
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import google.generativeai as genai

# -----------------------
# CONFIG
# -----------------------
ES_INDEX = "bi-assistant-1729"
ES_URL = "https://my-elasticsearch-project-bff9ea.es.us-central1.gcp.elastic.cloud:443"

es = Elasticsearch(
    ES_URL,
    api_key="ZVZyWEM1a0JMSUdVOUlUSnJrUU06SnhxN19DR0Q0RUZGbGUxZ25NdEFEdw=="
)

# Embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gemini API (you can swap with OpenAI if preferred)
genai.configure(api_key="AIzaSyDh524HLfa0vb2jffe2NCXvfCe227Ufpdo")
llm = genai.GenerativeModel("gemini-2.5-pro")

# -----------------------
# FastAPI Setup
# -----------------------
app = FastAPI(title="BI Assistant", version="1.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# -----------------------
# Helper: Search Elasticsearch (with feedback adjustment)
# -----------------------
def search_es(query, top_k=5):
    query_vector = model.encode(query).tolist()
    response = es.search(
        index=ES_INDEX,
        body={
            "size": top_k * 2,  # fetch more than needed, we’ll re-rank
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector},
                    },
                }
            },
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "title": source.get("title"),
            "company": source.get("company"),
            "url": source.get("url"),
            "content": source.get("content"),
            "score": hit["_score"],
        })

    # ---- Feedback-Aware Adjustment ----
    for r in results:
        url = r["url"]
        fb_stats = es.count(index=FEEDBACK_INDEX, body={"query": {"bool": {"must": [
            {"term": {"sources.keyword": url}},
            {"term": {"feedback": "positive"}}
        ]}}})
        pos = fb_stats["count"]

        fb_stats = es.count(index=FEEDBACK_INDEX, body={"query": {"bool": {"must": [
            {"term": {"sources.keyword": url}},
            {"term": {"feedback": "negative"}}
        ]}}})
        neg = fb_stats["count"]

        # simple adjustment: each positive adds +0.1, each negative -0.1
        r["score"] += (pos - neg) * 0.1

    # re-rank by adjusted score
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    return results


# -----------------------
# Endpoint: /ask
# -----------------------
@app.post("/ask")
async def ask(req: QueryRequest):
    # Step 1: Search ES
    docs = search_es(req.query, req.top_k)

    # Step 2: Build context for LLM
    context = "\n\n".join(
        [f"Source: {d['url']}\nContent: {d['content']}" for d in docs]
    )

    # Step 3: Prompt LLM
    prompt = f"""
    You are a business intelligence assistant.
    Answer the user query using the context below.
    Always include citations from the sources provided.

    User Query: {req.query}

    Context:
    {context}
    """

    response = llm.generate_content(prompt)

    return {
        "query": req.query,
        "answer": response.text,
        "sources": [d["url"] for d in docs],
        "scores": [d["score"] for d in docs]
    }

# -----------------------------
# Feedback Index
# -----------------------------
FEEDBACK_INDEX = "bi-assistant-feedback"

# Create feedback index if not exists
if not es.indices.exists(index=FEEDBACK_INDEX):
    es.indices.create(
        index=FEEDBACK_INDEX,
        body={
            "mappings": {
                "properties": {
                    "query": {"type": "text"},
                    "answer": {"type": "text"},
                    "sources": {"type": "keyword"},
                    "feedback": {"type": "keyword"},  # "positive" or "negative"
                    "timestamp": {"type": "date"}
                }
            }
        }
    )

@app.post("/feedback")
async def feedback(
    query: str = Body(...),
    answer: str = Body(...),
    sources: list = Body(...),
    feedback: str = Body(...)
):
    """
    Store user feedback (👍/👎) for answers.
    """
    doc = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "feedback": feedback,
        "timestamp": datetime.utcnow()
    }
    es.index(index=FEEDBACK_INDEX, body=doc)
    return {"status": "success", "message": "Feedback stored."}