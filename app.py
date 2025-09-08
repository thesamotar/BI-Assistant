from fastapi import FastAPI
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
# Helper: Search Elasticsearch
# -----------------------
def search_es(query, top_k=5):
    query_vector = model.encode(query).tolist()
    response = es.search(
        index=ES_INDEX,
        body={
            "size": top_k,
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
            "content": source.get("content")
        })
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
        "sources": [d["url"] for d in docs]
    }