from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np

# -----------------------
# CONFIG
# -----------------------
ES_INDEX = "bi-assistant-1729"
ES_URL = "https://my-elasticsearch-project-bff9ea.es.us-central1.gcp.elastic.cloud:443"

es = Elasticsearch(
    ES_URL,
    api_key="ZVZyWEM1a0JMSUdVOUlUSnJrUU06SnhxN19DR0Q0RUZGbGUxZ25NdEFEdw=="
)

# Load the same embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------
# Search Function
# -----------------------
def search_es(query, top_k=5):
    # Encode query
    query_vector = model.encode(query).tolist()

    # Elasticsearch vector search
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
        results.append(
            {
                "score": hit["_score"],
                "title": source.get("title"),
                "company": source.get("company"),
                "url": source.get("url"),
                "snippet": source.get("content")[:200] + "...",
            }
        )
    return results


# -----------------------
# Example Test
# -----------------------
if __name__ == "__main__":
    query = "What are the latest AI product launches by OpenAI?"
    results = search_es(query, top_k=5)

    print("\n🔍 Query:", query)
    print("Top Results:\n")
    for r in results:
        print(f"- ({r['company']}) {r['title']}")
        print(f"  URL: {r['url']}")
        print(f"  Snippet: {r['snippet']}")
        print(f"  Score: {r['score']:.2f}\n")