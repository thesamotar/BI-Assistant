from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import json

# -----------------------
# CONFIG
# -----------------------
JSON_FILE = "genai_competitors_articles_translated.json"
ES_INDEX = "bi-assistant-1729"

# Replace this with your actual Elasticsearch URL
ES_URL = "https://my-elasticsearch-project-bff9ea.es.us-central1.gcp.elastic.cloud:443"

# # If your ES cluster requires authentication:
# # Option 1: Basic Auth (username/password)
# es = Elasticsearch(
#     ES_URL,
#     basic_auth=("elastic", "your_password_here")
# )

# Option 2: API Key Auth (if preferred, comment above and use this)
es = Elasticsearch(
    ES_URL,
    api_key="ZVZyWEM1a0JMSUdVOUlUSnJrUU06SnhxN19DR0Q0RUZGbGUxZ25NdEFEdw=="
)

# -----------------------
# Load Sentence-BERT model
# -----------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  
# → 384-d embeddings


# -----------------------
# Chunking function
# -----------------------
def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into overlapping chunks of words."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap for context
    
    return chunks

# -----------------------
# Step 1: Delete old index and recreate
# -----------------------
if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)

es.indices.create(
    index=ES_INDEX,
    body={
        "mappings": {
            "properties": {
                "source": {"type": "keyword"},
                "company": {"type": "keyword"},
                "title": {"type": "text"},
                "date": {"type": "date"},
                "url": {"type": "keyword"},
                "content": {"type": "text"},
                "chunk_id": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": 384}
            }
        }
    }
)

# -----------------------
# Load JSON articles
# -----------------------
with open(JSON_FILE, "r", encoding="utf-8") as f:
    articles = json.load(f)

# -----------------------
# Step 3: Generate chunked embeddings & index
# -----------------------
doc_id = 0  # unique id across chunks
for article_id, article in enumerate(articles):
    content = article.get("content", "")
    if not content.strip():
        continue

    # Split into chunks
    chunks = chunk_text(content, chunk_size=800, overlap=100)

    for chunk_id, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        doc = {
            "source": article.get("source"),
            "company": article.get("company"),
            "title": article.get("title"),
            "date": article.get("date"),
            "url": article.get("url"),
            "content": chunk,         # only this chunk’s text
            "chunk_id": chunk_id,     # so we know which part of article
            "embedding": embedding,
        }

        es.index(index=ES_INDEX, id=doc_id, body=doc)
        doc_id += 1

print(f"✅ Re-indexed {doc_id} chunks into Elasticsearch index: {ES_INDEX}")