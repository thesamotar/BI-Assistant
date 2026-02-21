from eventregistry import EventRegistry, QueryArticlesIter
from datetime import datetime, timedelta
import json

# Create client (sign up at newsapi.ai for a free API key)
er = EventRegistry(apiKey="4e211009-aea6-4991-ad67-b00ef01a4f51", allowUseOfArchive=True)

# Date range: last 30 days
date_end = datetime.utcnow().strftime("%Y-%m-%d")
date_start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
print(f"Fetching articles from {date_start} to {date_end}")

# Major GenAI competitors
companies = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Meta AI",
    "Microsoft AI",
    "Mistral AI",
    "Cohere",
    "Hugging Face",
]

all_articles = []

for name in companies:
    print(f"  Fetching: {name} ...")
    query = QueryArticlesIter(
        keywords=name,
        dateStart=date_start,
        dateEnd=date_end,
        lang=["eng", "spa", "fra", "deu", "zho"],
    )
    count = 0
    for art in query.execQuery(er, maxItems=50):
        all_articles.append({
            "source": art.get("source", {}).get("title", ""),
            "company": name,
            "title": art.get("title"),
            "date": art.get("dateTime"),
            "url": art.get("url"),
            "content": art.get("body"),
        })
        count += 1
    print(f"    -> {count} articles")

print(f"\nTotal fetched: {len(all_articles)} articles.")

# Save to JSON for downstream pipeline
with open("genai_competitors_articles.json", "w", encoding="utf-8") as f:
    json.dump(all_articles, f, indent=2, ensure_ascii=False)

print("Saved to genai_competitors_articles.json")
