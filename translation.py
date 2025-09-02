import json
import google.generativeai as genai
from langdetect import detect

# 🔑 Configure Gemini with your API key
genai.configure(api_key="AIzaSyDh524HLfa0vb2jffe2NCXvfCe227Ufpdo")

# Load your dataset
with open("genai_competitors_articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Initialize Gemini model (text-only)
model = genai.GenerativeModel("gemini-1.5-flash")

translated_articles = []

for article in articles:
    content = article.get("content", "")
    try:
        # Detect language
        lang = detect(content) if content else "en"

        if lang != "en":
            # Translate using Gemini
            prompt = f"Translate the following text into English, keeping meaning intact:\n\n{content}"
            response = model.generate_content(prompt)
            translated_text = response.text.strip()
        else:
            translated_text = content

        cleaned_article = {
            "source": article.get("source"),
            "company": article.get("company"),
            "title": article.get("title"),
            "date": article.get("date"),
            "url": article.get("url"),
            "content": translated_text
        }
        translated_articles.append(cleaned_article)

    except Exception as e:
        print(f"⚠️ Error processing article {article.get('title')}: {e}")

# Save translated dataset
with open("genai_competitors_articles_translated.json", "w", encoding="utf-8") as f:
    json.dump(translated_articles, f, indent=2, ensure_ascii=False)

print(f"✅ Translation complete with Gemini. Saved {len(translated_articles)} articles to genai_competitors_articles_translated.json")