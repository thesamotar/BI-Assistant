import requests

BASE_URL = "http://127.0.0.1:8000"

# Step 1: Ask a question
query = "What are the latest AI product launches by OpenAI?"
ask_res = requests.post(f"{BASE_URL}/ask", json={"query": query})
ask_data = ask_res.json()
print("🔎 First Answer:")
print(ask_data)

# Step 2: Send feedback (simulate 👍 for first source, 👎 for second if available)
sources = ask_data.get("sources", [])
if sources:
    # Positive feedback for the first source
    fb1 = requests.post(f"{BASE_URL}/feedback", json={
        "query": query,
        "answer": ask_data["answer"],
        "sources": [sources[0]],
        "feedback": "positive"
    })
    print("✅ Feedback (positive) stored for:", sources[0], fb1.json())

    if len(sources) > 1:
        fb2 = requests.post(f"{BASE_URL}/feedback", json={
            "query": query,
            "answer": ask_data["answer"],
            "sources": [sources[1]],
            "feedback": "negative"
        })
        print("❌ Feedback (negative) stored for:", sources[1], fb2.json())

# Step 3: Ask again (should see re-ranking based on feedback)
ask_res2 = requests.post(f"{BASE_URL}/ask", json={"query": query})
ask_data2 = ask_res2.json()
print("\n🔄 Second Answer (after feedback):")
print(ask_data2)