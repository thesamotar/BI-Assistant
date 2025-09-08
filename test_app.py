import requests

res = requests.post("http://127.0.0.1:8000/ask",
                    json={"query": "What are the latest AI product launches by OpenAI?"})
print(res.json())