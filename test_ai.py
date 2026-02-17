import requests

prompt = "Analiza ventas mensuales: 1000, 2000, 3000, 1500"

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "deepseek-llm", "prompt": prompt, "stream": False},
)

print(response.json()["response"])
