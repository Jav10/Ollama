import ollama

response = ollama.chat(
    model="qwen3-vl:8b",
    messages=[
        {
            "role": "user",
            "content": "Describe esta imagen y responde en español.",
            "images": [r"docs\imagen.jpg"]
        }
    ]
)

print(response["message"]["content"])
