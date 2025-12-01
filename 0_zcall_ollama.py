import requests
import json
import time


inicio = time.time()

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.1:8b",
    "system": "Eres un experto en mecánica automotriz.",
    "prompt": "¿Qué es el sensor TPS y qué hace en un motor de gasolina?",
    "keep_alive": -1,
    "stream": False
}

response = requests.post(url, json=data)

# Leemos los chunks que llegan en tiempo real
#for line in response.iter_lines():
#    if line:
#        chunk = json.loads(line.decode('utf-8'))
#        # Cada chunk tiene: {"response": "...", "done": false}
#        if "response" in chunk:
#            print(chunk["response"], end="", flush=True)
#        if chunk.get("done"):
#            break

fin = time.time()

print(response.text)
print(f"\nTiempo total: {fin-inicio:.2f} segundos")
