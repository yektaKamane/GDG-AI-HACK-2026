import requests
import sys

def get_titolo(testo):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"Riassumi il contenuto di questo testo in una o due parole, adatte come nome di una cartella. Rispondi SOLO con le parole, niente altro.\n\nTesto: '{testo}'",
        "stream": False
    })
    return response.json()["response"].strip()

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    testo = f.read()

print(get_titolo(testo))
