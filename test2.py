import chromadb
import requests
import json
import os

# setup
client = chromadb.PersistentClient(path="./indice")
collection = client.get_or_create_collection("file_prova")

NAS_PATH = "/Users/riccardoinfascelli/Desktop/Hackathon/PROVA"

# ── funzioni base ──────────────────────────────────────────

def get_embedding(testo):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": testo
    })
    return response.json()["embedding"]


import json

def get_summary(testo):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"""Return ONLY valid JSON. No markdown.

{{
  "summary": "...",
  "keywords": ["..."]
}}

Text:
{testo}
""",
        "stream": False
    })

    raw = response.json()["response"]

    start = raw.find("{")
    end = raw.rfind("}")
    return json.loads(raw[start:end+1])

def get_cartelle_dal_disco():
    return [d for d in os.listdir(NAS_PATH) if os.path.isdir(os.path.join(NAS_PATH, d))]


def get_cartella(summary, cartelle_esistenti):
    cartelle_str = "\n".join(cartelle_esistenti)
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"Cartelle disponibili:\n{cartelle_str}\n\nDescrizione file: {summary}\n\nIn quale cartella va questo file? Scegli SOLO una delle cartelle disponibili. Rispondi SOLO con il nome esatto della cartella, nessun testo aggiuntivo, nessun slash.",
        "stream": False
    })
    return response.json()["response"].strip()


# ── pipeline ingestion ─────────────────────────────────────

def indicizza_file(testo, nome_file, file_id):
    print(f"\n→ File: {nome_file}")

    summary_obj = get_summary(testo)
    summary = summary_obj["summary"] + " " + " ".join(summary_obj["keywords"])
    print(f"  Summary: {summary_obj['summary']}")

    vettore = get_embedding(summary)

    cartelle = get_cartelle_con_descrizioni()
    print(f"  Cartelle disponibili: {cartelle}")

    cartella = get_cartella(summary, cartelle)
    print(f"  Cartella proposta: {cartella}")

    # sposta il file nella cartella giusta
    dest = os.path.join(NAS_PATH, cartella, nome_file)
    print(f"  → Destinazione: {dest}")

    collection.add(
        ids=[file_id],
        embeddings=[vettore],
        documents=[summary],
        metadatas=[{"path": dest, "cartella": cartella}]
    )
    print(f"  ✓ Salvato in indice")
    return cartella

def get_cartelle_con_descrizioni():
    cartelle = [d for d in os.listdir(NAS_PATH) if os.path.isdir(os.path.join(NAS_PATH, d))]
    
    json_path = os.path.join(NAS_PATH, "cartelle.json")
    descrizioni = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            descrizioni = json.load(f)
    
    risultato = []
    for c in cartelle:
        if c in descrizioni:
            risultato.append(f"{c}: {descrizioni[c]}")
        else:
            risultato.append(c)
    
    return risultato


def get_cartella(summary, cartelle_con_descrizioni):
    cartelle_str = "\n".join(cartelle_con_descrizioni)
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"Cartelle disponibili:\n{cartelle_str}\n\nDescrizione file: {summary}\n\nIn quale cartella va questo file? Scegli SOLO una delle cartelle disponibili. Rispondi SOLO con il nome esatto della cartella, nessun testo aggiuntivo, nessun slash.",
        "stream": False
    })
    return response.json()["response"].strip()


# ── test ───────────────────────────────────────────────────

if __name__ == "__main__":
    documenti = [
        ("Fattura n. 2024-089 del 15 marzo 2024. Cliente: Mario Rossi. Prestazione: visita odontoiatrica. Importo: 250 euro.", "fattura_dentista.txt", "file_001"),
        ("Ricetta: Pasta al pomodoro. Ingredienti: 320g spaghetti, 400g pomodori, aglio, basilico, olio extravergine. Procedimento: soffriggere aglio, aggiungere pomodori, cuocere pasta al dente.", "pasta_pomodoro.txt", "file_002"),
        ("Estratto conto gennaio 2024. Saldo iniziale: 3200 euro. Entrate: stipendio 1800 euro. Uscite: affitto 600 euro, utenze 120 euro. Saldo finale: 4280 euro.", "estratto_conto.txt", "file_003"),
    ]

    for testo, nome, fid in documenti:
        indicizza_file(testo, nome, fid)