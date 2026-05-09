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

def indicizza_cartella(cartella_input):
    print(f"\n→ Leggo file da: {cartella_input}")
    
    files = [f for f in os.listdir(cartella_input) if f.endswith(".txt")]
    print(f"  Trovati {len(files)} file: {files}")
    
    for i, nome_file in enumerate(files):
        path_completo = os.path.join(cartella_input, nome_file)
        
        with open(path_completo, "r", encoding="utf-8") as f:
            testo = f.read()
        
        indicizza_file(testo, nome_file, f"file_{i:03d}")


# ── pipeline ingestion ─────────────────────────────────────

def indicizza_file(testo, nome_file, file_id):
    print(f"\n→ File: {nome_file}")

    summary_obj = get_summary(testo)
    summary = summary_obj["summary"] + " " + " ".join(summary_obj["keywords"])
    print(f"  Summary: {summary_obj['summary']}")

    vettore = get_embedding(summary)

    cartelle = get_albero_cartelle()
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

def get_albero_cartelle():
    albero = []
    json_path = os.path.join(NAS_PATH, "cartelle.json")
    descrizioni = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            descrizioni = json.load(f)

    for root, dirs, files in os.walk(NAS_PATH):
        # ignora la cartella FileDaSistemare
        dirs[:] = [d for d in dirs if d != "FileDaSistemare"]
        
        path_relativo = os.path.relpath(root, NAS_PATH)
        if path_relativo == ".":
            continue
        
        desc = descrizioni.get(path_relativo, "")
        if desc:
            albero.append(f"{path_relativo}: {desc}")
        else:
            albero.append(path_relativo)

    return albero


# ── test ───────────────────────────────────────────────────

if __name__ == "__main__":
    cartella_input = "/Users/riccardoinfascelli/Desktop/Hackathon/FileDaSistemare"
    indicizza_cartella(cartella_input)