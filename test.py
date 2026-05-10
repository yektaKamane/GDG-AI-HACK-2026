import chromadb
import requests
import json

# setup
client = chromadb.PersistentClient(path="./indice")
collection = client.get_or_create_collection("file")

# ── funzioni base ──────────────────────────────────────────

def get_embedding(testo):
    response = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": testo
    })
    return response.json()["embedding"]


def get_summary(testo):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"Analizza questo testo e restituisci SOLO un JSON con due campi: summary (2-3 frasi che descrivono cosa è, di cosa parla, e includono TUTTE le entità rilevanti come nomi, date, luoghi, importi) e keywords (lista di 8-12 parole chiave inclusi nomi propri, date, importi). Nessun testo aggiuntivo, solo JSON.\n\nTesto: '{testo}'",
        "stream": False
    })
    raw = response.json()["response"]
    return json.loads(raw)


def get_cartelle_esistenti():
    risultati = collection.get()
    cartelle = set()
    for meta in risultati["metadatas"]:
        cartelle.add(meta["cartella"])
    return list(cartelle)


def get_cartella(summary, cartelle_esistenti):
    cartelle_str = "\n".join(cartelle_esistenti)
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:1b",
        "prompt": f"Cartelle esistenti:\n{cartelle_str}\n\nDescrizione file: {summary}\n\nIn quale cartella va questo file? Puoi usare una esistente o crearne una nuova. Rispondi SOLO con il path esatto, stesso formato delle cartelle esistenti, esempio: Lavoro/Fatture. Nessun testo aggiuntivo, nessun slash iniziale.",
        "stream": False
    })
    return response.json()["response"].strip()


# ── pipeline ingestion ─────────────────────────────────────

def indicizza_file(testo, path, file_id):
    print(f"\n→ Indicizzazione: {path}")

    summary_obj = get_summary(testo)
    summary = summary_obj["summary"] + " " + " ".join(summary_obj["keywords"])
    print(f"  Summary: {summary_obj['summary']}")

    vettore = get_embedding(summary)
    print(f"  Embedding: {len(vettore)} dimensioni")

    cartelle = get_cartelle_esistenti()
    cartella = get_cartella(summary, cartelle)
    print(f"  Cartella proposta: {cartella}")

    collection.add(
        ids=[file_id],
        embeddings=[vettore],
        documents=[summary],
        metadatas=[{"path": path, "cartella": cartella}]
    )
    print(f"  ✓ Salvato in indice")
    return cartella


# ── pipeline retrieval ─────────────────────────────────────

def cerca_file(query, n=3):
    print(f"\n→ Ricerca: '{query}'")

    vettore_query = get_embedding(query)
    risultati = collection.query(
        query_embeddings=[vettore_query],
        n_results=min(n, collection.count())
    )

    print(f"  Trovati {len(risultati['documents'][0])} risultati:")
    for doc, meta in zip(risultati["documents"][0], risultati["metadatas"][0]):
        print(f"  - {meta['path']} ({meta['cartella']})")
    
    return risultati


# ── test ───────────────────────────────────────────────────

if __name__ == "__main__":
    # ingestion
    indicizza_file(
        testo="Fattura n. 2024-089 del 15 marzo 2024. Cliente: Mario Rossi, Via Roma 12, Milano. Prestazione: visita specialistica odontoiatrica. Importo: 250 euro. Pagamento: bonifico bancario entro 30 giorni.",
        path="/nas/documenti/fattura_089.pdf",
        file_id="file_001"
    )

    # retrieval
    cerca_file("fattura del dentista")