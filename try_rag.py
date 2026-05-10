import chromadb
from sentence_transformers import SentenceTransformer

# 1. Inizializza il modello BGE (usiamo m3 per supporto multilingua e qualità alta)
# Se hai poca RAM, puoi usare 'BAAI/bge-small-en-v1.5' (solo inglese) 
model_name = 'BAAI/bge-m3'
model = SentenceTransformer(model_name)

# 2. Configura il VectorDB (ChromaDB)
# 'path="./my_vectordb"' salva i dati su disco; usa ':memory:' per dati volatili
client = chromadb.PersistentClient(path="./my_local_db")

# 3. Crea o recupera una collezione
collection = client.get_or_create_collection(name="documenti_aziendali")

# 4. Preparazione dei dati
documents = [
    "Il regolamento aziendale prevede ferie illimitate per i dipendenti.",
    "La cucina dell'ufficio viene pulita ogni venerdì sera.",
    "Il nuovo progetto di intelligenza artificiale partirà a settembre."
]
ids = ["doc1", "doc2", "doc3"]

# 5. Generazione degli Embeddings
# BGE suggerisce di aggiungere un'istruzione per i task di retrieval (facoltativo con m3)
embeddings = model.encode(documents, normalize_embeddings=True)

# 6. Caricamento nel VectorDB
collection.add(
    embeddings=embeddings.tolist(), # Converte array numpy in lista
    documents=documents,
    ids=ids
)

print(f"Caricati {len(documents)} documenti nel database locale.")

# --- Esempio di Ricerca ---
query = "Come funzionano le vacanze?"
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("\nRisultato della ricerca:")
print(f"Testo trovato: {results['documents'][0][0]}")
print(f"Distanza (punteggio): {results['distances'][0][0]}")
