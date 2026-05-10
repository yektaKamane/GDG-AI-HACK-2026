import sys
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    # 1. Controllo argomenti
    if len(sys.argv) < 2:
        print("Uso: python query.py \"tua domanda tra virgolette\"")
        sys.exit(1)

    query_text = sys.argv[1]

    # 2. Inizializzazione (stesso modello usato per l'ingest)
    # Nota: BGE-M3 è pesante, la prima volta caricherà per qualche secondo
    model = SentenceTransformer('BAAI/bge-m3')
    
    # 3. Connessione al DB locale
    client = chromadb.PersistentClient(path="./my_local_db")
    
    try:
        collection = client.get_collection(name="documenti_aziendali")
    except Exception:
        print("Errore: Il database non esiste. Esegui prima lo script di ingest.")
        sys.exit(1)

    # 4. Encoding della query
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    # 5. Ricerca (n_results=3 restituisce i 3 frammenti più simili)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # 6. Formattazione Output
    print(f"\n--- Risultati per: '{query_text}' ---\n")
    
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Più la distanza è bassa, più il risultato è simile (0.0 = identico)
        score = round((1 - distance) * 100, 2) 
        
        print(f"[{i+1}] Fonte: {metadata['source']} (Pagina: {metadata['page']})")
        print(f"    Score di affinità: {score}%")
        print(f"    Testo: {doc[:300]}...") # Mostra i primi 300 caratteri
        print("-" * 50)

if __name__ == "__main__":
    main()
