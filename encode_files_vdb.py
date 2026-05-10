import os
import sys
from pathlib import Path

# Nuovi import post-migrazione LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import chromadb

def main():
    # 1. Controllo argomenti
    if len(sys.argv) < 2:
        print("Uso: python ingest.py <percorso_cartella_documenti>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Errore: '{target_dir}' non è una directory valida.")
        sys.exit(1)

    # 2. Inizializzazione Modello e DB
    print("--- Inizializzazione modello BGE-M3 (CPU/GPU) ---")
    model = SentenceTransformer('BAAI/bge-m3')
    
    client = chromadb.PersistentClient(path="./my_local_db")
    collection = client.get_or_create_collection(name="documenti_aziendali")

    # 3. Configurazione Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 4. Scansione file
    extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader
    }

    files = [f for f in Path(target_dir).iterdir() if f.suffix in extensions]
    
    if not files:
        print("Nessun file supportato (.pdf, .txt, .docx) trovato.")
        return

    print(f"Trovati {len(files)} file. Inizio elaborazione...")

    for file_path in files:
        try:
            print(f"Lettura: {file_path.name}...")
            loader_class = extensions[file_path.suffix]
            loader = loader_class(str(file_path))
            
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            
            texts = [c.page_content for c in chunks]
            metadatas = [{"source": file_path.name, "page": c.metadata.get("page", 0)} for c in chunks]
            ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]

            # Generazione Embeddings
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            # Caricamento
            collection.upsert(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  -> OK: {len(chunks)} frammenti indicizzati.")

        except Exception as e:
            print(f"  -> ERRORE nel file {file_path.name}: {e}")

    print("\nCompletato! Il database è aggiornato in ./my_local_db")

if __name__ == "__main__":
    main()
