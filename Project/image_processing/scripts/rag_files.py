import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Configurazione Modello e DB
model = SentenceTransformer('BAAI/bge-m3')
client = chromadb.PersistentClient(path="./my_local_db")
collection = client.get_or_create_collection(name="documenti_file")

# 2. Funzione per caricare e processare i file
def process_files(directory_path):
    # Splitter: divide il testo in pezzi da 500 caratteri con un po' di sovrapposizione
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Gestione estensioni
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
            
        print(f"Processando: {filename}...")
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk.page_content)
            all_metadata.append({"source": filename, "page": chunk.metadata.get("page", 0)})
            all_ids.append(f"{filename}_{i}")

    # 3. Generazione Embeddings e inserimento (a blocchi per sicurezza)
    if all_chunks:
        embeddings = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=all_chunks,
            metadatas=all_metadata,
            ids=all_ids
        )
        print(f"Caricati {len(all_chunks)} frammenti nel database.")

# --- Esecuzione ---
# Crea una cartella 'documenti' e mettici dentro i tuoi PDF
if not os.path.exists("documenti"): os.makedirs("documenti")

process_files("./documenti")

# Prova di ricerca
query = "Cosa dice il documento riguardo ai costi?"
query_emb = model.encode(query, normalize_embeddings=True).tolist()
results = collection.query(query_embeddings=[query_emb], n_results=2)

for i, doc in enumerate(results['documents'][0]):
    fonte = results['metadatas'][0][i]['source']
    print(f"\nRisultato {i+1} (da {fonte}):\n{doc[:200]}...")
