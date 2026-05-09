import chromadb
from sentence_transformers import SentenceTransformer
import os

# Load same model and DB
embedder = SentenceTransformer('clip-ViT-B-32', local_files_only=True)
db_client = chromadb.PersistentClient(path="./local_db")
collection = db_client.get_or_create_collection(name="unified_storage")

def smart_search(query_text, file_type_filter):
    # 1. Embed the text query
    # CLIP allows text to search for images AND text!
    query_vector = embedder.encode(query_text).tolist()

    # 2. Query with metadata filter
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1,
        where={"type": file_type_filter}
    )

    if results['ids'][0]:
        best_match = results['ids'][0][0]
        metadata = results['metadatas'][0][0]
        print(f"\n🎯 Best Match Found in '{metadata['category']}':")
        print(f"📍 Location: {best_match}")
        
        # Automatically open the file on Mac
        # os.startfile(best_match)
    else:
        print(f"❌ No {file_type_filter} files found matching that description.")

if __name__ == "__main__":
    print("--- Local AI Search ---")
    query = input("What are you looking for? (e.g., 'A blue car' or 'Contract notes'): ")
    f_type = input("Search for 'image' or 'text'? ").strip().lower()
    
    if f_type in ['image', 'text']:
        smart_search(query, f_type)
    else:
        print("Invalid type. Please enter 'image' or 'text'.")