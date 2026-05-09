import os
import shutil
import ollama
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer

# 1. FORCE OFFLINE MODE (Prevents pings to Hugging Face)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 2. Setup Models and DB
print("🚀 Initializing local AI engines...")
# local_files_only=True ensures it doesn't look for updates online
embedder = SentenceTransformer('clip-ViT-B-32', local_files_only=True)
db_client = chromadb.PersistentClient(path="./local_db")
collection = db_client.get_or_create_collection(name="unified_storage")

def get_categories():
    """Returns a list of folders in the current directory, excluding system ones."""
    excluded = ['venv', 'local_db', '__pycache__', '.git']
    return [d for d in os.listdir('.') if os.path.isdir(d) and d not in excluded]

def process_all_files():
    categories = get_categories()
    if not categories:
        print("❌ No category folders found. Please create folders like 'Food', 'Tech', etc.")
        return

    # 3. Identify all supported files in the current folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.txt', '.md')
    # Filter files: must have supported extension AND not be a directory
    files_to_process = [f for f in os.listdir('.') 
                        if f.lower().endswith(supported_extensions) and os.path.isfile(f)]

    if not files_to_process:
        print("Empty folder! No images or text files found to organize.")
        return

    print(f"📂 Found {len(files_to_process)} files. Starting batch processing...\n")

    for filename in files_to_process:
        print(f"--- Processing: {filename} ---")
        ext = os.path.splitext(filename)[1].lower()
        selected_cat = "Uncategorized"
        file_type = ""

        try:
            # --- STEP A: CLASSIFICATION ---
            if ext in ('.jpg', '.jpeg', '.png'):
                file_type = "image"
                with open(filename, 'rb') as f:
                    response = ollama.generate(
                        model='moondream', 
                        prompt=f"Categorize this image into one of these: {categories}. Output ONLY the category name.",
                        images=[f.read()]
                    )
                ai_thought = response['response'].strip()
                
            else: # Text files
                file_type = "text"
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read(1500)
                response = ollama.generate(
                    model='phi3',
                    prompt=f"Text: '{content}'. Which category fits: {categories}? Output ONLY the category name."
                )
                ai_thought = response['response'].strip()

            # Robust matching: check if any category name appears in the AI's answer
            selected_cat = next((c for c in categories if c.lower() in ai_thought.lower()), "Uncategorized")
            print(f"🤖 AI Thought: '{ai_thought}' -> Decided Folder: {selected_cat}")

            # --- STEP B: ORGANIZE (Copy Only) ---
            if not os.path.exists(selected_cat): os.makedirs(selected_cat)
            dest_path = os.path.join(selected_cat, filename)
            shutil.copy2(filename, dest_path)

            # --- STEP C: EMBED & INDEX ---
            if file_type == "image":
                embedding = embedder.encode(Image.open(filename)).tolist()
            else:
                with open(filename, 'r', encoding='utf-8') as f:
                    embedding = embedder.encode(f.read()).tolist()

            collection.add(
                ids=[dest_path],
                embeddings=[embedding],
                metadatas=[{"type": file_type, "path": dest_path, "category": selected_cat}]
            )
            print(f"✅ Indexed and copied to {selected_cat}\n")

        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {e}")

    print("✨ Batch processing complete!")

if __name__ == "__main__":
    process_all_files()