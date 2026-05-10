from flask import Flask, render_template, jsonify, request
import os
import shutil
import fitz  # PyMuPDF
import ollama
import chromadb
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from pathlib import Path

# 1. ENVIRONMENT SETUP
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = Flask(__name__)

# 2. AI & DATABASE INITIALIZATION
print("🚀 Initializing AI Engines...")
embedder = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
db_client = chromadb.PersistentClient(path="./local_db")
collection = db_client.get_or_create_collection(name="unified_storage")

# --- FILE EXPLORER HELPERS ---

def get_node_data(path):
    """Recursive helper for the sidebar project tree."""
    name = os.path.basename(path) if os.path.basename(path) else path
    node = {
        "name": name,
        "path": path,
        "type": "folder" if os.path.isdir(path) else "file",
        "children": []
    }
    
    if node["type"] == "folder":
        # Exclude hidden folders and system directories
        excluded = {'.git', 'local_db', 'venv', '__pycache__', 'templates', 'static'}
        try:
            for entry in os.listdir(path):
                if entry not in excluded and not entry.startswith('.'):
                    full_path = os.path.join(path, entry)
                    node["children"].append(get_node_data(full_path))
        except PermissionError:
            pass
    return node

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tree')
def api_tree():
    """Returns the structure for the sidebar."""
    return jsonify(get_node_data("."))

@app.route('/api/files')
def api_files():
    """Returns files in a specific folder for the main panel."""
    path = request.args.get('path') or '.'
    if not os.path.exists(path):
        return jsonify({"error": "Path not found"}), 404

    files_data = []
    try:
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            stats = os.stat(full_path)
            files_data.append({
                "name": entry,
                "path": full_path,
                "type": "folder" if os.path.isdir(full_path) else "file",
                "size": stats.st_size
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"current_path": path, "files": files_data})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handles smart search via the chatbot input."""
    user_msg = request.json.get('message', '').lower()
    
    # Generate query vector from search text
    query_vector = embedder.encode(user_msg).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    if results['ids'] and results['ids'][0]:
        reply = "I found some relevant files based on your description:\n\n"
        for i in range(len(results['ids'][0])):
            path = results['ids'][0][i]
            meta = results['metadatas'][0][i]
            summary = meta.get('summary', 'No summary available')
            reply += f"📍 **{os.path.basename(path)}**\nSummary: {summary}\n\n"
        return jsonify({"reply": reply})
    
    return jsonify({"reply": "I couldn't find any specific files matching that description in my database."})

@app.route('/api/organize', methods=['POST'])
def api_organize():
    """The Summarization -> Embedding -> DBSCAN Cluster pipeline."""
    try:
        supported_ext = ('.jpg', '.jpeg', '.png', '.txt', '.md', '.pdf')
        # Only process files in the root to avoid recursive mess
        files = [f for f in os.listdir('.') if f.lower().endswith(supported_ext) and os.path.isfile(f)]
        
        if not files:
            return jsonify({"error": "No files found to organize."}), 400

        vectors = []
        file_data = [] # List of {filename, summary}

        for f in files:
            ext = f.lower().split('.')[-1]
            summary = ""
            
            # --- 1. SUMMARIZATION ---
            try:
                if ext in ['jpg', 'jpeg', 'png']:
                    with open(f, 'rb') as img_f:
                        res = ollama.generate(model='moondream', prompt="Describe this image in one short sentence.", images=[img_f.read()])
                        summary = res['response'].strip()
                elif ext == 'pdf':
                    doc = fitz.open(f)
                    text = "".join([page.get_text() for page in doc[:3]]) # First 3 pages
                    doc.close()
                    res = ollama.generate(model='gemma4:e2b', prompt=f"Summarize this document topic in 10 words: {text[:5000]}")
                    summary = res['response'].strip()
                else: # txt/md
                    with open(f, 'r', encoding='utf-8') as tf:
                        text = tf.read(2000)
                    res = ollama.generate(model='gemma4:e2b', prompt=f"Summarize this text topic in 10 words: {text}")
                    summary = res['response'].strip()

                # --- 2. EMBEDDING ---
                vector = embedder.encode(summary).tolist()
                
                # Save to DB for chat search
                collection.add(ids=[f], embeddings=[vector], metadatas=[{"type": ext, "summary": summary}])
                
                vectors.append(vector)
                file_data.append({"name": f, "summary": summary})
            except Exception as e:
                print(f"Error processing {f}: {e}")

        # --- 3. CLUSTERING (DBSCAN eps=0.25) ---
        X = np.array(vectors)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalize
        
        clustering = DBSCAN(eps=0.30, min_samples=2, metric='cosine').fit(X)
        labels = clustering.labels_

        # --- 4. NAMING & COPYING ---
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(file_data[idx])

        for label, items in groups.items():
            if label == -1:
                folder_name = "Misc_Files"
            else:
                # Name the folder based on the first item's summary
                name_res = ollama.generate(
                    model='gemma4:e2b', 
                    prompt=f"Based on these summaries: {[i['summary'] for i in items[:3]]}, give me a 1-word folder name."
                )
                folder_name = name_res['response'].strip().replace('.', '').replace(' ', '_')

            os.makedirs(folder_name, exist_ok=True)
            for item in items:
                shutil.move(item['name'], os.path.join(folder_name, item['name']))

        return jsonify({"reply": f"Success! I've analyzed {len(files)} files and grouped them into {len(set(labels)) - (1 if -1 in labels else 0)} categories."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)