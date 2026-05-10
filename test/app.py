from flask import Flask, render_template, jsonify, request
import os
import shutil
import fitz  # PyMuPDF
import ollama
import chromadb
import numpy as np
# from sklearn.cluster import HDBSCAN
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import AffinityPropagation
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from pathlib import Path

import re
import requests
import numpy as np
import torch
from PIL import Image

# Machine Learning & Vision
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
            
# --- PRE-SORT FILES TO PREVENT MODEL THRASHING ---
        images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        documents = [f for f in files if f.lower().endswith(('.txt', '.md', '.pdf'))]
        
        summaries_list = []
        ids_list = []
        metadatas_list = []

        # --- 1A. PROCESS ALL IMAGES (Moondream Batch) ---
        if images:
            print(f"📸 Processing {len(images)} images in batch...")
            for i, f in enumerate(images):
                # Only kill the model on the VERY LAST image to free RAM for Phi-3
                is_last = (i == len(images) - 1) 
                alive_status = "0m" if is_last else "5m"
                
                try:
                    with open(f, 'rb') as img_f:
                        res = ollama.generate(
                            model='moondream', 
                            prompt="Describe this image in one short sentence.", 
                            images=[img_f.read()],
                            keep_alive=alive_status 
                        )
                        summary = res['response'].strip()
                        summaries_list.append(summary)
                        ids_list.append(f)
                        metadatas_list.append({"type": f.lower().split('.')[-1], "summary": summary})
                        print(f"  ✅ {f}")
                except Exception as e:
                    print(f"⚠️ Error on {f}: {e}")

        # --- 1B. PROCESS ALL DOCUMENTS (Phi-3 Batch) ---
        if documents:
            print(f"📄 Processing {len(documents)} documents in batch...")
            for i, f in enumerate(documents):
                is_last = (i == len(documents) - 1)
                alive_status = "0m" if is_last else "5m"
                ext = f.lower().split('.')[-1]
                
                try:
                    if ext == 'pdf':
                        doc = fitz.open(f)
                        text = "".join([page.get_text() for page in doc[:2]])
                        doc.close()
                        text_to_summarize = text[:2000]
                    else:
                        with open(f, 'r', encoding='utf-8') as tf:
                            text_to_summarize = tf.read(1500)
                            
                    res = ollama.generate(
                        model='phi3', 
                        prompt=f"Summarize this text topic in 10 words: {text_to_summarize}",
                        options={"num_ctx": 1024},
                        keep_alive=alive_status
                    )
                    summary = res['response'].strip()
                    summaries_list.append(summary)
                    ids_list.append(f)
                    metadatas_list.append({"type": ext, "summary": summary})
                    print(f"  ✅ {f}")
                except Exception as e:
                    print(f"⚠️ Error on {f}: {e}")

        if not summaries_list:
            return jsonify({"error": "Failed to extract summaries from any files."}), 500

        # --- 2. BATCH EMBEDDING (Lightning Fast) ---
        print("\n🔢 Generating Vectors in one batch...")
        # Passing the whole list lets the SentenceTransformer utilize parallel matrix math!
        vectors = embedder.encode(summaries_list).tolist()
        
        # --- 2B. BATCH DATABASE INSERT (Lightning Fast) ---
        print("💾 Saving to Vector Database...")
        collection.add(
            ids=ids_list, 
            embeddings=vectors, 
            metadatas=metadatas_list
        )
        
        # Prepare file_data for the clustering block
        file_data = [{"name": ids_list[i], "summary": summaries_list[i]} for i in range(len(ids_list))]
        
        # # --- 3. CLUSTERING (DBSCAN eps=0.25) ---
        # X = np.array(vectors)
        # X = X / np.linalg.norm(X, axis=1, keepdims=True) # Normalize
        
        # # clustering = DBSCAN(eps=0.30, min_samples=2, metric='cosine').fit(X)
        # clustering = HDBSCAN(min_cluster_size=2, metric='euclidean').fit(X)
        # labels = clustering.labels_

        # # --- STEP 3: AFFINITY PROPAGATION ---
        # print("\n📊 Running Affinity Propagation...")
        # X = np.array(vectors)
        # X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # # damping is between 0.5 and 1.0. 
        # # It prevents the voting messages from bouncing back and forth endlessly.
        # clustering = AffinityPropagation(damping=0.75, random_state=42).fit(X)

        # labels = clustering.labels_
        

        # --- STEP 3: AGGLOMERATIVE CLUSTERING (Best for <100 files) ---
        print("\n📊 Running Agglomerative Clustering for small dataset...")
        X = np.array(vectors)
        
        # Normalize vectors so Euclidean acts like Cosine
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # distance_threshold is your "strictness" dial.
        # For L2 normalized vectors, the max distance is usually around 1.4.
        # A threshold of 0.6 or 0.7 is a great starting point for 33 files.
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.7, 
            metric='euclidean', 
            linkage='ward'
        ).fit(X)

        labels = clustering.labels_

# --- 4. NAMING & COPYING ---
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(file_data[idx])

        # Count how many files are in each cluster
        cluster_counts = Counter(labels)

        for label, items in groups.items():
            # If a cluster only has 1 file (or is labeled -1 from other algorithms), send to Misc
            if cluster_counts[label] == 1 or label == -1:
                folder_name = "Misc_Files"
            else:
                first_summary = items[0]['summary']
                name_res = ollama.generate(
                    model='phi3', 
                    prompt=f"Based on this summary: '{first_summary}', give me a ONE WORD folder name. DO NOT USE MORE THAN ONE WORD."
                )
                
                # 1. Clean out bad characters, spaces, and periods
                raw_name = name_res['response'].strip().replace('.', '').replace(' ', '_').replace('\n', '_')
                
                # 2. Keep only alphanumeric characters and underscores
                clean_name = "".join([c for c in raw_name if c.isalnum() or c == '_'])
                
                # 3. THE KILL SWITCH: Force the name to be 30 characters maximum
                folder_name = clean_name[:30]
                
                # Fallback if the AI gives us completely unusable garbage
                if not folder_name:
                    folder_name = f"Group_{label}"                
            print(f"📁 Routing to: {folder_name}")
            os.makedirs(folder_name, exist_ok=True)
            for item in items:
                shutil.move(item['name'], os.path.join(folder_name, item['name']))

        return jsonify({"reply": f"Success! I've analyzed {len(files)} files and grouped them into {len(set(labels)) - (1 if -1 in labels else 0)} categories."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/organize_photos', methods=['POST'])
def api_organize_photos():
    input_dir = "./input"
    output_dir = "./output"
    marker_name = "centroid_file"
    
    try:
        if not os.path.isdir(input_dir):
            return jsonify({"error": f"Directory {input_dir} not found"}), 400

        # 1. EXTRACTION OF EMBEDDINGS (EfficientNetB2)
        print("--- Step 1: Extracting Embeddings ---")
        ef_model = EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')
        
        valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
        files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]
        
        if not files:
            return jsonify({"error": "No images found"}), 400

        all_embeddings = []
        filenames = []

        for filename in files:
            img_path = os.path.join(input_dir, filename)
            img = keras_image.load_img(img_path, target_size=(260, 260))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            embedding = ef_model.predict(x, verbose=0)
            all_embeddings.append(embedding.flatten())
            filenames.append(filename)

        embeddings_array = np.array(all_embeddings)
        filenames_array = np.array(filenames)

        # 2. CLUSTERING (DBSCAN)
        print("--- Step 2: Clustering Images ---")
        X_scaled = StandardScaler().fit_transform(embeddings_array)
        # Using cosine metric as per your cluster_images.py
        db = DBSCAN(eps=1.0, min_samples=2, metric='cosine').fit(X_scaled)
        labels = db.labels_

        os.makedirs(output_dir, exist_ok=True)

        # # 3. SEMANTIC NAMING & REORGANIZATION
        # print("--- Step 3: Naming and Moving Clusters ---")
        
        # # Initialize Qwen2-VL for Description
        # v_model_id = "Qwen/Qwen2-VL-2B-Instruct"
        # v_model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     v_model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
        # )
        # v_processor = AutoProcessor.from_pretrained(v_model_id)

        # unique_labels = set(labels)
        # results = []

        # for label in unique_labels:
        #     indices = np.where(labels == label)[0]
        #     cluster_files = filenames_array[indices]
        #     cluster_embs = embeddings_array[indices]

        #     # Handle Outliers
        #     if label == -1:
        #         target_path = os.path.join(output_dir, "outliers")
        #         os.makedirs(target_path, exist_ok=True)
        #         for f in cluster_files:
        #             shutil.copy(os.path.join(input_dir, f), os.path.join(target_path, f))
        #         results.append({"label": "outliers", "count": len(cluster_files)})
        #         continue

        #     # Find Centroid (Representative Image)
        #     centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
        #     similarities = cosine_similarity(cluster_embs, centroid)
        #     closest_idx = np.argmax(similarities)
        #     representative_fname = cluster_files[closest_idx]
        #     rep_img_path = os.path.join(input_dir, representative_fname)

        #     # Generate Semantic Description (Qwen2-VL)
        #     raw_desc = "cluster_images"
        #     try:
        #         pil_img = Image.open(rep_img_path).convert("RGB")
        #         msgs = [{
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": pil_img},
        #                 {"type": "text", "text": "Descrivi brevemente cosa vedi in questa immagine."}
        #             ]
        #         }]
        #         v_text = v_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        #         i_in, v_in = process_vision_info(msgs)
        #         inputs = v_processor(text=[v_text], images=i_in, videos=v_in, padding=True, return_tensors="pt").to("cpu")
                
        #         with torch.no_grad():
        #             gen_ids = v_model.generate(**inputs, max_new_tokens=64)
                
        #         gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        #         raw_desc = v_processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
        #     except Exception as e:
        #         print(f"Vision error: {e}")

        # 3. SEMANTIC NAMING & REORGANIZATION
        print("--- Step 3: Naming and Moving Clusters ---")
        
        # Note: We completely removed the heavy Qwen2-VL initialization here!
        # Ollama will handle Moondream dynamically.

        unique_labels = set(labels)
        results = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_files = filenames_array[indices]
            cluster_embs = embeddings_array[indices]

            # Handle Outliers
            if label == -1:
                target_path = os.path.join(output_dir, "outliers")
                os.makedirs(target_path, exist_ok=True)
                for f in cluster_files:
                    shutil.copy(os.path.join(input_dir, f), os.path.join(target_path, f))
                results.append({"label": "outliers", "count": len(cluster_files)})
                continue

            # Find Centroid (Representative Image)
            centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
            similarities = cosine_similarity(cluster_embs, centroid)
            closest_idx = np.argmax(similarities)
            representative_fname = cluster_files[closest_idx]
            rep_img_path = os.path.join(input_dir, representative_fname)

            # Generate Semantic Description (Moondream via Ollama)
            raw_desc = "cluster_images"
            try:
                print(f"👀 Asking Moondream to look at centroid: {representative_fname}")
                with open(rep_img_path, 'rb') as img_f:
                    # Using the Italian prompt you had for Qwen
                    res = ollama.generate(
                        model='moondream', 
                        prompt="Descrivi brevemente cosa vedi in questa immagine.", 
                        images=[img_f.read()]
                    )
                    raw_desc = res['response'].strip()
            except Exception as e:
                print(f"Vision error (Moondream): {e}")

            # Generate Folder Title (Gemma 3 via Ollama)
            folder_title = "semantic_cluster"
            try:
                ollama_res = requests.post("http://localhost:11434/api/generate", json={
                    "model": "phi3",
                    "prompt": f"Riassumi il contenuto di questo testo in una o due parole, adatte come nome di una cartella. Rispondi SOLO con le parole, niente altro.\n\nTesto: '{raw_desc}'",
                    "stream": False
                })
                folder_title = ollama_res.json()["response"].strip()
            except Exception as e:
                print(f"Ollama error: {e}")

# --- Hardened Sanitization for Windows ---
            clean_title = folder_title.lower()
            
            # 1. Convert ALL weird whitespace (newlines \n, tabs \t, \r) into normal spaces
            clean_title = re.sub(r'\s+', ' ', clean_title)
            
            # 2. Remove all punctuation (keep only alphanumeric and spaces)
            clean_title = re.sub(r'[^\w\s]', '', clean_title) 
            
            # 3. Trim edges and replace spaces with underscores
            clean_title = clean_title.strip().replace(" ", "_")
            
            # 4. Remove double underscores
            clean_title = re.sub(r'_+', '_', clean_title) 
            
            # 5. THE KILL SWITCH: Force maximum length to 40 chars to prevent WinError 123
            clean_title = clean_title[:40].strip('_')
            
            # Fallback if the AI gives us completely unusable garbage
            if not clean_title:
                clean_title = "semantic_cluster"
            
            # Prevent collisions
            final_folder_name = clean_title
            counter = 1
            while os.path.exists(os.path.join(output_dir, final_folder_name)):
                final_folder_name = f"{clean_title}_{counter}"
                counter += 1
                        
            dest_path = os.path.join(output_dir, final_folder_name)
            os.makedirs(dest_path, exist_ok=True)

            # Move files and create Marker
            for f in cluster_files:
                shutil.move(os.path.join(input_dir, f), os.path.join(dest_path, f))
            
            with open(os.path.join(dest_path, marker_name), 'w') as m:
                m.write(representative_fname)
            
            results.append({"label": final_folder_name, "count": len(cluster_files), "description": raw_desc})

        return jsonify({"status": "success", "clusters": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True, port=5000)




