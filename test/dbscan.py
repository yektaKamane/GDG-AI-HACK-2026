import os
import shutil
import ollama
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# 1. FORCE OFFLINE MODE
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Load embedding model
embedder = SentenceTransformer('clip-ViT-B-32', local_files_only=True)

def dbscan_organizer(target_folder="."):
    supported = ('.jpg', '.jpeg', '.png', '.txt', '.md')
    files = [f for f in os.listdir(target_folder) if f.lower().endswith(supported) and os.path.isfile(f)]
    
    if not files:
        print("No files found.")
        return

    # --- STEP 1: GENERATE EMBEDDINGS ---
    print(f"🧠 Embedding {len(files)} files...")
    vectors = []
    for f in files:
        if f.lower().endswith(('.txt', '.md')):
            with open(f, 'r', encoding='utf-8') as file:
                vectors.append(embedder.encode(file.read(1000)))
        else:
            vectors.append(embedder.encode(Image.open(f)))
    
    vectors = np.array(vectors)

# ... (Previous code where you embed files into the `vectors` list) ...
    
    vectors = np.array(vectors)

    # --- THE MODALITY GAP FIX ---
    print("🔧 Aligning Image and Text vector spaces...")
    
    # 1. Figure out which vectors are images and which are text
    img_indices = [i for i, f in enumerate(files) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    txt_indices = [i for i, f in enumerate(files) if f.lower().endswith(('.txt', '.md'))]

    # Only apply the fix if the folder actually contains BOTH types of files
    if len(img_indices) > 0 and len(txt_indices) > 0:
        
        # 2. Find the mathematical center of the Image cloud and the Text cloud
        img_center = np.mean(vectors[img_indices], axis=0)
        txt_center = np.mean(vectors[txt_indices], axis=0)
        
        # 3. Calculate the distance between the two clouds
        modality_shift = img_center - txt_center
        
        # 4. Push all text vectors over so their center matches the image center
        vectors[txt_indices] += modality_shift
        
        # Re-normalize the vectors (Cosine distance requires length of 1)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # --- STEP 2: DBSCAN CLUSTERING ---
    # Now that the clouds are merged, DBSCAN will cluster by concept!
    print("📊 Running DBSCAN (Discovering natural folders)...")
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='cosine').fit(vectors)
    
    # ... (Continue to naming and moving) ...    
    labels = clustering.labels_

    # Organize files into groups based on their labels
    groups = {}
    for idx, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(files[idx])

# --- STEP 3: NAMING & MOVING ---
    for group_id, group_files in groups.items():
        
        # Label -1 is DBSCAN's mathematical label for "Noise" / Outliers
        if group_id == -1:
            folder_name = "Uncategorized_Outliers"
            print(f"\n👽 Found {len(group_files)} files that don't match any distinct group.")
        else:
            print(f"\n🏷️ Naming Cluster {group_id} ({len(group_files)} files)...")
            
            # Take ONLY the very first file in the cluster to determine the name
            sample_file = group_files[0]
            
            try:
                if sample_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    with open(sample_file, 'rb') as img_f:
                        # Ask Moondream directly for the folder name
                        res = ollama.generate(
                            model='moondream', 
                            prompt="""TASK: Categorize this image with a single general noun.
                        RULES:
                        1. Ignore any text, logos, or writing inside the image.
                        2. Do not describe the image. 
                        3. Reply with exactly one word.
                        EXAMPLES: Airplane, Dog, Car, Food, Building, Landscape.
                        CATEGORY:
                        """, 
                            images=[img_f.read()]
                        )
                        folder_name = res['response'].strip()
                else:
                    with open(sample_file, 'r', encoding='utf-8') as txt_f:
                        # Ask Phi-3 directly for the folder name
                        content = txt_f.read(500)
                        res = ollama.generate(
                            model='phi3', 
                            prompt=f"Based on this text: '{content}', what is a broad, single-word category name? Output ONLY the word."
                        )
                        folder_name = res['response'].strip()
                
                # Clean up the AI's output (remove punctuation, quotes, and spaces)
                folder_name = folder_name.replace(".", "").replace(" ", "_").replace('"', '').replace("'", "")
                print(f"  🧠 AI chose name: '{folder_name}' based on '{sample_file}'")

            except Exception as e:
                print(f"  ⚠️ Error naming cluster, defaulting to 'Group_{group_id}'. Error: {e}")
                folder_name = f"Group_{group_id}"
            
        # --- STEP 4: PHYSICAL MOVE ---
        if not os.path.exists(folder_name): 
            os.makedirs(folder_name)
            
        for f in group_files:
            shutil.copy2(f, os.path.join(folder_name, f))
            print(f"  ✅ {f} -> {folder_name}/")

if __name__ == "__main__":
    dbscan_organizer()