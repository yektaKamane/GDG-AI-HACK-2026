import os
import sys
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def main():
    if len(sys.argv) < 5:
        print("Utilizzo: python cluster_images.py <file.npy> <names.npy> <src> <out> <marker_name>")
        sys.exit(1)

    emb_path, name_path, src_folder, out_folder, marker_name = sys.argv[1:6]

    print("Caricamento dati...")
    embeddings = np.load(emb_path)
    filenames = np.load(name_path)

    print("Esecuzione DBSCAN...")
    X = StandardScaler().fit_transform(embeddings)
    db = DBSCAN(eps=1.0, min_samples=2, metric='cosine').fit(X)
    labels = db.labels_

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Organizzazione file e calcolo centroidi
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_name = f"cluster_{label}" if label != -1 else "outliers"
        target_dir = os.path.join(out_folder, cluster_name)
        os.makedirs(target_dir, exist_ok=True)

        # Indici degli elementi appartenenti a questo cluster
        indices = np.where(labels == label)[0]
        cluster_embeddings = embeddings[indices]
        cluster_filenames = filenames[indices]

        # Copia i file
        for fname in cluster_filenames:
            src_path = os.path.join(src_folder, fname)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(target_dir, fname))

        # Calcolo Centroide (se non è il cluster degli outlier)
        if label != -1 and len(indices) > 0:
            # Centroide matematico (media dei vettori)
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            
            # Trova il file più vicino al centroide usando la cosine similarity
            # (Coerente con la metrica del tuo DBSCAN)
            similarities = cosine_similarity(cluster_embeddings, centroid)
            closest_idx = np.argmax(similarities)
            representative_file = cluster_filenames[closest_idx]

            # Crea il file marker richiesto
            with open(os.path.join(target_dir, marker_name), 'w') as f:
                f.write(representative_file)

    print(f"Clustering completato in: {out_folder}")

if __name__ == "__main__":
    main()
