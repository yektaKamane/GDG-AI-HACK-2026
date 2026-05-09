import os
import sys
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def main():
    if len(sys.argv) < 3:
        print("Utilizzo: python cluster_images.py <file_embeddings.npy> <file_filenames.npy> <cartella_originale> <cartella_output>")
        sys.exit(1)

    emb_path = sys.argv[1]
    name_path = sys.argv[2]
    src_folder = sys.argv[3]
    out_folder = sys.argv[4]

    # 1. Caricamento dati
    print("Caricamento embeddings...")
    embeddings = np.load(emb_path)
    filenames = np.load(name_path)

    # 2. Normalizzazione (Fondamentale per DBSCAN)
    # Gli embeddings di EfficientNet possono avere range vari, lo scaling aiuta la metrica di distanza
    X = StandardScaler().fit_transform(embeddings)

    # 3. Esecuzione DBSCAN
    # eps: la distanza massima tra due campioni perché uno sia considerato vicino all'altro.
    # min_samples: il numero minimo di campioni in un vicinato per un core point.
    print("Esecuzione DBSCAN...")
    db = DBSCAN(eps=1.0, min_samples=2, metric='cosine').fit(X)
    labels = db.labels_

    # 4. Organizzazione in sottocartelle
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Trovati {n_clusters} cluster (Label -1 indica rumore/outlier).")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for filename, label in zip(filenames, labels):
        # Crea il nome della sottocartella (es: cluster_0, cluster_1, outlier)
        cluster_name = f"cluster_{label}" if label != -1 else "outliers"
        target_dir = os.path.join(out_folder, cluster_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Copia il file
        src_path = os.path.join(src_folder, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(target_dir, filename))
        else:
            print(f"Avviso: Immagine {filename} non trovata nella sorgente.")

    print(f"Finito! Foto suddivise in: {out_folder}")

if __name__ == "__main__":
    main()
