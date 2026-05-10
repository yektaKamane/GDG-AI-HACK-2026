#!/bin/bash

# ==========================================
# CONFIGURAZIONE
# ==========================================
INPUT_DIR="./immagini_originali"
OUTPUT_DIR="./organizzato_rag"

# Nomi file intermedi
EMB_NPY="embeddings_result.npy"
NAME_NPY="filenames.npy"
MARKER="centroid_file"

# Nomi script Python
SCRIPT_EXTRACT="estrai_embeddings.py"
SCRIPT_CLUSTER="cluster_images.py"
SCRIPT_DESCRIBE="descrivi_immagine.py"
SCRIPT_TITLER="sintetizza_testo.py"
SCRIPT_RAG="rag_script.py"

# ==========================================
# WORKFLOW
# ==========================================

# 1. Estrazione Embeddings
python3 "$SCRIPT_EXTRACT" "$INPUT_DIR"

# 2. Clustering (ora crea anche il centroid_file internamente)
python3 "$SCRIPT_CLUSTER" "$EMB_NPY" "$NAME_NPY" "$INPUT_DIR" "$OUTPUT_DIR" "$MARKER"

# 3. Descrizione e Rinomina semantica
for cluster_path in "$OUTPUT_DIR"/*/; do
    [ -d "$cluster_path" ] || continue
    
    # Se è la cartella degli outlier, la saltiamo o la gestiamo diversamente
    if [[ "$(basename "$cluster_path")" == "outliers" ]]; then continue; fi

    # Genera descrizioni per tutti i file nel cluster (per il RAG)
    for img in "$cluster_path"*; do
        filename=$(basename "$img")
        [[ "$filename" == "$MARKER" ]] && continue
        [[ "$img" == *.txt ]] && continue
        [ -f "$img" ] || continue

        # Crea il file descrizione (es. foto.jpg.txt)
        python3 "$SCRIPT_DESCRIBE" "$img" > "${img}.txt"
    done

    # Identifica il centroide e ottieni il titolo
    if [ -f "${cluster_path}${MARKER}" ]; then
        CENTROID_FNAME=$(cat "${cluster_path}${MARKER}")
        PATH_TESTO_CENTROIDE="${cluster_path}${CENTROID_FNAME}.txt"

        if [ -f "$PATH_TESTO_CENTROIDE" ]; then
            # Chiamata a sintetizza_testo passando il PATH del file
            TITOLO_RAW=$(python3 "$SCRIPT_TITLER" "$PATH_TESTO_CENTROIDE")
            TITOLO_CLEAN=$(echo "$TITOLO_RAW" | tr -d '[:punct:]' | tr ' ' '_')
            
            # Rinomina la cartella
            mv "$cluster_path" "${OUTPUT_DIR}/${TITOLO_CLEAN}"
            echo "Organizzato cluster: $TITOLO_CLEAN"
        fi
    fi
done

# 4. Esecuzione RAG finale
echo "Avvio indicizzazione RAG..."
python3 "$SCRIPT_RAG" "$OUTPUT_DIR"

echo "Fine processo."
