#!/bin/bash

# ==========================================
# CONTROLLO ARGOMENTI
# ==========================================

if [ $# -ne 1 ]; then
    echo "Uso: $0 <PATH>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$INPUT_DIR"

# ==========================================
# CONFIGURAZIONE
# ==========================================

# File intermedi
EMB_NPY="embeddings_result.npy"
NAME_NPY="filenames.npy"
MARKER="centroid_file"

# Script Python
SCRIPT_EXTRACT="extract_embeddings.py"
SCRIPT_CLUSTER="cluster_images.py"
SCRIPT_DESCRIBE="vision_cli.py"
SCRIPT_TITLER="script_resume.py"
SCRIPT_RAG="encode_files_vdb.py"

# ==========================================
# WORKFLOW
# ==========================================

echo "=========================================="
echo "1. Estrazione embeddings"
echo "=========================================="

python3 "$SCRIPT_EXTRACT" "$INPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Errore durante extract embeddings"
    exit 1
fi

# ==========================================

echo "=========================================="
echo "2. Clustering immagini"
echo "=========================================="

python3 "$SCRIPT_CLUSTER" \
    "$EMB_NPY" \
    "$NAME_NPY" \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    "$MARKER"

if [ $? -ne 0 ]; then
    echo "Errore durante clustering"
    exit 1
fi

# ==========================================

echo "=========================================="
echo "3. Naming semantico cluster"
echo "=========================================="

for cluster_path in "$OUTPUT_DIR"/*/; do

    [ -d "$cluster_path" ] || continue

    CLUSTER_NAME=$(basename "$cluster_path")

    echo ""
    echo "Cluster trovato: $CLUSTER_NAME"

    # Salta outliers
    if [[ "$CLUSTER_NAME" == "outliers" ]]; then
        echo "Cluster outliers saltato"
        continue
    fi

    # Verifica presenza marker centroide
    if [ ! -f "${cluster_path}${MARKER}" ]; then
        echo "Marker centroide non trovato"
        continue
    fi

    CENTROID_FNAME=$(cat "${cluster_path}${MARKER}")

    if [ -z "$CENTROID_FNAME" ]; then
        echo "Centroide vuoto"
        continue
    fi

    CENTROID_PATH="${cluster_path}${CENTROID_FNAME}"

    if [ ! -f "$CENTROID_PATH" ]; then
        echo "File centroide non trovato: $CENTROID_PATH"
        continue
    fi

    echo "Centroide: $CENTROID_FNAME"

    # ==========================================
    # DESCRIZIONE
    # ==========================================

    DESC_FILE="${INPUT_DIR}/${CENTROID_FNAME}.txt"

    echo "Genero descrizione semantica..."

    python3 "$SCRIPT_DESCRIBE" "$CENTROID_PATH" > "$DESC_FILE"

    if [ $? -ne 0 ]; then
        echo "Errore generazione descrizione"
        continue
    fi

    if [ ! -f "$DESC_FILE" ]; then
        echo "Descrizione non generata"
        continue
    fi

    # ==========================================
    # TITOLO
    # ==========================================

    echo "Genero titolo cluster..."

    TITOLO_RAW=$(python3 "$SCRIPT_TITLER" "$DESC_FILE")

    TITOLO_CLEAN=$(echo "$TITOLO_RAW" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -d '[:punct:]' \
        | tr ' ' '_' \
        | tr -s '_')

    if [ -z "$TITOLO_CLEAN" ]; then
        TITOLO_CLEAN="cluster_semantico"
    fi

    DEST_FOLDER="${OUTPUT_DIR}/${TITOLO_CLEAN}"

    COUNTER=1

    while [ -d "$DEST_FOLDER" ]; do
        DEST_FOLDER="${OUTPUT_DIR}/${TITOLO_CLEAN}_${COUNTER}"
        COUNTER=$((COUNTER + 1))
    done

    # ==========================================
    # RINOMINA
    # ==========================================

    mv "$cluster_path" "$DEST_FOLDER"

    echo "Cluster rinominato in:"
    echo "$DEST_FOLDER"

done

# ==========================================

echo ""
echo "=========================================="
echo "4. Indicizzazione RAG"
echo "=========================================="

python3 "$SCRIPT_RAG" "$INPUT_DIR"

if [ $? -ne 0 ]; then
    echo "Errore indicizzazione RAG"
    exit 1
fi

# ==========================================

echo ""
echo "=========================================="
echo "PROCESSO COMPLETATO"
echo "=========================================="