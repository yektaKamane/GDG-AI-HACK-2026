import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def main():
    # 1. Controllo degli argomenti da riga di comando
    if len(sys.argv) < 2:
        print("Errore: Devi fornire il percorso di una cartella.")
        print("Utilizzo: python extract_embeddings.py /percorso/alla/cartella")
        sys.exit(1)

    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Errore: '{folder_path}' non è una cartella valida.")
        sys.exit(1)

    # 2. Caricamento del modello (una sola volta)
    print("Caricamento del modello EfficientNetB2...")
    model = EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')

    all_embeddings = []
    filenames = []

    # 3. Iterazione sui file della cartella
    print(f"Inizio elaborazione immagini in: {folder_path}")
    
    # Filtriamo solo i file con estensione .jpg o .jpeg
    valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
    files = [f for f in os.listdir(folder_path) if f.endswith(valid_extensions)]

    if not files:
        print("Nessuna immagine JPG trovata nella cartella.")
        sys.exit(1)

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        try:
            # Caricamento e preprocessing
            img = image.load_img(img_path, target_size=(260, 260))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Estrazione embedding
            embedding = model.predict(x, verbose=0) # verbose=0 per non intasare la console
            
            # Aggiunta alla lista (flatten rimuove la dimensione batch extra)
            all_embeddings.append(embedding.flatten())
            filenames.append(filename)
            
            print(f"Processato: {filename}")
        except Exception as e:
            print(f"Errore nel processare {filename}: {e}")

    # 4. Conversione in array NumPy finale e salvataggio
    embeddings_array = np.array(all_embeddings)
    
    output_file = "embeddings_result.npy"
    np.save(output_file, embeddings_array)
    
    # Opzionale: salva anche i nomi dei file per sapere a cosa corrisponde ogni riga
    np.save("filenames.npy", np.array(filenames))

    print("-" * 30)
    print(f"Elaborazione completata!")
    print(f"Immagini processate: {embeddings_array.shape[0]}")
    print(f"Dimensione vettore embedding: {embeddings_array.shape[1]}")
    print(f"File salvato come: {output_file}")

if __name__ == "__main__":
    main()
