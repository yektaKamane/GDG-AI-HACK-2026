# FileMind 🧠📁

**FileMind** is a multimodal, unsupervised, local AI file manager. It transforms your messy directories into perfectly organized, logically named folders without ever sending your private data to the cloud.

By bridging the gap between images, PDFs, and text documents, FileMind uses local Large Language Models (LLMs) and Vision-Language Models (VLMs) to summarize your files, embed them into a mathematical vector space, and cluster them semantically. 

It doesn't just move your files; it understands what they *mean*.

## ✨ Features
* **100% Local & Private:** Runs entirely on your machine using Ollama and ChromaDB. No cloud APIs, no data leaks.
* **Smart Summarization:** Uses `Gemma_4` for document reading and `Qwen2-VL` for image vision.
* **Semantic Clustering:** Utilizes Agglomerative Clustering and DBSCAN to naturally discover the underlying structure of your data.
* **Intelligent Naming:** Automatically names new folders based on the semantic relationships of the files inside them.
* **Sleek Browser Interface:** A modern, dark-mode GUI to manage your file system effortlessly.


## ⚙️ Installation

### 1. Install Ollama (AI Engine)
FileMind relies on Ollama to run the AI models locally. 
1. Download and install [Ollama](https://ollama.com/).
2. Open your terminal and pull the required models by running:
   ```bash
   ollama pull gemmma4:1b
   ollama pull moondream

## 2. Install Python Dependencies

Ensure you have Python 3.8+ installed. Navigate to the project directory in your terminal and install the required packages:

```bash
pip install flask PyMuPDF ollama chromadb numpy scikit-learn sentence-transformers pillow requests
```

> **Note:** If you plan on using the EfficientNet or Qwen2-VL legacy routes, you may also need:
>
> ```bash
> pip install tensorflow torch transformers
> ```


## 🚀 How to Run

### 1. Start the Server

In your terminal, navigate to the FileMind directory and run:

```bash
python app.py
```


### 2. Open the Interface

Open your favorite web browser and go to:

```text
http://localhost:5000
```


## 3. Organize Your Files

Place the messy files you want to organize into the root folder (or use the UI).

Click one of the three buttons in the top right of the interface:

### 📂 Organize

The primary engine. Analyzes mixed documents and images, clusters them semantically, and creates newly named AI folders.

### 🖼️ Organize Photos

A specialized vision pipeline that groups similar images together and generates descriptive folder titles.

### 📁 Ex. Folders

The matching engine. Routes unorganized files into your already existing folder structure based on context.

## 🛠️ Built With

### Backend
- Python
- Flask

### Vector Database
- ChromaDB

### Embeddings
- SentenceTransformers (`clip-ViT-B-32`)

### Machine Learning
- Scikit-Learn
 - DBSCAN

### Frontend
- HTML
- CSS
- JavaScript