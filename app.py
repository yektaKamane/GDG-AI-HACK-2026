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
import json

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


app = Flask(__name__)


BASE_DIR = Path("/Users/riccardoinfascelli/Desktop/Hackathon/GDG-AI-HACK-2026/Project/PROVA").resolve()

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4:e2b"

MAX_FILE_PREVIEW_CHARS = 4000
MAX_TOTAL_FILES_FOR_AI = 80
DEBUG_ORGANIZER = True

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv",
    ".xml", ".yaml", ".yml", ".java", ".cpp", ".c", ".h", ".hpp",
    ".ts", ".tsx", ".jsx", ".sql", ".sh", ".ini", ".cfg", ".log"
}

IGNORED_DIRS = {
    "venv", ".venv", "__pycache__", ".git", "node_modules",
    ".idea", ".vscode", "dist", "build"
}


def debug(*args):
    if DEBUG_ORGANIZER:
        print("[ORGANIZER]", *args, flush=True)


def safe_path(relative_path=""):
    target = (BASE_DIR / relative_path).resolve()

    if not str(target).startswith(str(BASE_DIR)):
        raise ValueError("Invalid path")

    return target


def build_tree(path: Path):
    children = []

    try:
        for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue

            if item.is_dir() and item.name in IGNORED_DIRS:
                continue

            node = {
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)),
                "type": "folder" if item.is_dir() else "file",
            }

            if item.is_dir():
                node["children"] = build_tree(item)

            children.append(node)

    except PermissionError:
        debug("TREE SKIP permission denied:", path)

    return children


def is_probably_text_file(path: Path):
    return path.suffix.lower() in TEXT_EXTENSIONS


def read_file_preview(path: Path):
    try:
        if not path.is_file():
            return None

        if not is_probably_text_file(path):
            return None

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read(MAX_FILE_PREVIEW_CHARS)

    except Exception as e:
        debug("READ SKIP:", path.relative_to(BASE_DIR), "reason:", str(e))
        return None


def collect_files_for_ai():
    collected = []

    debug("Collecting files from:", BASE_DIR)

    for path in BASE_DIR.rglob("*"):
        rel = path.relative_to(BASE_DIR)

        if len(collected) >= MAX_TOTAL_FILES_FOR_AI:
            debug("STOP: max file limit reached:", MAX_TOTAL_FILES_FOR_AI)
            break

        if not path.is_file():
            debug("SKIP collect not file:", rel)
            continue

        if path.name.startswith("."):
            debug("SKIP collect hidden file:", rel)
            continue

        relative_parts = path.relative_to(BASE_DIR).parts

        if any(part in IGNORED_DIRS for part in relative_parts):
            debug("SKIP collect ignored dir:", rel)
            continue

        if not is_probably_text_file(path):
            debug("SKIP collect non-text extension:", rel, path.suffix)
            continue

        preview = read_file_preview(path)

        if preview is None:
            debug("SKIP collect unreadable preview:", rel)
            continue

        debug("COLLECT:", rel)

        relative_path = path.relative_to(BASE_DIR)

        collected.append({
            "path": str(relative_path),
            "current_folder": str(relative_path.parent) if str(relative_path.parent) != "." else "",
            "name": path.name,
            "extension": path.suffix.lower(),
            "preview": preview
        })

    debug("TOTAL COLLECTED:", len(collected))
    return collected


def extract_json_from_llm_response(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")

    possible_json = text[start:end + 1]
    return json.loads(possible_json)


def ask_ollama_for_organization(files):
    existing_dirs = sorted({
        str(path.relative_to(BASE_DIR))
        for path in BASE_DIR.rglob("*")
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in IGNORED_DIRS
    })

    prompt = f"""
You are an elite AI semantic file organizer.

Your ONLY goal is to organize files into REAL semantic categories.

You MUST understand:
- the topic
- the subject
- the meaning
- the domain
- the purpose

of every file.

You are NOT allowed to create generic folders.

==================================================
STRICT RULES
==================================================

NEVER create folders like:
- misc
- other
- random
- files
- documents
- docs
- text
- notes
- stuff
- generic
- uncategorized

These folders are FORBIDDEN.

You MUST ALWAYS choose:
- a real subject
- a real topic
- a real domain
- a meaningful semantic category

Even if uncertain.

==================================================
GOOD CATEGORY EXAMPLES
==================================================

GOOD:
- physics
- chemistry
- biology
- math
- calculus
- ai
- machine_learning
- neural_networks
- backend
- frontend
- finance
- taxes
- startup
- contracts
- recipes
- travel
- fitness
- university
- astronomy
- climate
- gaming
- cybersecurity
- legal
- invoices
- marketing
- psychology
- philosophy
- economics

BAD:
- misc
- docs
- files
- text
- notes
- random
- generic

==================================================
ORGANIZATION STRATEGY
==================================================

You must organize files by:
1. semantic topic
2. real-world subject
3. actual meaning
4. project domain

NOT by extension.

Examples:
- Python APIs -> backend
- React UI -> frontend
- Neural network experiments -> machine_learning
- Physics notes -> physics
- Recipes -> recipes
- Tax PDFs -> taxes
- Startup ideas -> startup
- Contracts -> legal
- CSV finance data -> finance
- AI prompts -> ai
- University exercises -> university

==================================================
FOLDER RULES
==================================================

- Folder names must be semantic.
- Folder names must describe the actual topic.
- Prefer 1 word.
- Maximum 2 words.
- Use lowercase.
- Use snake_case if needed.
- Maximum nesting depth: 2.
- Reuse existing folders if appropriate.
- Create new folders if necessary.
- NEVER leave a file uncategorized.
- EVERY file MUST belong to a meaningful category.

==================================================
CRITICAL RULES
==================================================

- You MUST return one move for EVERY file.
- Every file must appear EXACTLY ONCE.
- Never omit files.
- Never invent files.
- The "from" field must exactly match an input file.
- The "to" field must preserve the original filename.
- Do NOT rename files.
- Move ONLY files.
- Never move folders.
- Never use absolute paths.
- Never use ".."

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Example:

{{
  "summary": "Files organized by semantic topic.",
  "moves": [
    {{
      "from": "quantum_notes.txt",
      "to": "physics/quantum_notes.txt",
      "reason": "Contains quantum mechanics study notes."
    }},
    {{
      "from": "pizza_recipe.md",
      "to": "recipes/pizza_recipe.md",
      "reason": "Contains cooking instructions and ingredients."
    }}
  ]
}}

==================================================
EXISTING FOLDERS
==================================================

{json.dumps(existing_dirs, indent=2)}

==================================================
FILES TO ORGANIZE
==================================================

{json.dumps(files, indent=2)}
"""

    debug("Sending prompt to Ollama.")
    debug("Files sent to Ollama:", len(files))

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=180
    )

    debug("OLLAMA STATUS:", response.status_code)
    debug("OLLAMA RAW TEXT:", response.text[:5000])

    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")

    if not raw_text:
        raise ValueError("Empty response from Ollama")

    parsed = extract_json_from_llm_response(raw_text)

    debug("OLLAMA PARSED MOVES:", len(parsed.get("moves", [])))

    return parsed


def validate_relative_path(path_string):
    if not path_string:
        raise ValueError("Empty path")

    path = Path(path_string)

    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {path_string}")

    if ".." in path.parts:
        raise ValueError(f"Parent directory references are not allowed: {path_string}")

    return path


def clean_folder_part(name):
    name = str(name).strip().lower()
    name = name.replace("\\", "/")
    name = name.split("/")[-1]

    name = name.replace(" ", "_")
    name = "".join(ch for ch in name if ch.isalnum() or ch in "_-")

    chunks = [c for c in name.replace("-", "_").split("_") if c]
    chunks = chunks[:2]

    cleaned = "_".join(chunks)

    if not cleaned:
        return "misc"

    if len(cleaned) > 32:
        cleaned = cleaned[:32].rstrip("_-")

    return cleaned or "misc"


def clean_folder_path(path):
    path = str(path).strip().replace("\\", "/")
    parts = [p for p in path.split("/") if p and p not in {".", ".."}]

    cleaned_parts = [clean_folder_part(part) for part in parts[:2]]

    if not cleaned_parts:
        return Path("misc")

    return Path(*cleaned_parts)


def repair_missing_llm_moves(raw_plan, files):
    provided_paths = {f["path"] for f in files}
    returned_paths = {
        move.get("from")
        for move in raw_plan.get("moves", [])
        if move.get("from")
    }

    missing_paths = provided_paths - returned_paths

    for missing in sorted(missing_paths):
        debug("LLM OMITTED FILE, forcing misc:", missing)

        raw_plan.setdefault("moves", []).append({
            "from": missing,
            "to": f"misc/{Path(missing).name}",
            "reason": "The model omitted this file, so it was safely assigned to misc."
        })

    extra_paths = returned_paths - provided_paths

    for extra in sorted(extra_paths):
        debug("LLM RETURNED UNKNOWN FILE:", extra)

    return raw_plan


def normalize_plan(plan):
    normalized_dirs = set()
    normalized_moves = []

    moves = plan.get("moves", [])

    debug("Normalizing moves:", len(moves))

    seen_sources = set()

    for move in moves:
        source = move.get("from", "")
        destination = move.get("to", "")
        reason = move.get("reason", "")

        if not source or not destination:
            debug("NORMALIZE SKIP missing source/destination:", move)
            continue

        if source in seen_sources:
            debug("NORMALIZE SKIP duplicate source:", source)
            continue

        seen_sources.add(source)

        source_path = Path(source)
        destination_path = Path(destination)
        original_filename = source_path.name

        if destination_path.name != original_filename:
            debug(
                "NORMALIZE FIX filename mismatch:",
                source,
                "AI destination:",
                destination,
                "forced filename:",
                original_filename
            )

        if len(destination_path.parts) < 2:
            debug("NORMALIZE FIX no folder from AI, forcing misc:", source, "->", destination)
            destination_path = Path("sorted") / original_filename

        folder_path = clean_folder_path(destination_path.parent)

        if str(folder_path).split("/")[0] in IGNORED_DIRS:
            debug("NORMALIZE FIX ignored destination folder, forcing misc:", source, "->", folder_path)
            folder_path = Path("misc")

        final_destination = str(folder_path / original_filename)

        if source == final_destination:
            debug("NORMALIZE KEEP already good location:", source)
            continue

        debug("NORMALIZE MOVE:", source, "->", final_destination)

        normalized_dirs.add(str(folder_path))

        normalized_moves.append({
            "from": source,
            "to": final_destination,
            "reason": str(reason)[:160]
        })

    debug("NORMALIZED MOVE COUNT:", len(normalized_moves))

    return {
        "summary": plan.get("summary", "Files were reorganized by content and purpose."),
        "directories": sorted(normalized_dirs),
        "moves": normalized_moves
    }


def unique_destination_path(destination: Path):
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"

        if not candidate.exists():
            debug("DESTINATION EXISTS, using unique path:", candidate.relative_to(BASE_DIR))
            return candidate

        counter += 1


def remove_empty_folders(candidate_dirs):
    removed = []
    paths = []

    for directory in candidate_dirs:
        try:
            relative_dir = validate_relative_path(directory)
            path = safe_path(relative_dir)

            if path == BASE_DIR:
                continue

            if not path.exists() or not path.is_dir():
                continue

            if path.name in IGNORED_DIRS or path.name.startswith("."):
                continue

            paths.append(path)

        except Exception as e:
            debug("CLEANUP SKIP:", directory, "reason:", str(e))
            continue

    paths = sorted(set(paths), key=lambda p: len(p.parts), reverse=True)

    for path in paths:
        try:
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                removed.append(str(path.relative_to(BASE_DIR)))
                debug("REMOVED EMPTY FOLDER:", path.relative_to(BASE_DIR))
        except Exception as e:
            debug("REMOVE EMPTY FOLDER FAILED:", path.relative_to(BASE_DIR), "reason:", str(e))

    return removed


def apply_organization_plan(plan):
    created_dirs = set()
    moved_files = []
    skipped_files = []
    cleanup_candidates = set()

    moves = plan.get("moves", [])

    debug("Applying moves:", len(moves))

    for move in moves:
        source_relative = move.get("from", "")
        destination_relative = move.get("to", "")
        reason = move.get("reason", "")

        try:
            source_path = safe_path(validate_relative_path(source_relative))
            destination_path = safe_path(validate_relative_path(destination_relative))

            if not source_path.exists():
                debug("APPLY SKIP:", source_relative, "reason: Source file does not exist")
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Source file does not exist"
                })
                continue

            if not source_path.is_file():
                debug("APPLY SKIP:", source_relative, "reason: Source is not a file")
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Source is not a file"
                })
                continue

            if source_path == destination_path:
                debug("APPLY SKIP:", source_relative, "reason: Already in target location")
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Already in target location"
                })
                continue

            old_parent = source_path.parent

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(str(destination_path.parent.relative_to(BASE_DIR)))

            final_destination = unique_destination_path(destination_path)

            shutil.move(str(source_path), str(final_destination))

            debug("MOVED:", source_relative, "->", final_destination.relative_to(BASE_DIR))

            moved_files.append({
                "from": source_relative,
                "to": str(final_destination.relative_to(BASE_DIR)),
                "reason": reason
            })

            try:
                cleanup_candidates.add(str(old_parent.relative_to(BASE_DIR)))
            except ValueError:
                pass

        except Exception as e:
            debug("APPLY SKIP:", source_relative, "reason:", str(e))
            skipped_files.append({
                "path": source_relative,
                "reason": str(e)
            })

    removed_empty_dirs = remove_empty_folders(cleanup_candidates)

    real_created_dirs = []

    for directory in sorted(created_dirs):
        try:
            path = safe_path(validate_relative_path(directory))

            if path.exists() and path.is_dir() and any(path.iterdir()):
                real_created_dirs.append(directory)

        except Exception as e:
            debug("CREATED DIR CHECK SKIP:", directory, "reason:", str(e))

    debug("APPLY RESULT moved:", len(moved_files))
    debug("APPLY RESULT skipped:", len(skipped_files))
    debug("APPLY RESULT created dirs:", len(real_created_dirs))
    debug("APPLY RESULT removed empty dirs:", len(removed_empty_dirs))

    return {
        "created_dirs": real_created_dirs,
        "moved_files": moved_files,
        "skipped_files": skipped_files,
        "removed_empty_dirs": removed_empty_dirs
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tree")
def get_tree():
    return jsonify({
        "name": BASE_DIR.name,
        "path": "",
        "type": "folder",
        "children": build_tree(BASE_DIR)
    })


@app.route("/api/files")
def get_files():
    relative_path = request.args.get("path", "")

    try:
        folder = safe_path(relative_path)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Not a directory"}), 400

    files = []

    try:
        for item in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue

            if item.is_dir() and item.name in IGNORED_DIRS:
                continue

            stat = item.stat()

            files.append({
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)),
                "type": "folder" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else None,
            })

    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    return jsonify({
        "current_path": str(folder.relative_to(BASE_DIR)),
        "files": files
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    bot_reply = f"You said: {message}"

    return jsonify({
        "reply": bot_reply
    })


@app.route("/api/organize", methods=["POST"])
def organize_files():
    try:
        debug("===== ORGANIZATION STARTED =====")

        files = collect_files_for_ai()

        if not files:
            debug("No readable files found.")
            return jsonify({
                "reply": "I could not find readable text/code files to organize."
            })

        raw_plan = ask_ollama_for_organization(files)
        raw_plan = repair_missing_llm_moves(raw_plan, files)

        plan = normalize_plan(raw_plan)
        result = apply_organization_plan(plan)

        moved_count = len(result["moved_files"])
        dir_count = len(result["created_dirs"])
        skipped_count = len(result["skipped_files"])
        removed_empty_count = len(result.get("removed_empty_dirs", []))

        summary = plan.get("summary", "Organization completed.")

        reply = (
            f"{summary}\n\n"
            f"Created non-empty folders: {dir_count}\n"
            f"Moved files: {moved_count}\n"
            f"Skipped files: {skipped_count}\n"
            f"Removed empty folders: {removed_empty_count}"
        )

        if moved_count > 0:
            reply += "\n\nMain moves:\n"

            for move in result["moved_files"][:12]:
                reply += f"- {move['from']} → {move['to']}\n"

        if result.get("removed_empty_dirs"):
            reply += "\nRemoved empty folders:\n"

            for folder in result["removed_empty_dirs"][:12]:
                reply += f"- {folder}\n"

        if skipped_count > 0:
            reply += "\nSkipped files:\n"

            for skipped in result["skipped_files"][:12]:
                reply += f"- {skipped['path']}: {skipped['reason']}\n"

        debug("===== ORGANIZATION FINISHED =====")

        return jsonify({
            "reply": reply,
            "raw_plan": raw_plan,
            "plan": plan,
            "result": result
        })

    except requests.exceptions.ConnectionError:
        debug("ERROR: Could not connect to Ollama.")
        return jsonify({
            "error": "Could not connect to Ollama. Make sure Ollama is running with: ollama serve"
        }), 500

    except requests.exceptions.Timeout:
        debug("ERROR: Ollama timeout.")
        return jsonify({
            "error": "Ollama took too long to respond. Try with fewer files or a smaller directory."
        }), 500

    except Exception as e:
        debug("ERROR:", str(e))
        return jsonify({
            "error": f"Organization failed: {str(e)}"
        }), 500

@app.route('/api/organize_photos', methods=['POST'])
def api_organize_photos():
    os.system('sh pray_that_this_works.sh ./Project/PROVA/input')

# @app.route('/api/organize_photos', methods=['POST'])
# def api_organize_photos():
#     input_dir = "./input"
#     output_dir = "./output"
#     marker_name = "centroid_file"
    
#     try:
#         if not os.path.isdir(input_dir):
#             return jsonify({"error": f"Directory {input_dir} not found"}), 400

#         # 1. EXTRACTION OF EMBEDDINGS (EfficientNetB2)
#         print("--- Step 1: Extracting Embeddings ---")
#         ef_model = EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')
        
#         valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
#         files = [f for f in os.listdir(input_dir) if f.endswith(valid_extensions)]
        
#         if not files:
#             return jsonify({"error": "No images found"}), 400

#         all_embeddings = []
#         filenames = []

#         for filename in files:
#             img_path = os.path.join(input_dir, filename)
#             img = keras_image.load_img(img_path, target_size=(260, 260))
#             x = keras_image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = preprocess_input(x)
            
#             embedding = ef_model.predict(x, verbose=0)
#             all_embeddings.append(embedding.flatten())
#             filenames.append(filename)

#         embeddings_array = np.array(all_embeddings)
#         filenames_array = np.array(filenames)

#         # 2. CLUSTERING (DBSCAN)
#         print("--- Step 2: Clustering Images ---")
#         X_scaled = StandardScaler().fit_transform(embeddings_array)
#         # Using cosine metric as per your cluster_images.py
#         db = DBSCAN(eps=1.0, min_samples=2, metric='cosine').fit(X_scaled)
#         labels = db.labels_

#         os.makedirs(output_dir, exist_ok=True)

#         # 3. SEMANTIC NAMING & REORGANIZATION
#         print("--- Step 3: Naming and Moving Clusters ---")
        
#         # Initialize Qwen2-VL for Description
#         v_model_id = "Qwen/Qwen2-VL-2B-Instruct"
#         v_model = Qwen2VLForConditionalGeneration.from_pretrained(
#             v_model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
#         )
#         v_processor = AutoProcessor.from_pretrained(v_model_id)

#         unique_labels = set(labels)
#         results = []

#         for label in unique_labels:
#             indices = np.where(labels == label)[0]
#             cluster_files = filenames_array[indices]
#             cluster_embs = embeddings_array[indices]

#             # Handle Outliers
#             if label == -1:
#                 target_path = os.path.join(output_dir, "outliers")
#                 os.makedirs(target_path, exist_ok=True)
#                 for f in cluster_files:
#                     shutil.copy(os.path.join(input_dir, f), os.path.join(target_path, f))
#                 results.append({"label": "outliers", "count": len(cluster_files)})
#                 continue

#             # Find Centroid (Representative Image)
#             centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
#             similarities = cosine_similarity(cluster_embs, centroid)
#             closest_idx = np.argmax(similarities)
#             representative_fname = cluster_files[closest_idx]
#             rep_img_path = os.path.join(input_dir, representative_fname)

#             # Generate Semantic Description (Qwen2-VL)
#             raw_desc = "cluster_images"
#             try:
#                 pil_img = Image.open(rep_img_path).convert("RGB")
#                 msgs = [{
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": pil_img},
#                         {"type": "text", "text": "Descrivi brevemente cosa vedi in questa immagine."}
#                     ]
#                 }]
#                 v_text = v_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#                 i_in, v_in = process_vision_info(msgs)
#                 inputs = v_processor(text=[v_text], images=i_in, videos=v_in, padding=True, return_tensors="pt").to("cpu")
                
#                 with torch.no_grad():
#                     gen_ids = v_model.generate(**inputs, max_new_tokens=64)
                
#                 gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
#                 raw_desc = v_processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
#             except Exception as e:
#                 print(f"Vision error: {e}")

#         # # 3. SEMANTIC NAMING & REORGANIZATION
#         # print("--- Step 3: Naming and Moving Clusters ---")
        
#         # # Note: We completely removed the heavy Qwen2-VL initialization here!
#         # # Ollama will handle Moondream dynamically.

#         # unique_labels = set(labels)
#         # results = []

#         # for label in unique_labels:
#         #     indices = np.where(labels == label)[0]
#         #     cluster_files = filenames_array[indices]
#         #     cluster_embs = embeddings_array[indices]

#         #     # Handle Outliers
#         #     if label == -1:
#         #         target_path = os.path.join(output_dir, "outliers")
#         #         os.makedirs(target_path, exist_ok=True)
#         #         for f in cluster_files:
#         #             shutil.copy(os.path.join(input_dir, f), os.path.join(target_path, f))
#         #         results.append({"label": "outliers", "count": len(cluster_files)})
#         #         continue

#         #     # Find Centroid (Representative Image)
#         #     centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
#         #     similarities = cosine_similarity(cluster_embs, centroid)
#         #     closest_idx = np.argmax(similarities)
#         #     representative_fname = cluster_files[closest_idx]
#         #     rep_img_path = os.path.join(input_dir, representative_fname)

#         #     # Generate Semantic Description (Moondream via Ollama)
#         #     raw_desc = "cluster_images"
#         #     try:
#         #         print(f"👀 Asking Moondream to look at centroid: {representative_fname}")
#         #         with open(rep_img_path, 'rb') as img_f:
#         #             # Using the Italian prompt you had for Qwen
#         #             res = ollama.generate(
#         #                 model='moondream', 
#         #                 prompt="Descrivi brevemente cosa vedi in questa immagine.", 
#         #                 images=[img_f.read()]
#         #             )
#         #             raw_desc = res['response'].strip()
#         #     except Exception as e:
#         #         print(f"Vision error (Moondream): {e}")

#             # Generate Folder Title (Gemma 3 via Ollama)
#             folder_title = "semantic_cluster"
#             try:
#                 ollama_res = requests.post("http://localhost:11434/api/generate", json={
#                     "model": "gemma4:e2b",
#                     "prompt": f"Riassumi il contenuto di questo testo in una o due parole, adatte come nome di una cartella. Rispondi SOLO con le parole, niente altro.\n\nTesto: '{raw_desc}'",
#                     "stream": False
#                 })
#                 folder_title = ollama_res.json()["response"].strip()
#             except Exception as e:
#                 print(f"Ollama error: {e}")

# # --- Hardened Sanitization for Windows ---
#             clean_title = folder_title.lower()
            
#             # 1. Convert ALL weird whitespace (newlines \n, tabs \t, \r) into normal spaces
#             clean_title = re.sub(r'\s+', ' ', clean_title)
            
#             # 2. Remove all punctuation (keep only alphanumeric and spaces)
#             clean_title = re.sub(r'[^\w\s]', '', clean_title) 
            
#             # 3. Trim edges and replace spaces with underscores
#             clean_title = clean_title.strip().replace(" ", "_")
            
#             # 4. Remove double underscores
#             clean_title = re.sub(r'_+', '_', clean_title) 
            
#             # 5. THE KILL SWITCH: Force maximum length to 40 chars to prevent WinError 123
#             clean_title = clean_title[:40].strip('_')
            
#             # Fallback if the AI gives us completely unusable garbage
#             if not clean_title:
#                 clean_title = "semantic_cluster"
            
#             # Prevent collisions
#             final_folder_name = clean_title
#             counter = 1
#             while os.path.exists(os.path.join(output_dir, final_folder_name)):
#                 final_folder_name = f"{clean_title}_{counter}"
#                 counter += 1
                        
#             dest_path = os.path.join(output_dir, final_folder_name)
#             os.makedirs(dest_path, exist_ok=True)

#             # Move files and create Marker
#             for f in cluster_files:
#                 shutil.move(os.path.join(input_dir, f), os.path.join(dest_path, f))
            
#             with open(os.path.join(dest_path, marker_name), 'w') as m:
#                 m.write(representative_fname)
            
#             results.append({"label": final_folder_name, "count": len(cluster_files), "description": raw_desc})

#         return jsonify({"status": "success", "clusters": results})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 500



# @app.route("/api/organize", methods=["POST"])
# def organize_files():
#     try:
#         debug("===== ORGANIZATION STARTED =====")

#         files = collect_files_for_ai()

#         if not files:
#             debug("No readable files found.")
#             return jsonify({
#                 "reply": "I could not find readable text/code files to organize."
#             })

#         raw_plan = ask_ollama_for_organization(files)
#         raw_plan = repair_missing_llm_moves(raw_plan, files)

#         plan = normalize_plan(raw_plan)
#         result = apply_organization_plan(plan)

#         moved_count = len(result["moved_files"])
#         dir_count = len(result["created_dirs"])
#         skipped_count = len(result["skipped_files"])
#         removed_empty_count = len(result.get("removed_empty_dirs", []))

#         summary = plan.get("summary", "Organization completed.")

#         reply = (
#             f"{summary}\n\n"
#             f"Created non-empty folders: {dir_count}\n"
#             f"Moved files: {moved_count}\n"
#             f"Skipped files: {skipped_count}\n"
#             f"Removed empty folders: {removed_empty_count}"
#         )

#         if moved_count > 0:
#             reply += "\n\nMain moves:\n"

#             for move in result["moved_files"][:12]:
#                 reply += f"- {move['from']} → {move['to']}\n"

#         if result.get("removed_empty_dirs"):
#             reply += "\nRemoved empty folders:\n"

#             for folder in result["removed_empty_dirs"][:12]:
#                 reply += f"- {folder}\n"

#         if skipped_count > 0:
#             reply += "\nSkipped files:\n"

#             for skipped in result["skipped_files"][:12]:
#                 reply += f"- {skipped['path']}: {skipped['reason']}\n"

#         debug("===== ORGANIZATION FINISHED =====")

#         return jsonify({
#             "reply": reply,
#             "raw_plan": raw_plan,
#             "plan": plan,
#             "result": result
#         })

#     except requests.exceptions.ConnectionError:
#         debug("ERROR: Could not connect to Ollama.")
#         return jsonify({
#             "error": "Could not connect to Ollama. Make sure Ollama is running with: ollama serve"
#         }), 500

#     except requests.exceptions.Timeout:
#         debug("ERROR: Ollama timeout.")
#         return jsonify({
#             "error": "Ollama took too long to respond. Try with fewer files or a smaller directory."
#         }), 500

#     except Exception as e:
#         debug("ERROR:", str(e))
#         return jsonify({
#             "error": f"Organization failed: {str(e)}"
#         }), 500

if __name__ == "__main__":
    app.run(debug=True)