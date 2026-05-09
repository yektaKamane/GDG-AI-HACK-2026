from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os
import json
import shutil
import requests

app = Flask(__name__)

# Your target folder
BASE_DIR = Path("/home/yekta/AI-HACK/Project/PROVA").resolve()

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"  # change this if your local model has another name

MAX_FILE_PREVIEW_CHARS = 2500
MAX_TOTAL_FILES_FOR_AI = 80

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv",
    ".xml", ".yaml", ".yml", ".java", ".cpp", ".c", ".h", ".hpp",
    ".ts", ".tsx", ".jsx", ".sql", ".sh", ".ini", ".cfg", ".log"
}


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

            node = {
                "name": item.name,
                "path": str(item.relative_to(BASE_DIR)),
                "type": "folder" if item.is_dir() else "file",
            }

            if item.is_dir():
                node["children"] = build_tree(item)

            children.append(node)
    except PermissionError:
        pass

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

    except Exception:
        return None


def collect_files_for_ai():
    collected = []

    for path in BASE_DIR.rglob("*"):
        if len(collected) >= MAX_TOTAL_FILES_FOR_AI:
            break

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        # Avoid scanning virtual environments, caches, and hidden folders
        relative_parts = path.relative_to(BASE_DIR).parts
        ignored_dirs = {
            "venv", ".venv", "__pycache__", ".git", "node_modules",
            ".idea", ".vscode", "dist", "build"
        }

        if any(part in ignored_dirs for part in relative_parts):
            continue

        preview = read_file_preview(path)

        if preview is None:
            continue

        collected.append({
            "path": str(path.relative_to(BASE_DIR)),
            "name": path.name,
            "extension": path.suffix.lower(),
            "preview": preview
        })

    return collected


def extract_json_from_llm_response(text):
    """
    Ollama may return extra text around JSON.
    This tries to extract the first valid JSON object.
    """
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
    prompt = f"""
You are an AI file organizer.

You will receive a list of files from a project directory.
Each file includes:
- path
- name
- extension
- preview of its content

Your task:
Create the most logical folder organization based on file contents and purpose.

Important rules:
- Return ONLY valid JSON.
- Do not include markdown.
- Do not explain outside JSON.
- Do not move folders, only files.
- Do not rename files.
- Keep destination paths inside the project directory.
- Use simple folder names.
- Group similar files together.
- If a file is already in the right place, you may leave it unmoved.
- Avoid excessive nesting.

Return this exact JSON structure:

{{
  "summary": "Short human-readable summary of the organization.",
  "directories": [
    "folder_name",
    "another_folder"
  ],
  "moves": [
    {{
      "from": "old/relative/file.txt",
      "to": "new/relative/file.txt",
      "reason": "Why this move makes sense"
    }}
  ]
}}

Files:
{json.dumps(files, indent=2)}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()
    raw_text = data.get("response", "")

    if not raw_text:
        raise ValueError("Empty response from Ollama")

    return extract_json_from_llm_response(raw_text)


def validate_relative_path(path_string):
    if not path_string:
        raise ValueError("Empty path")

    path = Path(path_string)

    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {path_string}")

    if ".." in path.parts:
        raise ValueError(f"Parent directory references are not allowed: {path_string}")

    return path


def unique_destination_path(destination: Path):
    """
    Prevent overwriting files.
    If file.txt exists, create file_1.txt, file_2.txt, etc.
    """
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


def apply_organization_plan(plan):
    created_dirs = []
    moved_files = []
    skipped_files = []

    directories = plan.get("directories", [])
    moves = plan.get("moves", [])

    for directory in directories:
        try:
            relative_dir = validate_relative_path(directory)
            target_dir = safe_path(relative_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(relative_dir))
        except Exception as e:
            skipped_files.append({
                "path": directory,
                "reason": f"Could not create directory: {e}"
            })

    for move in moves:
        source_relative = move.get("from", "")
        destination_relative = move.get("to", "")
        reason = move.get("reason", "")

        try:
            source_path = safe_path(validate_relative_path(source_relative))
            destination_path = safe_path(validate_relative_path(destination_relative))

            if not source_path.exists():
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Source file does not exist"
                })
                continue

            if not source_path.is_file():
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Source is not a file"
                })
                continue

            if source_path == destination_path:
                skipped_files.append({
                    "path": source_relative,
                    "reason": "Already in target location"
                })
                continue

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            final_destination = unique_destination_path(destination_path)

            shutil.move(str(source_path), str(final_destination))

            moved_files.append({
                "from": str(source_path.relative_to(BASE_DIR)),
                "to": str(final_destination.relative_to(BASE_DIR)),
                "reason": reason
            })

        except Exception as e:
            skipped_files.append({
                "path": source_relative,
                "reason": str(e)
            })

    return {
        "created_dirs": created_dirs,
        "moved_files": moved_files,
        "skipped_files": skipped_files
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

    # Simple chatbot endpoint.
    # You can later connect this to Ollama too.
    bot_reply = f"You said: {message}"

    return jsonify({
        "reply": bot_reply
    })


@app.route("/api/organize", methods=["POST"])
def organize_files():
    try:
        files = collect_files_for_ai()

        if not files:
            return jsonify({
                "reply": "I could not find readable text/code files to organize."
            })

        plan = ask_ollama_for_organization(files)
        result = apply_organization_plan(plan)

        moved_count = len(result["moved_files"])
        dir_count = len(result["created_dirs"])
        skipped_count = len(result["skipped_files"])

        summary = plan.get("summary", "Organization completed.")

        reply = (
            f"{summary}\n\n"
            f"Created folders: {dir_count}\n"
            f"Moved files: {moved_count}\n"
            f"Skipped files: {skipped_count}"
        )

        if moved_count > 0:
            reply += "\n\nMain moves:\n"
            for move in result["moved_files"][:8]:
                reply += f"- {move['from']} → {move['to']}\n"

        if skipped_count > 0:
            reply += "\nSome files were skipped for safety or because they were already correctly placed."

        return jsonify({
            "reply": reply,
            "plan": plan,
            "result": result
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Could not connect to Ollama. Make sure Ollama is running with: ollama serve"
        }), 500

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Ollama took too long to respond. Try with fewer files or a smaller directory."
        }), 500

    except Exception as e:
        return jsonify({
            "error": f"Organization failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)