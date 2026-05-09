from flask import Flask, render_template, jsonify, request
from pathlib import Path
import json
import shutil
import requests

app = Flask(__name__)

# Your target folder
BASE_DIR = Path("/Users/riccardoinfascelli/Desktop/Hackathon/GDG-AI-HACK-2026/Project/PROVA").resolve()

# Ollama settings
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1"  # Change this if your local model has another name

MAX_FILE_PREVIEW_CHARS = 4000
MAX_TOTAL_FILES_FOR_AI = 80

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv",
    ".xml", ".yaml", ".yml", ".java", ".cpp", ".c", ".h", ".hpp",
    ".ts", ".tsx", ".jsx", ".sql", ".sh", ".ini", ".cfg", ".log"
}

IGNORED_DIRS = {
    "venv", ".venv", "__pycache__", ".git", "node_modules",
    ".idea", ".vscode", "dist", "build"
}

ALLOWED_FOLDER_REPLACEMENTS = {
    "python": "scripts",
    "python_files": "scripts",
    "javascript": "frontend",
    "html": "frontend",
    "css": "frontend",
    "markdown": "docs",
    "documents": "docs",
    "documentation": "docs",
    "configuration": "config",
    "configs": "config",
    "settings": "config",
    "datasets": "data",
    "dataset": "data",
    "csv": "data",
    "json_data": "data",
    "machine_learning": "models",
    "ml": "models",
    "ai": "models",
    "utilities": "scripts",
    "helpers": "scripts",
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

        relative_parts = path.relative_to(BASE_DIR).parts

        if any(part in IGNORED_DIRS for part in relative_parts):
            continue

        preview = read_file_preview(path)

        if preview is None:
            continue

        relative_path = path.relative_to(BASE_DIR)

        collected.append({
            "path": str(relative_path),
            "current_folder": str(relative_path.parent) if str(relative_path.parent) != "." else "",
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
    existing_dirs = sorted({
        str(path.relative_to(BASE_DIR))
        for path in BASE_DIR.rglob("*")
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in IGNORED_DIRS
    })

    prompt = f"""
You are an expert AI file organizer.

You will receive files from a project directory.
Each file includes:
- relative path
- filename
- extension
- current folder
- a preview of the file content

Your goal:
Reorganize files by CONTENT and PURPOSE, not just extension.

You must group files that belong together.

Examples:
- Python backend code -> backend
- Flask routes / API files -> backend
- HTML/CSS/JS UI files -> frontend
- text explanations / markdown -> docs
- datasets / csv / json data -> data
- config files -> config
- shell scripts / utility scripts -> scripts
- tests -> tests
- model files / ML logic -> models
- notebooks / experiments -> experiments
- images / static media -> assets

Folder naming rules:
- Folder names must be VERY SHORT.
- Maximum 2 words.
- Prefer 1 word.
- Use lowercase.
- Use simple names.
- Do NOT use long descriptive folder names.
- Do NOT use names like "python_scripts_related_to_ai".
- Good names: backend, frontend, docs, data, config, scripts, tests, models, experiments, assets, notes.
- Reuse existing folders if they are suitable.
- Create new folders only when needed.
- Avoid excessive nesting.
- Maximum nesting depth: 2 folders.
- Do not create folders for single files unless truly useful.
- Similar files should go into the same folder.
- If a file is already in a good location, leave it unmoved.

Safety rules:
- Return ONLY valid JSON.
- No markdown.
- No comments outside JSON.
- Do not rename files.
- Move only files, never folders.
- Destination paths must remain inside the project.
- Every "to" path must include the original filename.

Existing folders:
{json.dumps(existing_dirs, indent=2)}

Return exactly this JSON structure:

{{
  "summary": "Short summary of the organization.",
  "directories": [
    "short_folder_name"
  ],
  "moves": [
    {{
      "from": "old/relative/file.txt",
      "to": "short_folder_name/file.txt",
      "reason": "Short reason based on content similarity"
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
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1
            }
        },
        timeout=180
    )

    print("STATUS:", response.status_code)
    print("TEXT:", response.text)

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


def clean_folder_name(name):
    """
    Converts model folder names into short, safe folder names.

    Example:
    'python_scripts_related_to_ai' -> 'scripts'
    'Machine Learning Models' -> 'models'
    """
    name = str(name).strip().lower()
    name = name.replace("\\", "/")

    parts = [p for p in name.split("/") if p]
    name = parts[0] if parts else "misc"

    name = name.replace(" ", "_")
    name = "".join(ch for ch in name if ch.isalnum() or ch in "_-")

    if name in ALLOWED_FOLDER_REPLACEMENTS:
        return ALLOWED_FOLDER_REPLACEMENTS[name]

    if "frontend" in name or "ui" in name:
        return "frontend"

    if "backend" in name or "api" in name or "flask" in name:
        return "backend"

    if "doc" in name or "readme" in name:
        return "docs"

    if "data" in name or "dataset" in name or "csv" in name:
        return "data"

    if "config" in name or "setting" in name or "env" in name:
        return "config"

    if "test" in name:
        return "tests"

    if "script" in name or "utility" in name or "helper" in name:
        return "scripts"

    if "model" in name or "ml" in name or "ai" in name:
        return "models"

    if "asset" in name or "image" in name or "style" in name:
        return "assets"

    if "experiment" in name or "notebook" in name:
        return "experiments"

    if "note" in name:
        return "notes"

    chunks = [c for c in name.replace("-", "_").split("_") if c]

    if len(chunks) > 2:
        name = chunks[0]
    else:
        name = "_".join(chunks)

    if not name:
        return "misc"

    if len(name) > 16:
        return "misc"

    return name


def normalize_plan(plan):
    """
    Cleans the model plan before applying it.
    - Forces short folder names.
    - Ensures destination filename stays the same.
    - Prevents long generated directories.
    """
    normalized_dirs = set()
    normalized_moves = []

    for move in plan.get("moves", []):
        source = move.get("from", "")
        destination = move.get("to", "")
        reason = move.get("reason", "")

        if not source or not destination:
            continue

        source_path = Path(source)
        destination_path = Path(destination)

        original_filename = source_path.name

        if len(destination_path.parts) >= 2:
            folder = clean_folder_name(destination_path.parts[0])
        else:
            folder = clean_folder_name(destination_path.parent.name)

        final_destination = str(Path(folder) / original_filename)

        if source == final_destination:
            continue

        normalized_dirs.add(folder)

        normalized_moves.append({
            "from": source,
            "to": final_destination,
            "reason": reason[:160]
        })

    return {
        "summary": plan.get("summary", "Files were reorganized by content and purpose."),
        "directories": sorted(normalized_dirs),
        "moves": normalized_moves
    }


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
        files = collect_files_for_ai()

        if not files:
            return jsonify({
                "reply": "I could not find readable text/code files to organize."
            })

        raw_plan = ask_ollama_for_organization(files)
        plan = normalize_plan(raw_plan)
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
            "raw_plan": raw_plan,
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