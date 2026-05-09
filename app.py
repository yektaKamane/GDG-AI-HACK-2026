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
OLLAMA_MODEL = "gemma3:1b"  # Change this if your local model has another name

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
- Flask routes / API files -> api
- HTML templates -> templates
- CSS / JS UI code -> ui or frontend
- authentication files -> auth
- database files -> database
- scraping files -> scrapers
- AI prompts -> prompts
- text explanations / markdown -> docs or notes
- datasets / csv / json data -> datasets
- config files -> config
- tests -> tests
- images / static media -> assets
- notebooks / experiments -> experiments

Folder naming rules:
- You are free to invent folder names based on the actual content.
- Do NOT choose from a fixed set.
- Folder names should be meaningful and specific to the project.
- Folder names must be short.
- Maximum 2 words per folder.
- Prefer 1 word when possible.
- Use lowercase.
- Use snake_case if needed.
- Good examples: api, views, auth, datasets, prompts, invoices, notes, scraping, components, experiments.
- Bad examples: python_files, random_files, miscellaneous_documents, files_related_to_ai.
- Reuse existing folders only if they are actually suitable.
- Create new folders only if at least one file will be moved there.
- Do not create empty folders.
- Avoid excessive nesting.
- Maximum nesting depth: 2 folders.
- If a file is already in a good location, leave it unmoved.

Safety rules:
- Return ONLY valid JSON.
- No markdown.
- No comments outside JSON.
- Do not rename files.
- Move only files, never folders.
- Destination paths must remain inside the project.
- Every "to" path must include the original filename.
- Do not move files into ignored folders.
- Do not use absolute paths.
- Do not use ".." in paths.

Existing folders:
{json.dumps(existing_dirs, indent=2)}

Return exactly this JSON structure:

{{
  "summary": "Short summary of the organization.",
  "moves": [
    {{
      "from": "old/relative/file.txt",
      "to": "ai_chosen_folder/original_filename.txt",
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
            "format": "json"
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


def clean_folder_part(name):
    """
    Cleans one folder name part while preserving the AI's choice.

    Example:
    'Machine Learning Notes' -> 'machine_learning'
    'API Routes!!!' -> 'api_routes'
    """
    name = str(name).strip().lower()
    name = name.replace("\\", "/")
    name = name.split("/")[-1]

    name = name.replace(" ", "_")
    name = "".join(ch for ch in name if ch.isalnum() or ch in "_-")

    chunks = [c for c in name.replace("-", "_").split("_") if c]

    # Keep folder names short: max 2 words
    chunks = chunks[:2]

    cleaned = "_".join(chunks)

    if not cleaned:
        return "misc"

    # Avoid absurdly long folder names
    if len(cleaned) > 32:
        cleaned = cleaned[:32].rstrip("_-")

    return cleaned or "misc"


def clean_folder_path(path):
    """
    Cleans a model-generated folder path.
    Allows max nesting depth of 2.

    Example:
    'Backend/API Routes/file.py' -> 'backend/api_routes'
    """
    path = str(path).strip().replace("\\", "/")
    parts = [p for p in path.split("/") if p and p not in {".", ".."}]

    cleaned_parts = [clean_folder_part(part) for part in parts[:2]]

    if not cleaned_parts:
        return Path("misc")

    return Path(*cleaned_parts)


def normalize_plan(plan):
    """
    Cleans the model plan before applying it.
    - Preserves AI-generated folder names.
    - Sanitizes folder names for filesystem safety.
    - Ensures destination filename stays the same.
    - Derives directories only from actual moves.
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

        # If AI returns only a filename, skip because there is no folder decision
        if len(destination_path.parts) < 2:
            continue

        folder_path = clean_folder_path(destination_path.parent)

        if str(folder_path).split("/")[0] in IGNORED_DIRS:
            continue

        final_destination = str(folder_path / original_filename)

        if source == final_destination:
            continue

        normalized_dirs.add(str(folder_path))

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


def remove_empty_folders(candidate_dirs):
    """
    Removes empty folders caused by organization.
    Only removes folders inside BASE_DIR.
    Never removes BASE_DIR itself.
    Never removes ignored/system folders.
    """
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

        except Exception:
            continue

    # Remove deepest folders first
    paths = sorted(set(paths), key=lambda p: len(p.parts), reverse=True)

    for path in paths:
        try:
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                removed.append(str(path.relative_to(BASE_DIR)))
        except Exception:
            pass

    return removed


def apply_organization_plan(plan):
    created_dirs = set()
    moved_files = []
    skipped_files = []
    cleanup_candidates = set()

    moves = plan.get("moves", [])

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

            old_parent = source_path.parent

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.add(str(destination_path.parent.relative_to(BASE_DIR)))

            final_destination = unique_destination_path(destination_path)

            shutil.move(str(source_path), str(final_destination))

            moved_files.append({
                "from": str(source_path.relative_to(BASE_DIR)),
                "to": str(final_destination.relative_to(BASE_DIR)),
                "reason": reason
            })

            try:
                cleanup_candidates.add(str(old_parent.relative_to(BASE_DIR)))
            except ValueError:
                pass

        except Exception as e:
            skipped_files.append({
                "path": source_relative,
                "reason": str(e)
            })

    removed_empty_dirs = remove_empty_folders(cleanup_candidates)

    # Keep only folders that still exist and are not empty
    real_created_dirs = []

    for directory in sorted(created_dirs):
        try:
            path = safe_path(validate_relative_path(directory))

            if path.exists() and path.is_dir() and any(path.iterdir()):
                real_created_dirs.append(directory)

        except Exception:
            pass

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

            for move in result["moved_files"][:8]:
                reply += f"- {move['from']} → {move['to']}\n"

        if result.get("removed_empty_dirs"):
            reply += "\nRemoved empty folders:\n"

            for folder in result["removed_empty_dirs"][:8]:
                reply += f"- {folder}\n"

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