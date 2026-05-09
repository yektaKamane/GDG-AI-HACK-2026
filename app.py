from flask import Flask, render_template, jsonify, request
from pathlib import Path
import shutil
import requests
import json
import os

app = Flask(__name__)

BASE_DIR = Path("/home/yekta/AI-HACK/Project/PROVA").resolve()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

MAX_FILE_CHARS = 6000
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".html", ".css", ".json",
    ".csv", ".xml", ".yaml", ".yml", ".java", ".cpp", ".c",
    ".h", ".hpp", ".ts", ".tsx", ".jsx", ".sql"
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


def read_text_file(path: Path):
    try:
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return None

        if path.stat().st_size > 2_000_000:
            return None

        return path.read_text(encoding="utf-8", errors="ignore")[:MAX_FILE_CHARS]
    except Exception:
        return None


def collect_file_context():
    files = []

    for path in BASE_DIR.rglob("*"):
        if path.is_dir():
            continue

        if path.name.startswith("."):
            continue

        if any(part.startswith(".") for part in path.relative_to(BASE_DIR).parts):
            continue

        content = read_text_file(path)

        files.append({
            "path": str(path.relative_to(BASE_DIR)),
            "name": path.name,
            "extension": path.suffix.lower(),
            "content": content if content else "[Unreadable or skipped binary/large file]"
        })

    return files


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    return data["response"]


def get_ai_organization_plan(files):
    prompt = f"""
You are an AI file organization assistant.

You will receive a list of files from a project directory.
Analyze filenames and file contents.
Create the most logical folder organization.

Rules:
- Return ONLY valid JSON.
- Do not include markdown.
- Do not move files outside the base directory.
- Do not delete files.
- Prefer simple clear folder names.
- Similar files should go into the same folder.
- If files are already well placed, you may keep them where they are.
- Avoid overly deep nesting.

JSON format:
{{
  "summary": "short explanation of the organization",
  "moves": [
    {{
      "source": "old/relative/path/file.txt",
      "destination": "new/relative/path/file.txt"
    }}
  ]
}}

Files:
{json.dumps(files, indent=2)}
"""

    raw = call_ollama(prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "").strip()

        return json.loads(cleaned)


def apply_moves(plan):
    completed = []
    skipped = []

    for move in plan.get("moves", []):
        source_rel = move.get("source")
        dest_rel = move.get("destination")

        if not source_rel or not dest_rel:
            skipped.append({"move": move, "reason": "Missing source or destination"})
            continue

        try:
            source = safe_path(source_rel)
            destination = safe_path(dest_rel)
        except ValueError:
            skipped.append({"move": move, "reason": "Unsafe path"})
            continue

        if not source.exists():
            skipped.append({"move": move, "reason": "Source does not exist"})
            continue

        if source.is_dir():
            skipped.append({"move": move, "reason": "Moving directories is not allowed"})
            continue

        if source.resolve() == destination.resolve():
            skipped.append({"move": move, "reason": "Source and destination are the same"})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        final_destination = destination
        counter = 1

        while final_destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            final_destination = destination.parent / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.move(str(source), str(final_destination))

        completed.append({
            "source": str(source.relative_to(BASE_DIR)),
            "destination": str(final_destination.relative_to(BASE_DIR))
        })

    return completed, skipped


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

    prompt = f"""
You are a helpful assistant inside a file explorer app.
The current base directory is: {BASE_DIR}

User message:
{message}
"""

    try:
        reply = call_ollama(prompt)
    except Exception as e:
        reply = f"Error calling Ollama: {str(e)}"

    return jsonify({"reply": reply})


@app.route("/api/organize", methods=["POST"])
def organize_files():
    try:
        files = collect_file_context()

        if not files:
            return jsonify({
                "summary": "No readable files were found to organize.",
                "completed": [],
                "skipped": []
            })

        plan = get_ai_organization_plan(files)
        completed, skipped = apply_moves(plan)

        summary = plan.get("summary", "Organization completed.")

        return jsonify({
            "summary": summary,
            "completed": completed,
            "skipped": skipped
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)