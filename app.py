from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os

app = Flask(__name__)

# run pwd and copy the output here, replace the existing
BASE_DIR = Path("/home/yekta/AI-HACK/Project/PROVA").resolve()

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

    # Replace this with your real chatbot logic.
    bot_reply = f"You said: {message}"

    return jsonify({
        "reply": bot_reply
    })


if __name__ == "__main__":
    app.run(debug=True)