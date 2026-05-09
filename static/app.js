const treeEl = document.getElementById("tree");
const filesEl = document.getElementById("files");
const currentPathEl = document.getElementById("currentPath");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

const organizeBtn = document.getElementById("organizeBtn");

async function loadTree() {
    const res = await fetch("/api/tree");
    const data = await res.json();

    treeEl.innerHTML = "";
    treeEl.appendChild(renderTreeNode(data));

    loadFiles("");
}

function renderTreeNode(node) {
    const wrapper = document.createElement("div");

    const item = document.createElement("div");
    item.className = "tree-node";
    item.textContent = `${node.type === "folder" ? "📁" : "📄"} ${node.name}`;

    item.addEventListener("click", (event) => {
        event.stopPropagation();

        if (node.type === "folder") {
            loadFiles(node.path);
        }
    });

    wrapper.appendChild(item);

    if (node.children && node.children.length > 0) {
        const children = document.createElement("div");
        children.className = "tree-children";

        node.children.forEach(child => {
            children.appendChild(renderTreeNode(child));
        });

        wrapper.appendChild(children);
    }

    return wrapper;
}

async function loadFiles(path) {
    const res = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
    const data = await res.json();

    filesEl.innerHTML = "";

    if (data.error) {
        filesEl.innerHTML = `<div class="file-meta">${data.error}</div>`;
        return;
    }

    currentPathEl.textContent = data.current_path || "/";

    data.files.forEach(file => {
        const row = document.createElement("div");
        row.className = "file-row";

        row.innerHTML = `
            <div>
                <div class="file-name">
                    ${file.type === "folder" ? "📁" : "📄"} ${file.name}
                </div>
                <div class="file-meta">${file.path}</div>
            </div>
            <div class="file-meta">
                ${file.type === "folder" ? "Folder" : formatBytes(file.size)}
            </div>
        `;

        if (file.type === "folder") {
            row.addEventListener("click", () => loadFiles(file.path));
        }

        filesEl.appendChild(row);
    });
}

function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    if (!bytes) return "";

    const sizes = ["B", "KB", "MB", "GB"];
    const index = Math.floor(Math.log(bytes) / Math.log(1024));

    return `${(bytes / Math.pow(1024, index)).toFixed(1)} ${sizes[index]}`;
}

function addMessage(role, text) {
    const message = document.createElement("div");
    message.className = `message ${role}`;
    message.textContent = text;

    messagesEl.appendChild(message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = chatInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    chatInput.value = "";

    const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: text })
    });

    const data = await res.json();

    if (data.reply) {
        addMessage("bot", data.reply);
    } else {
        addMessage("bot", "Something went wrong.");
    }
});

organizeBtn.addEventListener("click", async () => {
    organizeBtn.disabled = true;
    organizeBtn.textContent = "Organizing...";

    addMessage("bot", "I am analyzing the files and organizing the directory...");

    try {
        const res = await fetch("/api/organize", {
            method: "POST"
        });

        const data = await res.json();

        if (data.error) {
            addMessage("bot", `Organization failed: ${data.error}`);
        } else {
            const movedCount = data.completed.length;
            const skippedCount = data.skipped.length;

            addMessage(
                "bot",
                `${data.summary}\n\nMoved files: ${movedCount}\nSkipped files: ${skippedCount}`
            );

            await loadTree();
            await loadFiles("");
        }
    } catch (error) {
        addMessage("bot", `Organization failed: ${error.message}`);
    }

    organizeBtn.disabled = false;
    organizeBtn.textContent = "Organize";
});

loadTree();