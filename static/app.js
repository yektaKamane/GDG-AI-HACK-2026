const treeEl = document.getElementById("tree");
const filesEl = document.getElementById("files");
const currentPathEl = document.getElementById("currentPath");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");

// Button elements
const organizeButton = document.getElementById("organizeButton");
const organizePhotosButton = document.getElementById("organizePhotosButton");
const organizeExFolderButton = document.getElementById("organizeExFolderButton");

let selectedPath = "";

async function loadTree() {
    const res = await fetch("/api/tree");
    const data = await res.json();

    treeEl.innerHTML = "";
    treeEl.appendChild(renderTreeNode(data));
}

function renderTreeNode(node) {
    const wrapper = document.createElement("div");

    const item = document.createElement("div");
    item.className = "tree-node";
    item.textContent = `${node.type === "folder" ? "📁" : "📄"} ${node.name}`;

    if (node.path === selectedPath) {
        item.classList.add("selected");
    }

    item.addEventListener("click", (event) => {
        event.stopPropagation();

        if (node.type === "folder") {
            selectedPath = node.path;
            loadFiles(node.path);
            loadTree();
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

    selectedPath = data.current_path || "";
    currentPathEl.textContent = data.current_path || "/";

    if (data.files.length === 0) {
        filesEl.innerHTML = `<div class="empty-state">This folder is empty.</div>`;
        return;
    }

    data.files.forEach(file => {
        const row = document.createElement("div");
        row.className = "file-row";

        row.innerHTML = `
            <div>
                <div class="file-name">
                    ${file.type === "folder" ? "📁" : "📄"} ${escapeHtml(file.name)}
                </div>
                <div class="file-meta">${escapeHtml(file.path)}</div>
            </div>
            <div class="file-meta">
                ${file.type === "folder" ? "Folder" : formatBytes(file.size)}
            </div>
        `;

        if (file.type === "folder") {
            row.addEventListener("click", () => {
                selectedPath = file.path;
                loadFiles(file.path);
                loadTree();
            });
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

// --- Loading States for Buttons ---

function setOrganizeLoading(isLoading) {
    organizeButton.disabled = isLoading;
    organizeButton.textContent = isLoading ? "Organizing..." : "Organize";
}

function setOrganizePhotosLoading(isLoading) {
    organizePhotosButton.disabled = isLoading;
    organizePhotosButton.textContent = isLoading ? "Organizing..." : "Organize Photos";
}

function setOrganizeExFolderLoading(isLoading) {
    organizeExFolderButton.disabled = isLoading;
    organizeExFolderButton.textContent = isLoading ? "Organizing..." : "Ex. Folders";
}

// --- Refresh Helper ---

async function refreshExplorer() {
    await loadTree();
    await loadFiles(selectedPath);
}

// --- Chat Form Listener ---

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = chatInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    chatInput.value = "";

    try {
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
            addMessage("bot", data.error || "Something went wrong.");
        }
    } catch (error) {
        addMessage("bot", "Could not contact the backend.");
    }
});

// --- Organize Buttons Listeners ---

// 1. Original Organize Button
organizeButton.addEventListener("click", async () => {
    addMessage("user", "Organize this project directory.");
    addMessage("bot", "I am analyzing the files and preparing an organization plan...");

    setOrganizeLoading(true);

    try {
        const res = await fetch("/api/organize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await res.json();

        if (!res.ok) {
            addMessage("bot", data.error || "Organization failed.");
            return;
        }

        addMessage("bot", data.reply || "Organization completed.");
        await refreshExplorer();

    } catch (error) {
        addMessage("bot", "Could not contact the backend while organizing.");
    } finally {
        setOrganizeLoading(false);
    }
});

// 2. Organize Photos Button
organizePhotosButton.addEventListener("click", async () => {
    addMessage("user", "Organize the photos in this directory.");
    addMessage("bot", "I am running the vision model and clustering the photos...");

    setOrganizePhotosLoading(true);

    try {
        const res = await fetch("/api/organize_photos", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await res.json();

        if (!res.ok) {
            addMessage("bot", data.error || data.message || "Photo organization failed.");
            return;
        }

        // Handle the specific JSON response structure from api_organize_photos
        if (data.status === "success") {
            addMessage("bot", `Success! Organized photos into ${data.clusters.length} clusters.`);
        } else {
            addMessage("bot", data.reply || "Photo organization completed.");
        }

        await refreshExplorer();

    } catch (error) {
        addMessage("bot", "Could not contact the backend while organizing photos.");
    } finally {
        setOrganizePhotosLoading(false);
    }
});

// 3. Organize with Existing Folders Button
organizeExFolderButton.addEventListener("click", async () => {
    addMessage("user", "Organize files into the existing folders.");
    addMessage("bot", "I am matching files to your current folder structure...");

    setOrganizeExFolderLoading(true);

    try {
        const res = await fetch("/api/organize_with_ex_folder", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await res.json();

        if (!res.ok) {
            addMessage("bot", data.error || data.message || "Organization failed.");
            return;
        }

        addMessage("bot", data.reply || "Organization into existing folders completed.");
        await refreshExplorer();

    } catch (error) {
        addMessage("bot", "Could not contact the backend while organizing.");
    } finally {
        setOrganizeExFolderLoading(false);
    }
});

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function init() {
    await loadTree();
    await loadFiles("");
}

init();