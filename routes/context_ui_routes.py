"""Manual Context Roots inspector UI.

This is a human-operated inspection surface for configured context roots.

It does not:
- call the model
- inject context into chat
- start LangGraph/tool routing
- write files
- index content
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["context-ui"])


@router.get("/context-roots", response_class=HTMLResponse)
def context_roots_page() -> str:
    return _HTML


_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Context Roots Inspector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e8ecf3;
    }

    body {
      margin: 0;
      padding: 24px;
      background: #0f1117;
    }

    main {
      max-width: 1120px;
      margin: 0 auto;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 28px;
    }

    .subtle {
      color: #9aa4b2;
      margin-bottom: 24px;
      line-height: 1.5;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(260px, 360px) 1fr;
      gap: 16px;
      align-items: start;
    }

    .card {
      background: #171a23;
      border: 1px solid #2b3140;
      border-radius: 14px;
      padding: 16px;
    }

    .root-card {
      cursor: pointer;
      margin-bottom: 12px;
    }

    .root-card:hover,
    .root-card.active {
      border-color: #6f8cff;
      background: #1d2230;
    }

    .root-title {
      font-weight: 700;
      margin-bottom: 4px;
    }

    .meta {
      color: #aeb7c5;
      font-size: 13px;
      line-height: 1.5;
      word-break: break-word;
    }

    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      color: #b7c0ce;
    }

    input, button {
      font: inherit;
    }

    input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #384154;
      border-radius: 10px;
      background: #0f121a;
      color: #e8ecf3;
      padding: 10px 12px;
      outline: none;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    button {
      border: 1px solid #4a5872;
      border-radius: 10px;
      background: #20283a;
      color: #e8ecf3;
      padding: 9px 12px;
      cursor: pointer;
    }

    button:hover {
      background: #2a344b;
    }

    button.primary {
      border-color: #6f8cff;
      background: #33428a;
    }

    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #0b0d13;
      border: 1px solid #2b3140;
      border-radius: 12px;
      padding: 12px;
      min-height: 280px;
      overflow: auto;
      line-height: 1.45;
      font-size: 13px;
    }

    .pill {
      display: inline-block;
      border: 1px solid #3a455b;
      background: #111620;
      color: #c8d1df;
      border-radius: 999px;
      padding: 2px 8px;
      margin: 2px 4px 2px 0;
      font-size: 12px;
    }

    .error {
      color: #ff9b9b;
    }

    @media (max-width: 860px) {
      body {
        padding: 14px;
      }

      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Context Roots Inspector</h1>
    <div class="subtle">
      Manual read-only inspection for configured context roots.
      This page does not call the model, inject chat context, write files, or run tools.
    </div>

    <div class="grid">
      <section class="card">
        <div class="actions" style="margin-top:0">
          <button class="primary" id="refreshRoots">Refresh roots</button>
        </div>
        <div id="rootsStatus" class="meta" style="margin-top:12px">Loading roots...</div>
        <div id="roots"></div>
      </section>

      <section class="card">
        <div class="meta">Selected root</div>
        <h2 id="selectedTitle" style="margin-top:4px">None</h2>
        <div id="selectedMeta" class="meta"></div>

        <label for="selectorInput">Selector</label>
        <input id="selectorInput" value="." placeholder="README.md or docs" />

        <label for="queryInput">Search query</label>
        <input id="queryInput" placeholder="context" />

        <div class="actions">
          <button id="listBtn">List</button>
          <button id="readBtn">Read</button>
          <button id="searchBtn">Search</button>
        </div>

        <label for="output">Output</label>
        <pre id="output">Choose a root, then list/read/search.</pre>
      </section>
    </div>
  </main>

  <script>
    let roots = [];
    let selectedRootId = null;

    const rootsEl = document.getElementById("roots");
    const rootsStatusEl = document.getElementById("rootsStatus");
    const selectedTitleEl = document.getElementById("selectedTitle");
    const selectedMetaEl = document.getElementById("selectedMeta");
    const outputEl = document.getElementById("output");
    const selectorInput = document.getElementById("selectorInput");
    const queryInput = document.getElementById("queryInput");

    function setOutput(value) {
      outputEl.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });

      const text = await response.text();
      let payload;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { raw: text };
      }

      if (!response.ok) {
        const error = new Error(`HTTP ${response.status}`);
        error.payload = payload;
        throw error;
      }

      return payload;
    }

    function renderRoots() {
      rootsEl.innerHTML = "";

      if (!roots.length) {
        rootsStatusEl.textContent = "No roots configured.";
        selectedRootId = null;
        renderSelected();
        return;
      }

      rootsStatusEl.textContent = `${roots.length} root(s) configured.`;

      for (const root of roots) {
        const div = document.createElement("div");
        div.className = "card root-card" + (root.id === selectedRootId ? " active" : "");
        div.onclick = () => {
          selectedRootId = root.id;
          renderRoots();
          renderSelected();
        };

        const caps = (root.allowed_capabilities || [])
          .map(cap => `<span class="pill">${escapeHtml(cap)}</span>`)
          .join("");

        div.innerHTML = `
          <div class="root-title">${escapeHtml(root.display_name || root.id)}</div>
          <div class="meta">
            id: ${escapeHtml(root.id)}<br>
            provider: ${escapeHtml(root.provider_type)}<br>
            enabled: ${String(root.enabled)}<br>
            scope: ${escapeHtml(root.safety_scope || "")}<br>
            verification: ${escapeHtml(root.verification?.status || "unknown")}<br>
            adapter: ${escapeHtml(root.adapter_status?.status || "unknown")}<br>
            ${caps}
          </div>
        `;

        rootsEl.appendChild(div);
      }

      renderSelected();
    }

    function renderSelected() {
      const root = roots.find(item => item.id === selectedRootId);

      if (!root) {
        selectedTitleEl.textContent = "None";
        selectedMetaEl.textContent = "No root selected.";
        return;
      }

      selectedTitleEl.textContent = root.display_name || root.id;
      selectedMetaEl.innerHTML = `
        <span class="pill">${escapeHtml(root.id)}</span>
        <span class="pill">${escapeHtml(root.provider_type)}</span>
        <span class="pill">${escapeHtml(root.safety_scope || "")}</span>
      `;
    }

    async function refreshRoots() {
      rootsStatusEl.textContent = "Loading roots...";
      try {
        const payload = await requestJson("/api/context/roots");
        roots = payload.roots || [];
        if (!selectedRootId && roots.length) {
          selectedRootId = roots[0].id;
        }
        renderRoots();
        setOutput(payload);
      } catch (err) {
        rootsStatusEl.innerHTML = `<span class="error">Failed to load roots.</span>`;
        setOutput(err.payload || String(err));
      }
    }

    function requireRoot() {
      if (!selectedRootId) {
        throw new Error("No root selected.");
      }
      return selectedRootId;
    }

    async function listSelected() {
      try {
        const payload = await requestJson("/api/context/list", {
          method: "POST",
          body: JSON.stringify({
            root_id: requireRoot(),
            selector: selectorInput.value || ".",
          }),
        });
        setOutput(payload);
      } catch (err) {
        setOutput(err.payload || String(err));
      }
    }

    async function readSelected() {
      try {
        const payload = await requestJson("/api/context/read", {
          method: "POST",
          body: JSON.stringify({
            root_id: requireRoot(),
            selector: selectorInput.value || ".",
          }),
        });
        setOutput(payload);
      } catch (err) {
        setOutput(err.payload || String(err));
      }
    }

    async function searchSelected() {
      try {
        const payload = await requestJson("/api/context/search", {
          method: "POST",
          body: JSON.stringify({
            root_id: requireRoot(),
            query: queryInput.value || "",
          }),
        });
        setOutput(payload);
      } catch (err) {
        setOutput(err.payload || String(err));
      }
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    document.getElementById("refreshRoots").onclick = refreshRoots;
    document.getElementById("listBtn").onclick = listSelected;
    document.getElementById("readBtn").onclick = readSelected;
    document.getElementById("searchBtn").onclick = searchSelected;

    refreshRoots();
  </script>
</body>
</html>
"""
