// atlas.js — Atlas, an Obsidian-style markdown vault.
//
// A modal feature (same lifecycle as Gallery/Calendar) with three panes:
//   • left   — note list + search + new-note + import/export
//   • center — split markdown editor / live preview ([[wikilinks]] clickable)
//   • right  — backlinks for the open note
// Graph is a dockable panel that slides in on the right (widening the dialog)
// next to the open note: a dependency-free force-directed view of the link
// network on a <canvas>. Clicking a node opens that note. Templates live under
// templates/ and inject at the cursor; import is a drag-and-drop dialog.
//
// The vault lives as real .md files on disk; this module is a thin client over
// the /api/atlas/* endpoints (routes/atlas_routes.py).

import * as Modals from './modalManager.js';
import { mdToHtml } from './markdown.js';
import { showToast, showError, styledPrompt } from './ui.js';
import { makeWindowDraggable } from './windowDrag.js';
import { getWorkspace } from './workspace.js';

// Width (px) the graph panel adds to the dialog when docked open.
const GRAPH_PANEL_W = 340;

// Notes under this folder act as reusable templates (Obsidian convention).
const TEMPLATE_DIR = 'templates/';

const API = '/api/atlas';

function _atlasEsc(v) {
  const div = document.createElement('div');
  div.textContent = String(v ?? '');
  return div.innerHTML;
}

function _pathBaseName(p) {
  const s = String(p || '').replace(/[\\/]+$/, '');
  if (!s) return '';
  const parts = s.split(/[\\/]/);
  return parts[parts.length - 1] || s;
}

function _activeWorkspaceRoot() {
  try { return getWorkspace ? (getWorkspace() || '') : ''; }
  catch (_) { return ''; }
}


function _renderRootSection(container) {
  const root = _activeWorkspaceRoot();
  if (!root || !container) return;

  const section = document.createElement('section');
  section.className = 'atlas-files-section atlas-root-section';
  section.innerHTML = `
    <div class="atlas-files-section-head">
      <span>Root</span>
    </div>
    <div class="atlas-root-placeholder">
      <div class="atlas-root-card" title="${_atlasEsc(root)}">
        <div class="atlas-root-name">${_atlasEsc(_pathBaseName(root))}</div>
        <div class="atlas-root-path">${_atlasEsc(root)}</div>
        <div class="atlas-root-note">Workspace root is active. File view and indexing come later.</div>
      </div>
    </div>`;
  container.appendChild(section);
}

function _wireFilesPanelActions() {
  if (!_modal) return;

  const list = _modal.querySelector('.atlas-list');
  if (list && !list._atlasResetDelegate) {
    list._atlasResetDelegate = true;
    list.addEventListener('click', (e) => {
      const rootBtn = e.target.closest('[data-act="session-root"]');
      if (rootBtn) {
        e.preventDefault();
        e.stopPropagation();
        _clearSelectedFolder();
        return;
      }
      const btn = e.target.closest('[data-act="reset-session"]');
      if (!btn) return;
      e.preventDefault();
      e.stopPropagation();
      _resetSessionNotes();
    });
  }

  const btn = _modal.querySelector('[data-act="reset-session"]');
  if (!btn || btn._atlasResetWired) return;
  btn._atlasResetWired = true;
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    _resetSessionNotes();
  });
}

async function _resetSessionNotes() {
  const notes = Array.isArray(_notes) ? [..._notes].filter(n => n && n.path) : [];
  const folders = _sessionFolderList();
  const noteCount = notes.length;
  const folderCount = folders.length;

  if (!noteCount && !folderCount) {
    showToast('Nothing to reset.');
    return;
  }

  const ok = window.confirm(
    `Reset Atlas session notes and folders?\n\nThis will remove ${noteCount} note${noteCount === 1 ? '' : 's'} and ${folderCount} folder${folderCount === 1 ? '' : 's'}.\nIt will not touch the active Workspace root.`
  );
  if (!ok) return;

  try {
    await apiSend('/session/reset', 'POST', {});
  } catch (e) {
    showError('Reset failed: ' + e.message);
    return;
  }

  _current = null;
  _dirty = false;
  _selectedFolder = '';
  _emptyFolders.clear();
  _notes = [];

  const ta = _modal?.querySelector('textarea');
  if (ta) ta.value = '';

  const pathEl = _modal?.querySelector('[data-el="path"]');
  if (pathEl) pathEl.textContent = 'No note open';

  const preview = _modal?.querySelector('.atlas-preview');
  if (preview) preview.innerHTML = '';

  _renderBacklinks([]);

  await _reloadNotes();
  _renderList(_notes);
  _refreshAtlasLayoutAfterFileChange();
  if (_graphOpen) _openGraph();

  showToast('Reset session notes and folders.');
}

function _syncAtlasWorkspaceTitle() {
  if (!_modal) return;
  const subtitle = _modal.querySelector('[data-el="workspace-title"]');
  if (!subtitle) return;

  const root = _activeWorkspaceRoot();
  if (root) {
    subtitle.textContent = `Root: ${_pathBaseName(root)}`;
    subtitle.title = root;
  } else {
    subtitle.textContent = 'Session Notes';
    subtitle.title = 'Atlas session notes';
  }

  if (Array.isArray(_notes)) _renderList(_notes);

  const importBtn = _modal.querySelector('[data-act="import"]');
  if (importBtn) {
    importBtn.classList.toggle('disabled', !!root);
    importBtn.setAttribute('aria-disabled', root ? 'true' : 'false');
    importBtn.title = root
      ? 'Import is disabled while an active Workspace root is selected'
      : 'Import markdown files or a zip into Atlas session notes';
  }
}

function _guardAtlasImportAllowed() {
  const root = _activeWorkspaceRoot();
  if (!root) return true;
  showToast("Can't switch Atlas root while an active Workspace is selected. Clear Workspace first.");
  _syncAtlasWorkspaceTitle();
  return false;
}


// ── module state ────────────────────────────────────────────────────────
let _modal = null;
let _open = false;
let _notes = [];          // [{path, title, tags, size, mtime}]
let _current = null;      // path of the open note
let _dirty = false;
let _saveTimer = null;
let _graphOpen = false;   // graph docked as a right-side panel
let _graph = null;        // GraphView instance (lazy)

// ── tiny fetch helpers ──────────────────────────────────────────────────
async function apiGet(path, params) {
  const url = new URL(API + path, location.origin);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const r = await fetch(url, { credentials: 'same-origin' });
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}
async function _refreshFilesTree() {
  try {
    await _loadFolders();
  } catch (_) {}
  try {
    _renderList(Array.isArray(_notes) ? _notes : []);
  } catch (_) {}
  try {
    _refreshAtlasLayoutAfterFileChange();
  } catch (_) {}
}

async function _loadFolders() {
  try {
    const data = await apiGet('/folders');
    const folders = Array.isArray(data.folders) ? data.folders : [];
    _emptyFolders.clear();
    folders.filter(Boolean).forEach((folder) => _emptyFolders.add(folder));
  } catch (_) {
    // Keep the last known folders if the folder endpoint is temporarily unavailable.
    if (_emptyFolders && typeof _emptyFolders.clear === 'function') _emptyFolders.clear();
  }
}

async function apiSend(path, method, body) {
  const r = await fetch(API + path, {
    method,
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

// ── modal lifecycle ───────────────────────────────────────────────────────
function isAtlasOpen() { return _open; }

function _buildModal() {
  if (_modal) return _modal;
  _modal = document.createElement('div');
  _modal.id = 'atlas-modal';
  _modal.className = 'modal';
  _modal.style.display = 'none';
  _modal.innerHTML = `
    <style>
      #atlas-modal { --atlas-left-w:260px; --atlas-right-w:260px; --atlas-graph-w:340px; }
      #atlas-modal .modal-content { width: min(1200px, 96vw); height: 88vh; display:flex; flex-direction:column; border-radius:16px 10px 10px 10px; overflow:hidden; clip-path:inset(0 round 16px 10px 10px 10px); position:relative; }
      #atlas-modal .modal-content::after { content:""; position:absolute; inset:0; pointer-events:none; border:1px solid rgba(220,150,210,.28); border-radius:16px 10px 10px 10px; box-shadow:inset 0 1px 0 rgba(255,255,255,.025); z-index:5; }
      #atlas-modal .atlas-head { display:flex; align-items:center; gap:8px; padding:8px 12px; border-bottom:1px solid var(--bubble-border,#333); border-top-left-radius:16px; }
      #atlas-modal .atlas-title-wrap { display:flex; flex-direction:column; gap:1px; min-width:0; }
      #atlas-modal .atlas-title { background:none; border:0; color:inherit; padding:0; margin:0; font-family:var(--ui-font,Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif); font-size:16px; font-weight:800; letter-spacing:.16px; cursor:pointer; text-align:left; }
      #atlas-modal .atlas-title:hover { text-decoration:none; filter:none; }
      #atlas-modal .atlas-subtitle { font-size:11px; opacity:.58; max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      #atlas-modal .atlas-head .grow { flex:1; }
      #atlas-modal .atlas-tabs { display:flex; align-items:center; gap:4px; padding:6px 10px; border-bottom:1px solid var(--bubble-border,#333); }
      #atlas-modal .atlas-tab { background:transparent; border:0; border-bottom:2px solid transparent; color:inherit; opacity:.62; border-radius:0; padding:6px 10px 7px; cursor:pointer; font-size:12px; }
      #atlas-modal .atlas-tab:hover { background:transparent; opacity:.9; }
      #atlas-modal .atlas-tab.active { background:transparent; border-bottom-color:var(--brand-color,#f19ad7); opacity:1; }
      #atlas-modal .atlas-body { flex:1; display:flex; min-height:0; overflow:hidden; }
      #atlas-modal .atlas-splitter { flex:0 0 5px; width:5px; cursor:col-resize; background:transparent; border-left:1px solid transparent; border-right:1px solid transparent; }
      #atlas-modal .atlas-splitter:hover, #atlas-modal .atlas-splitter.dragging { background:rgba(241,154,215,.14); border-color:rgba(241,154,215,.35); }
      #atlas-modal[data-files="closed"] .atlas-splitter-left { display:none; }
      #atlas-modal:not([data-side="backlinks"]):not([data-side="graph"]) .atlas-splitter-side { display:none; }
      #atlas-modal[data-files="closed"] .atlas-left { display:none; }
      #atlas-modal[data-view="edit"] .atlas-preview { display:none; }
      #atlas-modal[data-view="preview"] .atlas-editor { display:none; }
      #atlas-modal[data-view="preview"] .atlas-preview { border-left:0; }
      #atlas-modal[data-side="backlinks"] .atlas-right { display:block; }
      #atlas-modal[data-side="graph"] .atlas-graph-wrap { display:flex !important; }
      #atlas-modal .atlas-left { width:var(--atlas-left-w,260px); flex:0 0 var(--atlas-left-w,260px); border-right:1px solid var(--bubble-border,#333); display:flex; flex-direction:column; min-height:0; }
      #atlas-modal .atlas-list { overflow:auto; flex:1; padding:2px 0; }
      #atlas-modal .atlas-files-section { border-bottom:1px solid color-mix(in srgb,var(--bubble-border,#333) 72%,transparent); }
      #atlas-modal .atlas-files-section-head { display:flex; align-items:center; gap:8px; padding:8px 8px 6px; font-size:11px; text-transform:uppercase; letter-spacing:.055em; opacity:.68; }
      #atlas-modal .atlas-files-section-head .grow { flex:1; }
      #atlas-modal .atlas-reset-btn { border:1px solid var(--input-border,#444); background:transparent; color:inherit; border-radius:6px; padding:3px 7px; font-size:11px; cursor:pointer; text-transform:none; letter-spacing:0; opacity:.78; }
      #atlas-modal .atlas-reset-btn:hover { opacity:1; background:rgba(127,127,127,.14); }
      #atlas-modal .atlas-section-root-btn { border:0; background:transparent; color:inherit; padding:0; margin:0; font:inherit; text-transform:inherit; letter-spacing:inherit; cursor:pointer; opacity:1; }

      #atlas-modal .atlas-session-list { min-height:20px; }
      #atlas-modal .atlas-root-placeholder { padding:2px 8px 10px; }
      #atlas-modal .atlas-root-card { border:1px solid color-mix(in srgb,var(--bubble-border,#333) 72%,transparent); border-radius:8px; padding:8px 9px; background:rgba(127,127,127,.06); }
      #atlas-modal .atlas-root-name { font-size:12px; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
      #atlas-modal .atlas-root-path { font-size:10px; opacity:.56; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:2px; }
      #atlas-modal .atlas-root-note { font-size:11px; opacity:.52; margin-top:6px; line-height:1.35; }

      #atlas-modal .atlas-row { display:flex; align-items:center; gap:4px; padding:4px 8px; cursor:pointer; font-size:13px; white-space:nowrap; overflow:hidden; border-radius:5px; }
      #atlas-modal .atlas-row:hover { background:rgba(127,127,127,.12); }
      #atlas-modal .atlas-row.active { background:var(--section-accent,rgba(120,140,255,.18)); }
      #atlas-modal .atlas-row.folder { opacity:.85; font-weight:500; }
      #atlas-modal .atlas-row.drop-target { outline:1px dashed var(--brand-color,#48f); outline-offset:-1px; }
      #atlas-modal .atlas-row .twisty { width:12px; flex:0 0 auto; opacity:.6; font-size:10px; }
      #atlas-modal .atlas-row .label { overflow:hidden; text-overflow:ellipsis; flex:1; }
      #atlas-modal .atlas-row .base-badge { font-size:9px; opacity:.55; border:1px solid currentColor; border-radius:3px; padding:0 3px; letter-spacing:.5px; }
      #atlas-modal .atlas-menu { position:absolute; z-index:30; background:var(--input-bg,#222); border:1px solid var(--input-border,#444); border-radius:8px; padding:4px; min-width:170px; box-shadow:0 6px 20px rgba(0,0,0,.4); }
      #atlas-modal .atlas-menu div { padding:7px 10px; border-radius:6px; cursor:pointer; font-size:13px; }
      #atlas-modal .atlas-menu div:hover { background:rgba(127,127,127,.18); }
      .atlas-floating-menu { z-index:2147483647 !important; position:fixed; z-index:10000; background:var(--input-bg,#222); border:1px solid var(--input-border,#444); border-radius:8px; padding:4px; min-width:170px; box-shadow:0 6px 20px rgba(0,0,0,.4); }
      .atlas-floating-menu div { padding:7px 10px; border-radius:6px; cursor:pointer; font-size:13px; }
      .atlas-floating-menu div:hover { background:rgba(127,127,127,.18); }
      #atlas-modal .atlas-preview a.atlas-tag { color:var(--brand-color,#6cf); background:rgba(110,140,255,.14); border-radius:10px; padding:0 7px; font-size:.9em; text-decoration:none; white-space:nowrap; }
      #atlas-modal .atlas-preview a.atlas-tag:hover { background:rgba(110,140,255,.28); }
      #atlas-modal .atlas-center { flex:1; display:flex; min-height:0; }
      #atlas-modal .atlas-editor, #atlas-modal .atlas-preview { flex:1; overflow:auto; padding:14px; min-width:0; }
      #atlas-modal .atlas-editor { display:flex; flex-direction:column; border-right:1px solid var(--bubble-border,#333); }
      #atlas-modal .atlas-editor textarea { flex:1; width:100%; resize:none; border:0; background:transparent; color:inherit; font-family:var(--code-font,monospace); font-size:13px; line-height:1.55; outline:none; }
      #atlas-modal .atlas-preview a.atlas-wikilink { color:var(--brand-color,#6cf); text-decoration:none; border-bottom:1px dashed currentColor; cursor:pointer; }
      #atlas-modal .atlas-preview a.atlas-wikilink.missing { color:var(--accent-error,#e66); opacity:.8; }
      #atlas-modal .atlas-right { display:none; width:var(--atlas-right-w,260px); flex:0 0 var(--atlas-right-w,260px); border-left:1px solid var(--bubble-border,#333); overflow:auto; padding:10px; font-size:12px; }
      #atlas-modal .atlas-right h4 { margin:6px 0; font-size:11px; text-transform:uppercase; opacity:.6; }
      #atlas-modal .atlas-bl { padding:4px 6px; cursor:pointer; border-radius:5px; }
      #atlas-modal .atlas-bl:hover { background:rgba(127,127,127,.15); }
      #atlas-modal .atlas-graph-wrap { width:var(--atlas-graph-w,340px); flex:0 0 var(--atlas-graph-w,340px); border-left:1px solid var(--bubble-border,#333); position:relative; display:flex; flex-direction:column; overflow:hidden; }
      #atlas-modal .atlas-graph-bar { display:flex; align-items:center; gap:8px; padding:6px 10px; border-bottom:1px solid var(--bubble-border,#333); font-size:11px; text-transform:uppercase; opacity:.7; }
      #atlas-modal canvas.atlas-graph { flex:1; width:100%; display:block; cursor:grab; }
      #atlas-modal .atlas-ed-bar { display:flex; align-items:center; gap:8px; padding:6px 12px 8px; }
      #atlas-modal .atlas-drop { border:2px dashed var(--input-border,#555); border-radius:10px; padding:28px 18px; text-align:center; display:flex; flex-direction:column; gap:10px; align-items:center; }
      #atlas-modal .atlas-drop.dragover, #atlas-import-overlay .atlas-drop.dragover { border-color:var(--brand-color,#48f); background:rgba(80,120,255,.08); }
      #atlas-import-overlay .atlas-drop { border:2px dashed var(--input-border,#555); border-radius:10px; padding:28px 18px; text-align:center; display:flex; flex-direction:column; gap:10px; align-items:center; }
      .atlas-btn { background:var(--input-bg,#222); border:1px solid var(--input-border,#444); color:inherit; border-radius:6px; padding:4px 9px; cursor:pointer; font-size:12px; }
      .atlas-btn:hover { background:rgba(127,127,127,.2); }
      .atlas-pick-item { padding:8px 10px; cursor:pointer; border-radius:6px; font-size:13px; }
      .atlas-pick-item:hover { background:rgba(127,127,127,.15); }
      #atlas-modal .atlas-btn { background:var(--input-bg,#222); border:1px solid var(--input-border,#444); color:inherit; border-radius:6px; padding:4px 9px; cursor:pointer; font-size:12px; }
      #atlas-modal .atlas-btn:hover { background:rgba(127,127,127,.2); }
      #atlas-modal .atlas-btn.active { background:transparent; color:inherit; border-color:var(--input-border,#444); box-shadow:inset 0 -2px 0 var(--brand-color,#f19ad7); }
      #atlas-modal input.atlas-search { width:100%; box-sizing:border-box; background:var(--input-bg,#222); border:1px solid var(--input-border,#444); color:inherit; border-radius:6px; padding:5px 8px; font-size:12px; }
      #atlas-modal .atlas-status { font-size:11px; opacity:.6; padding:0 12px 6px; }
    </style>
    <div class="modal-content">
      <div class="atlas-head">
        <div class="atlas-title-wrap">
          <button class="atlas-title" data-act="home" title="Clear the open note view">Ideaverse</button>
          <span class="atlas-subtitle" data-el="workspace-title">Atlas workspace</span>
        </div>
        <span class="grow"></span>
        <button class="atlas-btn" data-act="refresh" title="Refresh Ideaverse UI">↻</button>
        <button class="atlas-btn" data-act="graph-toggle" title="Show the link graph alongside your note">Graph</button>
        <button class="atlas-btn" data-act="backlinks-toggle" title="Show backlinks alongside your note">Backlinks</button>
        <button class="atlas-btn" data-act="import">Import</button>
        <button class="atlas-btn" data-act="export">Export</button>
        <button class="atlas-btn" data-act="delete" title="Delete the open note">Delete</button>
        <button class="atlas-btn" id="atlas-close" title="Close">&#x2715;</button>
      </div>
      <div class="atlas-tabs">
        <button class="atlas-tab active" data-pane="files">Files</button>
        <button class="atlas-tab active" data-pane="edit">Edit</button>
        <button class="atlas-tab" data-pane="preview">Preview</button>
      </div>
      <div class="atlas-body">
        <div class="atlas-left">
          <div style="padding:8px; display:flex; gap:6px;">
            <input class="atlas-search" placeholder="Search… (#tag)" />
            <button class="atlas-btn" data-act="new" title="New note">+</button>
          </div>
          <div class="atlas-list"></div>
        </div>
        <div class="atlas-splitter atlas-splitter-left" data-split="left" title="Resize file explorer"></div>
        <div class="atlas-center">
          <div class="atlas-editor">
            <div class="atlas-ed-bar">
              <span class="atlas-status" data-el="path">No note open</span>
            </div>
            <textarea spellcheck="false" placeholder="# Write markdown…  link with [[Other note]]"></textarea>
          </div>
          <div class="atlas-preview"></div>
        </div>
        <div class="atlas-splitter atlas-splitter-side" data-split="side" title="Resize side panel"></div>
        <div class="atlas-right">
          <h4>Backlinks</h4>
          <div data-el="backlinks"><div style="opacity:.5;">—</div></div>
        </div>
        <div class="atlas-graph-wrap" style="display:none;">
          <div class="atlas-graph-bar"><span>Graph</span><span style="flex:1"></span><button class="atlas-btn" data-act="graph-close" title="Hide graph">&#x2715;</button></div>
          <canvas class="atlas-graph"></canvas>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(_modal);

  // header actions
  _modal.querySelector('#atlas-close').onclick = closeAtlas;
  _modal.querySelector('[data-act="home"]').onclick = () => _clearOpenNoteView();
  _modal.querySelector('[data-act="new"]').onclick = (e) => _openPlusMenu(e.currentTarget);
  _modal.querySelector('[data-act="refresh"]').onclick = () => _refreshAtlasUi();
  _modal.querySelector('[data-act="graph-toggle"]').onclick = () => _toggleGraph();
  _modal.querySelector('[data-act="backlinks-toggle"]').onclick = () => _toggleBacklinks();
  _modal.querySelector('[data-act="graph-close"]').onclick = () => _toggleGraph(false);
  _modal.querySelector('[data-act="export"]').onclick = () => {
    const ok = window.confirm('Export the current Atlas vault as a zip file?');
    if (!ok) return;
    window.open(API + '/export', '_blank');
  };
  _modal.querySelector('[data-act="import"]').onclick = () => {
    if (!_guardAtlasImportAllowed()) return;
    _openImportDialog();
  };
  _modal.querySelector('[data-act="delete"]').onclick = () => _deleteCurrentNote();

  // pane tabs
  _modal.querySelectorAll('.atlas-tab').forEach((tab) => {
    tab.onclick = () => _setAtlasPane(tab.dataset.pane);
  });
  _setAtlasPane('edit', { initial: true });
  _setSidePanel(null);
  _syncAtlasWorkspaceTitle();
  _wireAtlasSplitters();

  // search
  const search = _modal.querySelector('.atlas-search');
  let st;
  search.oninput = () => { clearTimeout(st); st = setTimeout(() => _runSearch(search.value), 200); };

  // editor: debounced save
  const ta = _modal.querySelector('textarea');
  ta.addEventListener('input', () => {
    _dirty = true;
    _renderPreview(ta.value);
    clearTimeout(_saveTimer);
    _saveTimer = setTimeout(_saveCurrent, 600);
  });

  // preview: clicking a [[wikilink]] navigates
  _modal.querySelector('.atlas-preview').addEventListener('click', (e) => {
    const tag = e.target.closest('a.atlas-tag');
    if (tag) {
      e.preventDefault();
      const search = _modal.querySelector('.atlas-search');
      search.value = '#' + tag.dataset.tag;
      _runSearch(search.value);
      return;
    }
    const a = e.target.closest('a.atlas-wikilink');
    if (!a) return;
    e.preventDefault();
    const target = a.dataset.target;
    const resolved = a.dataset.resolved;
    if (resolved) openNote(resolved);
    else _newNote(target);  // ghost link → create it
  });

  return _modal;
}

async function openAtlas() {
  _buildModal();
  _modal.classList.remove('hidden');
  _modal.style.display = 'flex';
  _open = true;
  Modals.register('atlas-modal', {
    railBtnId: 'rail-atlas',
    sidebarBtnId: 'tool-atlas-btn',
    closeFn: () => _doClose(),
    restoreFn: () => {},
  });
  // Hook into Odysseus window management: one call gives drag, edge/corner
  // resize, and tile/dock-snap — same as Gallery/Calendar. Bound once.
  if (!_modal._dragWired) {
    const content = _modal.querySelector('.modal-content');
    const header = _modal.querySelector('.atlas-head');
    makeWindowDraggable(_modal, { content, header, minWidth: 720, minHeight: 420 });
    _modal._dragWired = true;
  }
  document.addEventListener('keydown', _escHandler);
  document.addEventListener('overflow-state-change', _syncAtlasWorkspaceTitle);
  await _reloadNotes();
  // Restart the graph sim if the panel was left open from a previous session.
  if (_graphOpen) _openGraph();
}

function _clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function _wireAtlasSplitters() {
  if (!_modal || _modal._atlasSplittersWired) return;
  _modal._atlasSplittersWired = true;

  const body = _modal.querySelector('.atlas-body');
  if (!body) return;

  const startDrag = (kind, e) => {
    e.preventDefault();
    const splitter = e.currentTarget;
    splitter.classList.add('dragging');

    const onMove = (ev) => {
      const rect = body.getBoundingClientRect();
      if (!rect.width) return;

      if (kind === 'left') {
        const width = _clamp(ev.clientX - rect.left, 180, 420);
        _modal.style.setProperty('--atlas-left-w', width + 'px');
        return;
      }

      const side = _modal.dataset.side;
      if (side !== 'graph' && side !== 'backlinks') return;

      const max = side === 'graph' ? 520 : 420;
      const min = side === 'graph' ? 280 : 220;
      const width = _clamp(rect.right - ev.clientX, min, max);

      if (side === 'graph') {
        _modal.style.setProperty('--atlas-graph-w', width + 'px');
        if (_graph) requestAnimationFrame(() => _graph._resize());
      } else {
        _modal.style.setProperty('--atlas-right-w', width + 'px');
      }
    };

    const stop = () => {
      splitter.classList.remove('dragging');
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', stop);
      document.removeEventListener('pointercancel', stop);
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', stop, { once: true });
    document.addEventListener('pointercancel', stop, { once: true });
  };

  _modal.querySelector('[data-split="left"]')?.addEventListener('pointerdown', (e) => startDrag('left', e));
  _modal.querySelector('[data-split="side"]')?.addEventListener('pointerdown', (e) => startDrag('side', e));
}

function _setAtlasPane(pane, opts = {}) {
  if (!_modal) return;
  if (pane === 'files') {
    const closed = _modal.dataset.files === 'closed';
    _modal.dataset.files = closed ? 'open' : 'closed';
  } else if (pane === 'edit' || pane === 'preview') {
    _modal.dataset.view = pane;
  }
  const filesOpen = _modal.dataset.files !== 'closed';
  const view = _modal.dataset.view || 'edit';
  _modal.querySelectorAll('.atlas-tab').forEach((tab) => {
    const p = tab.dataset.pane;
    tab.classList.toggle('active', p === 'files' ? filesOpen : p === view);
  });
}

function _setSidePanel(side) {
  if (!_modal) return;
  _modal.dataset.side = side || '';
  _modal.querySelector('[data-act="graph-toggle"]')?.classList.toggle('active', side === 'graph');
  _modal.querySelector('[data-act="backlinks-toggle"]')?.classList.toggle('active', side === 'backlinks');
}

function _toggleBacklinks(force) {
  const next = (typeof force === 'boolean') ? force : _modal.dataset.side !== 'backlinks';
  if (next) {
    if (_graphOpen) _toggleGraph(false);
    _setSidePanel('backlinks');
  } else {
    _setSidePanel(null);
  }
}

function _clearOpenNoteView() {
  // UI-only reset: clear the currently displayed note/editor state.
  // This does not change the selected workspace/vault and does not touch chat context.
  if (!_modal) return;
  if (_saveTimer) { clearTimeout(_saveTimer); _saveTimer = null; }
  _current = null;
  _dirty = false;
  const ta = _modal.querySelector('textarea');
  if (ta) ta.value = '';
  const path = _modal.querySelector('[data-el="path"]');
  if (path) path.textContent = 'No note open';
  const preview = _modal.querySelector('.atlas-preview');
  if (preview) preview.innerHTML = '';
  _renderBacklinks([]);
  _renderList(_notes);
}

async function _refreshAtlasUi() {
  // UI hygiene refresh only. Re-read the note list and graph, then clear the
  // open board view. Workspace selection and chat context are intentionally untouched.
  await _reloadNotes();
  await _loadFolders();
  _renderList(_notes);
  _clearOpenNoteView();
  if (_graphOpen) _openGraph();
}

function _escHandler(e) { if (e.key === 'Escape' && _open) closeAtlas(); }

function _doClose() {
  if (_saveTimer) { clearTimeout(_saveTimer); _saveCurrent(); }
  if (_graph) { _graph.stop(); }
  _open = false;
  document.removeEventListener('keydown', _escHandler);
  document.removeEventListener('overflow-state-change', _syncAtlasWorkspaceTitle);
  if (_modal) { _modal.style.display = 'none'; _modal.classList.add('hidden'); }
}
function closeAtlas() { _doClose(); }

// ── note list ─────────────────────────────────────────────────────────────
async function _reloadNotes() {
  try {
    const data = await apiGet('/notes');
    _notes = data.notes || [];
  await _loadFolders();
  } catch { _notes = []; }
  _renderList(_notes);
}

// Folder collapse state + the folder the user last clicked (drives the
// default location for "New note" / "New folder").
const _collapsed = new Set();
// Just-created folders that are still empty — the /notes API only lists files,
// so without this a new empty folder would vanish until it holds a note.
const _emptyFolders = new Set();
let _selectedFolder = '';

function _ensureDir(root, dirPath) {
  let node = root;
  const parts = dirPath.split('/');
  for (let i = 0; i < parts.length; i++) {
    const seg = parts[i];
    if (!node.dirs.has(seg)) node.dirs.set(seg, { dirs: new Map(), files: [], path: parts.slice(0, i + 1).join('/') });
    node = node.dirs.get(seg);
  }
  return node;
}

// Build a nested tree { dirs: Map<name, node>, files: [item] } from flat paths.

function _sessionFolderList() {
  if (!_emptyFolders) return [];
  if (typeof _emptyFolders.forEach === 'function') {
    const out = [];
    _emptyFolders.forEach((f) => { if (f) out.push(String(f)); });
    return out;
  }
  if (Array.isArray(_emptyFolders)) return _emptyFolders.filter(Boolean).map(String);
  return [];
}

function _resolvePathForSelectedFolder(input) {
  let value = String(input || '').trim().replace(/\\/g, '/');
  if (!value) return '';

  // Leading slash means "create at Atlas session root".
  if (value.startsWith('/')) return value.replace(/^\/+/, '');

  // Explicit paths are respected.
  if (value.includes('/')) return value.replace(/^\/+/, '');

  // Plain names inherit the selected folder.
  if (_selectedFolder) return `${_selectedFolder}/${value}`;

  return value;
}

function _clearSelectedFolder() {
  _selectedFolder = '';
  try { _renderList(_notes); } catch (_) {}
  showToast('Session root selected');
}

function _buildTree(items) {
  const root = { dirs: new Map(), files: [] };
  for (const it of items) {
    const parts = it.path.split('/');
    const dir = parts.slice(0, -1).join('/');
    const node = dir ? _ensureDir(root, dir) : root;
    node.files.push(it);
  }
  for (const f of _emptyFolders) if (f) _ensureDir(root, f);
  return root;
}

function _renderList(items) {
  const list = _modal.querySelector('.atlas-list');
  list.innerHTML = '';

  const sessionSection = document.createElement('section');
  sessionSection.className = 'atlas-files-section atlas-session-section';
  sessionSection.innerHTML = `
    <div class="atlas-files-section-head">
      <button type="button" class="atlas-section-root-btn" data-act="session-root" title="Create new notes at the Atlas session root">Session Notes</button>
      <span class="grow"></span>
      <button type="button" class="atlas-reset-btn" data-act="reset-session" title="Reset Atlas session notes and folders">Reset</button>
    </div>
    <div class="atlas-session-list"></div>`;
  list.appendChild(sessionSection);

  const sessionList = sessionSection.querySelector('.atlas-session-list');
  const hasNotes = Array.isArray(items) && items.length > 0;
  const hasFolders = !!(_emptyFolders && (
    (_emptyFolders.size && _emptyFolders.size > 0) ||
    (Array.isArray(_emptyFolders) && _emptyFolders.length > 0)
  ));

  if (!hasNotes && !hasFolders) {
    sessionList.innerHTML = '<div style="opacity:.5;padding:10px;">No session notes</div>';
  } else {
    const tree = _buildTree(Array.isArray(items) ? items : []);
    _renderTreeLevel(sessionList, tree, 0);
    // Dropping on empty session-list space moves a note to the Atlas session root.
    _wireDropTarget(sessionList, '');
  }

  _renderRootSection(list);
  _wireFilesPanelActions();
  try { _refreshDockHandlesSoon(); } catch (_) {}
}

function _renderTreeLevel(container, node, depth) {
  // Folders first (alphabetical), then files.
  const folders = [...node.dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  for (const [name, child] of folders) {
    const collapsed = _collapsed.has(child.path);
    const row = document.createElement('div');
    row.className = 'atlas-row folder' + (_selectedFolder === child.path ? ' active' : '');
    row.style.paddingLeft = (8 + depth * 12) + 'px';
    row.innerHTML = `<span class="twisty">${collapsed ? '▸' : '▾'}</span><span class="label"></span>`;
    const twisty = row.querySelector('.twisty');
    const label = row.querySelector('.label');
    label.textContent = name;
    if (twisty) {
      twisty.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (collapsed) _collapsed.delete(child.path); else _collapsed.add(child.path);
        _renderList(_notes);
      };
    }
    row.onclick = () => {
      _selectedFolder = child.path;
      if (collapsed) _collapsed.delete(child.path);
      _renderList(_notes);
    };
    _wireDropTarget(row, child.path);
    container.appendChild(row);
    if (!collapsed) _renderTreeLevel(container, child, depth + 1);
  }
  for (const it of node.files.sort((a, b) => (a.title || a.path).localeCompare(b.title || b.path))) {
    const row = document.createElement('div');
    const isBase = it.path.endsWith('.base');
    row.className = 'atlas-row' + (it.path === _current ? ' active' : '');
    row.style.paddingLeft = (8 + depth * 12 + 14) + 'px';
    row.draggable = true;
    row.innerHTML = `<span class="label"></span>${isBase ? '<span class="base-badge">BASE</span>' : ''}`;
    row.querySelector('.label').textContent = it.title || it.path.split('/').pop();
    row.title = it.path;
    row.onclick = () => openNote(it.path);
    row.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/atlas-path', it.path); });
    container.appendChild(row);
  }
}

// Make `el` accept a dropped note, moving it into `folder` via /rename.
function _wireDropTarget(el, folder) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('drop-target'); });
  el.addEventListener('dragleave', () => el.classList.remove('drop-target'));
  el.addEventListener('drop', async (e) => {
    e.preventDefault(); e.stopPropagation();
    el.classList.remove('drop-target');
    const src = e.dataTransfer.getData('text/atlas-path');
    if (!src) return;
    const base = src.split('/').pop();
    const dest = folder ? `${folder}/${base}` : base;
    if (dest === src) return;
    try {
      const res = await apiSend('/rename', 'POST', { path: src, new_path: dest });
      if (_current === src) _current = res.path;
      await _reloadNotes();
    } catch (err) { showError('Move failed: ' + err.message); }
  });
}

async function _runSearch(q) {
  q = (q || '').trim();
  if (!q) { _renderList(_notes); return; }
  try {
    const data = await apiGet('/search', { q });
    _renderList((data.results || []).map((r) => ({ path: r.path, title: r.title })));
  } catch { /* keep current list */ }
}

// ── open / edit / save a note ───────────────────────────────────────────────
async function openNote(path) {
  if (_dirty) { clearTimeout(_saveTimer); await _saveCurrent(); }
  try {
    const data = await apiGet('/note', { path });
    _current = data.path;
    _selectedFolder = data.path.includes('/') ? data.path.slice(0, data.path.lastIndexOf('/')) : '';
    _dirty = false;
    _setAtlasPane('edit');
    _modal.querySelector('textarea').value = data.content || '';
    _modal.querySelector('[data-el="path"]').textContent = data.path;
    _renderPreview(data.content || '', data.outlinks);
    _renderBacklinks(data.backlinks || []);
    _renderList(_notes);
  } catch (e) { console.error('openNote', e); }
}

async function _deleteCurrentNote() {
  if (_current) {
    const path = _current;
    const name = path.split('/').pop();
    const dirtyNote = _dirty ? '\n\nUnsaved edits in this note will be discarded.' : '';
    const ok = window.confirm(`Delete note "${name}"?\n\nThis removes the note from the Atlas vault.${dirtyNote}`);
    if (!ok) return;

    try {
      if (_saveTimer) { clearTimeout(_saveTimer); _saveTimer = null; }
      await apiSend('/delete', 'POST', { path });
      _current = null;
      _dirty = false;

      const ta = _modal.querySelector('textarea');
      if (ta) ta.value = '';

      const status = _modal.querySelector('[data-el="path"]');
      if (status) status.textContent = 'No note open';

      const preview = _modal.querySelector('.atlas-preview');
      if (preview) preview.innerHTML = '';

      _renderBacklinks([]);
      await _reloadNotes();
  await _refreshFilesTree();
      if (_graphOpen) _openGraph();
      showToast(`Deleted ${name}`);
    } catch (e) {
      showError('Delete failed: ' + e.message);
    }
    return;
  }

  if (_selectedFolder) {
    const folder = _selectedFolder;
    const ok = window.confirm(`Delete folder "${folder}" and everything inside it?\n\nThis removes all notes/files in that folder from the Atlas vault.`);
    if (!ok) return;

    try {
      await apiSend('/folder/delete', 'POST', { path: folder });
      _selectedFolder = '';
      _collapsed.clear();
      _emptyFolders.delete(folder);
      await _reloadNotes();
      if (_graphOpen) _openGraph();
      showToast(`Deleted folder ${folder}`);
    } catch (e) {
      showError('Folder delete failed: ' + e.message);
    }
    return;
  }

  showToast('Open a note or select a folder first.');
}

// The "+" button: a small menu so creating a note no longer demands you
// decide its folder up front, and template insertion lives here too.
function _openPlusMenu(anchor) {
  const existing = document.querySelector('.atlas-floating-menu[data-owner="atlas"]');
  if (existing) { existing.remove(); return; }

  const menu = document.createElement('div');
  menu.className = 'atlas-floating-menu';
  menu.style.zIndex = '2147483647';
  menu.dataset.owner = 'atlas';

  const here = _selectedFolder ? ` (in ${_selectedFolder}/)` : '';
  menu.innerHTML = `
    <div data-m="note">New note${here}</div>
    <div data-m="folder">New folder${here}</div>
    <div data-m="template">Insert template…</div>`;

  const r = anchor.getBoundingClientRect();
  menu.style.top = Math.min(window.innerHeight - 12, r.bottom + 4) + 'px';
  menu.style.left = Math.max(8, Math.min(window.innerWidth - 190, r.right - 175)) + 'px';

  document.body.appendChild(menu);

  const close = () => {
    menu.remove();
    document.removeEventListener('click', onDoc, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const onDoc = (e) => {
    if (!menu.contains(e.target) && e.target !== anchor) close();
  };
  const onKey = (e) => {
    if (e.key === 'Escape') close();
  };

  setTimeout(() => {
    document.addEventListener('click', onDoc, true);
    document.addEventListener('keydown', onKey, true);
  }, 0);

  menu.querySelector('[data-m="note"]').onclick = () => { close(); _newNote(); };
  menu.querySelector('[data-m="folder"]').onclick = () => { close(); _newFolder(); };
  menu.querySelector('[data-m="template"]').onclick = () => { close(); _insertTemplate(); };
}

// New note: default into the selected folder so there's no "which folder?"
// friction — the prompt is pre-filled and you can still edit the subpath.
async function _newNote(name) {
  const folder = _selectedFolder ? _selectedFolder + '/' : '';
  const def = typeof name === 'string' && name ? name : folder;
  const path = await styledPrompt('', {
    title: 'New note',
    defaultValue: def,
    placeholder: 'e.g. Ideas/Big Idea',
    confirmText: 'Create',
    maxLength: 200,
  });
  if (!path || !path.replace(/\/+$/, '')) return;
  try {
    const res = await apiSend('/note', 'PUT', { path, content: `# ${path.split('/').pop()}\n\n` });
    await _reloadNotes();
  await _refreshFilesTree();
    openNote(res.path);
  } catch (e) { showError('Could not create note: ' + e.message); }
}

async function _newFolder() {
  const folder = _selectedFolder ? _selectedFolder + '/' : '';
  const path = await styledPrompt('', {
    title: 'New folder',
    defaultValue: folder,
    placeholder: 'e.g. Projects/2026',
    confirmText: 'Create',
    maxLength: 200,
  });
  if (!path || !path.replace(/\/+$/, '')) return;
  try {
    const res = await apiSend('/folder', 'POST', { path });
    _selectedFolder = res.path;
    _collapsed.delete(res.path);
    _emptyFolders.add(res.path);
    await _loadFolders();
    _emptyFolders.add(res.path);
    _renderList(_notes);
    _refreshAtlasLayoutAfterFileChange();
  } catch (e) { showError('Could not create folder: ' + e.message); }
}
async function _saveCurrent() {
  if (!_current || !_dirty) return;
  const content = _modal.querySelector('textarea').value;
  try {
    await apiSend('/note', 'PUT', { path: _current, content });
    _dirty = false;
    // Refresh the cached list (title/mtime may have changed) without stealing focus.
    apiGet('/notes').then((d) => { _notes = d.notes || []; }).catch(() => {});
  } catch (e) { console.error('save', e); }
}

// ── preview (with [[wikilink]] + ![[embed]] rendering) ───────────────────────
// Map of raw target → resolved relpath for the OPEN note, so wikilinks know
// whether they point at a real note or a ghost.
let _outlinkMap = {};

function _renderPreview(src, outlinks) {
  if (Array.isArray(outlinks)) {
    _outlinkMap = {};
    for (const o of outlinks) _outlinkMap[o.target] = o.resolved;
  }
  const prepared = _preprocessMarkdown(src || '');
  const box = _modal.querySelector('.atlas-preview');
  box.innerHTML = mdToHtml(prepared, { sanitize: true });
  // mdToHtml turns [label](#atlas-link-<target>) into a normal <a>. Tag those
  // as wikilinks and mark whether they resolve to a real note (vs a ghost).
  box.querySelectorAll('a[href^="#atlas-link-"]').forEach((a) => {
    const target = decodeURIComponent(a.getAttribute('href').slice('#atlas-link-'.length));
    const resolved = _outlinkMap[target] || '';
    a.classList.add('atlas-wikilink');
    if (!resolved) a.classList.add('missing');
    a.dataset.target = target;
    a.dataset.resolved = resolved;
  });
  // #tag anchors → pills.
  box.querySelectorAll('a[href^="#atlas-tag-"]').forEach((a) => {
    a.classList.add('atlas-tag');
    a.dataset.tag = decodeURIComponent(a.getAttribute('href').slice('#atlas-tag-'.length));
  });
}

// Mirror of src/atlas_links: a fenced/inline code span we must NOT scan for
// links or tags (so `# heading` and example snippets stay literal).
const _CODE_SPAN = /```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`/g;

function _preprocessMarkdown(src) {
  // Protect code spans from the link/tag passes by stashing them behind
  // placeholders, then restoring verbatim at the end.
  const stash = [];
  src = src.replace(_CODE_SPAN, (m) => { stash.push(m); return ` ${stash.length - 1} `; });

  // Image embeds: ![[pic.png]] → markdown image served from the raw endpoint.
  src = src.replace(/!\[\[([^\]|#]+?)\]\]/g, (m, target) => {
    const t = target.trim();
    if (/\.(png|jpe?g|gif|webp|svg|bmp)$/i.test(t)) {
      const path = t.includes('/') ? t : `_attachments/${t}`;
      return `![${t}](${API}/raw?path=${encodeURIComponent(path)})`;
    }
    return `[[${target}]]`; // non-image embed → render as a normal link
  });
  // [[Target|alias]] / [[Target#heading]] / [[Target]] → a standard markdown
  // link with an #atlas-link- href (decoded + classified after render).
  src = src.replace(/\[\[([^\]]+?)\]\]/g, (m, inner) => {
    const [left, alias] = inner.split('|');
    const target = left.split('#')[0].trim();
    const label = (alias || left).trim().replace(/[[\]]/g, '');
    return `[${label}](#atlas-link-${encodeURIComponent(target)})`;
  });
  // #tag / #parent/child → a pill link. Anchored to line-start or whitespace
  // (so `# Heading` and `url#frag` are skipped) and not purely numeric — same
  // rules as the Python parser.
  src = src.replace(/(^|\s)#([A-Za-z0-9_][\w/-]*)/g, (m, pre, raw) => {
    const tag = raw.replace(/\/+$/, '');
    if (!tag || tag.replace(/\//g, '').match(/^\d+$/)) return m;
    return `${pre}[#${tag}](#atlas-tag-${encodeURIComponent(tag)})`;
  });

  return src.replace(/ (\d+) /g, (m, i) => stash[+i]);
}

function _renderBacklinks(list) {
  const box = _modal.querySelector('[data-el="backlinks"]');
  if (!list.length) { box.innerHTML = '<div style="opacity:.5;">None</div>'; return; }
  box.innerHTML = '';
  for (const path of list) {
    const d = document.createElement('div');
    d.className = 'atlas-bl';
    d.textContent = (_notes.find((n) => n.path === path) || {}).title || path;
    d.title = path;
    d.onclick = () => openNote(path);
    box.appendChild(d);
  }
}

// ── graph: a dockable panel on the right that widens the dialog ───────────────
// Width is adjusted in JS (not via a CSS class) so it composes with the inline
// width the resize handles set — the dialog grows by the panel width on open
// and shrinks back on close, capped to the viewport.
function _toggleGraph(force) {
  _graphOpen = (typeof force === 'boolean') ? force : !_graphOpen;
  const wrap = _modal.querySelector('.atlas-graph-wrap');

  if (_graphOpen) {
    _setSidePanel('graph');
    wrap.style.display = '';
    _openGraph();
  } else {
    if (_modal?.dataset.side === 'graph') _setSidePanel(null);
    wrap.style.display = 'none';
    if (_graph) _graph.stop();
  }

  _modal.querySelector('[data-act="graph-toggle"]').classList.toggle('active', _graphOpen);
}
async function _openGraph() {
  const canvas = _modal.querySelector('canvas.atlas-graph');
  let data;
  try { data = await apiGet('/graph'); } catch { data = { nodes: [], links: [] }; }
  if (!_graph) _graph = new GraphView(canvas, (id) => { if (!id.startsWith('::missing::')) openNote(id); });
  // Wait one frame so the freshly-shown panel has its real width before the
  // simulation seeds node positions / sizes the canvas.
  requestAnimationFrame(() => {
    _graph.setData(data.nodes, data.links);
    _graph.start();
  });
}

// ── templates ─────────────────────────────────────────────────────────────
function _insertAtCursor(ta, text) {
  const s = ta.selectionStart ?? ta.value.length;
  const e = ta.selectionEnd ?? ta.value.length;
  ta.value = ta.value.slice(0, s) + text + ta.value.slice(e);
  ta.selectionStart = ta.selectionEnd = s + text.length;
  ta.focus();
  ta.dispatchEvent(new Event('input'));   // re-render preview + queue save
}

async function _insertTemplate() {
  if (!_current) { showToast('Open or create a note first.'); return; }
  const templates = _notes.filter((n) => n.path.startsWith(TEMPLATE_DIR));
  if (!templates.length) {
    showToast(`No templates yet — make a note under "${TEMPLATE_DIR}" and it shows up here.`);
    return;
  }
  const pick = await _listDialog('Insert template', templates.map((t) => ({
    label: t.title || t.path.slice(TEMPLATE_DIR.length),
    value: t.path,
  })));
  if (!pick) return;
  try {
    const data = await apiGet('/note', { path: pick });
    const ta = _modal.querySelector('textarea');
    _insertAtCursor(ta, _expandTemplate(data.content || ''));
  } catch (e) { showError('Could not load template: ' + e.message); }
}

// Tiny placeholder expansion so templates can carry a live date / title,
// mirroring the most-used Obsidian core-template tokens.
function _expandTemplate(text) {
  const now = new Date();
  const iso = now.toISOString().slice(0, 10);
  const title = _current ? _current.split('/').pop().replace(/\.md$/i, '') : '';
  return text
    .replace(/\{\{\s*date\s*\}\}/gi, iso)
    .replace(/\{\{\s*time\s*\}\}/gi, now.toTimeString().slice(0, 5))
    .replace(/\{\{\s*title\s*\}\}/gi, title);
}

// ── import (drop-in dialog) / export ─────────────────────────────────────────
let _importOverlay = null;

function _refreshAtlasLayoutAfterFileChange() {
  // The shared dock seam can stale after Atlas rebuilds its file tree
  // through create/delete/reset actions. Keep this Atlas-local.
  try { _syncAtlasWorkspaceTitle(); } catch (_) {}
  try { _refreshDockHandlesSoon(); } catch (_) {
    requestAnimationFrame(() => {
      window.dispatchEvent(new Event('resize'));
      requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    });
  }
}

function _refreshDockHandlesSoon() {
  // Atlas Import is a transient overlay. The shared dock seam can hide while
  // another modal is visible, so nudge layout after Import opens/closes.
  requestAnimationFrame(() => {
    window.dispatchEvent(new Event('resize'));
    requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
  });
}

function _buildImportOverlay() {
  if (_importOverlay) return _importOverlay;
  const ov = document.createElement('div');
  ov.id = 'atlas-import-overlay';
  ov.className = 'modal';
  ov.style.display = 'none';
  ov.innerHTML = `
    <div class="modal-content" style="width:min(520px,92vw);">
      <div class="modal-header"><h4>Import to Atlas</h4></div>
      <div class="modal-body">
        <div class="atlas-drop" data-el="drop">
          <div style="font-size:13px;">Drop Markdown files or a <code>.zip</code> here</div>
          <div style="opacity:.55;font-size:11px;">a single .md works too</div>
          <div style="display:flex;gap:8px;">
            <button class="atlas-btn" data-act="pick-files">Choose files…</button>
            <button class="atlas-btn" data-act="pick-folder">Choose folder…</button>
          </div>
        </div>
      </div>
      <div class="modal-footer"><button class="confirm-btn confirm-btn-secondary" data-act="cancel">Close</button></div>
    </div>
    <input type="file" data-el="files-input" accept=".md,.markdown,.txt,.csv,.json,.zip,.png,.jpg,.jpeg,.gif,.webp,.svg" multiple style="display:none;">
    <input type="file" data-el="folder-input" webkitdirectory multiple style="display:none;">
  `;
  document.body.appendChild(ov);
  _importOverlay = ov;

  const close = () => {
    ov.style.display = 'none';
    ov.classList.add('hidden');
    _refreshDockHandlesSoon();
  };
  ov.querySelector('[data-act="cancel"]').onclick = close;
  ov.addEventListener('click', (e) => { if (e.target === ov) close(); });

  const filesInput = ov.querySelector('[data-el="files-input"]');
  const folderInput = ov.querySelector('[data-el="folder-input"]');
  ov.querySelector('[data-act="pick-files"]').onclick = () => filesInput.click();
  ov.querySelector('[data-act="pick-folder"]').onclick = () => folderInput.click();
  filesInput.onchange = (e) => { close(); _doImport(e.target.files); };
  folderInput.onchange = (e) => { close(); _doImport(e.target.files); };

  const drop = ov.querySelector('[data-el="drop"]');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove('dragover');
  }));
  drop.addEventListener('drop', (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) { close(); _doImport(files); }
  });
  return ov;
}

function _openImportDialog() {
  if (!_guardAtlasImportAllowed()) return;
  const ov = _buildImportOverlay();
  ov.classList.remove('hidden');
  ov.style.display = 'flex';
  _refreshDockHandlesSoon();
}

async function _doImport(files) {
  if (!files || !files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name);
  try {
    const r = await fetch(API + '/import', { method: 'POST', body: fd, credentials: 'same-origin' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const res = await r.json();
    showToast(`Imported ${res.imported || 0} file(s)` + (res.skipped ? `, skipped ${res.skipped}.` : '.'));
    await _reloadNotes();
  } catch (e) { showError('Import failed: ' + e.message); }
}

// Small styled single-choice picker (used for templates). Resolves the chosen
// item's value, or null on cancel. Reuses the app's modal classes so it matches
// the rest of the UI.
function _listDialog(title, items) {
  return new Promise((resolve) => {
    const ov = document.createElement('div');
    ov.className = 'modal';
    ov.style.display = 'flex';
    ov.innerHTML = `
      <div class="modal-content" style="width:min(420px,90vw);max-height:70vh;display:flex;flex-direction:column;">
        <div class="modal-header"><h4>${title}</h4></div>
        <div class="modal-body" data-el="list" style="overflow:auto;"></div>
        <div class="modal-footer"><button class="confirm-btn confirm-btn-secondary" data-act="cancel">Cancel</button></div>
      </div>`;
    document.body.appendChild(ov);
    const done = (v) => { ov.remove(); resolve(v); };
    const list = ov.querySelector('[data-el="list"]');
    for (const it of items) {
      const row = document.createElement('div');
      row.className = 'atlas-pick-item';
      row.textContent = it.label;
      row.onclick = () => done(it.value);
      list.appendChild(row);
    }
    ov.querySelector('[data-act="cancel"]').onclick = () => done(null);
    ov.addEventListener('click', (e) => { if (e.target === ov) done(null); });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// GraphView — a small dependency-free force-directed graph on <canvas>.
// O(n²) repulsion is fine for typical personal vaults (hundreds of notes);
// the simulation cools via `alpha` and is paused when not visible.
// ═══════════════════════════════════════════════════════════════════════════
class GraphView {
  constructor(canvas, onNodeClick) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onNodeClick = onNodeClick;
    this.nodes = [];
    this.links = [];
    this.alpha = 1;
    this.raf = null;
    this.scale = 1;
    this.offset = { x: 0, y: 0 };
    this.drag = null;       // node being dragged
    this.pan = null;        // background pan start
    this.hover = null;
    this._bind();
  }

  setData(nodes, links) {
    const w = this.canvas.clientWidth || 600, h = this.canvas.clientHeight || 400;
    const byId = {};
    this.nodes = nodes.map((n) => {
      const node = { ...n, x: w / 2 + (Math.random() - 0.5) * 200, y: h / 2 + (Math.random() - 0.5) * 200, vx: 0, vy: 0, deg: 0 };
      byId[n.id] = node;
      return node;
    });
    this.links = links
      .map((l) => ({ source: byId[l.source], target: byId[l.target] }))
      .filter((l) => l.source && l.target);
    this.links.forEach((l) => { l.source.deg++; l.target.deg++; });
    this.alpha = 1;
  }

  start() {
    this._resize();
    if (!this.raf) this._loop();
  }
  stop() { if (this.raf) { cancelAnimationFrame(this.raf); this.raf = null; } }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.canvas.width = w * dpr; this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  _tick() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const REPULSE = 2200, SPRING = 0.02, LEN = 70;
    const nodes = this.nodes;
    // Repulsion (every pair).
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        const f = (REPULSE / d2) * this.alpha;
        const d = Math.sqrt(d2);
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    // Spring along links.
    for (const l of this.links) {
      let dx = l.target.x - l.source.x, dy = l.target.y - l.source.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - LEN) * SPRING * this.alpha;
      const fx = (dx / d) * f, fy = (dy / d) * f;
      l.source.vx += fx; l.source.vy += fy; l.target.vx -= fx; l.target.vy -= fy;
    }
    // Gentle centering + integrate.
    for (const n of nodes) {
      if (n === this.drag) { n.vx = 0; n.vy = 0; continue; }
      n.vx += (w / 2 - n.x) * 0.0008 * this.alpha;
      n.vy += (h / 2 - n.y) * 0.0008 * this.alpha;
      n.vx *= 0.85; n.vy *= 0.85;
      n.x += n.vx; n.y += n.vy;
    }
    this.alpha *= 0.995;
    if (this.alpha < 0.02) this.alpha = 0.02; // keep a little life for dragging
  }

  _loop() {
    this._tick();
    this._draw();
    this.raf = requestAnimationFrame(() => this._loop());
  }

  _toScreen(n) { return { x: n.x * this.scale + this.offset.x, y: n.y * this.scale + this.offset.y }; }
  _toWorld(px, py) { return { x: (px - this.offset.x) / this.scale, y: (py - this.offset.y) / this.scale }; }

  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    const cs = getComputedStyle(this.canvas);
    const fg = cs.color || '#ccc';
    // edges
    ctx.strokeStyle = 'rgba(127,127,127,.35)';
    ctx.lineWidth = 1;
    for (const l of this.links) {
      const a = this._toScreen(l.source), b = this._toScreen(l.target);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    // nodes
    for (const n of this.nodes) {
      const p = this._toScreen(n);
      const r = (4 + Math.min(n.deg, 8)) * this.scale;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = n.missing ? 'rgba(230,100,100,.65)' : (n === this.hover ? '#6cf' : 'rgba(120,150,255,.85)');
      ctx.fill();
      if (this.scale > 0.6 || n === this.hover) {
        ctx.fillStyle = fg; ctx.font = `${11}px sans-serif`; ctx.textAlign = 'center';
        ctx.fillText(n.title || n.id, p.x, p.y - r - 3);
      }
    }
  }

  _nodeAt(px, py) {
    const wp = this._toWorld(px, py);
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      const r = 6 + Math.min(n.deg, 8);
      if ((n.x - wp.x) ** 2 + (n.y - wp.y) ** 2 <= r * r) return n;
    }
    return null;
  }

  _bind() {
    const c = this.canvas;
    const pos = (e) => { const r = c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    c.addEventListener('mousedown', (e) => {
      const p = pos(e); const n = this._nodeAt(p.x, p.y);
      if (n) { this.drag = n; this._moved = false; }
      else { this.pan = { x: p.x - this.offset.x, y: p.y - this.offset.y }; }
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.drag && !this.pan) {
        const p = pos(e); this.hover = this._nodeAt(p.x, p.y);
        c.style.cursor = this.hover ? 'pointer' : 'grab';
        return;
      }
      const p = pos(e); this._moved = true;
      if (this.drag) { const wp = this._toWorld(p.x, p.y); this.drag.x = wp.x; this.drag.y = wp.y; this.alpha = Math.max(this.alpha, 0.3); }
      else if (this.pan) { this.offset.x = p.x - this.pan.x; this.offset.y = p.y - this.pan.y; }
    });
    window.addEventListener('mouseup', (e) => {
      if (this.drag && !this._moved) this.onNodeClick(this.drag.id);
      this.drag = null; this.pan = null;
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const p = pos(e); const before = this._toWorld(p.x, p.y);
      this.scale *= e.deltaY < 0 ? 1.1 : 0.9;
      this.scale = Math.max(0.2, Math.min(4, this.scale));
      const after = this._toWorld(p.x, p.y);
      this.offset.x += (after.x - before.x) * this.scale;
      this.offset.y += (after.y - before.y) * this.scale;
    }, { passive: false });
  }
}

const atlasModule = { openAtlas, closeAtlas, isAtlasOpen };
export { openAtlas, closeAtlas, isAtlasOpen };
export default atlasModule;
