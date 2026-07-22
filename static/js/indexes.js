import { getWorkspace } from './workspace.js';

const $ = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const response = await fetch(url, {
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  let data = {};
  try {
    data = await response.json();
  } catch (_) {
    data = {};
  }

  if (!response.ok) {
    throw new Error(
      data.detail || data.error || `Request failed (${response.status})`,
    );
  }
  return data;
}

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setStatus(text, kind = '') {
  const element = $('indexes-current-status');
  if (!element) return;
  element.textContent = text;
  element.dataset.kind = kind;
}

function setBusy(busy) {
  for (const id of [
    'indexes-use-active-btn',
    'indexes-register-btn',
    'indexes-index-btn',
    'indexes-reindex-btn',
    'indexes-refresh-btn',
    'indexes-check-stale-btn',
  ]) {
    const button = $(id);
    if (button) button.disabled = busy;
  }
}

async function prepareActiveWorkspace() {
  const pathInput = $('indexes-workspace-path');
  const rootInput = $('indexes-root-id');
  if (!pathInput || !rootInput) return null;

  const active = String(getWorkspace() || '').trim();
  pathInput.value = active;
  rootInput.value = '';

  if (!active) {
    setStatus('No active workspace. Select one from the chat toolbar first.');
    return null;
  }

  const data = await api(
    `/api/indexes/suggest-root-id?path=${encodeURIComponent(active)}`,
  );
  rootInput.value = data.root_id || '';
  setStatus('Active workspace detected and ready to register.', 'success');
  return { path: active, rootId: rootInput.value };
}

function currentSelection() {
  const path = $('indexes-workspace-path')?.value.trim() || '';
  const rootId = $('indexes-root-id')?.value.trim() || '';
  if (!path || !rootId) {
    throw new Error('Select an active workspace before indexing.');
  }
  return { path, rootId };
}

async function registerCurrent() {
  const { path, rootId } = currentSelection();
  const result = await api('/api/indexes/register', {
    method: 'POST',
    body: JSON.stringify({
      root_id: rootId,
      path,
    }),
  });
  setStatus(`Registered ${result.root_id}.`, 'success');
  await refreshRoots();
  return result;
}

async function indexCurrent(rebuild) {
  setBusy(true);
  try {
    const registered = await registerCurrent();
    setStatus(
      rebuild ? 'Reindexing workspace...' : 'Indexing workspace...',
    );
    const result = await api(
      rebuild ? '/api/indexes/reindex' : '/api/indexes/index',
      {
        method: 'POST',
        body: JSON.stringify({ root_id: registered.root_id }),
      },
    );
    const stats = result.stats || {};
    setStatus(
      `Indexed ${stats.note_count ?? 0} files into ${result.root_id} ` +
      `(${stats.chunk_count ?? 0} chunks, ${stats.embedding_model || 'nomic-embed-text:latest'}).`,
      'success',
    );
    await refreshRoots();
    return result;
  } finally {
    setBusy(false);
  }
}

function formatEmbeddingSize(root) {
  const sizeMb = Number(root.embedding_size_mb || 0);
  if (!root.is_indexed && sizeMb <= 0) return 'Not indexed';
  if (sizeMb > 0 && sizeMb < 0.01) return `${sizeMb.toFixed(4)} MB`;
  return `${sizeMb.toFixed(2)} MB`;
}

function rootCard(root) {
  const rootId = esc(root.root_id || root.id || '');
  const path = esc(root.path || root.local_path || '');
  const model = esc(root.embedding_model || 'nomic-embed-text:latest');
  const displayName = esc(root.display_name || rootId);
  return `
    <div class="indexes-root-card" data-root-id="${rootId}">
      <div class="indexes-root-summary">
        <strong title="${displayName}">${displayName}</strong>
        <button type="button" class="indexes-expand-button" data-index-action="toggle" aria-expanded="false" title="Show index details" aria-label="Show details for ${displayName}">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
        </button>
      </div>
      <div class="indexes-root-details" hidden>
        <div class="indexes-root-info">
          <span class="indexes-root-path" title="${path}">${path}</span>
          <span>Root ID: <code>${rootId}</code></span>
          <span>Embedding model: <code>${model}</code></span>
          <span title="Estimated float32 vector payload stored for this isolated index">Embedding size: ${esc(formatEmbeddingSize(root))}</span>
          <span>Last indexed: ${esc(root.last_indexed || 'Never')}</span>
          <span>Last queried: ${esc(root.last_queried || 'Never')}</span>
        </div>
        <div class="indexes-root-actions">
          <button type="button" data-index-action="stats" title="Show the file, chunk, and query statistics for this index">Stats</button>
          <button type="button" data-index-action="reindex" title="Clear this index and rebuild it from the saved workspace">Reindex</button>
          <button type="button" data-index-action="flush" class="danger" title="Delete indexed chunks while keeping the saved workspace root">Flush Index</button>
          <button type="button" data-index-action="remove" class="danger" title="Remove this saved workspace root, with an option to flush its index">Remove Root</button>
        </div>
        <pre class="indexes-root-stats" hidden></pre>
      </div>
    </div>`;
}

async function refreshRoots() {
  const list = $('indexes-root-list');
  if (!list) return;
  const data = await api('/api/indexes');
  const roots = data.roots || [];
  list.innerHTML = roots.length
    ? roots.map(rootCard).join('')
    : '<div class="indexes-empty">No saved roots yet.</div>';
}

async function rootAction(event) {
  const button = event.target.closest('[data-index-action]');
  if (!button) return;
  const card = button.closest('[data-root-id]');
  const rootId = card?.dataset.rootId || '';
  const action = button.dataset.indexAction;
  if (!rootId || !action) return;

  if (action === 'toggle') {
    const details = card.querySelector('.indexes-root-details');
    if (!details) return;
    const expanded = details.hidden;
    details.hidden = !expanded;
    button.setAttribute('aria-expanded', String(expanded));
    button.setAttribute('title', expanded ? 'Hide index details' : 'Show index details');
    const name = card.querySelector('.indexes-root-summary strong')?.textContent || rootId;
    button.setAttribute('aria-label', `${expanded ? 'Hide' : 'Show'} details for ${name}`);
    return;
  }

  setBusy(true);
  try {
    if (action === 'stats') {
      const result = await api(`/api/indexes/${encodeURIComponent(rootId)}/stats`);
      const output = card.querySelector('.indexes-root-stats');
      if (output) {
        output.hidden = false;
        output.textContent = JSON.stringify(result.index || {}, null, 2);
      }
      return;
    }

    if (action === 'reindex') {
      setStatus(`Reindexing ${rootId}...`);
      const result = await api('/api/indexes/reindex', {
        method: 'POST',
        body: JSON.stringify({ root_id: rootId }),
      });
      const stats = result.stats || {};
      setStatus(
        `Reindexed ${rootId}: ${stats.note_count ?? 0} files, ${stats.chunk_count ?? 0} chunks.`,
        'success',
      );
      await refreshRoots();
      return;
    }

    if (action === 'flush') {
      if (!window.confirm(
        `Flush index "${rootId}"?\n\nWorkspace files will not be deleted.`,
      )) return;
      const result = await api(
        `/api/indexes/${encodeURIComponent(rootId)}/index`,
        { method: 'DELETE' },
      );
      setStatus(
        `Flushed ${result.deleted_chunks || 0} chunks from ${rootId}.`,
        'success',
      );
      await refreshRoots();
      return;
    }

    if (action === 'remove') {
      if (!window.confirm(
        `Remove root "${rootId}"?\n\nWorkspace files will not be deleted.`,
      )) return;
      const flush = window.confirm('Also flush its isolated ChromaDB index?');
      await api(
        `/api/indexes/${encodeURIComponent(rootId)}?flush_index=${flush}`,
        { method: 'DELETE' },
      );
      setStatus(`Removed root ${rootId}.`, 'success');
      await refreshRoots();
    }
  } finally {
    setBusy(false);
  }
}

async function showStaleRoots() {
  const list = $('indexes-stale-list');
  if (!list) return;
  const days = Number($('indexes-stale-days')?.value || 30);
  const data = await api(
    `/api/indexes/stale?older_than_days=${encodeURIComponent(days)}`,
  );
  const roots = data.roots || [];
  list.innerHTML = roots.length
    ? roots.map(rootCard).join('')
    : '<div class="indexes-empty">No stale indexes found.</div>';
}

function safely(task) {
  return (...args) => {
    Promise.resolve(task(...args)).catch((error) => {
      setBusy(false);
      setStatus(error.message || String(error), 'error');
    });
  };
}

function bind() {
  $('indexes-use-active-btn')?.addEventListener('click', safely(prepareActiveWorkspace));
  $('indexes-register-btn')?.addEventListener('click', safely(registerCurrent));
  $('indexes-index-btn')?.addEventListener('click', safely(() => indexCurrent(false)));
  $('indexes-reindex-btn')?.addEventListener('click', safely(() => indexCurrent(true)));
  $('indexes-refresh-btn')?.addEventListener('click', safely(refreshRoots));
  $('indexes-check-stale-btn')?.addEventListener('click', safely(showStaleRoots));
  $('indexes-root-list')?.addEventListener('click', safely(rootAction));
  $('indexes-stale-list')?.addEventListener('click', safely(rootAction));

  document.addEventListener('workspace-changed', safely(prepareActiveWorkspace));
  document.addEventListener('click', (event) => {
    if (!event.target.closest('.settings-nav-item[data-settings-tab="indexes"]')) return;
    safely(async () => {
      await prepareActiveWorkspace();
      await refreshRoots();
    })();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bind, { once: true });
} else {
  bind();
}
