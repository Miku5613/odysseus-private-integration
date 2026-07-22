const INDEX_CACHE_MS = 10_000;

let cachedRoots = [];
let cachedAt = 0;
let inFlight = null;

export function normalizeWorkspacePath(path) {
  return String(path || '')
    .replaceAll('\\', '/')
    .replace(/\/+$/, '')
    .toLowerCase();
}

export function findIndexForPath(roots, path) {
  const wanted = normalizeWorkspacePath(path);
  if (!wanted) return null;
  return (roots || []).find((root) => normalizeWorkspacePath(
    root.path || root.local_path || '',
  ) === wanted) || null;
}

export function indexIconMarkup(title = 'Workspace is indexed') {
  const safeTitle = String(title || 'Workspace is indexed')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
  return `<span class="workspace-index-icon" title="${safeTitle}" aria-label="${safeTitle}">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>
    </svg>
  </span>`;
}

export async function loadIndexRoots({ force = false } = {}) {
  if (!force && cachedAt && Date.now() - cachedAt < INDEX_CACHE_MS) {
    return cachedRoots;
  }
  if (inFlight) return inFlight;

  inFlight = fetch('/api/indexes', { credentials: 'same-origin' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`Unable to load saved roots (${response.status})`);
      const data = await response.json();
      cachedRoots = Array.isArray(data.roots) ? data.roots : [];
      cachedAt = Date.now();
      return cachedRoots;
    })
    .finally(() => { inFlight = null; });
  return inFlight;
}
