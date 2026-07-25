// static/js/runtimeControl.js
//
// Standalone Agent-runtime selector and OpenCode UI adapter.
// OpenCode remains an external peer runtime. This module never falls back to
// Legacy or LangGraph when an OpenCode run fails.

import uiModule from './ui.js';
import sessionModule from './sessions.js';
import chatRenderer from './chatRenderer.js';
import markdownModule from './markdown.js';

const DEFAULT_RUNTIME_KEY = 'odysseus-agent-runtime-default-v1';
const SESSION_RUNTIME_KEY = 'odysseus-agent-runtime-sessions-v1';
const REUSE_SESSION_KEY = 'odysseus-opencode-reuse-session-v1';
const RESET_ON_SWITCH_KEY = 'odysseus-opencode-reset-on-switch-v1';
const OPENCODE_MODEL_KEY = 'odysseus-opencode-model-v1';
const OPENCODE_WORKSPACE_KEY = 'odysseus-opencode-workspace-v1';
const OPENCODE_PERMISSION_MODE_KEY = 'odysseus-opencode-permission-mode-v1';

const RUNTIMES = new Set(['legacy', 'langgraph', 'opencode']);
let _pendingRuntime = null;
let _activeRun = null;
let _initialized = false;
let _runtimePanel = null;
let _runtimeTab = null;
let _indicator = null;
let _quickResetButton = null;
let _healthMessage = null;
let _defaultMessage = null;

function safeStorageGet(key) {
  try { return window.localStorage.getItem(key); } catch (_) { return null; }
}

function safeStorageSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch (_) {}
}

function normalizeRuntime(value, fallback = 'legacy') {
  const runtime = String(value || '').trim().toLowerCase();
  return RUNTIMES.has(runtime) ? runtime : fallback;
}

function loadSessionMap() {
  try {
    const parsed = JSON.parse(safeStorageGet(SESSION_RUNTIME_KEY) || '{}');
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed
      : {};
  } catch (_) {
    return {};
  }
}

function saveSessionMap(map) {
  safeStorageSet(SESSION_RUNTIME_KEY, JSON.stringify(map || {}));
}

function boolSetting(key, fallback) {
  const raw = safeStorageGet(key);
  if (raw == null) return fallback;
  return raw === 'true';
}

function permissionMode() {
  const value = String(
    safeStorageGet(OPENCODE_PERMISSION_MODE_KEY) || 'ask'
  ).trim().toLowerCase();
  return value === 'workspace' ? 'workspace' : 'ask';
}

function currentOdysseusSessionId() {
  return String(
    (
      sessionModule.getCurrentSessionId
      && sessionModule.getCurrentSessionId()
    )
    || ''
  ).trim();
}

export function getDefaultRuntime() {
  return normalizeRuntime(safeStorageGet(DEFAULT_RUNTIME_KEY), 'langgraph');
}

export function isAgentMode() {
  const button = document.getElementById('mode-agent-btn');
  if (!button) return true;
  return button.classList.contains('active')
    || button.getAttribute('aria-pressed') === 'true';
}

export function ensureSessionRuntime(sessionId) {
  const id = String(sessionId || '').trim();
  if (!id) return normalizeRuntime(_pendingRuntime, getDefaultRuntime());

  const map = loadSessionMap();
  if (!RUNTIMES.has(map[id])) {
    map[id] = normalizeRuntime(_pendingRuntime, getDefaultRuntime());
    saveSessionMap(map);
  }
  _pendingRuntime = null;
  refreshRuntimeIndicator();
  return map[id];
}

export function getRuntimeForSession(sessionId) {
  const id = String(sessionId || '').trim();
  if (!id) return normalizeRuntime(_pendingRuntime, getDefaultRuntime());
  const map = loadSessionMap();
  return normalizeRuntime(map[id], getDefaultRuntime());
}

async function postLangGraphSetting(runtime) {
  const selected = normalizeRuntime(runtime);
  const legacyInput = document.getElementById('set-agentLangGraphEnabled');
  if (legacyInput) legacyInput.checked = selected === 'langgraph';

  try {
    const response = await fetch('/api/auth/settings', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_langgraph_enabled: selected === 'langgraph',
      }),
    });
    if (!response.ok) {
      throw new Error(`Settings returned HTTP ${response.status}`);
    }
    return true;
  } catch (error) {
    console.warn('Runtime setting sync failed:', error);
    return false;
  }
}

export async function prepareNativeRuntime(runtime) {
  const selected = normalizeRuntime(runtime);
  if (selected === 'opencode') return true;
  return postLangGraphSetting(selected);
}

function runtimeLabel(runtime) {
  if (runtime === 'opencode') return 'OpenCode';
  if (runtime === 'langgraph') return 'LangGraph';
  return 'Legacy';
}

function dispatchRuntimeChanged(runtime, sessionId = '') {
  try {
    window.dispatchEvent(new CustomEvent('odysseus:runtime-changed', {
      detail: {
        runtime,
        sessionId,
        defaultRuntime: getDefaultRuntime(),
      },
    }));
  } catch (_) {}
}

export async function setDefaultRuntime(runtime) {
  const selected = normalizeRuntime(runtime);
  safeStorageSet(DEFAULT_RUNTIME_KEY, selected);
  await postLangGraphSetting(selected);

  if (selected === 'legacy') {
    await setCurrentRuntime('legacy');
  }

  syncSettingsControls();
  refreshRuntimeIndicator();
  dispatchRuntimeChanged(selected, '');
  return selected;
}

export async function setCurrentRuntime(runtime, sessionId = null) {
  const selected = normalizeRuntime(runtime);
  const id = String(
    sessionId
      || (sessionModule.getCurrentSessionId && sessionModule.getCurrentSessionId())
      || ''
  ).trim();

  const previous = getRuntimeForSession(id);
  if (id) {
    const map = loadSessionMap();
    map[id] = selected;
    saveSessionMap(map);
  } else {
    _pendingRuntime = selected;
  }

  if (
    previous === 'opencode'
    && selected !== 'opencode'
    && boolSetting(RESET_ON_SWITCH_KEY, false)
  ) {
    await resetOpenCodeSession(id);
  }

  if (selected !== 'opencode') {
    await postLangGraphSetting(selected);
  }

  syncSettingsControls();
  refreshRuntimeIndicator();
  dispatchRuntimeChanged(selected, id);
  return selected;
}

export async function cycleCurrentRuntime() {
  const id = sessionModule.getCurrentSessionId
    ? sessionModule.getCurrentSessionId()
    : '';
  const current = getRuntimeForSession(id);
  const order = ['legacy', 'langgraph', 'opencode'];
  const index = Math.max(0, order.indexOf(current));
  const next = order[(index + 1) % order.length];
  return setCurrentRuntime(next, id);
}

function hideLegacyLangGraphSetting() {
  const checkbox = document.getElementById('set-agentLangGraphEnabled');
  if (!checkbox) return;
  const row = checkbox.closest('.settings-row');
  if (row) row.style.display = 'none';
  const help = row && row.nextElementSibling;
  if (
    help
    && String(help.textContent || '').toLowerCase().includes('legacy runtime')
  ) {
    help.style.display = 'none';
  }
}

function runtimeIconSvg(size = 16) {
  return `
    <svg width="${size}" height="${size}" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round"
         aria-hidden="true">
      <circle cx="6" cy="6" r="2"></circle>
      <circle cx="18" cy="6" r="2"></circle>
      <circle cx="12" cy="18" r="2"></circle>
      <path d="M7.7 7.2 10.8 16"></path>
      <path d="m16.3 7.2-3.1 8.8"></path>
      <path d="M8 6h8"></path>
    </svg>`;
}

function replaceIndexesAttributes(root) {
  for (const element of [root, ...root.querySelectorAll('*')]) {
    for (const attribute of Array.from(element.attributes || [])) {
      if (!/indexes/i.test(attribute.value || '')) continue;
      element.setAttribute(
        attribute.name,
        String(attribute.value).replace(/indexes/ig, 'runtime')
      );
    }
    if (element.id && /indexes/i.test(element.id)) {
      element.id = element.id.replace(/indexes/ig, 'runtime');
    }
  }
}

function findIndexesTab(modal) {
  const preferred = modal.querySelectorAll(
    'button,[role="tab"],[data-settings-tab],[data-tab],' +
    '[data-settings-section],.settings-tab,.settings-nav-item,' +
    '.settings-sidebar-item,.settings-menu-item'
  );
  const direct = Array.from(preferred).find(
    element => String(element.textContent || '').trim() === 'Indexes'
  );
  if (direct) return direct;

  return Array.from(modal.querySelectorAll('*')).find(element => {
    if (String(element.textContent || '').trim() !== 'Indexes') return false;
    if (element.closest('[data-settings-panel="indexes"]')) return false;
    return element.children.length <= 3;
  }) || null;
}

function createRuntimePanel() {
  const panel = document.createElement('div');
  panel.dataset.settingsPanel = 'runtime';
  panel.className = 'hidden admin-only runtime-settings-panel';
  panel.innerHTML = `
    <div class="admin-card runtime-settings-card runtime-selector-card">
      <div class="runtime-card-head">
        <div class="runtime-card-heading">
          <span class="runtime-card-kicker">Agent orchestration</span>
          <h2 class="runtime-card-title">
            ${runtimeIconSvg(15)}
            <span>Runtime Selector</span>
          </h2>
          <p class="settings-help-text">
            Choose the default Agent engine for newly created chats.
            Legacy remains available from the composer runtime control.
          </p>
        </div>
        <div class="runtime-default-pill" aria-live="polite">
          <span class="runtime-status-dot"></span>
          <span>Default</span>
          <strong id="runtime-default-label">Legacy</strong>
        </div>
      </div>

      <div class="runtime-selector-grid">
        <label class="runtime-choice-row runtime-choice-langgraph">
          <span class="runtime-choice-icon">
            ${runtimeIconSvg(17)}
          </span>
          <span class="runtime-choice-copy">
            <strong>LangGraph</strong>
            <small>Odysseus-native supervised Agent runtime.</small>
          </span>
          <span class="runtime-toggle">
            <input id="set-runtime-langgraph" type="checkbox"
                   role="switch" aria-label="Enable LangGraph">
            <span class="runtime-toggle-track" aria-hidden="true">
              <span class="runtime-toggle-thumb"></span>
            </span>
          </span>
        </label>

        <label class="runtime-choice-row runtime-choice-opencode">
          <span class="runtime-choice-icon">
            <svg width="17" height="17" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true">
              <path d="M8 9 4 12l4 3"></path>
              <path d="m16 9 4 3-4 3"></path>
              <path d="m14 5-4 14"></path>
            </svg>
          </span>
          <span class="runtime-choice-copy">
            <strong>OpenCode</strong>
            <small>Standalone coding runtime with native tools.</small>
          </span>
          <span class="runtime-toggle">
            <input id="set-runtime-opencode" type="checkbox"
                   role="switch" aria-label="Enable OpenCode">
            <span class="runtime-toggle-track" aria-hidden="true">
              <span class="runtime-toggle-thumb"></span>
            </span>
          </span>
        </label>
      </div>

      <div id="runtime-default-message"
           class="runtime-settings-message runtime-inline-message"></div>
    </div>

    <div class="admin-card runtime-settings-card runtime-console-card">
      <div class="runtime-card-head runtime-console-head">
        <div class="runtime-card-heading">
          <span class="runtime-card-kicker">Standalone runtime</span>
          <h2 class="runtime-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2"
                 stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true">
              <ellipse cx="12" cy="5" rx="7" ry="3"></ellipse>
              <path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5"></path>
              <path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"></path>
            </svg>
            <span>OpenCode Console</span>
          </h2>
          <p class="settings-help-text">
            Configure the managed server, workspace policy, and native
            conversation lifecycle.
          </p>
        </div>
        <div class="runtime-health-pill">
          <span class="runtime-health-dot"></span>
          <span>Server</span>
          <strong id="runtime-health-message">Not checked</strong>
        </div>
      </div>

      <div class="runtime-console-layout">
        <section class="runtime-section runtime-environment-section">
          <div class="runtime-section-head">
            <div>
              <strong>Environment</strong>
              <small>Applied when the managed server restarts.</small>
            </div>
          </div>

          <div class="runtime-config-grid">
            <label class="runtime-field">
              <span>Model</span>
              <div class="runtime-field-shell">
                <input id="runtime-opencode-model" type="text"
                       placeholder="qwen3:8b" autocomplete="off">
                <button id="runtime-use-active-model-btn" type="button"
                        class="runtime-field-action"
                        title="Use active chat model">
                  Use chat
                </button>
              </div>
            </label>

            <label class="runtime-field">
              <span>Workspace</span>
              <div class="runtime-field-shell">
                <input id="runtime-opencode-workspace" type="text"
                       placeholder="D:\\path\\to\\workspace"
                       autocomplete="off">
                <button id="runtime-use-active-workspace-btn" type="button"
                        class="runtime-field-action"
                        title="Use active workspace">
                  Use active
                </button>
              </div>
            </label>
          </div>

          <div class="runtime-primary-row">
            <button id="runtime-apply-restart-btn" type="button"
                    class="runtime-primary-action"
                    aria-label="Apply &amp; restart OpenCode">
              Apply changes &amp; restart
            </button>
            <div id="runtime-config-message"
                 class="runtime-settings-message"></div>
          </div>
        </section>

        <section class="runtime-section runtime-permission-section">
          <div class="runtime-section-head">
            <div>
              <strong>Permission policy</strong>
              <small>Odysseus owns approvals across OpenCode restarts.</small>
            </div>
            <span class="runtime-section-badge">Session scoped</span>
          </div>

          <label class="runtime-field">
            <span>Handling mode</span>
            <select id="runtime-opencode-permission-mode"
                    class="runtime-permission-select">
              <option value="ask">Ask for Approval</option>
              <option value="workspace">
                Allow safe workspace file handling
              </option>
            </select>
          </label>

          <p class="runtime-policy-note">
            Workspace mode auto-approves ordinary file operations only when
            their resolved paths remain inside the configured workspace.
            Shell, external, sensitive, repeated, and unknown actions still
            require approval.
          </p>

          <div class="runtime-approval-row">
            <div id="runtime-permission-message"
                 class="runtime-settings-message">
              New chats ask for approval.
            </div>
            <button id="runtime-clear-permission-approvals-btn"
                    type="button" class="runtime-subtle-action">
              Clear approvals
            </button>
          </div>
        </section>

        <section class="runtime-section runtime-behavior-section">
          <div class="runtime-section-head">
            <div>
              <strong>Conversation lifecycle</strong>
              <small>Control how native OpenCode context is retained.</small>
            </div>
          </div>

          <div class="runtime-session-options">
            <label class="runtime-choice-row runtime-compact-choice">
              <span class="runtime-choice-copy">
                <strong>Reuse per chat</strong>
                <small>Continue the same native conversation.</small>
              </span>
              <span class="runtime-toggle">
                <input id="set-opencode-reuse-session" type="checkbox"
                       checked role="switch"
                       aria-label="Reuse OpenCode session">
                <span class="runtime-toggle-track" aria-hidden="true">
                  <span class="runtime-toggle-thumb"></span>
                </span>
              </span>
            </label>

            <label class="runtime-choice-row runtime-compact-choice">
              <span class="runtime-choice-copy">
                <strong>Reset when leaving</strong>
                <small>Start fresh after switching runtimes.</small>
              </span>
              <span class="runtime-toggle">
                <input id="set-opencode-reset-on-switch" type="checkbox"
                       role="switch"
                       aria-label="Reset when leaving OpenCode">
                <span class="runtime-toggle-track" aria-hidden="true">
                  <span class="runtime-toggle-thumb"></span>
                </span>
              </span>
            </label>
          </div>
        </section>
      </div>

      <div class="runtime-action-dock">
        <div class="runtime-action-dock-copy">
          <strong>Runtime controls</strong>
          <small>New session preserves the Odysseus chat.</small>
        </div>
        <div class="runtime-session-actions">
          <button id="runtime-health-btn" type="button">
            Check server
          </button>
          <button id="runtime-reset-session-btn" type="button"
                  title="Restart current session">
            New session
          </button>
          <button id="runtime-log-terminal-btn" type="button"
                  title="Open log terminal">
            Open logs
          </button>
          <button id="runtime-stop-server-btn" type="button"
                  class="runtime-danger-action"
                  title="Close OpenCode">
            Close server
          </button>
        </div>
      </div>
    </div>`;
  return panel;
}

function activateRuntimePanel() {
  const modal = document.getElementById('settings-modal');
  if (!modal || !_runtimePanel || !_runtimeTab) return;

  modal.querySelectorAll('[data-settings-panel]').forEach(panel => {
    panel.classList.add('hidden');
  });
  _runtimePanel.classList.remove('hidden');

  const parent = _runtimeTab.parentElement;
  if (parent) {
    Array.from(parent.children).forEach(item => {
      item.classList.remove('active');
      if (item.hasAttribute('aria-selected')) {
        item.setAttribute('aria-selected', 'false');
      }
    });
  }
  _runtimeTab.classList.add('active');
  if (_runtimeTab.hasAttribute('aria-selected')) {
    _runtimeTab.setAttribute('aria-selected', 'true');
  }
  refreshPermissionState();
}

function buildRuntimeSettings() {
  const modal = document.getElementById('settings-modal');
  if (!modal || modal.querySelector('[data-settings-panel="runtime"]')) {
    _runtimePanel = modal && modal.querySelector(
      '[data-settings-panel="runtime"]'
    );
    return;
  }

  const indexesPanel = modal.querySelector(
    '[data-settings-panel="indexes"]'
  );
  const indexesTab = findIndexesTab(modal);
  if (!indexesPanel || !indexesTab || !indexesTab.parentNode) {
    console.warn('Runtime settings insertion point was not found.');
    return;
  }

  const runtimeTab = indexesTab.cloneNode(true);
  replaceIndexesAttributes(runtimeTab);
  runtimeTab.innerHTML = `${runtimeIconSvg(14)}<span>Runtime</span>`;
  runtimeTab.classList.remove('active');
  runtimeTab.setAttribute('aria-selected', 'false');
  runtimeTab.addEventListener('click', event => {
    event.preventDefault();
    event.stopPropagation();
    activateRuntimePanel();
  });

  const runtimePanel = createRuntimePanel();
  indexesTab.parentNode.insertBefore(runtimeTab, indexesTab);
  indexesPanel.parentNode.insertBefore(runtimePanel, indexesPanel);
  _runtimeTab = runtimeTab;
  _runtimePanel = runtimePanel;

  modal.addEventListener('click', event => {
    const candidate = event.target.closest(
      'button,[role="tab"],[data-settings-tab],[data-tab],' +
      '[data-settings-section],.settings-tab,.settings-nav-item,' +
      '.settings-sidebar-item,.settings-menu-item'
    );
    if (!candidate || candidate === runtimeTab) return;
    if (runtimePanel.contains(candidate)) return;
    setTimeout(() => {
      if (!runtimeTab.classList.contains('active')) {
        runtimePanel.classList.add('hidden');
      }
    }, 0);
  });

  wireRuntimeSettings();
}

function syncSettingsControls() {
  const runtime = getDefaultRuntime();
  const langGraph = document.getElementById('set-runtime-langgraph');
  const openCode = document.getElementById('set-runtime-opencode');
  const label = document.getElementById('runtime-default-label');

  if (langGraph) langGraph.checked = runtime === 'langgraph';
  if (openCode) openCode.checked = runtime === 'opencode';
  if (label) label.textContent = runtimeLabel(runtime);

  const reuse = document.getElementById('set-opencode-reuse-session');
  const reset = document.getElementById('set-opencode-reset-on-switch');
  if (reuse) reuse.checked = boolSetting(REUSE_SESSION_KEY, true);
  if (reset) reset.checked = boolSetting(RESET_ON_SWITCH_KEY, false);

  const permissionSelect = document.getElementById(
    'runtime-opencode-permission-mode'
  );
  if (permissionSelect && document.activeElement !== permissionSelect) {
    permissionSelect.value = permissionMode();
  }

  const model = document.getElementById('runtime-opencode-model');
  const workspace = document.getElementById('runtime-opencode-workspace');
  if (model && document.activeElement !== model) {
    model.value = safeStorageGet(OPENCODE_MODEL_KEY) || '';
  }
  if (workspace && document.activeElement !== workspace) {
    workspace.value = safeStorageGet(OPENCODE_WORKSPACE_KEY) || '';
  }
}

function showDefaultMessage(message, error = false) {
  if (!_defaultMessage) {
    _defaultMessage = document.getElementById('runtime-default-message');
  }
  if (!_defaultMessage) return;
  _defaultMessage.textContent = message || '';
  _defaultMessage.classList.toggle('error', !!error);
}


function showConfigMessage(message, error = false) {
  const target = document.getElementById('runtime-config-message');
  if (!target) return;
  target.textContent = message || '';
  target.classList.toggle('error', !!error);
}

function showPermissionMessage(message, error = false) {
  const target = document.getElementById('runtime-permission-message');
  if (!target) return;
  target.textContent = message || '';
  target.classList.toggle('error', !!error);
}

async function refreshPermissionState() {
  const sessionId = currentOdysseusSessionId();
  if (!sessionId) {
    showPermissionMessage(
      permissionMode() === 'workspace'
        ? 'New chats will auto-allow ordinary file handling inside the workspace.'
        : 'New chats will ask for approval.'
    );
    return null;
  }

  try {
    const payload = await runtimeJsonRequest(
      `/api/opencode/runtime/permissions/${encodeURIComponent(sessionId)}`
    );
    const select = document.getElementById(
      'runtime-opencode-permission-mode'
    );
    if (select && document.activeElement !== select) {
      select.value = payload.mode || permissionMode();
    }
    const approvalCount = Array.isArray(payload.approvals)
      ? payload.approvals.length
      : 0;
    const pendingCount = Array.isArray(payload.pending)
      ? payload.pending.length
      : 0;
    showPermissionMessage(
      `${approvalCount} chat approval${approvalCount === 1 ? '' : 's'}`
      + (pendingCount ? ` · ${pendingCount} pending` : '')
    );
    return payload;
  } catch (error) {
    showPermissionMessage(
      `Permission state unavailable: ${error.message || error}`,
      true
    );
    return null;
  }
}

async function savePermissionMode(mode) {
  const normalized = mode === 'workspace' ? 'workspace' : 'ask';
  safeStorageSet(OPENCODE_PERMISSION_MODE_KEY, normalized);
  const sessionId = currentOdysseusSessionId();
  if (!sessionId) {
    showPermissionMessage(
      normalized === 'workspace'
        ? 'New chats will auto-allow ordinary file handling inside the workspace.'
        : 'New chats will ask for approval.'
    );
    return { ok: true, mode: normalized };
  }

  const payload = await runtimeJsonRequest(
    `/api/opencode/runtime/permissions/${encodeURIComponent(sessionId)}/mode`,
    {
      method: 'PUT',
      body: JSON.stringify({ mode: normalized }),
    }
  );
  await refreshPermissionState();
  return payload;
}

async function clearCurrentPermissionApprovals() {
  const sessionId = currentOdysseusSessionId();
  if (!sessionId) {
    showPermissionMessage('No active Odysseus chat.', true);
    return null;
  }
  const payload = await runtimeJsonRequest(
    `/api/opencode/runtime/permissions/${encodeURIComponent(sessionId)}/approvals`,
    { method: 'DELETE' }
  );
  showPermissionMessage(
    `Cleared ${Number(payload.removed || 0)} chat approval`
    + (Number(payload.removed || 0) === 1 ? '' : 's')
  );
  await refreshPermissionState();
  return payload;
}

function configuredOpenCodeValues() {
  const modelInput = document.getElementById('runtime-opencode-model');
  const workspaceInput = document.getElementById(
    'runtime-opencode-workspace'
  );
  return {
    model_id: String(modelInput && modelInput.value || '').trim(),
    workspace: String(workspaceInput && workspaceInput.value || '').trim(),
  };
}

function storeConfiguredOpenCodeValues(values) {
  if (values.model_id) {
    safeStorageSet(OPENCODE_MODEL_KEY, values.model_id);
  }
  if (values.workspace) {
    safeStorageSet(OPENCODE_WORKSPACE_KEY, values.workspace);
  }
}

function activeChatModel() {
  const value = sessionModule.getCurrentModel
    ? sessionModule.getCurrentModel()
    : '';
  const model = String(value || '').trim();
  return model && model !== 'Select model' ? model : '';
}

async function activeWorkspacePath() {
  const pathInput = document.getElementById('indexes-workspace-path');
  const useActive = document.getElementById('indexes-use-active-btn');

  if (useActive) {
    useActive.click();
    await new Promise(resolve => setTimeout(resolve, 180));
  }

  return String(pathInput && pathInput.value || '').trim();
}

async function runtimeJsonRequest(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) {}
  if (!response.ok || payload.ok === false) {
    const detail = payload.detail || payload.error || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }
  return payload;
}

export async function restartOpenCodeServer() {
  if (_activeRun) {
    showConfigMessage('Stop the active OpenCode run before restarting.', true);
    return null;
  }

  const values = configuredOpenCodeValues();
  if (!values.model_id) {
    showConfigMessage('Choose an OpenCode model first.', true);
    return null;
  }
  if (!values.workspace) {
    showConfigMessage('Choose an OpenCode workspace first.', true);
    return null;
  }

  showConfigMessage('Restarting OpenCode…');
  try {
    const payload = await runtimeJsonRequest(
      '/api/opencode/runtime/server/restart',
      {
        method: 'POST',
        body: JSON.stringify(values),
      }
    );
    storeConfiguredOpenCodeValues(values);
    showConfigMessage(
      'OpenCode restarted. Existing native sessions were cleared.'
    );
    await refreshOpenCodeHealth();
    return payload;
  } catch (error) {
    showConfigMessage(`Restart failed: ${error.message || error}`, true);
    return null;
  }
}

export async function stopOpenCodeServer() {
  if (_activeRun) {
    setHealthMessage('Stop the active run first', true);
    return null;
  }
  setHealthMessage('Closing…');
  try {
    const payload = await runtimeJsonRequest(
      '/api/opencode/runtime/server/stop',
      { method: 'POST', body: '{}' }
    );
    setHealthMessage(payload.stopped ? 'Closed' : 'Already closed');
    return payload;
  } catch (error) {
    setHealthMessage(`Close failed: ${error.message || error}`, true);
    return null;
  }
}

export async function openOpenCodeLogTerminal() {
  try {
    const sessionId = String(
      (
        sessionModule.getCurrentSessionId
        && sessionModule.getCurrentSessionId()
      )
      || ''
    ).trim();
    const payload = await runtimeJsonRequest(
      '/api/opencode/runtime/server/log-terminal',
      {
        method: 'POST',
        body: JSON.stringify({
          odysseus_session_id: sessionId,
        }),
      }
    );
    setHealthMessage(
      payload.scope === 'session'
        ? 'Session log terminal opened'
        : 'Server log terminal opened'
    );
    return payload;
  } catch (error) {
    setHealthMessage(`Log terminal failed: ${error.message || error}`, true);
    return null;
  }
}

function wireRuntimeSettings() {
  const langGraph = document.getElementById('set-runtime-langgraph');
  const openCode = document.getElementById('set-runtime-opencode');
  const reuse = document.getElementById('set-opencode-reuse-session');
  const reset = document.getElementById('set-opencode-reset-on-switch');
  const modelInput = document.getElementById('runtime-opencode-model');
  const workspaceInput = document.getElementById(
    'runtime-opencode-workspace'
  );
  const useModel = document.getElementById(
    'runtime-use-active-model-btn'
  );
  const useWorkspace = document.getElementById(
    'runtime-use-active-workspace-btn'
  );
  const applyRestart = document.getElementById(
    'runtime-apply-restart-btn'
  );
  const health = document.getElementById('runtime-health-btn');
  const resetSession = document.getElementById(
    'runtime-reset-session-btn'
  );
  const stopServer = document.getElementById(
    'runtime-stop-server-btn'
  );
  const logTerminal = document.getElementById(
    'runtime-log-terminal-btn'
  );
  const permissionSelect = document.getElementById(
    'runtime-opencode-permission-mode'
  );
  const clearApprovals = document.getElementById(
    'runtime-clear-permission-approvals-btn'
  );

  if (langGraph) {
    langGraph.addEventListener('change', async () => {
      showDefaultMessage('Saving…');
      const runtime = langGraph.checked ? 'langgraph' : 'legacy';
      if (openCode && langGraph.checked) openCode.checked = false;
      await setDefaultRuntime(runtime);
      showDefaultMessage(`New chats will use ${runtimeLabel(runtime)}.`);
    });
  }

  if (openCode) {
    openCode.addEventListener('change', async () => {
      showDefaultMessage('Saving…');
      const runtime = openCode.checked ? 'opencode' : 'legacy';
      if (langGraph && openCode.checked) langGraph.checked = false;
      await setDefaultRuntime(runtime);
      showDefaultMessage(`New chats will use ${runtimeLabel(runtime)}.`);
    });
  }

  if (reuse) {
    reuse.addEventListener('change', () => {
      safeStorageSet(REUSE_SESSION_KEY, String(!!reuse.checked));
    });
  }

  if (reset) {
    reset.addEventListener('change', () => {
      safeStorageSet(RESET_ON_SWITCH_KEY, String(!!reset.checked));
    });
  }

  for (const input of [modelInput, workspaceInput]) {
    if (!input) continue;
    input.addEventListener('input', () => {
      showConfigMessage('Configuration changed — apply and restart.');
    });
  }

  if (useModel) {
    useModel.addEventListener('click', () => {
      const model = activeChatModel();
      if (!model || !modelInput) {
        showConfigMessage('No active chat model is selected.', true);
        return;
      }
      modelInput.value = model;
      showConfigMessage('Active chat model copied. Apply to restart.');
    });
  }

  if (useWorkspace) {
    useWorkspace.addEventListener('click', async () => {
      useWorkspace.disabled = true;
      const workspace = await activeWorkspacePath();
      useWorkspace.disabled = false;
      if (!workspace || !workspaceInput) {
        showConfigMessage(
          'No active workspace path is available. Enter it manually.',
          true
        );
        return;
      }
      workspaceInput.value = workspace;
      showConfigMessage('Active workspace copied. Apply to restart.');
    });
  }

  if (applyRestart) {
    applyRestart.addEventListener('click', async () => {
      applyRestart.disabled = true;
      await restartOpenCodeServer();
      applyRestart.disabled = false;
    });
  }

  if (health) {
    health.addEventListener('click', refreshOpenCodeHealth);
  }

  if (resetSession) {
    resetSession.addEventListener('click', async () => {
      const id = sessionModule.getCurrentSessionId
        ? sessionModule.getCurrentSessionId()
        : '';
      if (!id) {
        setHealthMessage('No active chat', true);
        return;
      }
      resetSession.disabled = true;
      const ok = await resetOpenCodeSession(id);
      resetSession.disabled = false;
      setHealthMessage(
        ok
          ? 'Current OpenCode session closed; next message starts fresh'
          : 'Session restart failed',
        !ok
      );
    });
  }

  if (stopServer) {
    stopServer.addEventListener('click', stopOpenCodeServer);
  }

  if (logTerminal) {
    logTerminal.addEventListener('click', openOpenCodeLogTerminal);
  }

  if (permissionSelect) {
    permissionSelect.addEventListener('change', async () => {
      permissionSelect.disabled = true;
      try {
        await savePermissionMode(permissionSelect.value);
      } catch (error) {
        showPermissionMessage(
          `Permission mode failed: ${error.message || error}`,
          true
        );
      } finally {
        permissionSelect.disabled = false;
      }
    });
  }

  if (clearApprovals) {
    clearApprovals.addEventListener('click', async () => {
      clearApprovals.disabled = true;
      try {
        await clearCurrentPermissionApprovals();
      } catch (error) {
        showPermissionMessage(
          `Clear approvals failed: ${error.message || error}`,
          true
        );
      } finally {
        clearApprovals.disabled = false;
      }
    });
  }

  syncSettingsControls();
  refreshPermissionState();
}

function setHealthMessage(message, error = false) {
  if (!_healthMessage) {
    _healthMessage = document.getElementById('runtime-health-message');
  }
  if (!_healthMessage) return;
  _healthMessage.textContent = message || '';
  _healthMessage.classList.toggle('error', !!error);
}

export async function refreshOpenCodeHealth() {
  setHealthMessage('Checking…');
  try {
    const response = await fetch('/api/opencode/runtime/health', {
      credentials: 'same-origin',
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    const launch = payload.launch || {};
    const server = payload.server || {};
    const model = payload.model || {};
    const ownership = launch.owned ? 'managed' : 'external';
    const version = server.version ? ` v${server.version}` : '';
    const modelName = model.model_id ? ` · ${model.model_id}` : '';
    setHealthMessage(`${ownership}${version}${modelName}`);

    const modelInput = document.getElementById('runtime-opencode-model');
    const workspaceInput = document.getElementById(
      'runtime-opencode-workspace'
    );
    const configuredModel = String(model.model_id || '').trim();
    const configuredWorkspace = String(payload.workspace || '').trim();

    if (modelInput && !modelInput.value && configuredModel) {
      modelInput.value = configuredModel;
      safeStorageSet(OPENCODE_MODEL_KEY, configuredModel);
    }
    if (workspaceInput && !workspaceInput.value && configuredWorkspace) {
      workspaceInput.value = configuredWorkspace;
      safeStorageSet(OPENCODE_WORKSPACE_KEY, configuredWorkspace);
    }
    return payload;
  } catch (error) {
    setHealthMessage(`Unavailable: ${error.message || error}`, true);
    return null;
  }
}

export async function resetOpenCodeSession(sessionId) {
  const id = String(sessionId || '').trim();
  if (!id) return false;
  try {
    const response = await fetch(
      `/api/opencode/runtime/conversation/${encodeURIComponent(id)}`,
      {
        method: 'DELETE',
        credentials: 'same-origin',
      }
    );
    return response.ok;
  } catch (_) {
    return false;
  }
}

function createRuntimeIndicator() {
  if (_indicator || document.getElementById('runtime-indicator-btn')) {
    _indicator = document.getElementById('runtime-indicator-btn');
    return;
  }

  const right = document.querySelector('.chat-input-right');
  if (!right) return;

  const button = document.createElement('button');
  button.type = 'button';
  button.id = 'runtime-indicator-btn';
  button.className =
    'input-icon-btn tool-indicator runtime-indicator-btn';
  button.style.display = 'none';
  button.innerHTML = `
    <span class="runtime-indicator-icon">${runtimeIconSvg(14)}</span>
    <span class="runtime-indicator-label">Legacy</span>`;

  button.addEventListener('click', async event => {
    event.preventDefault();
    event.stopPropagation();
    await cycleCurrentRuntime();
  });

  const modeToggle = right.querySelector('.mode-toggle');
  if (modeToggle) {
    right.insertBefore(button, modeToggle);
  } else {
    right.insertBefore(button, right.firstChild);
  }
  _indicator = button;
  refreshRuntimeIndicator();
}

function createRuntimeQuickReset() {
  if (
    _quickResetButton
    || document.getElementById('runtime-quick-reset-btn')
  ) {
    _quickResetButton = document.getElementById(
      'runtime-quick-reset-btn'
    );
    return;
  }

  const right = document.querySelector('.chat-input-right');
  if (!right) return;

  const button = document.createElement('button');
  button.type = 'button';
  button.id = 'runtime-quick-reset-btn';
  button.className =
    'input-icon-btn tool-indicator runtime-quick-reset-btn';
  button.style.display = 'none';
  button.title = 'Start a fresh OpenCode session';
  button.setAttribute(
    'aria-label',
    'Start a fresh OpenCode session'
  );
  button.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round"
         aria-hidden="true">
      <path d="M20 11a8.1 8.1 0 0 0-15.5-2"></path>
      <path d="M4 4v5h5"></path>
      <path d="M4 13a8.1 8.1 0 0 0 15.5 2"></path>
      <path d="M20 20v-5h-5"></path>
    </svg>`;

  button.addEventListener('click', async event => {
    event.preventDefault();
    event.stopPropagation();
    if (_activeRun) return;

    const id = sessionModule.getCurrentSessionId
      ? sessionModule.getCurrentSessionId()
      : '';
    if (!id) return;

    button.disabled = true;
    button.classList.add('is-busy');
    const ok = await resetOpenCodeSession(id);
    button.classList.remove('is-busy');
    button.classList.toggle('is-success', !!ok);
    button.disabled = false;
    button.title = ok
      ? 'Fresh OpenCode session ready'
      : 'OpenCode session reset failed';
    setTimeout(() => {
      button.classList.remove('is-success');
      button.title = 'Start a fresh OpenCode session';
    }, 1800);
  });

  const modeToggle = right.querySelector('.mode-toggle');
  if (modeToggle) {
    right.insertBefore(button, modeToggle);
  } else {
    right.appendChild(button);
  }
  _quickResetButton = button;
}

export function refreshRuntimeIndicator() {
  if (!_indicator) {
    _indicator = document.getElementById('runtime-indicator-btn');
  }
  if (!_indicator) return;

  const id = sessionModule.getCurrentSessionId
    ? sessionModule.getCurrentSessionId()
    : '';
  const runtime = getRuntimeForSession(id);
  const selectorEnabled = getDefaultRuntime() !== 'legacy';
  const visible = isAgentMode() && selectorEnabled;

  _indicator.style.display = visible ? '' : 'none';
  _indicator.classList.toggle('runtime-opencode', runtime === 'opencode');
  _indicator.classList.toggle('runtime-langgraph', runtime === 'langgraph');
  _indicator.classList.toggle('runtime-legacy', runtime === 'legacy');

  const label = _indicator.querySelector('.runtime-indicator-label');
  if (label) label.textContent = runtimeLabel(runtime);

  const next = runtime === 'legacy'
    ? 'LangGraph'
    : runtime === 'langgraph'
      ? 'OpenCode'
      : 'Legacy';
  _indicator.title =
    `${runtimeLabel(runtime)} active — click to switch to ${next}`;

  if (!_quickResetButton) {
    _quickResetButton = document.getElementById(
      'runtime-quick-reset-btn'
    );
  }
  if (_quickResetButton) {
    const resetVisible = visible && runtime === 'opencode';
    _quickResetButton.style.display = resetVisible ? '' : 'none';
    _quickResetButton.disabled = !!_activeRun;
  }
}

function watchComposerState() {
  const modeAgent = document.getElementById('mode-agent-btn');
  const modeChat = document.getElementById('mode-chat-btn');
  for (const button of [modeAgent, modeChat]) {
    if (!button) continue;
    button.addEventListener('click', () => {
      setTimeout(refreshRuntimeIndicator, 0);
    });
  }

  document.addEventListener('click', event => {
    if (
      event.target.closest('.list-item[data-session-id],.session-item,' +
      '.new-chat-btn,.send-btn.newchat-mode')
    ) {
      setTimeout(refreshRuntimeIndicator, 0);
    }
  });

  window.addEventListener('hashchange', refreshRuntimeIndicator);
  window.addEventListener('popstate', refreshRuntimeIndicator);
  window.addEventListener(
    'odysseus:runtime-changed',
    refreshRuntimeIndicator
  );

  const target = document.getElementById('chat-history');
  if (target && 'MutationObserver' in window) {
    new MutationObserver(() => {
      requestAnimationFrame(refreshRuntimeIndicator);
    }).observe(target, { childList: true });
  }
}

function assistantHolder() {
  const box = document.getElementById('chat-history');
  if (!box) throw new Error('Chat history is unavailable.');

  const holder = document.createElement('div');
  holder.className = 'msg msg-ai opencode-message';
  holder.dataset.runtime = 'opencode';

  const role = document.createElement('div');
  role.className = 'role opencode-message-role';
  role.textContent = 'OpenCode';

  const status = document.createElement('span');
  status.className = 'opencode-message-status';
  status.textContent = 'Starting';
  role.appendChild(status);

  const body = document.createElement('div');
  body.className = 'body opencode-message-body';

  holder.appendChild(role);
  holder.appendChild(body);
  box.appendChild(holder);
  uiModule.scrollHistory();
  return { box, holder, body, status };
}

function ensureToolThread(context) {
  if (context.toolThread) return context.toolThread;
  const thread = document.createElement('div');
  thread.className = 'agent-thread streaming';
  context.box.insertBefore(thread, context.holder);
  context.toolThread = thread;
  return thread;
}

function createToolNode(context, event) {
  const callId = String(event.call_id || event.id || '');
  if (context.toolNodes.has(callId)) {
    return context.toolNodes.get(callId);
  }

  const thread = ensureToolThread(context);
  const node = document.createElement('div');
  node.className = 'agent-thread-node running';
  node.dataset.callId = callId;

  const header = document.createElement('div');
  header.className = 'agent-thread-header';

  const icon = document.createElement('span');
  icon.className = 'agent-thread-icon';
  icon.textContent = '◌';

  const name = document.createElement('span');
  name.className = 'agent-thread-name';
  name.textContent = String(event.tool || 'OpenCode tool');

  const state = document.createElement('span');
  state.className = 'agent-thread-status';
  state.textContent = 'running';

  header.append(icon, name, state);

  const details = document.createElement('div');
  details.className = 'agent-thread-content';

  const summary = document.createElement('pre');
  summary.className = 'agent-thread-cmd';
  summary.textContent = String(event.summary || 'OpenCode tool call');
  details.appendChild(summary);

  node.append(header, details);
  thread.appendChild(node);

  const record = { node, icon, state, details, summary };
  context.toolNodes.set(callId, record);
  return record;
}

function completeToolNode(context, event) {
  const record = createToolNode(context, event);
  const success = event.success !== false;
  record.node.classList.remove('running');
  record.node.classList.toggle('error', !success);
  record.icon.textContent = success ? '✓' : '✗';
  record.state.textContent = success ? 'done' : 'failed';

  const output = String(event.output_summary || '').trim();
  if (output) {
    const result = document.createElement('pre');
    result.className = 'agent-tool-output';
    result.textContent = output;
    record.details.appendChild(result);
  }
}

function permissionResourceText(event) {
  const resources = Array.isArray(event.resources)
    ? event.resources.map(value => String(value || '').trim()).filter(Boolean)
    : [];
  if (resources.length) return resources.join('\n');
  return String(event.title || event.action || 'OpenCode permission').trim();
}

function createPermissionNode(context, event) {
  const requestId = String(event.request_id || '').trim();
  if (!requestId) return null;
  if (context.permissionNodes.has(requestId)) {
    return context.permissionNodes.get(requestId);
  }

  const thread = ensureToolThread(context);
  const node = document.createElement('div');
  node.className = 'agent-thread-node open waiting';
  node.dataset.permissionRequestId = requestId;

  const header = document.createElement('div');
  header.className = 'agent-thread-header';

  const icon = document.createElement('span');
  icon.className = 'agent-thread-icon';
  icon.textContent = '?';

  const name = document.createElement('span');
  name.className = 'agent-thread-name';
  name.textContent = `Permission: ${String(event.action || 'OpenCode')}`;

  const state = document.createElement('span');
  state.className = 'agent-thread-status';
  state.textContent = 'approval required';

  header.append(icon, name, state);

  const content = document.createElement('div');
  content.className = 'agent-thread-content';

  const summary = document.createElement('pre');
  summary.className = 'agent-thread-cmd';
  summary.textContent = permissionResourceText(event);
  content.appendChild(summary);

  const actions = document.createElement('div');
  actions.className = 'runtime-config-actions';

  const once = document.createElement('button');
  once.type = 'button';
  once.textContent = 'Allow once';

  const session = document.createElement('button');
  session.type = 'button';
  session.textContent = 'Allow for this chat';

  const reject = document.createElement('button');
  reject.type = 'button';
  reject.textContent = 'Reject';

  actions.append(once, session, reject);
  content.appendChild(actions);
  node.append(header, content);
  thread.appendChild(node);

  const record = {
    node,
    icon,
    state,
    actions,
    buttons: [once, session, reject],
  };
  context.permissionNodes.set(requestId, record);

  const decide = async decision => {
    record.buttons.forEach(button => { button.disabled = true; });
    state.textContent = 'sending decision';
    try {
      const payload = await runtimeJsonRequest(
        '/api/opencode/runtime/permissions/reply',
        {
          method: 'POST',
          body: JSON.stringify({
            odysseus_session_id: context.sessionId,
            request_id: requestId,
            decision,
          }),
        }
      );
      node.classList.remove('waiting');
      node.classList.toggle('error', decision === 'reject');
      icon.textContent = decision === 'reject' ? '✗' : '✓';
      state.textContent = decision === 'session'
        ? 'allowed for chat'
        : (decision === 'reject' ? 'rejected' : 'allowed once');
      actions.remove();
      showPermissionMessage(
        `${Number(payload.approval_count || 0)} chat approval`
        + (Number(payload.approval_count || 0) === 1 ? '' : 's')
      );
    } catch (error) {
      record.buttons.forEach(button => { button.disabled = false; });
      node.classList.add('error');
      icon.textContent = '!';
      state.textContent = 'reply failed';
      showPermissionMessage(
        `Permission reply failed: ${error.message || error}`,
        true
      );
    }
  };

  once.addEventListener('click', () => decide('once'));
  session.addEventListener('click', () => decide('session'));
  reject.addEventListener('click', () => decide('reject'));

  return record;
}

function renderResolvedPermission(context, event) {
  const requestId = String(event.request_id || '').trim();
  let record = requestId
    ? context.permissionNodes.get(requestId)
    : null;
  if (!record) {
    record = createPermissionNode(context, {
      ...event,
      request_id: requestId || `auto-${Date.now()}`,
      title: event.source === 'workspace_policy'
        ? 'Allowed by workspace policy'
        : 'Allowed by Odysseus chat approval',
    });
  }
  if (!record) return;
  record.node.classList.remove('waiting', 'open');
  record.icon.textContent = '✓';
  record.state.textContent = event.source === 'workspace_policy'
    ? 'allowed in workspace'
    : 'allowed for chat';
  if (record.actions && record.actions.isConnected) {
    record.actions.remove();
  }
}

function renderCompletedMessage(context, content) {
  const finalText = String(content || context.accumulated || '').trim();
  context.holder.dataset.raw = finalText;
  if (!finalText) {
    context.body.textContent = 'OpenCode completed without a text response.';
  } else {
    try {
      context.body.innerHTML = markdownModule.processWithThinking(
        markdownModule.squashOutsideCode(finalText)
      );
    } catch (_) {
      context.body.textContent = finalText;
    }
  }

  if (
    chatRenderer.createMsgFooter
    && !context.holder.querySelector('.msg-footer')
  ) {
    try {
      context.holder.appendChild(
        chatRenderer.createMsgFooter(context.holder)
      );
    } catch (_) {}
  }

  if (window.hljs) {
    context.holder.querySelectorAll('pre code').forEach(block => {
      try { window.hljs.highlightElement(block); } catch (_) {}
    });
  }

  if (context.toolThread) {
    context.toolThread.classList.remove('streaming');
  }
  context.status.textContent = 'completed';
  uiModule.scrollHistory();
}

function renderTerminalFailure(context, message, cancelled = false) {
  const text = String(message || '').trim()
    || (cancelled ? 'Cancelled by user.' : 'OpenCode runtime failed.');
  if (context.accumulated) {
    context.body.textContent = context.accumulated;
    const note = document.createElement('div');
    note.className = cancelled
      ? 'opencode-terminal-note'
      : 'opencode-terminal-note error';
    note.textContent = text;
    context.body.appendChild(note);
  } else {
    context.body.textContent = text;
  }
  context.holder.classList.toggle('opencode-failed', !cancelled);
  context.holder.classList.toggle('opencode-cancelled', cancelled);
  context.status.textContent = cancelled ? 'cancelled' : 'failed';
  if (context.toolThread) {
    context.toolThread.classList.remove('streaming');
  }
  uiModule.scrollHistory();
}

function parseSseFrame(frame) {
  const lines = String(frame || '').split(/\r?\n/);
  const data = lines
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trimStart())
    .join('\n');
  if (!data || data === '[DONE]') return null;
  try { return JSON.parse(data); } catch (_) { return null; }
}

function handleRuntimeEvent(context, event) {
  if (!event || typeof event !== 'object') return false;
  const type = String(event.type || '');

  if (type === 'runtime.started') {
    _activeRun.runId = String(event.run_id || '');
    context.status.textContent = 'running';
    return false;
  }

  if (type === 'runtime.status') {
    context.status.textContent = String(
      event.label || event.status || 'working'
    );
    return false;
  }

  if (type === 'tool.started') {
    createToolNode(context, event);
    uiModule.scrollHistory();
    return false;
  }

  if (type === 'tool.completed') {
    completeToolNode(context, event);
    uiModule.scrollHistory();
    return false;
  }

  if (type === 'runtime.permission_requested') {
    createPermissionNode(context, event);
    context.status.textContent = 'waiting for approval';
    uiModule.scrollHistory();
    return false;
  }

  if (type === 'runtime.permission_resolved') {
    renderResolvedPermission(context, event);
    context.status.textContent = 'working';
    uiModule.scrollHistory();
    return false;
  }

  if (type === 'assistant.delta') {
    context.accumulated += String(event.delta || '');
    context.body.textContent = context.accumulated;
    context.status.textContent = 'streaming';
    uiModule.scrollHistory();
    return false;
  }

  if (type === 'runtime.completed') {
    renderCompletedMessage(context, event.content);
    context.terminal = 'completed';
    return true;
  }

  if (type === 'runtime.cancelled') {
    renderTerminalFailure(
      context,
      event.message || 'Cancelled by user.',
      true
    );
    context.terminal = 'cancelled';
    return true;
  }

  if (type === 'runtime.failed') {
    renderTerminalFailure(
      context,
      event.error || event.message || 'OpenCode runtime failed.',
      false
    );
    context.terminal = 'failed';
    return true;
  }

  return false;
}

export function isOpenCodeStreaming() {
  return !!_activeRun;
}

export async function stopActiveOpenCodeRun() {
  const active = _activeRun;
  if (!active) return false;

  active.context.status.textContent = 'cancelling';
  if (!active.runId) {
    active.controller.abort();
    renderTerminalFailure(
      active.context,
      'Cancelled before OpenCode started.',
      true
    );
    _activeRun = null;
    refreshRuntimeIndicator();
    return true;
  }

  try {
    const response = await fetch(
      `/api/opencode/runtime/cancel/${encodeURIComponent(active.runId)}`,
      {
        method: 'POST',
        credentials: 'same-origin',
      }
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return true;
  } catch (error) {
    active.controller.abort();
    renderTerminalFailure(
      active.context,
      `Cancellation failed: ${error.message || error}`,
      false
    );
    _activeRun = null;
    refreshRuntimeIndicator();
    return false;
  }
}

export async function streamOpenCodeTurn({
  message,
  sessionId,
  title = 'OpenCode chat',
} = {}) {
  const id = String(sessionId || '').trim();
  const prompt = String(message || '').trim();
  if (!id) throw new Error('OpenCode requires an Odysseus chat session.');
  if (!prompt) return { status: 'empty' };
  if (_activeRun) throw new Error('An OpenCode run is already active.');

  ensureSessionRuntime(id);
  if (!boolSetting(REUSE_SESSION_KEY, true)) {
    await resetOpenCodeSession(id);
  }

  chatRenderer.addMessage('user', prompt, '', {
    runtime: 'opencode',
    timestamp: new Date().toISOString(),
  });

  const context = {
    ...assistantHolder(),
    sessionId: id,
    accumulated: '',
    terminal: '',
    toolThread: null,
    toolNodes: new Map(),
    permissionNodes: new Map(),
  };
  const controller = new AbortController();
  _activeRun = {
    runId: '',
    sessionId: id,
    controller,
    context,
  };
  refreshRuntimeIndicator();

  try {
    const response = await fetch('/api/opencode/runtime/stream', {
      method: 'POST',
      credentials: 'same-origin',
      headers: {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: prompt,
        odysseus_session_id: id,
        title,
        permission_mode: permissionMode(),
      }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`OpenCode stream returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });

      let boundary;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const frame = buffer.slice(0, boundary);
        const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/);
        buffer = buffer.slice(boundary + (separator ? separator[0].length : 2));
        const event = parseSseFrame(frame);
        if (handleRuntimeEvent(context, event)) {
          try { await reader.cancel(); } catch (_) {}
          break;
        }
      }

      if (context.terminal) break;
    }

    if (!context.terminal) {
      throw new Error('OpenCode stream ended without a terminal event.');
    }
    return {
      status: context.terminal,
      runId: _activeRun && _activeRun.runId,
      content: context.accumulated,
    };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      if (!context.terminal) {
        renderTerminalFailure(context, 'Cancelled by user.', true);
        context.terminal = 'cancelled';
      }
      return { status: 'cancelled' };
    }

    if (!context.terminal) {
      renderTerminalFailure(
        context,
        error.message || String(error),
        false
      );
      context.terminal = 'failed';
    }
    return {
      status: 'failed',
      error: error.message || String(error),
    };
  } finally {
    _activeRun = null;
    refreshRuntimeIndicator();
  }
}

async function hydrateDefaultRuntime() {
  if (safeStorageGet(DEFAULT_RUNTIME_KEY)) return;
  try {
    const response = await fetch('/api/auth/settings', {
      credentials: 'same-origin',
    });
    const settings = await response.json();
    safeStorageSet(
      DEFAULT_RUNTIME_KEY,
      settings.agent_langgraph_enabled === false ? 'legacy' : 'langgraph'
    );
  } catch (_) {
    safeStorageSet(DEFAULT_RUNTIME_KEY, 'langgraph');
  }
}

export async function initRuntimeControl() {
  if (_initialized) return;
  _initialized = true;
  await hydrateDefaultRuntime();
  hideLegacyLangGraphSetting();
  buildRuntimeSettings();
  createRuntimeIndicator();
  createRuntimeQuickReset();
  watchComposerState();
  syncSettingsControls();

  const modelInput = document.getElementById('runtime-opencode-model');
  if (modelInput && !modelInput.value) {
    const model = activeChatModel();
    if (model) modelInput.value = model;
  }

  refreshRuntimeIndicator();
}

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    () => initRuntimeControl(),
    { once: true }
  );
} else {
  queueMicrotask(() => initRuntimeControl());
}

const runtimeControl = {
  initRuntimeControl,
  getDefaultRuntime,
  getRuntimeForSession,
  ensureSessionRuntime,
  setDefaultRuntime,
  setCurrentRuntime,
  cycleCurrentRuntime,
  prepareNativeRuntime,
  refreshRuntimeIndicator,
  refreshOpenCodeHealth,
  restartOpenCodeServer,
  stopOpenCodeServer,
  openOpenCodeLogTerminal,
  resetOpenCodeSession,
  refreshPermissionState,
  savePermissionMode,
  clearCurrentPermissionApprovals,
  streamOpenCodeTurn,
  stopActiveOpenCodeRun,
  isOpenCodeStreaming,
  isAgentMode,
};

window.runtimeControl = runtimeControl;
export default runtimeControl;
