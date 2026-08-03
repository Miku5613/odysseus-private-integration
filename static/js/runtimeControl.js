// static/js/runtimeControl.js
//
// Standalone Agent-runtime selector and OpenCode UI adapter.
// OpenCode remains an external peer runtime. This module never falls back to
// Legacy or LangGraph when an OpenCode run fails.

import uiModule from './ui.js';
import sessionModule from './sessions.js';
import chatRenderer from './chatRenderer.js';
import markdownModule from './markdown.js';
import { getWorkspace } from './workspace.js';

import openCodeTaskPlannerAdapter from './openCodeTaskPlannerAdapter.js';
import multiGraphTracker from './multiGraphTracker.js';

const DEFAULT_RUNTIME_KEY = 'odysseus-agent-runtime-default-v1';
const SESSION_RUNTIME_KEY = 'odysseus-agent-runtime-sessions-v1';
const REUSE_SESSION_KEY = 'odysseus-opencode-reuse-session-v1';
const RESET_ON_SWITCH_KEY = 'odysseus-opencode-reset-on-switch-v1';
const TASK_PLANNER_UI_KEY = 'odysseus-task-planner-ui-enabled-v1';
const RUNTIME_SECTION_STATE_PREFIX = 'odysseus-runtime-section-collapsed-v1:';
const OPENCODE_MODEL_KEY = 'odysseus-opencode-model-v1';
const OPENCODE_ENDPOINT_KEY = 'odysseus-opencode-endpoint-v1';
const OPENCODE_WORKSPACE_KEY = 'odysseus-opencode-workspace-v1';
const OPENCODE_PERMISSION_MODE_KEY = 'odysseus-opencode-permission-mode-v1';
const LANGGRAPH_SETTINGS_DEFAULTS = {
  version: 1,
  execution_mode: 'automatic',
  permission_level: 'guided',
  runtime_model_policy: {
    mode: 'single_compatible',
    model: '',
    role_models: {},
  },
  legacy_step_fallback_policy: 'ask_first',
  interrupted_run_policy: 'ask_before_resume',
  effort_preset: 'balanced',
  research_approval_policy: 'ask_before_long_research',
  research_reuse_policy: 'prefer_existing',
  context_source_policy: {
    conversation: true,
    workspace: true,
    indexed_knowledge: true,
    completed_research: true,
    personal_memory: false,
  },
  parallel_worker_limit: 'automatic',
  task_planner_visibility: 'show',
  completed_run_retention: 30,
};

const RUNTIMES = new Set(['legacy', 'langgraph', 'opencode']);
let _runtimeAvailability = {
  legacy: { available: true, label: 'Legacy' },
  langgraph: { available: true, label: 'LangGraph' },
  opencode: { available: true, label: 'OpenCode' },
};
let _runtimeAvailabilityLoaded = false;
let _pendingRuntime = null;
let _openCodeReadySessions = new Map();
let _openCodeSessionManagerBusy = false;
let _openCodePreparePromise = null;
let _lastOpenCodeFailure = { key: '', timestamp: 0 };
let _activeRun = null;
let _initialized = false;
let _runtimePanel = null;
let _runtimeTab = null;
let _indicator = null;
let _sessionManagerButton = null;
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
  const candidate = RUNTIMES.has(runtime) ? runtime : fallback;
  if (!_runtimeAvailabilityLoaded || runtimeAvailable(candidate)) {
    return candidate;
  }
  return runtimeAvailable(fallback) ? fallback : 'legacy';
}

function runtimeAvailable(runtime) {
  const value = _runtimeAvailability[String(runtime || '').toLowerCase()];
  return !value || value.available !== false;
}

async function loadRuntimeAvailability() {
  try {
    const response = await fetch('/api/runtime/availability', {
      credentials: 'same-origin',
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    _runtimeAvailability = {
      ..._runtimeAvailability,
      ...(payload.runtimes || {}),
    };
  } catch (error) {
    console.warn('Optional runtime availability check failed:', error);
    _runtimeAvailability = {
      legacy: { available: true, label: 'Legacy' },
      langgraph: { available: false, label: 'LangGraph' },
      opencode: { available: false, label: 'OpenCode' },
    };
  } finally {
    _runtimeAvailabilityLoaded = true;
  }
}

function applyRuntimeAvailabilityToControls() {
  const select = document.getElementById('set-runtime-default');
  if (select) {
    for (const option of select.options) {
      const runtime = option.value;
      const available = runtimeAvailable(runtime);
      const label = runtimeLabel(runtime);
      option.disabled = !available;
      option.textContent = available ? label : `${label} — Not Installed`;
    }
  }

  const summary = document.getElementById('runtime-availability-summary');
  if (summary) {
    const missing = ['langgraph', 'opencode']
      .filter(runtime => !runtimeAvailable(runtime))
      .map(runtimeLabel);
    summary.textContent = missing.length
      ? `${missing.join(' and ')} ${missing.length === 1 ? 'is' : 'are'} Not Installed. Legacy remains available.`
      : 'Legacy, LangGraph, and OpenCode are installed and selectable.';
  }
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
  openCodeTaskPlannerAdapter.handleRuntimeChanged(runtime, {
    sessionId,
  });
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
  const order = ['legacy', 'langgraph', 'opencode'].filter(runtimeAvailable);
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
    <div class="admin-card runtime-settings-card runtime-selector-card runtime-minimal-card">
      <div class="runtime-minimal-heading">
        <span class="runtime-card-kicker">Agent orchestration</span>
        <h2 class="runtime-card-title">
          ${runtimeIconSvg(16)}
          <span>Runtime Selector</span>
        </h2>
        <p class="settings-help-text">
          Choose which Agent runtime newly created chats use by default. Existing chats keep their assigned runtime.
        </p>
      </div>

      <label class="runtime-select-row" for="set-runtime-default">
        <span class="runtime-select-copy">
          <strong>Runtime</strong>
           <small>Choose which Agent runtime newly created chats use by default. Existing chats keep their assigned runtime.</small>
        </span>
        <select id="set-runtime-default" class="runtime-select-control"
                aria-label="Default runtime">
          <option value="legacy">Legacy</option>
          <option value="langgraph">LangGraph</option>
          <option value="opencode">OpenCode</option>
        </select>
        <small id="runtime-availability-summary"
               class="settings-help-text" aria-live="polite"></small>
      </label>

      <label class="runtime-lifecycle-option runtime-global-planner-toggle">
        <span class="runtime-choice-copy">
          <strong>Task Planner UI</strong>
          <small>Show live OpenCode and LangGraph plan progress above the composer.</small>
        </span>
        <span class="runtime-toggle">
          <input id="set-task-planner-ui-enabled" type="checkbox" checked
                 role="switch" aria-label="Show Task Planner UI">
          <span class="runtime-toggle-track" aria-hidden="true">
            <span class="runtime-toggle-thumb"></span>
          </span>
        </span>
      </label>

      <div id="runtime-default-message"
           class="runtime-settings-message runtime-inline-message"></div>
    </div>

    <div id="runtime-opencode-panel"
         class="admin-card runtime-settings-card runtime-console-card runtime-minimal-card">
      <div class="runtime-minimal-heading">
        <span class="runtime-card-kicker">Standalone runtime</span>
        <div class="runtime-card-title-row">
          <h2 class="runtime-card-title">OpenCode Console</h2>
          <button type="button" class="runtime-card-collapse"
                  data-runtime-collapse="opencode" aria-expanded="true"
                  aria-label="Collapse OpenCode settings">⌃</button>
        </div>
        <p class="settings-help-text">
          Control how OpenCode conversation context is retained.
        </p>
      </div>

      <section class="runtime-lifecycle-panel" aria-labelledby="runtime-lifecycle-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="runtime-lifecycle-title">Conversation Lifecycle</h3>
          <p>Choose whether each chat keeps or refreshes its native OpenCode context.</p>
        </div>

        <div class="runtime-lifecycle-options">
          <label class="runtime-lifecycle-option">
            <span class="runtime-choice-copy">
              <strong>Reuse conversation per chat</strong>
               <small>Keeps one native OpenCode conversation attached to each Odysseus chat so its context continues between messages.</small>
            </span>
            <span class="runtime-toggle">
              <input id="set-opencode-reuse-session" type="checkbox" checked
                     role="switch" aria-label="Reuse OpenCode conversation per chat">
              <span class="runtime-toggle-track" aria-hidden="true">
                <span class="runtime-toggle-thumb"></span>
              </span>
            </span>
          </label>

          <label class="runtime-lifecycle-option">
            <span class="runtime-choice-copy">
              <strong>Reset after switching runtime</strong>
               <small>Starts a fresh OpenCode conversation after leaving OpenCode and returning.</small>
            </span>
            <span class="runtime-toggle">
              <input id="set-opencode-reset-on-switch" type="checkbox"
                     role="switch" aria-label="Reset OpenCode after switching runtime">
              <span class="runtime-toggle-track" aria-hidden="true">
                <span class="runtime-toggle-thumb"></span>
              </span>
            </span>
          </label>
        </div>
      </section>

      <section class="runtime-session-manager-card">
        <div class="runtime-session-manager-copy">
          <h3>OpenCode Session Manager</h3>
          <p>Opens the native OpenCode console for sessions, permissions, MCP, plugins, extensions, and commands.</p>
        </div>
        <button id="runtime-session-manager-settings-btn" type="button"
                class="runtime-session-manager-launch">
          Open Session Manager
        </button>
      </section>

      <!-- Keep configuration state available to existing runtime code without
           exposing model or workspace controls in this minimal settings view. -->
      <div class="runtime-internal-controls" hidden aria-hidden="true">
        <select id="runtime-opencode-model" tabindex="-1"></select>
        <input id="runtime-opencode-workspace" type="text" tabindex="-1">
      </div>
    </div>

    <div id="runtime-langgraph-panel" hidden aria-hidden="true"
         class="admin-card runtime-settings-card runtime-console-card runtime-minimal-card hidden">
      <div class="runtime-minimal-heading">
        <span class="runtime-card-kicker">Native agent runtime</span>
        <div class="runtime-card-title-row">
          <h2 class="runtime-card-title">LangGraph Runtime</h2>
          <button type="button" class="runtime-card-collapse"
                  data-runtime-collapse="langgraph" aria-expanded="true"
                  aria-label="Collapse LangGraph settings">⌃</button>
        </div>
        <p class="settings-help-text">
          These global defaults apply to newly created Agent chats. Existing
          chats keep their assigned runtime.
        </p>
      </div>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-execution-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-execution-title">Execution</h3>
          <p>Choose planning depth and conservative local concurrency.</p>
        </div>
        <div class="runtime-selector-grid">
          ${langGraphSelect('runtime-langgraph-execution', 'Planning mode', [
            ['automatic', 'Automatic'],
            ['direct_only', 'Direct only'],
            ['multi_step_allowed', 'Multi-step allowed'],
            ['ask_before_multi_step', 'Ask before multi-step'],
          ])}
          ${langGraphSelect('runtime-langgraph-effort', 'Effort', [
            ['quick', 'Quick'], ['balanced', 'Balanced'],
            ['thorough', 'Thorough'], ['custom', 'Custom'],
          ])}
          ${langGraphSelect('runtime-langgraph-parallel', 'Parallel workers', [
            ['automatic', 'Automatic'], ['1', '1'], ['2', '2'],
            ['3', '3'], ['4', '4'],
          ])}
        </div>
      </section>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-permissions-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-permissions-title">Permissions</h3>
          <p>Guided asks before mutations. Active still obeys tool safety and exact approvals.</p>
        </div>
        <div class="runtime-selector-grid">
          ${langGraphSelect('runtime-langgraph-permission', 'Action level', [
            ['read_only', 'Read only'], ['guided', 'Guided'], ['active', 'Active'],
          ])}
          ${langGraphSelect('runtime-langgraph-fallback', 'Legacy step fallback', [
            ['never', 'Never'], ['ask_first', 'Ask first'],
            ['allow_bounded', 'Allow bounded'],
          ])}
          ${langGraphSelect('runtime-langgraph-research-approval', 'Long research', [
            ['ask_before_long_research', 'Ask first'],
            ['allow_within_budget', 'Allow within budget'],
            ['disabled', 'Disabled'],
          ])}
        </div>
      </section>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-models-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-models-title">Models</h3>
          <p>Only enabled text-capable models are offered.</p>
        </div>
        <p class="runtime-policy-note">Use one model for all roles uses the same compatible model for planning, workers, and final synthesis.</p>
        ${langGraphSelect('runtime-langgraph-model', 'Runtime model', [
          ['', 'Use active chat model'],
        ])}
        <details class="runtime-advanced">
          <summary>Advanced role assignment</summary>
          <div class="runtime-selector-grid">
            ${['planner', 'general_worker', 'knowledge_worker', 'code_worker',
              'research_worker', 'final_synthesis'].map(role =>
                langGraphSelect(`runtime-langgraph-role-${role}`,
                  role.replaceAll('_', ' '), [['', 'Use runtime model']],
                  `data-langgraph-role="${role}"`)).join('')}
          </div>
        </details>
      </section>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-recovery-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-recovery-title">Recovery and research</h3>
          <p>Automatic resume is limited to durable, safe read-only PlanStep boundaries.</p>
        </div>
        <div class="runtime-selector-grid">
          ${langGraphSelect('runtime-langgraph-interrupted', 'Interrupted runs', [
            ['ask_before_resume', 'Ask before resume'],
            ['never_auto_resume', 'Never auto-resume'],
            ['auto_resume_safe_read_only', 'Auto-resume safe read-only'],
          ])}
          ${langGraphSelect('runtime-langgraph-retention', 'Completed run retention', [
            ['7', '7 days'], ['30', '30 days'], ['90', '90 days'],
          ])}
          ${langGraphSelect('runtime-langgraph-reuse', 'Reuse completed reports', [
            ['prefer_existing', 'Prefer existing'], ['ask', 'Ask'],
            ['always_new_when_requested', 'New when requested'],
          ])}
        </div>
      </section>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-context-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-context-title">Context</h3>
          <p>Only authorized sources are eligible; these switches never bypass access checks.</p>
        </div>
        <div class="runtime-context-grid">
          ${langGraphContextToggle('conversation', 'Conversation context')}
          ${langGraphContextToggle('workspace', 'Selected workspace')}
          ${langGraphContextToggle('indexed_knowledge', 'Ideaverse knowledge')}
          ${langGraphContextToggle('completed_research', 'Completed research')}
          ${langGraphContextToggle('personal_memory', 'Personal memory')}
        </div>
      </section>

      <section class="runtime-lifecycle-panel" aria-labelledby="langgraph-visibility-title">
        <div class="runtime-lifecycle-heading">
          <h3 id="langgraph-visibility-title">Task visibility</h3>
          <p>This changes presentation only; backend state remains authoritative.</p>
        </div>
        ${langGraphSelect('runtime-langgraph-planner-visibility', 'Live Task Planner', [
          ['show', 'Show'], ['show_collapsed', 'Show collapsed'], ['hidden', 'Hidden'],
        ])}
      </section>

      <div class="runtime-config-actions">
        <button id="runtime-langgraph-save" type="button"
                class="runtime-session-manager-launch">Save LangGraph settings</button>
        <div id="runtime-langgraph-message"
             class="runtime-settings-message runtime-inline-message"
             role="status" aria-live="polite"></div>
      </div>
    </div>`;
  return panel;
}

const LANGGRAPH_SETTING_HELP = {
  'runtime-langgraph-execution': 'Controls whether LangGraph answers directly, uses one specialist, or creates a coordinated multi-step plan.',
  'runtime-langgraph-effort': 'Controls how much planning, validation, and tool work LangGraph may use before producing the final answer.',
  'runtime-langgraph-parallel': 'Controls how many independent workers may run simultaneously; lower values use fewer local resources.',
  'runtime-langgraph-permission': 'Controls what LangGraph may do without approval. Guided asks before mutations; Active still obeys tool safety and exact approvals.',
  'runtime-langgraph-fallback': 'Controls whether one failed PlanStep may use Legacy fallback. It never falls back the entire query.',
  'runtime-langgraph-research-approval': 'Controls whether LangGraph must ask before starting a longer Deep Research job.',
  'runtime-langgraph-model': 'Selects the model used for all LangGraph roles when single-model mode is active.',
  'runtime-langgraph-role-planner': 'Selects the model responsible for planning, worker selection, and replanning.',
  'runtime-langgraph-role-general_worker': 'Selects the model used for general reasoning and direct read-only tasks.',
  'runtime-langgraph-role-knowledge_worker': 'Selects the model used for workspace, indexed knowledge, and Ideaverse retrieval.',
  'runtime-langgraph-role-code_worker': 'Selects the model used for source inspection, coding, and technical tasks.',
  'runtime-langgraph-role-research_worker': 'Selects the model used to coordinate and interpret research work.',
  'runtime-langgraph-role-final_synthesis': 'Selects the model that writes the final user-facing answer from verified worker results.',
  'runtime-langgraph-interrupted': 'Controls what Odysseus offers when a LangGraph run was interrupted by a backend restart.',
  'runtime-langgraph-retention': 'Controls how long completed LangGraph run history remains before safe cleanup.',
  'runtime-langgraph-reuse': 'Uses a suitable completed report before starting a duplicate research job.',
  'runtime-langgraph-planner-visibility': 'Shows live plan and step progress while LangGraph is working.',
};

const LANGGRAPH_CONTEXT_HELP = {
  conversation: 'Allows LangGraph to use messages from the current chat.',
  workspace: 'Allows LangGraph to read from the currently selected authorized workspace.',
  indexed_knowledge: 'Allows LangGraph to retrieve authorized indexed Ideaverse knowledge.',
  completed_research: 'Allows LangGraph to use previously completed research reports.',
  personal_memory: 'Allows LangGraph to use saved personal memory when available and authorized.',
};

function langGraphSelect(id, label, options, attributes = '') {
  const help = LANGGRAPH_SETTING_HELP[id] || '';
  return `<label class="runtime-select-row" for="${id}">
    <span class="runtime-select-copy"><strong>${label}</strong><small id="${id}-help">${help}</small></span>
    <select id="${id}" class="runtime-select-control" aria-describedby="${id}-help" ${attributes}>
      ${options.map(([value, text]) =>
        `<option value="${value}">${text}</option>`).join('')}
    </select>
  </label>`;
}

function langGraphContextToggle(key, label) {
  return `<label class="runtime-lifecycle-option">
    <span class="runtime-choice-copy"><strong>${label}</strong><small id="runtime-langgraph-context-${key}-help">${LANGGRAPH_CONTEXT_HELP[key] || ''}</small></span>
    <span class="runtime-toggle">
      <input type="checkbox" role="switch" data-langgraph-context="${key}"
             aria-label="${label}" aria-describedby="runtime-langgraph-context-${key}-help">
      <span class="runtime-toggle-track" aria-hidden="true">
        <span class="runtime-toggle-thumb"></span>
      </span>
    </span>
  </label>`;
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
  applyRuntimeAvailabilityToControls();
  const runtime = getDefaultRuntime();
  const runtimeSelect = document.getElementById('set-runtime-default');
  if (runtimeSelect && document.activeElement !== runtimeSelect) {
    runtimeSelect.value = runtime;
  }

  const reuse = document.getElementById('set-opencode-reuse-session');
  const reset = document.getElementById('set-opencode-reset-on-switch');
  if (reuse) reuse.checked = boolSetting(REUSE_SESSION_KEY, true);
  if (reset) reset.checked = boolSetting(RESET_ON_SWITCH_KEY, false);
  const plannerToggle = document.getElementById('set-task-planner-ui-enabled');
  const plannerEnabled = boolSetting(TASK_PLANNER_UI_KEY, true);
  if (plannerToggle) plannerToggle.checked = plannerEnabled;
  multiGraphTracker.setEnabled(plannerEnabled);

  const permissionSelect = document.getElementById(
    'runtime-opencode-permission-mode'
  );
  if (permissionSelect && document.activeElement !== permissionSelect) {
    permissionSelect.value = permissionMode();
  }

  const model = document.getElementById('runtime-opencode-model');
  const workspace = document.getElementById('runtime-opencode-workspace');
  if (model && document.activeElement !== model) {
    restoreOpenCodeModelSelection();
  }
  if (workspace && document.activeElement !== workspace) {
    workspace.value = safeStorageGet(OPENCODE_WORKSPACE_KEY) || '';
  }
  syncRuntimeSettingsPanelVisibility(runtime);
}

function syncRuntimeSettingsPanelVisibility(runtime = getDefaultRuntime()) {
  const openCodePanel = document.getElementById('runtime-opencode-panel');
  const langGraphPanel = document.getElementById('runtime-langgraph-panel');
  if (openCodePanel) {
    const inactive = runtime !== 'opencode';
    openCodePanel.hidden = inactive;
    openCodePanel.classList.toggle('hidden', inactive);
    openCodePanel.setAttribute('aria-hidden', String(inactive));
  }
  if (langGraphPanel) {
    const inactive = runtime !== 'langgraph';
    langGraphPanel.hidden = inactive;
    langGraphPanel.classList.toggle('hidden', inactive);
    langGraphPanel.setAttribute('aria-hidden', String(inactive));
  }
}

function setLangGraphMessage(message, error = false) {
  const target = document.getElementById('runtime-langgraph-message');
  if (!target) return;
  target.textContent = message || '';
  target.classList.toggle('error', !!error);
}

function mergedLangGraphSettings(value) {
  const input = value && typeof value === 'object' ? value : {};
  return {
    ...LANGGRAPH_SETTINGS_DEFAULTS,
    ...input,
    runtime_model_policy: {
      ...LANGGRAPH_SETTINGS_DEFAULTS.runtime_model_policy,
      ...(input.runtime_model_policy || {}),
      role_models: {
        ...(input.runtime_model_policy?.role_models || {}),
      },
    },
    context_source_policy: {
      ...LANGGRAPH_SETTINGS_DEFAULTS.context_source_policy,
      ...(input.context_source_policy || {}),
    },
  };
}

function setControlValue(id, value) {
  const control = document.getElementById(id);
  if (control && document.activeElement !== control) {
    control.value = String(value ?? '');
  }
}

function applyLangGraphSettingsToControls(value) {
  const config = mergedLangGraphSettings(value);
  setControlValue('runtime-langgraph-execution', config.execution_mode);
  setControlValue('runtime-langgraph-effort', config.effort_preset);
  setControlValue('runtime-langgraph-parallel', config.parallel_worker_limit);
  setControlValue('runtime-langgraph-permission', config.permission_level);
  setControlValue(
    'runtime-langgraph-fallback',
    config.legacy_step_fallback_policy
  );
  setControlValue(
    'runtime-langgraph-research-approval',
    config.research_approval_policy
  );
  setControlValue(
    'runtime-langgraph-interrupted',
    config.interrupted_run_policy
  );
  setControlValue(
    'runtime-langgraph-retention',
    config.completed_run_retention
  );
  setControlValue('runtime-langgraph-reuse', config.research_reuse_policy);
  setControlValue(
    'runtime-langgraph-planner-visibility',
    config.task_planner_visibility
  );
  setControlValue(
    'runtime-langgraph-model',
    config.runtime_model_policy.model
  );
  document.querySelectorAll('[data-langgraph-role]').forEach(control => {
    control.value = String(
      config.runtime_model_policy.role_models[control.dataset.langgraphRole]
      || ''
    );
  });
  document.querySelectorAll('[data-langgraph-context]').forEach(control => {
    control.checked = !!(
      config.context_source_policy[control.dataset.langgraphContext]
    );
  });
  return config;
}

function collectLangGraphSettings() {
  const value = id => String(
    document.getElementById(id)?.value || ''
  ).trim();
  const roleModels = {};
  document.querySelectorAll('[data-langgraph-role]').forEach(control => {
    if (control.value) {
      roleModels[control.dataset.langgraphRole] = control.value;
    }
  });
  const context = {};
  document.querySelectorAll('[data-langgraph-context]').forEach(control => {
    context[control.dataset.langgraphContext] = !!control.checked;
  });
  return {
    version: 1,
    execution_mode: value('runtime-langgraph-execution'),
    permission_level: value('runtime-langgraph-permission'),
    runtime_model_policy: {
      mode: Object.keys(roleModels).length
        ? 'advanced_roles'
        : 'single_compatible',
      model: value('runtime-langgraph-model'),
      role_models: roleModels,
    },
    legacy_step_fallback_policy: value('runtime-langgraph-fallback'),
    interrupted_run_policy: value('runtime-langgraph-interrupted'),
    effort_preset: value('runtime-langgraph-effort'),
    research_approval_policy: value(
      'runtime-langgraph-research-approval'
    ),
    research_reuse_policy: value('runtime-langgraph-reuse'),
    context_source_policy: context,
    parallel_worker_limit: value('runtime-langgraph-parallel'),
    task_planner_visibility: value(
      'runtime-langgraph-planner-visibility'
    ),
    completed_run_retention: Number(
      value('runtime-langgraph-retention')
    ),
  };
}

async function populateLangGraphModels() {
  const response = await fetch('/api/model-endpoints', {
    credentials: 'same-origin',
  });
  if (!response.ok) throw new Error(`Model registry HTTP ${response.status}`);
  const payload = await response.json();
  const modelIds = [];
  for (const endpoint of Array.isArray(payload) ? payload : []) {
    if (
      !endpoint
      || endpoint.is_enabled === false
      || (endpoint.model_type || 'llm') !== 'llm'
    ) continue;
    for (const model of Array.isArray(endpoint.models) ? endpoint.models : []) {
      const id = String(model || '').trim();
      if (id && !modelIds.includes(id)) modelIds.push(id);
    }
  }
  const controls = document.querySelectorAll(
    '#runtime-langgraph-model,[data-langgraph-role]'
  );
  controls.forEach(control => {
    const selected = control.value;
    const defaultText = control.dataset.langgraphRole
      ? 'Use runtime model'
      : 'Use active chat model';
    control.replaceChildren(new Option(defaultText, ''));
    modelIds.forEach(model => control.add(new Option(model, model)));
    if (modelIds.includes(selected)) control.value = selected;
  });
}

async function loadLangGraphSettings() {
  try {
    const payload = await runtimeJsonRequest('/api/auth/settings');
    const config = applyLangGraphSettingsToControls(
      payload.langgraph_runtime
    );
    await populateLangGraphModels();
    applyLangGraphSettingsToControls(config);
    setLangGraphMessage('LangGraph settings loaded.');
    return config;
  } catch (error) {
    applyLangGraphSettingsToControls(null);
    setLangGraphMessage(
      `LangGraph settings unavailable: ${error.message || error}`,
      true
    );
    return null;
  }
}

async function saveLangGraphSettings() {
  const config = collectLangGraphSettings();
  setLangGraphMessage('Saving...');
  const payload = await runtimeJsonRequest('/api/auth/settings', {
    method: 'POST',
    body: JSON.stringify({ langgraph_runtime: config }),
  });
  applyLangGraphSettingsToControls(payload.langgraph_runtime);
  setLangGraphMessage('LangGraph settings saved for new Agent chats.');
  return payload.langgraph_runtime;
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

let _openCodeModelEndpoints = [];

function selectedOpenCodeModelOption() {
  const select = document.getElementById('runtime-opencode-model');
  if (!select || select.selectedIndex < 0) return null;
  return select.options[select.selectedIndex] || null;
}

function selectOpenCodeModel(modelId, endpointId = '') {
  const select = document.getElementById('runtime-opencode-model');
  if (!select) return false;
  const targetModel = String(modelId || '').trim();
  const targetEndpoint = String(endpointId || '').trim();
  const option = Array.from(select.options).find(item => (
    item.value === targetModel &&
    (!targetEndpoint || item.dataset.endpointId === targetEndpoint)
  ));
  if (!option) return false;
  select.selectedIndex = Array.from(select.options).indexOf(option);
  return select.selectedIndex >= 0;
}

function restoreOpenCodeModelSelection() {
  return selectOpenCodeModel(
    safeStorageGet(OPENCODE_MODEL_KEY) || '',
    safeStorageGet(OPENCODE_ENDPOINT_KEY) || ''
  );
}

async function loadOpenCodeModelOptions() {
  const select = document.getElementById('runtime-opencode-model');
  if (!select) return [];

  const response = await fetch('/api/model-endpoints', {
    credentials: 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Model registry HTTP ${response.status}`);
  }

  const payload = await response.json();
  _openCodeModelEndpoints = (Array.isArray(payload) ? payload : [])
    .filter(endpoint => (
      endpoint &&
      endpoint.is_enabled !== false &&
      (endpoint.model_type || 'llm') === 'llm'
    ));

  select.innerHTML = '';
  for (const endpoint of _openCodeModelEndpoints) {
    const models = Array.isArray(endpoint.models) ? endpoint.models : [];
    if (!models.length) continue;

    const group = document.createElement('optgroup');
    group.label = endpoint.name || endpoint.base_url || 'Model endpoint';

    for (const modelId of models) {
      const option = document.createElement('option');
      option.value = String(modelId);
      option.textContent = String(modelId);
      option.dataset.endpointId = String(endpoint.id || '');
      option.dataset.endpointName = String(endpoint.name || '');
      group.appendChild(option);
    }
    select.appendChild(group);
  }

  if (!restoreOpenCodeModelSelection()) {
    const first = select.options[0];
    if (first) first.selected = true;
  }
  return _openCodeModelEndpoints;
}

function configuredOpenCodeValues() {
  const modelInput = document.getElementById('runtime-opencode-model');
  const workspaceInput = document.getElementById(
    'runtime-opencode-workspace'
  );
  const option = selectedOpenCodeModelOption();
  return {
    endpoint_id: String(option?.dataset.endpointId || '').trim(),
    model_id: String(modelInput && modelInput.value || '').trim(),
    workspace: String(workspaceInput && workspaceInput.value || '').trim(),
  };
}

function storeConfiguredOpenCodeValues(values) {
  if (values.endpoint_id) {
    safeStorageSet(OPENCODE_ENDPOINT_KEY, values.endpoint_id);
  }
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

async function openCodeResponseError(response, fallback = '') {
  const defaultMessage = fallback || `HTTP ${response.status}`;
  const contentType = String(response.headers?.get('content-type') || '').toLowerCase();
  if (contentType.includes('text/html')) return defaultMessage;
  let detail = '';
  try {
    const payload = await response.clone().json();
    const value = payload?.detail ?? payload?.error ?? payload?.message;
    detail = typeof value === 'string' ? value : (value ? JSON.stringify(value) : '');
  } catch (_) {
    try { detail = await response.text(); } catch (_) {}
  }
  detail = String(detail || '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 300);
  return detail || defaultMessage;
}

export async function restartOpenCodeServer() {
  if (_activeRun) {
    showConfigMessage('Stop the active OpenCode run before restarting.', true);
    return null;
  }

  const values = configuredOpenCodeValues();
  if (!values.endpoint_id || !values.model_id) {
    showConfigMessage(
      'Choose an enabled Odysseus model for OpenCode.',
      true
    );
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

async function openOpenCodeLogTerminal() {
  const sessionId = currentOdysseusSessionId();
  const payload = await runtimeJsonRequest(
    '/api/opencode/runtime/server/log-terminal',
    {
      method: 'POST',
      body: JSON.stringify(
        sessionId ? { odysseus_session_id: sessionId } : {}
      ),
    }
  );

  const scope = payload.scope === 'session' ? 'active chat' : 'server';
  setHealthMessage(
    `Opened OpenCode ${scope} logs${payload?.path ? `: ${payload.path}` : ''}`
  );
  return payload;
}

function setRuntimeSectionCollapsed(section, collapsed) {
  const panel = document.getElementById(`runtime-${section}-panel`);
  const button = document.querySelector(`[data-runtime-collapse="${section}"]`);
  if (!panel || !button) return;
  panel.classList.toggle('is-collapsed', collapsed);
  button.setAttribute('aria-expanded', String(!collapsed));
  button.textContent = collapsed ? '⌄' : '⌃';
  safeStorageSet(`${RUNTIME_SECTION_STATE_PREFIX}${section}`, String(collapsed));
}

function restoreRuntimeSectionStates() {
  for (const section of ['opencode', 'langgraph']) {
    setRuntimeSectionCollapsed(
      section,
      boolSetting(`${RUNTIME_SECTION_STATE_PREFIX}${section}`, false)
    );
  }
}

function wireRuntimeSettings() {
  const runtimeSelect = document.getElementById('set-runtime-default');
  const reuse = document.getElementById('set-opencode-reuse-session');
  const reset = document.getElementById('set-opencode-reset-on-switch');
  const plannerToggle = document.getElementById(
    'set-task-planner-ui-enabled'
  );
  const sessionManagerButton = document.getElementById(
    'runtime-session-manager-settings-btn'
  );
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
  const saveLangGraph = document.getElementById('runtime-langgraph-save');

  if (runtimeSelect) {
    runtimeSelect.addEventListener('change', async () => {
      const runtime = normalizeRuntime(runtimeSelect.value, 'legacy');
      showDefaultMessage('Saving…');
      await setDefaultRuntime(runtime);
      runtimeSelect.value = runtime;
      syncRuntimeSettingsPanelVisibility(runtime);
      if (runtime === 'langgraph') await loadLangGraphSettings();
      showDefaultMessage(`New chats will use ${runtimeLabel(runtime)}.`);
    });
  }
  if (saveLangGraph) {
    saveLangGraph.addEventListener('click', async () => {
      saveLangGraph.disabled = true;
      try {
        await saveLangGraphSettings();
      } catch (error) {
        setLangGraphMessage(
          `Save failed: ${error.message || error}`,
          true
        );
      } finally {
        saveLangGraph.disabled = false;
      }
    });
  }
  if (plannerToggle) {
    plannerToggle.addEventListener('change', () => {
      const enabled = !!plannerToggle.checked;
      safeStorageSet(TASK_PLANNER_UI_KEY, String(enabled));
      multiGraphTracker.setEnabled(enabled);
    });
  }

  document.querySelectorAll('[data-runtime-collapse]').forEach(button => {
    button.addEventListener('click', () => {
      const section = String(button.dataset.runtimeCollapse || '');
      const panel = document.getElementById(`runtime-${section}-panel`);
      if (!panel) return;
      setRuntimeSectionCollapsed(
        section,
        !panel.classList.contains('is-collapsed')
      );
    });
  });
  restoreRuntimeSectionStates();

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

  if (sessionManagerButton) {
    sessionManagerButton.addEventListener('click', () => {
      handleOpenCodeSessionManagerClick().catch(() => {});
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
      if (!selectOpenCodeModel(model)) {
        showConfigMessage(
          'The active chat model is not enabled in the Odysseus registry.',
          true
        );
        return;
      }
      showConfigMessage('Active chat model selected. Apply to restart.');
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
  loadLangGraphSettings().catch(() => {});
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

    if (modelInput && configuredModel) {
      const endpointId = String(model.endpoint_id || '').trim();
      if (selectOpenCodeModel(configuredModel, endpointId)) {
        safeStorageSet(OPENCODE_MODEL_KEY, configuredModel);
        if (endpointId) {
          safeStorageSet(OPENCODE_ENDPOINT_KEY, endpointId);
        }
      }
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

function activeOpenCodeWorkspace() {
  return String(getWorkspace() || '').trim();
}

function normalizeOpenCodeModelId(value) {
  return String(value || '').trim();
}

function activeOpenCodeSelection() {
  const pending = sessionModule.getPendingChat
    ? sessionModule.getPendingChat()
    : null;
  const pendingModel = normalizeOpenCodeModelId(pending?.modelId);
  const pendingEndpoint = String(pending?.endpointId || '').trim();
  if (pendingModel) {
    return { modelId: pendingModel, endpointId: pendingEndpoint };
  }

  const visibleModel = normalizeOpenCodeModelId(activeChatModel());
  if (!visibleModel) return { modelId: '', endpointId: '' };

  const exact = [];
  const suffix = [];
  for (const endpoint of _openCodeModelEndpoints) {
    const models = Array.isArray(endpoint?.models) ? endpoint.models : [];
    for (const rawModel of models) {
      const modelId = normalizeOpenCodeModelId(rawModel);
      if (!modelId) continue;
      const item = {
        modelId,
        endpointId: String(endpoint?.id || '').trim(),
      };
      if (modelId === visibleModel) {
        exact.push(item);
      } else if (
        modelId.endsWith(`/${visibleModel}`)
        || visibleModel.endsWith(`/${modelId}`)
      ) {
        suffix.push(item);
      }
    }
  }

  const matches = exact.length ? exact : suffix;
  return matches.length === 1
    ? matches[0]
    : { modelId: visibleModel, endpointId: '' };
}

function activeOpenCodeEndpointId(modelId) {
  if (!selectOpenCodeModel(modelId)) return '';
  const option = selectedOpenCodeModelOption();
  return String(option?.dataset.endpointId || '').trim();
}

async function materializedOpenCodeSessionId() {
  if (
    sessionModule.hasPendingChat
    && sessionModule.hasPendingChat()
    && sessionModule.materializePendingSession
  ) {
    const created = await sessionModule.materializePendingSession();
    if (!created) {
      throw new Error('The new Odysseus chat could not be created.');
    }
  }

  const sessionId = String(
    sessionModule.getCurrentSessionId
      ? sessionModule.getCurrentSessionId()
      : ''
  ).trim();
  if (!sessionId) {
    throw new Error('OpenCode requires an Odysseus chat session.');
  }
  return sessionId;
}

function openCodeReadyKey(sessionId, _modelId, workspace) {
  return [
    String(sessionId || '').trim(),
    String(workspace || '').trim().toLowerCase(),
  ].join('|');
}

function currentOpenCodeReadyState() {
  const sessionId = String(
    sessionModule.getCurrentSessionId
      ? sessionModule.getCurrentSessionId()
      : ''
  ).trim();
  const modelId = activeChatModel();
  const workspace = activeOpenCodeWorkspace();
  if (!sessionId || !modelId || !workspace) return null;

  const key = openCodeReadyKey(sessionId, modelId, workspace);
  return _openCodeReadySessions.get(key) || null;
}

function renderOpenCodeRuntimeStatus(message, metadata = {}) {
  const text = String(message || '').trim();
  if (!text) return false;

  if (metadata.failed) {
    const sessionId = String(
      sessionModule.getCurrentSessionId
        ? sessionModule.getCurrentSessionId()
        : ''
    ).trim();
    const key = [sessionId, text].join('|');
    const now = Date.now();
    if (
      _lastOpenCodeFailure.key === key
      && now - _lastOpenCodeFailure.timestamp < 5000
    ) {
      return false;
    }
    _lastOpenCodeFailure = { key, timestamp: now };
  }

  chatRenderer.addMessage('assistant', text, 'OpenCode', {
    runtime: 'opencode',
    kind: 'runtime_status',
    deterministic: true,
    timestamp: new Date().toISOString(),
    ...metadata,
  });
  const box = document.getElementById('chat-history');
  if (box) box.scrollTop = box.scrollHeight;
  return true;
}

async function prepareOpenCodeForCurrentChat() {
  if (_openCodePreparePromise) return _openCodePreparePromise;

  _openCodePreparePromise = (async () => {
    const sessionId = await materializedOpenCodeSessionId();
    const workspace = activeOpenCodeWorkspace();
    if (!workspace) {
      throw new Error('Select an active workspace before starting OpenCode.');
    }
    if (!_openCodeModelEndpoints.length) {
      await loadOpenCodeModelOptions();
    }

    const selection = activeOpenCodeSelection();
    if (!selection.modelId) {
      throw new Error('Select a chat model before starting OpenCode.');
    }
    if (!selection.endpointId) {
      throw new Error(
        'The active chat model is not enabled in the Odysseus registry.'
      );
    }

    const payload = await runtimeJsonRequest(
      '/api/opencode/runtime/prepare',
      {
        method: 'POST',
        body: JSON.stringify({
          odysseus_session_id: sessionId,
          title: 'OpenCode Session',
          permission_mode: permissionMode(),
          endpoint_id: selection.endpointId,
          model_id: selection.modelId,
          workspace,
        }),
      }
    );

    const ready = {
      sessionId,
      modelId: selection.modelId,
      workspace,
      payload,
    };
    _openCodeReadySessions.set(
      openCodeReadyKey(sessionId, selection.modelId, workspace),
      ready
    );
    renderOpenCodeRuntimeStatus(
      payload.message || 'OpenCode is ready for this chat.',
      { ready: true, model_id: selection.modelId, workspace }
    );
    refreshRuntimeIndicator();
    return ready;
  })();

  try {
    return await _openCodePreparePromise;
  } finally {
    _openCodePreparePromise = null;
  }
}

async function openOpenCodeSessionManagerTerminal(sessionId) {
  return runtimeJsonRequest(
    '/api/opencode/runtime/server/session-manager-terminal',
    {
      method: 'POST',
      body: JSON.stringify({
        odysseus_session_id: sessionId,
      }),
    }
  );
}

async function handleOpenCodeSessionManagerClick() {
  if (_activeRun || _openCodeSessionManagerBusy) return null;
  const ready = currentOpenCodeReadyState();
  if (!ready) return null;

  _openCodeSessionManagerBusy = true;
  refreshRuntimeIndicator();
  try {
    const payload = await openOpenCodeSessionManagerTerminal(ready.sessionId);
    return {
      action: payload.focused ? 'focused' : 'opened',
      ...payload,
    };
  } finally {
    _openCodeSessionManagerBusy = false;
    refreshRuntimeIndicator();
  }
}

function simplifyOpenCodeSettings() {
  const workspaceInput = document.getElementById(
    'runtime-opencode-workspace'
  );
  const workspaceField = workspaceInput?.closest('.runtime-field');
  if (workspaceField) {
    workspaceField.innerHTML = `
      <span>Workspace</span>
      <output id="runtime-opencode-active-workspace"
              class="runtime-field-readonly"></output>
    `;
  }

  const output = document.getElementById(
    'runtime-opencode-active-workspace'
  );
  if (output) {
    output.textContent = activeOpenCodeWorkspace()
      || 'No active workspace selected';
  }

  document.getElementById(
    'runtime-apply-restart-btn'
  )?.closest('.runtime-primary-row')?.remove();

  document.querySelector(
    '.runtime-permission-section'
  )?.remove();

  const actionDock = document.querySelector('.runtime-action-dock');
  if (actionDock) {
    actionDock.innerHTML = `
      <div class="runtime-action-dock-copy">
        <strong>OpenCode Session Manager</strong>
        <small>Live session, lifecycle, permissions, MCP, plugins,
        extensions, and commands.</small>
      </div>
      <div class="runtime-session-actions">
        <button id="runtime-session-manager-settings-btn"
                type="button">
          Open Session Manager
        </button>
      </div>
    `;
    actionDock.querySelector(
      '#runtime-session-manager-settings-btn'
    )?.addEventListener(
      'click',
      () => handleOpenCodeSessionManagerClick().catch(() => {})
    );
  }

  document.addEventListener(
    'workspace-changed',
    event => {
      const current = document.getElementById(
        'runtime-opencode-active-workspace'
      );
      if (current) {
        current.textContent = String(
          event?.detail?.path || activeOpenCodeWorkspace()
          || 'No active workspace selected'
        );
      }
      _openCodeReadySessions.clear();
      refreshRuntimeIndicator();
    },
    { passive: true }
  );
}


function escapeOpenCodeSessionText(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function closeOpenCodeSessionPanel() {
  document.getElementById('opencode-session-panel-backdrop')?.remove();
}

function openSessionNameDialog({ title = '', heading = 'New OpenCode Session' } = {}) {
  return new Promise(resolve => {
    const backdrop = document.createElement('div');
    backdrop.className = 'opencode-session-dialog-backdrop';
    backdrop.innerHTML = `
      <div class="opencode-session-dialog" role="dialog" aria-modal="true">
        <h3>${escapeOpenCodeSessionText(heading)}</h3>
        <label>
          <span>Session name</span>
          <input type="text" maxlength="160" value="${escapeOpenCodeSessionText(title)}"
                 placeholder="Gauntlet of Gods - Main Project">
        </label>
        <div class="opencode-session-dialog-actions">
          <button type="button" data-action="cancel">Cancel</button>
          <button type="button" class="primary" data-action="save">${title ? 'Rename' : 'Create'}</button>
        </div>
      </div>`;
    const finish = value => {
      backdrop.remove();
      resolve(value);
    };
    backdrop.addEventListener('click', event => {
      if (event.target === backdrop || event.target.closest('[data-action="cancel"]')) finish('');
      if (event.target.closest('[data-action="save"]')) {
        const value = backdrop.querySelector('input')?.value.trim() || '';
        if (value) finish(value);
      }
    });
    backdrop.addEventListener('keydown', event => {
      if (event.key === 'Escape') finish('');
      if (event.key === 'Enter') {
        const value = backdrop.querySelector('input')?.value.trim() || '';
        if (value) finish(value);
      }
    });
    document.body.appendChild(backdrop);
    const input = backdrop.querySelector('input');
    input?.focus();
    input?.select();
  });
}

function openSessionConfirmDialog({ heading, message, confirmLabel = 'Delete' }) {
  return new Promise(resolve => {
    const backdrop = document.createElement('div');
    backdrop.className = 'opencode-session-dialog-backdrop';
    backdrop.innerHTML = `
      <div class="opencode-session-dialog opencode-session-confirm" role="alertdialog" aria-modal="true">
        <h3>${escapeOpenCodeSessionText(heading)}</h3>
        <p>${escapeOpenCodeSessionText(message)}</p>
        <div class="opencode-session-dialog-actions">
          <button type="button" data-action="cancel">Cancel</button>
          <button type="button" class="danger" data-action="confirm">${escapeOpenCodeSessionText(confirmLabel)}</button>
        </div>
      </div>`;
    const finish = value => {
      backdrop.remove();
      resolve(value);
    };
    backdrop.addEventListener('click', event => {
      if (event.target === backdrop || event.target.closest('[data-action="cancel"]')) finish(false);
      if (event.target.closest('[data-action="confirm"]')) finish(true);
    });
    backdrop.addEventListener('keydown', event => {
      if (event.key === 'Escape') finish(false);
    });
    document.body.appendChild(backdrop);
    backdrop.querySelector('[data-action="cancel"]')?.focus();
  });
}

function openCodeSessionRelativeTime(value) {
  const when = new Date(value || 0).getTime();
  if (!Number.isFinite(when) || when <= 0) return '';
  const seconds = Math.max(1, Math.round((Date.now() - when) / 1000));
  if (seconds < 60) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

async function loadNativeOpenCodeSessions() {
  const sessionId = await materializedOpenCodeSessionId();
  const workspace = activeOpenCodeWorkspace();
  const payload = await runtimeJsonRequest(
    `/api/opencode/runtime/native-sessions?odysseus_session_id=${encodeURIComponent(sessionId)}&workspace=${encodeURIComponent(workspace)}`
  );
  const selection = activeOpenCodeSelection();
  const readyKey = openCodeReadyKey(sessionId, selection.modelId, workspace);
  if (payload?.active_opencode_session_id) {
    _openCodeReadySessions.set(readyKey, { sessionId, modelId: selection.modelId, workspace, payload });
  } else {
    _openCodeReadySessions.delete(readyKey);
  }
  refreshRuntimeIndicator();
  return payload;
}

async function activateNativeOpenCodeSession(nativeSessionId) {
  const sessionId = await materializedOpenCodeSessionId();
  const workspace = activeOpenCodeWorkspace();
  if (!workspace) throw new Error('Select an active workspace before starting an OpenCode session.');
  if (!_openCodeModelEndpoints.length) await loadOpenCodeModelOptions();
  const selection = activeOpenCodeSelection();
  if (!selection.modelId || !selection.endpointId) {
    throw new Error('Select an enabled model before starting an OpenCode session.');
  }
  const payload = await runtimeJsonRequest('/api/opencode/runtime/native-sessions/activate', {
    method: 'POST',
    body: JSON.stringify({
      odysseus_session_id: sessionId,
      opencode_session_id: nativeSessionId,
      permission_mode: permissionMode(),
      endpoint_id: selection.endpointId,
      model_id: selection.modelId,
      workspace,
    }),
  });
  const ready = { sessionId, modelId: selection.modelId, workspace, payload };
  _openCodeReadySessions.set(openCodeReadyKey(sessionId, selection.modelId, workspace), ready);
  renderOpenCodeRuntimeStatus(payload.message || 'OpenCode session activated.', {
    ready: true,
    opencode_session_id: payload.opencode_session_id,
  });
  refreshRuntimeIndicator();
  return payload;
}

async function createNativeOpenCodeSession(title) {
  const sessionId = await materializedOpenCodeSessionId();
  const workspace = activeOpenCodeWorkspace();
  if (!workspace) throw new Error('Select an active workspace before creating an OpenCode session.');
  if (!_openCodeModelEndpoints.length) await loadOpenCodeModelOptions();
  const selection = activeOpenCodeSelection();
  if (!selection.modelId || !selection.endpointId) {
    throw new Error('Select an enabled model before creating an OpenCode session.');
  }
  const payload = await runtimeJsonRequest('/api/opencode/runtime/native-sessions', {
    method: 'POST',
    body: JSON.stringify({
      odysseus_session_id: sessionId,
      title,
      permission_mode: permissionMode(),
      endpoint_id: selection.endpointId,
      model_id: selection.modelId,
      workspace,
    }),
  });
  const ready = { sessionId, modelId: selection.modelId, workspace, payload };
  _openCodeReadySessions.set(openCodeReadyKey(sessionId, selection.modelId, workspace), ready);
  renderOpenCodeRuntimeStatus(payload.message || `OpenCode session created: "${title}".`, {
    ready: true,
    opencode_session_id: payload.opencode_session_id,
  });
  refreshRuntimeIndicator();
  return payload;
}

async function renameNativeOpenCodeSession(nativeSessionId, currentTitle) {
  const title = await openSessionNameDialog({ title: currentTitle, heading: 'Rename OpenCode Session' });
  if (!title) return false;
  const sessionId = await materializedOpenCodeSessionId();
  await runtimeJsonRequest(`/api/opencode/runtime/native-sessions/${encodeURIComponent(nativeSessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ odysseus_session_id: sessionId, title }),
  });
  return true;
}

async function deleteNativeOpenCodeSession(nativeSessionId, currentTitle, { announce = true } = {}) {
  const sessionId = await materializedOpenCodeSessionId();
  const payload = await runtimeJsonRequest(
    `/api/opencode/runtime/native-sessions/${encodeURIComponent(nativeSessionId)}`,
    {
      method: 'DELETE',
      body: JSON.stringify({ odysseus_session_id: sessionId, title: currentTitle }),
    }
  );
  if (payload.active_cleared) {
    const selection = activeOpenCodeSelection();
    const workspace = activeOpenCodeWorkspace();
    _openCodeReadySessions.delete(openCodeReadyKey(sessionId, selection.modelId, workspace));
    refreshRuntimeIndicator();
  }
  if (announce) {
    renderOpenCodeRuntimeStatus(payload.message || `OpenCode session deleted: "${currentTitle}".`, {
      ready: false,
      opencode_session_id: '',
    });
  }
  return payload;
}

async function openNativeOpenCodeTui(nativeSessionId) {
  const sessionId = await materializedOpenCodeSessionId();
  const workspace = activeOpenCodeWorkspace();
  if (!workspace) {
    throw new Error('Select an active workspace before opening OpenCode TUI.');
  }
  const payload = await runtimeJsonRequest(
    `/api/opencode/runtime/native-sessions/${encodeURIComponent(nativeSessionId)}/open-tui`,
    {
      method: 'POST',
      body: JSON.stringify({
        odysseus_session_id: sessionId,
        workspace,
      }),
    }
  );
  renderOpenCodeRuntimeStatus('Opened the selected session in OpenCode TUI.', {
    deterministic: true,
    opencode_session_id: nativeSessionId,
  });
  return payload;
}

function nativeOpenCodeSessionMatches(item, search) {
  if (!search) return true;
  return [item?.title, item?.model_name, item?.preview, item?.id]
    .map(value => String(value || '').toLowerCase())
    .some(value => value.includes(search));
}

function renderNativeOpenCodeSessions(panel, payload, state) {
  const list = panel.querySelector('.opencode-session-list');
  const search = String(panel.querySelector('.opencode-session-search')?.value || '').trim().toLowerCase();
  const sessions = Array.isArray(payload?.sessions) ? payload.sessions : [];
  const filtered = sessions.filter(item => nativeOpenCodeSessionMatches(item, search));
  const availableIds = new Set(sessions.map(item => String(item.id || '')));
  for (const selectedId of state.selected) {
    if (!availableIds.has(selectedId)) state.selected.delete(selectedId);
  }
  panel.querySelector('.opencode-session-count').textContent = `${sessions.length} session${sessions.length === 1 ? '' : 's'}`;
  panel.querySelector('.opencode-session-mode-toggle').textContent = state.selectMode ? 'Cancel' : 'Select';
  panel.querySelector('.opencode-session-select-summary').hidden = !state.selectMode;
  panel.querySelector('.opencode-session-selected-count').textContent = `${state.selected.size} Selected`;
  panel.querySelector('.opencode-session-batch-delete').disabled = state.selected.size === 0;
  const visibleIds = filtered.map(item => String(item.id || ''));
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every(id => state.selected.has(id));
  panel.querySelector('.opencode-session-select-all').setAttribute('aria-pressed', String(allVisibleSelected));
  if (!filtered.length) {
    list.innerHTML = '<div class="opencode-session-empty">No OpenCode sessions found.</div>';
    return;
  }
  list.innerHTML = filtered.map(item => {
    const active = !!item.active;
    const time = openCodeSessionRelativeTime(item.updated_at || item.created_at);
    const modelName = String(item.model_name || '').trim() || 'Unknown model';
    const messageCount = Math.max(0, Number(item.message_count) || 0);
    const shortId = String(item.id || '').slice(-8);
    const preview = String(item.preview || '').trim()
      || `Native session ...${shortId}`;
    const meta = [
      modelName,
      `${messageCount} message${messageCount === 1 ? '' : 's'}`,
      time || 'time unavailable',
    ].join(' | ');
    return `
      <article class="opencode-session-row${active ? ' is-active' : ''}" data-session-id="${escapeOpenCodeSessionText(item.id)}">
        ${state.selectMode ? `<input type="checkbox" class="opencode-session-select-control" data-action="select" aria-label="Select ${escapeOpenCodeSessionText(item.title)}" ${state.selected.has(String(item.id || '')) ? 'checked' : ''}>` : ''}
        <div class="opencode-session-row-copy">
          <div class="opencode-session-title">${escapeOpenCodeSessionText(item.title)}</div>
          <div class="opencode-session-meta">${escapeOpenCodeSessionText(meta)}</div>
          <div class="opencode-session-preview">${escapeOpenCodeSessionText(preview)}</div>
        </div>
        ${active ? '<span class="opencode-session-active-pill">ACTIVE</span>' : ''}
        <button type="button" class="opencode-session-more" aria-label="Session actions">
          <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="3" r="1.2"></circle><circle cx="8" cy="8" r="1.2"></circle><circle cx="8" cy="13" r="1.2"></circle></svg>
        </button>
        <div class="opencode-session-menu" hidden>
          ${active ? '' : `<button type="button" data-action="activate">${payload.active_opencode_session_id ? 'Switch' : 'Start'}</button>`}
          <button type="button" data-action="open-native-tui">Open in OpenCode TUI</button>
          <button type="button" data-action="rename">Rename</button>
          <button type="button" class="danger" data-action="delete">Delete</button>
        </div>
      </article>`;
  }).join('');
}

async function openOpenCodeSessionPanel() {
  if (_activeRun) return;
  closeOpenCodeSessionPanel();
  const backdrop = document.createElement('div');
  backdrop.id = 'opencode-session-panel-backdrop';
  backdrop.className = 'opencode-session-panel-backdrop';
  backdrop.innerHTML = `
    <section class="opencode-session-panel" role="dialog" aria-modal="true" aria-label="OpenCode Sessions">
      <header>
        <div><h2>OpenCode Sessions</h2><span class="opencode-session-count">Loading...</span></div>
        <button type="button" class="opencode-session-close" aria-label="Close">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3l10 10M13 3L3 13"></path></svg>
        </button>
      </header>
      <div class="opencode-session-toolbar">
        <button type="button" class="opencode-session-recent" data-action="recent" aria-pressed="true">Recent</button>
        <button type="button" class="opencode-session-mode-toggle" data-action="select-mode">Select</button>
        <button type="button" class="opencode-session-new">+ New Session</button>
      </div>
      <div class="opencode-session-select-summary" hidden>
        <button type="button" class="opencode-session-select-all" data-action="select-all" aria-pressed="false">All</button>
        <span class="opencode-session-selected-count">0 Selected</span>
        <button type="button" class="opencode-session-batch-delete danger" data-action="batch-delete" disabled>Delete</button>
        <button type="button" class="opencode-session-select-close" data-action="select-close" aria-label="Exit selection mode">X</button>
      </div>
      <div class="opencode-session-search-wrap">
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.5"></circle><path d="M10.5 10.5L14 14"></path></svg>
        <input class="opencode-session-search" type="search" placeholder="Search sessions...">
      </div>
      <div class="opencode-session-list"><div class="opencode-session-empty">Loading sessions...</div></div>
    </section>`;
  document.body.appendChild(backdrop);
  const panel = backdrop.querySelector('.opencode-session-panel');
  const state = { selectMode: false, selected: new Set() };
  let payload = await loadNativeOpenCodeSessions();
  renderNativeOpenCodeSessions(panel, payload, state);
  backdrop.addEventListener('click', async event => {
    if (event.target === backdrop || event.target.closest('.opencode-session-close')) {
      closeOpenCodeSessionPanel();
      return;
    }
    const more = event.target.closest('.opencode-session-more');
    if (more) {
      const row = more.closest('.opencode-session-row');
      panel.querySelectorAll('.opencode-session-menu').forEach(menu => {
        if (menu !== row.querySelector('.opencode-session-menu')) menu.hidden = true;
      });
      const menu = row.querySelector('.opencode-session-menu');
      menu.hidden = !menu.hidden;
      return;
    }
    const action = event.target.closest('[data-action]');
    if (!action) return;
    if (action.dataset.action === 'recent') {
      const searchInput = panel.querySelector('.opencode-session-search');
      if (searchInput) searchInput.value = '';
      renderNativeOpenCodeSessions(panel, payload, state);
      return;
    }
    if (action.dataset.action === 'select-mode' || action.dataset.action === 'select-close') {
      state.selectMode = action.dataset.action === 'select-mode' ? !state.selectMode : false;
      state.selected.clear();
      renderNativeOpenCodeSessions(panel, payload, state);
      return;
    }
    if (action.dataset.action === 'select-all') {
      const search = String(panel.querySelector('.opencode-session-search')?.value || '').trim().toLowerCase();
      const visible = (Array.isArray(payload?.sessions) ? payload.sessions : [])
        .filter(item => nativeOpenCodeSessionMatches(item, search))
        .map(item => String(item.id || ''));
      const allSelected = visible.length > 0 && visible.every(id => state.selected.has(id));
      visible.forEach(id => allSelected ? state.selected.delete(id) : state.selected.add(id));
      renderNativeOpenCodeSessions(panel, payload, state);
      return;
    }
    if (action.dataset.action === 'batch-delete') {
      const selectedIds = [...state.selected];
      if (!selectedIds.length) return;
      const confirmed = await openSessionConfirmDialog({
        heading: 'Delete OpenCode Sessions',
        message: `Permanently delete ${selectedIds.length} native OpenCode session${selectedIds.length === 1 ? '' : 's'} and their history?`,
        confirmLabel: `Delete ${selectedIds.length}`,
      });
      if (!confirmed) return;
      let success = 0;
      let failed = 0;
      const byId = new Map((payload.sessions || []).map(item => [String(item.id || ''), item]));
      for (const selectedId of selectedIds) {
        try {
          await deleteNativeOpenCodeSession(selectedId, String(byId.get(selectedId)?.title || ''), { announce: false });
          success += 1;
        } catch (_) {
          failed += 1;
        }
      }
      const message = failed
        ? `Deleted ${success} OpenCode sessions. ${failed} could not be deleted.`
        : `Deleted ${success} OpenCode sessions.`;
      renderOpenCodeRuntimeStatus(message, { deterministic: true });
      state.selected.clear();
      state.selectMode = false;
      payload = await loadNativeOpenCodeSessions();
      renderNativeOpenCodeSessions(panel, payload, state);
      return;
    }
    const row = action.closest('.opencode-session-row');
    const nativeId = row?.dataset.sessionId || '';
    const currentTitle = row?.querySelector('.opencode-session-title')?.textContent?.trim() || '';
    try {
      if (action.dataset.action === 'select') {
        if (action.checked) state.selected.add(nativeId);
        else state.selected.delete(nativeId);
        renderNativeOpenCodeSessions(panel, payload, state);
        return;
      } else if (action.dataset.action === 'activate') {
        await activateNativeOpenCodeSession(nativeId);
      } else if (action.dataset.action === 'open-native-tui') {
        await openNativeOpenCodeTui(nativeId);
      } else if (action.dataset.action === 'rename') {
        if (!await renameNativeOpenCodeSession(nativeId, currentTitle)) return;
      } else if (action.dataset.action === 'delete') {
        const confirmed = await openSessionConfirmDialog({
          heading: 'Delete OpenCode Session',
          message: `Permanently delete "${currentTitle}" and its native OpenCode history?`,
        });
        if (!confirmed) return;
        await deleteNativeOpenCodeSession(nativeId, currentTitle);
      }
      payload = await loadNativeOpenCodeSessions();
      renderNativeOpenCodeSessions(panel, payload, state);
    } catch (error) {
      renderOpenCodeRuntimeStatus(`OpenCode session action failed: ${error.message || error}`, { failed: true });
    }
  });
  panel.querySelector('.opencode-session-search')?.addEventListener('input', () => {
    renderNativeOpenCodeSessions(panel, payload, state);
  });
  panel.querySelector('.opencode-session-new')?.addEventListener('click', async () => {
    const title = await openSessionNameDialog();
    if (!title) return;
    try {
      await createNativeOpenCodeSession(title);
      payload = await loadNativeOpenCodeSessions();
      renderNativeOpenCodeSessions(panel, payload, state);
    } catch (error) {
      renderOpenCodeRuntimeStatus(`OpenCode session creation failed: ${error.message || error}`, { failed: true });
    }
  });
}

function createOpenCodeSessionManagerButton() {
  if (_sessionManagerButton || document.getElementById('opencode-session-manager-btn')) {
    _sessionManagerButton = document.getElementById('opencode-session-manager-btn');
    return;
  }
  const right = document.querySelector('.chat-input-right');
  if (!right) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.id = 'opencode-session-manager-btn';
  button.className = 'input-icon-btn tool-indicator opencode-session-manager-btn';
  button.style.display = 'none';
  button.title = 'OpenCode Sessions';
  button.setAttribute('aria-label', 'OpenCode Sessions');
  button.innerHTML = `
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="18" height="5" rx="1"></rect>
      <rect x="3" y="10" width="18" height="5" rx="1"></rect>
      <rect x="3" y="16" width="18" height="4" rx="1"></rect>
    </svg>`;
  button.addEventListener('click', () => openOpenCodeSessionPanel().catch(error => {
    renderOpenCodeRuntimeStatus(`OpenCode sessions could not load: ${error.message || error}`, { failed: true });
  }));
  const tui = document.getElementById('runtime-quick-reset-btn');
  if (tui) right.insertBefore(button, tui);
  else {
    const modeToggle = right.querySelector('.mode-toggle');
    if (modeToggle) right.insertBefore(button, modeToggle);
    else right.appendChild(button);
  }
  _sessionManagerButton = button;
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
  button.title = 'Start or select an OpenCode session first.';
  button.setAttribute('aria-label', 'Open active OpenCode session in TUI');
  button.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round"
         aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="1"></rect>
      <path d="m7 9 3 3-3 3"></path>
      <path d="M13 15h4"></path>
    </svg>
  `;
  button.addEventListener('click', () => {
    handleOpenCodeSessionManagerClick().catch(() => {});
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
  _indicator.classList.toggle('active', runtime === 'opencode');
  _indicator.classList.toggle('runtime-langgraph', runtime === 'langgraph');
  _indicator.classList.toggle('runtime-legacy', runtime === 'legacy');

  const label = _indicator.querySelector('.runtime-indicator-label');
  if (label) label.textContent = runtimeLabel(runtime);

  const availableOrder = ['legacy', 'langgraph', 'opencode']
    .filter(runtimeAvailable);
  const runtimeIndex = Math.max(0, availableOrder.indexOf(runtime));
  const nextRuntime = availableOrder[
    (runtimeIndex + 1) % availableOrder.length
  ] || 'legacy';
  const next = runtimeLabel(nextRuntime);
  _indicator.title =
    `${runtimeLabel(runtime)} active — click to switch to ${next}`;

  if (!_quickResetButton) {
    _quickResetButton = document.getElementById(
      'runtime-quick-reset-btn'
    );
  }
  const managerVisible = visible && runtime === 'opencode';
  const ready = !!currentOpenCodeReadyState();
  if (!_sessionManagerButton) {
    _sessionManagerButton = document.getElementById('opencode-session-manager-btn');
  }
  if (_sessionManagerButton) {
    _sessionManagerButton.style.display = managerVisible ? '' : 'none';
    _sessionManagerButton.disabled = !!_activeRun;
  }
  if (_quickResetButton) {
    _quickResetButton.style.display = managerVisible ? '' : 'none';
    _quickResetButton.disabled =
      !ready || !!_activeRun || _openCodeSessionManagerBusy;
    _quickResetButton.classList.toggle('is-ready', ready);
    _quickResetButton.classList.toggle('active', ready);
    _quickResetButton.setAttribute(
      'aria-pressed',
      ready ? 'true' : 'false'
    );
    _quickResetButton.title = ready
      ? 'Open active OpenCode session in TUI'
      : 'Start or select an OpenCode session first.';
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

  openCodeTaskPlannerAdapter.handleRuntimeEvent(event, {
    sessionId: context?.sessionId || '',
    title: context?.title || '',
  });

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

function captureOpenCodeTranscriptMetadata(context, event) {
  if (!context || !event || typeof event !== 'object') return;

  context.runId = String(
    event.run_id || event.runId || context.runId || ''
  ).trim();
  context.opencodeSessionId = String(
    event.opencode_session_id
    || event.opencodeSessionId
    || context.opencodeSessionId
    || ''
  ).trim();
  context.providerId = String(
    event.provider_id || event.providerId || context.providerId || ''
  ).trim();
  context.modelId = String(
    event.model_id || event.modelId || context.modelId || ''
  ).trim();
  context.opencodeMessageId = String(
    event.message_id
    || event.messageId
    || context.opencodeMessageId
    || ''
  ).trim();
}

export async function streamOpenCodeTurn({
  message,
  sessionId,
  title = 'OpenCode chat',
  attachments = [],
  attachmentInfo = [],
  onAccepted = null,
} = {}) {
  const id = String(sessionId || '').trim();
  const prompt = String(message || '').trim();
  const attachmentIds = Array.isArray(attachments)
    ? attachments.map(value => String(value || '').trim()).filter(Boolean)
    : [];
  if (!id) throw new Error('OpenCode requires an Odysseus chat session.');
  if (!prompt && !attachmentIds.length) return { status: 'empty', accepted: false };
  if (_activeRun) throw new Error('An OpenCode run is already active.');

  ensureSessionRuntime(id);
  const ready = currentOpenCodeReadyState();
  if (!ready) {
    throw new Error('Start or create an OpenCode session before sending a message.');
  }
  const selection = activeOpenCodeSelection();
  if (!selection.modelId || !selection.endpointId) {
    throw new Error('Select an enabled model for the next OpenCode query.');
  }
  if (!boolSetting(REUSE_SESSION_KEY, true)) {
    await resetOpenCodeSession(id);
  }

  const userTimestamp = new Date().toISOString();
  chatRenderer.addMessage('user', prompt || '[Image attachment]', '', {
    runtime: 'opencode',
    timestamp: userTimestamp,
    ...(Array.isArray(attachmentInfo) && attachmentInfo.length
      ? { attachments: attachmentInfo }
      : {}),
  });

  const context = {
    ...assistantHolder(),
    sessionId: id,
    accumulated: '',
    terminal: '',
    toolThread: null,
    toolNodes: new Map(),
    permissionNodes: new Map(),
    runId: '',
    opencodeSessionId: '',
    providerId: '',
    modelId: '',
    opencodeMessageId: '',
  };
  const controller = new AbortController();
  _activeRun = {
    runId: '',
    sessionId: id,
    controller,
    context,
  };
  refreshRuntimeIndicator();
  let accepted = false;
  const acceptOnce = () => {
    if (accepted) return;
    accepted = true;
    if (typeof onAccepted === 'function') {
      try { onAccepted(); } catch (_) {}
    }
  };

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
        endpoint_id: selection.endpointId,
        model_id: selection.modelId,
        attachments: attachmentIds,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await openCodeResponseError(
        response,
        `OpenCode stream returned HTTP ${response.status}`
      );
      throw new Error(`OpenCode failed: ${detail}`);
    }
    if (!response.body) {
      throw new Error('OpenCode failed: the stream response had no body.');
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
        if (event.type === 'runtime.accepted') acceptOnce();
        captureOpenCodeTranscriptMetadata(context, event);
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

    if (context.terminal === 'completed') {
      const finalText = String(
        context.holder?.dataset?.raw || context.accumulated || ''
      ).trim();
      if (finalText) {
      }
    }

    return {
      status: context.terminal,
      accepted,
      runId: _activeRun && _activeRun.runId,
      content: context.accumulated,
    };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      if (!context.terminal) {
        renderTerminalFailure(context, 'Cancelled by user.', true);
        context.terminal = 'cancelled';
      }
      return { status: 'cancelled', accepted };
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
      accepted,
      error: error.message || String(error),
    };
  } finally {
    _activeRun = null;
    refreshRuntimeIndicator();
  }
}

async function hydrateDefaultRuntime() {
  const stored = safeStorageGet(DEFAULT_RUNTIME_KEY);
  if (stored) {
    safeStorageSet(
      DEFAULT_RUNTIME_KEY,
      normalizeRuntime(stored, 'legacy')
    );
    return;
  }
  try {
    const response = await fetch('/api/auth/settings', {
      credentials: 'same-origin',
    });
    const settings = await response.json();
    safeStorageSet(
      DEFAULT_RUNTIME_KEY,
      normalizeRuntime(
        settings.agent_langgraph_enabled === false ? 'legacy' : 'langgraph'
      )
    );
  } catch (_) {
    safeStorageSet(DEFAULT_RUNTIME_KEY, normalizeRuntime('langgraph'));
  }
}

export async function initRuntimeControl() {
  if (_initialized) return;
  _initialized = true;
  await loadRuntimeAvailability();
  await hydrateDefaultRuntime();
  hideLegacyLangGraphSetting();
  buildRuntimeSettings();
  applyRuntimeAvailabilityToControls();
  simplifyOpenCodeSettings();
  createRuntimeIndicator();
  createRuntimeQuickReset();
  createOpenCodeSessionManagerButton();
  watchComposerState();

  try {
    await loadOpenCodeModelOptions();
  } catch (error) {
    showConfigMessage(
      `Model registry failed: ${error.message || error}`,
      true
    );
  }

  syncSettingsControls();

  const modelInput = document.getElementById('runtime-opencode-model');
  if (modelInput && !modelInput.value) {
    const model = activeChatModel();
    if (model) selectOpenCodeModel(model);
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
