const MULTI_GRAPH_RUNTIME = 'p3c5b2-multi-graph-v1';
const SINGLE_GRAPH_RUNTIME = 'general-supervisor-v1';
const TASK_PLANNER_RUNTIMES = new Set([
  MULTI_GRAPH_RUNTIME,
  SINGLE_GRAPH_RUNTIME,
]);

const TASK_EVENT_TYPES = Object.freeze({
  PLAN_UPDATED: 'task.plan.updated',
  PLAN_COMPLETED: 'task.plan.completed',
  PLAN_FAILED: 'task.plan.failed',
  PLAN_CANCELLED: 'task.plan.cancelled',
  STEP_STARTED: 'task.step.started',
  STEP_COMPLETED: 'task.step.completed',
  STEP_FAILED: 'task.step.failed',
  STEP_PARTIAL: 'task.step.partial',
  STEP_CANCELLED: 'task.step.cancelled',
  RESET: 'task.reset',
});

const TERMINAL_PLAN_STATUSES = new Set(['completed', 'failed', 'cancelled']);
const TERMINAL_STEP_STATUSES = new Set([
  'completed',
  'failed',
  'partial',
  'blocked',
  'cancelled',
  'skipped',
]);

function createInitialState() {
  return {
    active: false,
    visible: false,
    pinned: false,
    runtime: '',
    planKey: '',
    overallStatus: '',
    revision: -1,
    objective: '',
    reasoningSummary: '',
    detail: '',
    steps: [],
    statuses: new Map(),
    currentStepId: null,
    hideTimer: null,
  };
}

const state = createInitialState();

let host = null;
let chip = null;
let label = null;
let list = null;
let title = null;
let subtitle = null;
let statusBadge = null;
let spinner = null;
let composerObserver = null;

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function normalizeStatus(value, fallback = '') {
  const status = String(value || '').trim().toLowerCase();
  if (status === 'canceled') return 'cancelled';
  if (status === 'success' || status === 'succeeded' || status === 'done') {
    return 'completed';
  }
  if (status === 'error') return 'failed';
  return status || fallback;
}

function normalizedRevision(value) {
  const revision = Number(value);
  return Number.isFinite(revision) && revision >= 0 ? revision : 0;
}

function planKeyFor(data, objective) {
  const explicit =
    data.plan_id ??
    data.planId ??
    data.id ??
    data.plan?.id;

  if (
    explicit !== undefined &&
    explicit !== null &&
    String(explicit).trim()
  ) {
    return String(explicit).trim();
  }

  return `${MULTI_GRAPH_RUNTIME}:${String(objective || '').trim()}`;
}

function normalizePlanUpdate(payload) {
  const data = payload?.data || payload || {};

  if (
    !TASK_PLANNER_RUNTIMES.has(data.runtime) ||
    !Array.isArray(data.steps) ||
    !data.steps.length
  ) {
    return null;
  }

  const objective = String(data.objective || '');
  const steps = data.steps
    .map((step) => ({
      id: String(step?.id || ''),
      objective: String(step?.objective || step?.id || ''),
      depends_on: Array.isArray(step?.depends_on)
        ? step.depends_on.map(String)
        : [],
      work_domains: Array.isArray(step?.work_domains)
        ? step.work_domains.map(String)
        : [],
      worker_label: String(step?.worker_label || ''),
      progress: String(step?.progress || ''),
      result_summary: String(step?.result_summary || ''),
      status: normalizeStatus(step?.status, 'pending'),
    }))
    .filter((step) => step.id);

  if (!steps.length) return null;

  return {
    runtime: data.runtime,
    planKey: planKeyFor(data, objective),
    revision: normalizedRevision(data.revision),
    status: normalizeStatus(data.status, 'planned'),
    objective,
    reasoningSummary: String(data.reasoning_summary || ''),
    detail: String(data.detail || data.message || ''),
    steps,
  };
}

function cloneState(source) {
  return {
    ...source,
    steps: source.steps.map((step) => ({
      ...step,
      depends_on: [...step.depends_on],
      work_domains: [...step.work_domains],
    })),
    statuses: new Map(source.statuses),
  };
}

function resetState(target) {
  const clean = createInitialState();
  const timer = target.hideTimer;
  Object.assign(target, clean);
  target.hideTimer = timer;
  return target;
}

function stepIdFromPayload(payload) {
  return String(
    payload?.plan_step_id ??
    payload?.step_id ??
    payload?.data?.plan_step_id ??
    payload?.data?.step_id ??
    ''
  );
}

function applyPlan(target, plan) {
  const isSamePlan =
    target.active &&
    target.planKey === plan.planKey;

  if (isSamePlan && plan.revision < target.revision) {
    return target;
  }

  const previousStatuses = isSamePlan
    ? new Map(target.statuses)
    : new Map();

  const previousCurrentStep = isSamePlan
    ? target.currentStepId
    : null;

  target.active = true;
  target.visible = true;
  target.runtime = plan.runtime;
  target.planKey = plan.planKey;
  target.revision = plan.revision;
  target.overallStatus = plan.status;
  target.objective = plan.objective;
  target.reasoningSummary = plan.reasoningSummary;
  target.detail = plan.detail;
  target.steps = plan.steps;
  target.statuses = new Map();

  for (const step of target.steps) {
    const prior = previousStatuses.get(step.id);
    target.statuses.set(
      step.id,
      TERMINAL_STEP_STATUSES.has(prior)
        ? prior
        : normalizeStatus(step.status, prior || 'pending'),
    );
  }

  target.currentStepId = (
    previousCurrentStep &&
    target.statuses.has(previousCurrentStep)
  )
    ? previousCurrentStep
    : null;

  if (TERMINAL_PLAN_STATUSES.has(target.overallStatus)) {
    target.currentStepId = null;
  }

  return target;
}

function setStepStatus(target, stepId, status) {
  if (
    !target.active ||
    !stepId ||
    !target.statuses.has(stepId)
  ) {
    return target;
  }

  const normalized = normalizeStatus(status);
  if (!normalized) return target;

  target.statuses.set(stepId, normalized);

  if (normalized === 'running') {
    target.currentStepId = stepId;
  } else if (target.currentStepId === stepId) {
    target.currentStepId = null;
  }

  return target;
}

function reduceTaskEvent(source, event) {
  const target = cloneState(source);
  const type = String(event?.type || '');
  const payload = event?.payload ?? event?.data ?? {};

  if (type === TASK_EVENT_TYPES.RESET) {
    return resetState(target);
  }

  if (type === TASK_EVENT_TYPES.PLAN_UPDATED) {
    const plan = event?.plan || normalizePlanUpdate(payload);
    return plan ? applyPlan(target, plan) : target;
  }

  if (!target.active) return target;

  const stepId = String(
    event?.stepId ||
    stepIdFromPayload(payload)
  );

  if (type === TASK_EVENT_TYPES.STEP_STARTED) {
    return setStepStatus(target, stepId, 'running');
  }

  if (type === TASK_EVENT_TYPES.STEP_COMPLETED) {
    return setStepStatus(target, stepId, 'completed');
  }

  if (type === TASK_EVENT_TYPES.STEP_FAILED) {
    target.detail = String(
      event?.detail ||
      payload?.error ||
      payload?.message ||
      target.detail ||
      ''
    );
    return setStepStatus(target, stepId, 'failed');
  }

  if (type === TASK_EVENT_TYPES.STEP_PARTIAL) {
    return setStepStatus(target, stepId, 'partial');
  }

  if (type === TASK_EVENT_TYPES.STEP_CANCELLED) {
    return setStepStatus(target, stepId, 'cancelled');
  }

  if (
    type === TASK_EVENT_TYPES.PLAN_COMPLETED ||
    type === TASK_EVENT_TYPES.PLAN_FAILED ||
    type === TASK_EVENT_TYPES.PLAN_CANCELLED
  ) {
    const planStatus = type.split('.').at(-1);
    target.overallStatus = planStatus;
    target.currentStepId = null;
    target.detail = String(
      event?.detail ||
      payload?.error ||
      payload?.message ||
      target.detail ||
      ''
    );

    if (planStatus === 'completed') {
      for (const step of target.steps) {
        const current = target.statuses.get(step.id);
        if (current === 'running') {
          target.statuses.set(step.id, 'completed');
        }
      }
    } else {
      for (const step of target.steps) {
        if (target.statuses.get(step.id) === 'running') {
          target.statuses.set(
            step.id,
            planStatus === 'failed'
              ? 'failed'
              : 'cancelled',
          );
        }
      }
    }
  }

  return target;
}

function copyReducedState(next) {
  const timer = state.hideTimer;
  Object.assign(state, next);
  state.hideTimer = timer;
}

function visibleDockChips() {
  return Array
    .from(document.querySelectorAll('.minimized-dock-chip'))
    .filter((el) => {
      const style = getComputedStyle(el);
      const rect = el.getBoundingClientRect();

      return (
        style.display !== 'none' &&
        style.visibility !== 'hidden' &&
        rect.width > 0 &&
        rect.height > 0
      );
    });
}

function elevationFromMinimizedUi() {
  const candidates = [
    ...visibleDockChips(),
    document.querySelector('.minimized-dock'),
    document.getElementById('minimized-dock'),
  ].filter(Boolean);

  for (const start of candidates) {
    let el = start;

    while (
      el &&
      el !== document.documentElement
    ) {
      const z = getComputedStyle(el).zIndex;

      if (z && z !== 'auto') {
        const number = Number(z);
        if (Number.isFinite(number)) return number;
      }

      el = el.parentElement;
    }
  }

  return 80;
}

function rectanglesOverlap(a, b, padding = 6) {
  return !(
    a.right + padding <= b.left ||
    a.left >= b.right + padding ||
    a.bottom + padding <= b.top ||
    a.top >= b.bottom + padding
  );
}

function composerRect() {
  const composer =
    document.querySelector('.chat-input-bar') ||
    document.getElementById('chat-form') ||
    document.getElementById('message')?.closest('form');

  if (!composer) return null;

  const rect = composer.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0
    ? rect
    : null;
}

function positionTracker() {
  if (!host || !state.visible) return;

  const rect = composerRect();
  if (!rect) return;

  host.style.zIndex = String(elevationFromMinimizedUi());

  const bottom = Math.max(
    8,
    window.innerHeight - rect.top + 10,
  );
  host.style.bottom = `${bottom}px`;

  const minX = rect.left + 56;
  const maxX = rect.right - 56;
  const candidates = [
    rect.left + rect.width * 0.50,
    rect.left + rect.width * 0.38,
    rect.left + rect.width * 0.62,
    rect.left + rect.width * 0.28,
    rect.left + rect.width * 0.72,
  ].map((x) => Math.max(minX, Math.min(maxX, x)));

  const dockRects = visibleDockChips()
    .map((el) => el.getBoundingClientRect());

  let bestX = candidates[0];
  let bestOverlap = Number.POSITIVE_INFINITY;

  for (const x of candidates) {
    host.style.left = `${x}px`;

    const trackerRect = chip?.getBoundingClientRect();
    if (!trackerRect) break;

    let overlapArea = 0;

    for (const dockRect of dockRects) {
      if (!rectanglesOverlap(trackerRect, dockRect)) {
        continue;
      }

      const width = Math.max(
        0,
        Math.min(trackerRect.right, dockRect.right) -
        Math.max(trackerRect.left, dockRect.left),
      );
      const height = Math.max(
        0,
        Math.min(trackerRect.bottom, dockRect.bottom) -
        Math.max(trackerRect.top, dockRect.top),
      );
      overlapArea += width * height;
    }

    if (overlapArea === 0) {
      bestX = x;
      bestOverlap = 0;
      break;
    }

    if (overlapArea < bestOverlap) {
      bestOverlap = overlapArea;
      bestX = x;
    }
  }

  host.style.left = `${bestX}px`;
  host.style.setProperty(
    '--mg-tracker-max-width',
    `${Math.max(
      240,
      Math.min(420, rect.width - 24),
    )}px`,
  );
}

function ensureMount() {
  if (
    host &&
    document.body.contains(host)
  ) {
    return;
  }

  host = document.createElement('div');
  host.id = 'multi-graph-tracker';
  host.setAttribute('aria-live', 'polite');
  host.innerHTML = `
    <div
      class="multi-graph-tracker-panel"
      role="dialog"
      aria-label="Task plan"
    >
      <div class="multi-graph-tracker-panel-head">
        <div class="multi-graph-tracker-heading">
          <div class="multi-graph-tracker-title">Plan</div>
          <div class="multi-graph-tracker-subtitle"></div>
        </div>
        <span class="multi-graph-tracker-status">Planning</span>
      </div>
      <div class="multi-graph-tracker-list"></div>
    </div>
    <button
      type="button"
      class="multi-graph-tracker-chip"
      aria-expanded="false"
      aria-label="Show task plan"
    >
      <span
        class="multi-graph-tracker-spinner"
        aria-hidden="true"
      ></span>
      <span class="multi-graph-tracker-label">Step 1/1</span>
      <svg
        class="multi-graph-tracker-caret"
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <polyline points="18 15 12 9 6 15"></polyline>
      </svg>
    </button>
  `;

  document.body.appendChild(host);

  chip = host.querySelector('.multi-graph-tracker-chip');
  label = host.querySelector('.multi-graph-tracker-label');
  list = host.querySelector('.multi-graph-tracker-list');
  title = host.querySelector('.multi-graph-tracker-title');
  subtitle = host.querySelector('.multi-graph-tracker-subtitle');
  statusBadge = host.querySelector('.multi-graph-tracker-status');
  spinner = host.querySelector('.multi-graph-tracker-spinner');

  chip.addEventListener('click', (event) => {
    event.stopPropagation();
    state.pinned = !state.pinned;
    host.classList.toggle('is-open', state.pinned);
    chip.setAttribute(
      'aria-expanded',
      state.pinned ? 'true' : 'false',
    );
    requestAnimationFrame(positionTracker);
  });

  host.addEventListener('mouseenter', () => {
    if (!state.visible) return;
    host.classList.add('is-previewing');
    requestAnimationFrame(positionTracker);
  });

  host.addEventListener('mouseleave', () => {
    host.classList.remove('is-previewing');
  });

  document.addEventListener('click', (event) => {
    if (
      !state.pinned ||
      host.contains(event.target)
    ) {
      return;
    }

    state.pinned = false;
    host.classList.remove('is-open');
    chip.setAttribute('aria-expanded', 'false');
  });

  document.addEventListener('keydown', (event) => {
    if (
      event.key !== 'Escape' ||
      !state.pinned
    ) {
      return;
    }

    state.pinned = false;
    host.classList.remove('is-open');
    chip.setAttribute('aria-expanded', 'false');
  });

  window.addEventListener(
    'resize',
    () => requestAnimationFrame(positionTracker),
  );

  window.addEventListener(
    'scroll',
    () => requestAnimationFrame(positionTracker),
    true,
  );

  const composer =
    document.querySelector('.chat-input-bar') ||
    document.getElementById('chat-form');

  if (
    composer &&
    'ResizeObserver' in window
  ) {
    composerObserver?.disconnect();
    composerObserver = new ResizeObserver(
      () => requestAnimationFrame(positionTracker),
    );
    composerObserver.observe(composer);
  }
}

function statusFor(stepId) {
  return state.statuses.get(stepId) || 'pending';
}

function activeStepIndex() {
  if (!state.steps.length) return 0;

  if (state.currentStepId) {
    const index = state.steps.findIndex(
      (step) => step.id === state.currentStepId,
    );

    if (index >= 0) return index;
  }

  const next = state.steps.findIndex(
    (step) => !TERMINAL_STEP_STATUSES.has(
      statusFor(step.id),
    ),
  );

  return next >= 0
    ? next
    : state.steps.length - 1;
}

function statusGlyph(status) {
  if (status === 'completed') {
    return '<span class="mg-step-glyph is-completed">✓</span>';
  }
  if (status === 'running') {
    return '<span class="mg-step-glyph is-running"><span></span></span>';
  }
  if (status === 'failed') {
    return '<span class="mg-step-glyph is-failed">×</span>';
  }
  if (status === 'partial') {
    return '<span class="mg-step-glyph is-partial">!</span>';
  }
  if (status === 'blocked') {
    return '<span class="mg-step-glyph is-partial">!</span>';
  }
  if (status === 'cancelled') {
    return '<span class="mg-step-glyph is-cancelled">−</span>';
  }
  if (status === 'skipped') {
    return '<span class="mg-step-glyph is-skipped">↷</span>';
  }

  return '<span class="mg-step-glyph is-pending"></span>';
}

function statusLabel() {
  const status = normalizeStatus(
    state.overallStatus,
    'running',
  );

  if (status === 'completed') return 'Complete';
  if (status === 'failed') return 'Failed';
  if (status === 'cancelled') return 'Cancelled';
  if (status === 'planned') return 'Planned';

  return 'Running';
}

function chipLabel() {
  const status = normalizeStatus(state.overallStatus);

  if (status === 'completed') return 'Plan complete';
  if (status === 'failed') return 'Plan failed';
  if (status === 'cancelled') return 'Plan cancelled';

  const index = activeStepIndex();
  const current = Math.min(
    state.steps.length,
    index + 1,
  );

  return `Step ${current}/${state.steps.length}`;
}

function render() {
  ensureMount();

  if (
    !state.active ||
    !state.steps.length
  ) {
    state.visible = false;
    state.pinned = false;
    host.classList.remove(
      'is-open',
      'is-previewing',
    );
    host.style.display = 'none';
    chip.setAttribute('aria-expanded', 'false');
    return;
  }

  state.visible = true;
  host.style.display = 'block';

  label.textContent = chipLabel();
  statusBadge.textContent = statusLabel();

  const terminal = TERMINAL_PLAN_STATUSES.has(
    normalizeStatus(state.overallStatus),
  );
  spinner.hidden = terminal;

  host.classList.toggle(
    'is-complete',
    state.overallStatus === 'completed',
  );
  host.classList.toggle(
    'is-failed',
    state.overallStatus === 'failed',
  );
  host.classList.toggle(
    'is-cancelled',
    state.overallStatus === 'cancelled',
  );

  title.textContent = state.objective || 'Plan';
  subtitle.textContent =
    state.detail ||
    state.reasoningSummary ||
    '';

  list.innerHTML = state.steps
    .map((step, index) => {
      const status = statusFor(step.id);
      const dependencies = (
        Array.isArray(step.depends_on) &&
        step.depends_on.length
      )
        ? (
          '<div class="mg-step-deps">' +
          'after ' +
          escapeHtml(step.depends_on.join(', ')) +
          '</div>'
        )
        : '';
      const context = [
        step.worker_label,
        status === 'completed'
          ? step.result_summary
          : step.progress,
      ].filter(Boolean).join(' · ');

      return `
        <div
          class="multi-graph-tracker-step is-${escapeHtml(status)}"
          data-step-id="${escapeHtml(step.id)}"
        >
          ${statusGlyph(status)}
          <div class="mg-step-copy">
            <div class="mg-step-line">
              <span class="mg-step-number">${index + 1}</span>
              <span class="mg-step-objective">${
                escapeHtml(step.objective || step.id)
              }</span>
            </div>
            ${dependencies}
            ${context ? `<div class="mg-step-deps">${escapeHtml(context)}</div>` : ''}
          </div>
        </div>
      `;
    })
    .join('');

  requestAnimationFrame(positionTracker);
}

function clearHideTimer() {
  if (!state.hideTimer) return;

  clearTimeout(state.hideTimer);
  state.hideTimer = null;
}

function scheduleHide(ms = 7000) {
  clearHideTimer();

  state.hideTimer = setTimeout(() => {
    reset();
  }, ms);
}

function handleEvent(event) {
  clearHideTimer();

  const next = reduceTaskEvent(state, event);
  copyReducedState(next);
  render();

  if (
    TERMINAL_PLAN_STATUSES.has(
      normalizeStatus(state.overallStatus),
    )
  ) {
    scheduleHide();
  }
}

function handlePlanUpdate(payload) {
  const plan = normalizePlanUpdate(payload);
  if (!plan) return;

  handleEvent({
    type: TASK_EVENT_TYPES.PLAN_UPDATED,
    plan,
  });
}

function handleToolStart(payload) {
  const stepId = stepIdFromPayload(payload);
  if (!stepId) return;

  handleEvent({
    type: TASK_EVENT_TYPES.STEP_STARTED,
    stepId,
    payload,
  });
}

function handleToolOutput(payload) {
  const stepId = stepIdFromPayload(payload);
  if (!stepId) return;

  const explicit = normalizeStatus(
    payload?.status ||
    payload?.data?.status,
  );

  let type = TASK_EVENT_TYPES.STEP_COMPLETED;

  if (explicit === 'partial') {
    type = TASK_EVENT_TYPES.STEP_PARTIAL;
  } else if (explicit === 'cancelled') {
    type = TASK_EVENT_TYPES.STEP_CANCELLED;
  } else if (
    explicit === 'failed' ||
    payload?.success === false ||
    payload?.data?.success === false
  ) {
    type = TASK_EVENT_TYPES.STEP_FAILED;
  }

  handleEvent({
    type,
    stepId,
    detail:
      payload?.error ||
      payload?.data?.error ||
      '',
    payload,
  });
}

function handleMetrics(metrics, options = {}) {
  if (options.background) return;

  if (!metrics?.multi_graph && !metrics?.task_planner) {
    if (state.active) reset();
    return;
  }

  const executionOrder = Array.isArray(
    metrics.execution_order,
  )
    ? metrics.execution_order.map(String)
    : [];

  for (const stepId of executionOrder) {
    if (
      state.statuses.has(stepId) &&
      statusFor(stepId) === 'pending'
    ) {
      handleEvent({
        type: TASK_EVENT_TYPES.STEP_COMPLETED,
        stepId,
      });
    }
  }

  const planStatus = normalizeStatus(
    metrics.plan_status ||
    state.overallStatus,
  );

  if (planStatus === 'completed') {
    handleEvent({
      type: TASK_EVENT_TYPES.PLAN_COMPLETED,
      payload: metrics,
    });
  } else if (planStatus === 'failed') {
    handleEvent({
      type: TASK_EVENT_TYPES.PLAN_FAILED,
      detail:
        metrics.error ||
        metrics.message ||
        '',
      payload: metrics,
    });
  } else if (planStatus === 'cancelled') {
    handleEvent({
      type: TASK_EVENT_TYPES.PLAN_CANCELLED,
      detail: metrics.message || '',
      payload: metrics,
    });
  } else {
    state.overallStatus =
      planStatus ||
      state.overallStatus;
    render();
  }
}

function reset() {
  clearHideTimer();

  const clean = createInitialState();
  Object.assign(state, clean);
  render();
}

function getStateSnapshot() {
  return cloneState(state);
}

export {
  MULTI_GRAPH_RUNTIME,
  SINGLE_GRAPH_RUNTIME,
  TASK_EVENT_TYPES,
  createInitialState,
  getStateSnapshot,
  normalizePlanUpdate,
  reduceTaskEvent,
  reset,
};

export default {
  handleEvent,
  handlePlanUpdate,
  handleToolStart,
  handleToolOutput,
  handleMetrics,
  reset,
};
