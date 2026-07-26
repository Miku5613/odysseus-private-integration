// static/js/openCodeTaskPlannerAdapter.js
//
// OpenCode event adapter for the shared Task Planner UI.
// OpenCode remains a peer runtime. This module translates only OpenCode-owned
// runtime events into the runtime-agnostic Task Planner contract.

import multiGraphTracker, {
  MULTI_GRAPH_RUNTIME,
  TASK_EVENT_TYPES,
} from './multiGraphTracker.js';

const OPENCODE_OWNER = 'opencode-task-planner-adapter-v1';
const PERMISSION_EVENTS = new Set([
  'runtime.permission_requested',
  'runtime.permission_resolved',
]);

let adapterState = createAdapterState();

function createAdapterState() {
  return {
    ownsTracker: false,
    sessionId: '',
    runId: '',
    planId: '',
    revision: 0,
    objective: '',
    hasNativePlan: false,
    steps: [],
    currentStepId: '',
  };
}

function stringValue(...values) {
  for (const value of values) {
    if (value === undefined || value === null) continue;
    const text = String(value).trim();
    if (text) return text;
  }
  return '';
}

function eventData(event) {
  return event?.data && typeof event.data === 'object'
    ? event.data
    : {};
}

function eventType(event) {
  return stringValue(event?.type).toLowerCase();
}

function eventRunId(event, context = {}) {
  const data = eventData(event);
  return stringValue(
    event?.run_id,
    event?.runId,
    data.run_id,
    data.runId,
    event?.id,
    data.id,
    context?.runId,
    context?.sessionId,
    'opencode-run',
  );
}

function eventObjective(event, context = {}) {
  const data = eventData(event);
  return stringValue(
    event?.objective,
    data.objective,
    event?.title,
    data.title,
    context?.title,
    event?.prompt,
    data.prompt,
    event?.message,
    data.message,
    'OpenCode execution',
  );
}

function normalizeStatus(value) {
  const status = stringValue(value).toLowerCase();

  if (
    status === 'done' ||
    status === 'success' ||
    status === 'succeeded' ||
    status === 'complete'
  ) {
    return 'completed';
  }

  if (status === 'error' || status === 'errored') {
    return 'failed';
  }

  if (status === 'canceled') return 'cancelled';

  if (
    status === 'in_progress' ||
    status === 'in-progress' ||
    status === 'working' ||
    status === 'started'
  ) {
    return 'running';
  }

  return status;
}

function stepIdFromEvent(event) {
  const data = eventData(event);
  return stringValue(
    event?.plan_step_id,
    event?.step_id,
    event?.stepId,
    data.plan_step_id,
    data.step_id,
    data.stepId,
    event?.call_id,
    event?.callId,
    data.call_id,
    data.callId,
    event?.tool_call_id,
    event?.toolCallId,
    data.tool_call_id,
    data.toolCallId,
  );
}

function stepObjective(step, fallback = '') {
  return stringValue(
    step?.objective,
    step?.title,
    step?.name,
    step?.tool,
    step?.action,
    step?.summary,
    step?.id,
    fallback,
  );
}


// STEP3L_USER_FACING_PLANNER
const PLANNER_TECHNICAL_PATTERNS = [
  /(?:[A-Za-z]:\\|\/)[^\s"'`]+/g,
  /(?:[\w.-]+[\\/])+[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|html|css)\b/gi,
  /\b(?:read|write|edit|bash|grep|glob|todowrite)\b/gi,
  /\bqwen[\w.:-]*\b/gi,
  /\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b/gi,
];

function cleanPlannerObjective(value, fallback = 'Complete the requested task') {
  let text = stringValue(value).replace(/\s+/g, ' ').trim();
  if (!text) return fallback;

  text = text
    .replace(/^inspect\s+.+?\s+and\s+/i, '')
    .replace(/^review\s+.+?\s+and\s+/i, '')
    .replace(/^read\s+.+?\s+and\s+/i, '')
    .replace(/^check\s+.+?\s+and\s+/i, '');

  for (const pattern of PLANNER_TECHNICAL_PATTERNS) {
    text = text.replace(pattern, ' ');
  }

  text = text
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:])/g, '$1')
    .trim();

  if (!text || text.length < 4) return fallback;
  return (text.charAt(0).toUpperCase() + text.slice(1)).slice(0, 180);
}

function objectiveIsFinal(value) {
  return /\b(final|verdict|summary|summar(?:y|ize)|synthesi[sz]e|conclusion|answer)\b/i
    .test(String(value || ''));
}

function fallbackStepsFromObjective(objective) {
  const numbered = [];
  const objectiveLines = String(objective || '')
    .replace(/\\n/g, '\n')
    .replace(/\r\n/g, '\n')
    .split('\n');

  for (const line of objectiveLines) {
    const match = line.match(/^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$/);
    if (!match) continue;
    const cleaned = cleanPlannerObjective(match[1], '');
    if (cleaned) numbered.push(cleaned);
  }

  const objectives = numbered.length
    ? numbered
    : [cleanPlannerObjective(objective)];

  if (
    objectives.length > 1 &&
    !objectiveIsFinal(objectives[objectives.length - 1])
  ) {
    objectives.push('Prepare the final answer');
  }

  return objectives.map((item, index) => ({
    id: `opencode-objective-${index + 1}`,
    objective: item,
    depends_on: index ? [`opencode-objective-${index}`] : [],
    work_domains: ['general'],
    status: index === 0 ? 'running' : 'pending',
  }));
}

function nextFallbackStepId() {
  const current = adapterState.steps.find(
    step => step.id === adapterState.currentStepId
  );
  if (
    current &&
    ['pending', 'running'].includes(normalizeStatus(current.status))
  ) {
    return current.id;
  }

  return (
    adapterState.steps.find(
      step => normalizeStatus(step.status) === 'running'
    )?.id ||
    adapterState.steps.find(
      step => normalizeStatus(step.status) === 'pending'
    )?.id ||
    ''
  );
}

function stepsFromEvent(event, fallbackObjective = '') {
  const data = eventData(event);
  const candidates = (
    Array.isArray(event?.steps)
      ? event.steps
      : Array.isArray(event?.plan?.steps)
        ? event.plan.steps
        : Array.isArray(data.steps)
          ? data.steps
          : Array.isArray(data.plan?.steps)
            ? data.plan.steps
            : []
  );

  const normalized = candidates
    .map((step, index) => ({
      id: stringValue(
        step?.id,
        step?.step_id,
        step?.stepId,
        `opencode-step-${index + 1}`,
      ),
      objective: cleanPlannerObjective(
        stepObjective(step, ''),
        `Complete objective ${index + 1}`,
      ),
      depends_on: Array.isArray(step?.depends_on)
        ? step.depends_on.map(String)
        : [],
      work_domains: ['general'],
      status: normalizeStatus(step?.status) || 'pending',
    }))
    .filter(step => step.id && step.objective);

  return normalized.length
    ? normalized
    : fallbackStepsFromObjective(fallbackObjective);
}

function cloneSteps(steps) {
  return steps.map(step => ({
    ...step,
    depends_on: [...(step.depends_on || [])],
    work_domains: [...(step.work_domains || [])],
  }));
}

function planAction(status = 'running', detail = '') {
  return {
    kind: 'plan',
    payload: {
      data: {
        runtime: MULTI_GRAPH_RUNTIME,
        plan_id: adapterState.planId,
        revision: adapterState.revision,
        status,
        objective: adapterState.objective,
        reasoning_summary: 'OpenCode peer runtime',
        detail,
        owner: OPENCODE_OWNER,
        steps: cloneSteps(adapterState.steps),
      },
    },
  };
}

function eventAction(type, stepId = '', detail = '', payload = {}) {
  return {
    kind: 'event',
    event: {
      type,
      stepId,
      detail,
      payload,
    },
  };
}

function ensureStepForStatus(event) {
  const explicitId = stepIdFromEvent(event);

  if (adapterState.hasNativePlan && explicitId) {
    return adapterState.steps.some(step => step.id === explicitId)
      ? explicitId
      : (
        adapterState.currentStepId ||
        adapterState.steps.find(
          step => normalizeStatus(step.status) === 'running'
        )?.id ||
        ''
      );
  }

  const fallbackId = nextFallbackStepId();
  adapterState.currentStepId = fallbackId;
  return fallbackId;
}

function startRun(event, context = {}) {
  const runId = eventRunId(event, context);
  const objective = eventObjective(event, context);

  adapterState = {
    ownsTracker: true,
    sessionId: stringValue(context?.sessionId),
    runId,
    planId: `opencode:${runId}`,
    revision: 1,
    objective,
    hasNativePlan: false,
    steps: stepsFromEvent(event, objective),
    currentStepId: '',
  };

  const firstStepId = (
    stepIdFromEvent(event) ||
    adapterState.steps.find(step => (
      normalizeStatus(step.status) === 'running'
    ))?.id ||
    adapterState.steps[0]?.id ||
    ''
  );

  adapterState.currentStepId = firstStepId;

  const actions = [planAction('running')];
  if (firstStepId) {
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_STARTED,
      firstStepId,
      '',
      event,
    ));
  }
  return actions;
}

function applyNativePlanUpdate(event) {
  if (!adapterState.ownsTracker) return [];

  const data = eventData(event);
  const hasIncomingSteps = (
    Array.isArray(event?.steps) ||
    Array.isArray(event?.plan?.steps) ||
    Array.isArray(data.steps) ||
    Array.isArray(data.plan?.steps)
  );
  if (!hasIncomingSteps) return [];

  const steps = stepsFromEvent(event, adapterState.objective);
  if (!steps.length) return [];

  adapterState.steps = steps;
  adapterState.hasNativePlan = true;
  adapterState.revision += 1;
  adapterState.currentStepId = (
    steps.find(step => normalizeStatus(step.status) === 'running')?.id ||
    ''
  );

  const detail = stringValue(
    event?.detail,
    data.detail,
    event?.message,
    data.message,
    'OpenCode checklist updated',
  );

  return [planAction('running', detail)];
}

function translateStatus(event) {
  if (!adapterState.ownsTracker) return [];

  const data = eventData(event);
  const type = eventType(event);

  if (
    adapterState.hasNativePlan &&
    (type === 'tool.started' || type === 'tool.completed')
  ) {
    const toolName = stringValue(event?.tool, data.tool, 'tool');
    const summary = stringValue(
      event?.summary,
      data.summary,
      event?.title,
      data.title,
    );
    const detail = type === 'tool.completed'
      ? 'Objective activity completed'
      : 'Working on the current objective';

    adapterState.revision += 1;
    return [planAction('running', detail)];
  }
  const hasIncomingSteps = (
    Array.isArray(event?.steps) ||
    Array.isArray(event?.plan?.steps) ||
    Array.isArray(data.steps) ||
    Array.isArray(data.plan?.steps)
  );

  if (hasIncomingSteps) {
    adapterState.steps = stepsFromEvent(event, adapterState.objective);
    adapterState.revision += 1;
  }

  const revisionBeforeStep = adapterState.revision;
  const stepId = ensureStepForStatus(event);
  const addedDynamicStep = adapterState.revision > revisionBeforeStep;
  const status = normalizeStatus(
    event?.status ||
    data.status ||
    event?.phase ||
    data.phase ||
    event?.state ||
    data.state,
  );
  const detail = stringValue(
    event?.error,
    data.error,
    event?.message,
    data.message,
    event?.summary,
    data.summary,
  );

  const actions = [];

  if (!adapterState.hasNativePlan && stepId) {
    const current = adapterState.steps.find(step => step.id === stepId);
    if (current) {
      if (status === 'completed' || status === 'failed') {
        current.status = status;
        const next = adapterState.steps.find(
          step => normalizeStatus(step.status) === 'pending'
        );
        if (next) {
          next.status = 'running';
          adapterState.currentStepId = next.id;
        } else {
          adapterState.currentStepId = '';
        }
        adapterState.revision += 1;
        actions.push(planAction('running', 'Objective completed'));
      } else if (status === 'running') {
        current.status = 'running';
        adapterState.currentStepId = current.id;
      }
    }
  }

  if (hasIncomingSteps || addedDynamicStep) {
    actions.push(planAction('running', detail));
  }

  if (!stepId) return actions;

  if (status === 'completed') {
    adapterState.currentStepId = '';
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_COMPLETED,
      stepId,
      detail,
      event,
    ));
  } else if (status === 'failed') {
    adapterState.currentStepId = '';
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_FAILED,
      stepId,
      detail,
      event,
    ));
  } else if (status === 'partial') {
    adapterState.currentStepId = '';
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_PARTIAL,
      stepId,
      detail,
      event,
    ));
  } else if (status === 'cancelled') {
    adapterState.currentStepId = '';
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_CANCELLED,
      stepId,
      detail,
      event,
    ));
  } else if (status === 'running' || !status) {
    adapterState.currentStepId = stepId;
    actions.push(eventAction(
      TASK_EVENT_TYPES.STEP_STARTED,
      stepId,
      detail,
      event,
    ));
  }

  return actions;
}

function terminalAction(type, event) {
  if (!adapterState.ownsTracker) return [];

  const data = eventData(event);
  const detail = stringValue(
    event?.error,
    data.error,
    event?.message,
    data.message,
    event?.summary,
    data.summary,
  );

  adapterState.currentStepId = '';
  return [eventAction(type, '', detail, event)];
}

function translateOpenCodeRuntimeEvent(event, context = {}) {
  const type = eventType(event);

  if (!type || PERMISSION_EVENTS.has(type)) return [];

  if (type === 'runtime.started') {
    return startRun(event, context);
  }

  if (type === 'runtime.plan.updated') {
    return applyNativePlanUpdate(event);
  }

  if (type === 'runtime.status') {
    return translateStatus(event);
  }

  if (type === 'tool.started') {
    return translateStatus({
      ...event,
      status: 'running',
    });
  }

  if (type === 'tool.completed') {
    const data = eventData(event);
    const completedStatus = normalizeStatus(
      event?.status ||
      data.status ||
      (
        event?.error || data.error
          ? 'failed'
          : 'completed'
      ),
    ) || 'completed';

    return translateStatus({
      ...event,
      status: completedStatus,
    });
  }

  if (type === 'runtime.completed') {
    adapterState.steps = adapterState.steps.map(step => ({
      ...step,
      status: 'completed',
    }));
    adapterState.currentStepId = '';
    adapterState.status = 'completed';
    adapterState.revision += 1;
    return [planAction('completed', 'Completed')];
  }

  if (type === 'runtime.failed') {
    return terminalAction(
      TASK_EVENT_TYPES.PLAN_FAILED,
      event,
    );
  }

  if (type === 'runtime.cancelled') {
    return terminalAction(
      TASK_EVENT_TYPES.PLAN_CANCELLED,
      event,
    );
  }

  return [];
}

function applyAction(action) {
  if (action.kind === 'plan') {
    multiGraphTracker.handlePlanUpdate(action.payload);
    return;
  }

  if (action.kind === 'event') {
    multiGraphTracker.handleEvent(action.event);
  }
}

function handleRuntimeEvent(event, context = {}) {
  const actions = translateOpenCodeRuntimeEvent(event, context);
  for (const action of actions) applyAction(action);
  return actions.length > 0;
}

function handleRuntimeChanged(runtime, context = {}) {
  const selected = stringValue(runtime).toLowerCase();

  if (
    adapterState.ownsTracker &&
    selected !== 'opencode'
  ) {
    multiGraphTracker.reset();
    adapterState = createAdapterState();
    return true;
  }

  if (
    selected === 'opencode' &&
    context?.sessionId
  ) {
    adapterState.sessionId = String(context.sessionId);
  }

  return false;
}

function resetOwnedTracker() {
  if (adapterState.ownsTracker) {
    multiGraphTracker.reset();
  }
  adapterState = createAdapterState();
}

function resetAdapterStateForTests() {
  adapterState = createAdapterState();
}

function getAdapterStateSnapshot() {
  return {
    ...adapterState,
    steps: cloneSteps(adapterState.steps),
  };
}

export {
  OPENCODE_OWNER,
  getAdapterStateSnapshot,
  handleRuntimeChanged,
  handleRuntimeEvent,
  resetAdapterStateForTests,
  resetOwnedTracker,
  translateOpenCodeRuntimeEvent,
};

export default {
  getAdapterStateSnapshot,
  handleRuntimeChanged,
  handleRuntimeEvent,
  resetOwnedTracker,
};
