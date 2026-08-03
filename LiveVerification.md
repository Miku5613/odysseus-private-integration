# LiveVerification

- Date: 2026-08-03
- Branch: `feature/langgraph-product-runtime-hardening-20260729`
- Disposable server: `http://127.0.0.1:8001` with auth disabled
- Provider: `qwen3.5:9b` via local Ollama OpenAI-compatible endpoint
- Prompt: one bounded three-goal workspace run covering an expected missing file,
  `requirements.txt` dependency categorization, and grounded coordinator inspection.

Result: passed. Admission selected `multi_graph`; the expected missing file did
not block the other goals; the runtime directly read all 57 lines of the
workspace-root `requirements.txt`; all three acquisition steps completed; and
Qwen emitted a non-empty selected-model synthesis. The result persisted under
session `b26a1a3a-c108-4de4-a031-857a91c444c0` / message
`77ac38a7-4763-4437-986d-602a2dc13c59`.

Observed terminal metrics: `response_origin=selected_model`,
`presenting_model=qwen3.5:9b`, `fallback_used=false`, `fallback_scope=none`,
`synthesis_error=null`, and terminal `COMPLETED`. No web-search widening or
whole-query Legacy fallback occurred. Resume from cursor 0 replayed the
authoritative plan and terminal through cursor 5 with `run_completed=true`.

Focused verification: 151 provider, admission, planner, worker, stream,
recovery, and persistence checks passed.
