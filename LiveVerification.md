# LiveVerification

- Date: 2026-07-29
- Branch: `feature/langgraph-product-runtime-hardening-20260729`
- Disposable server: `http://127.0.0.1:8001` with auth disabled
- Provider: `qwen3.5:9b` via local Ollama OpenAI-compatible endpoint
- Prompt: `Read requirements.txt and summarize the main dependency categories. Do not modify anything.`

Result: passed. Structured admission and supervisor calls used JSON mode with `reasoning_effort: none`; the runtime directly read all 57 lines of the workspace-root `requirements.txt`, emitted a non-empty selected-model dependency summary, and persisted it under session `af79a149-7c84-443e-9dc1-0afc1cc59231` / message `dcebd859-a843-4af7-a2a4-d020fa22d5e9`.

Observed terminal metrics: `response_origin=selected_model`, `presenting_model=qwen3.5:9b`, `fallback_used=false`, `fallback_scope=none`, `synthesis_error=null`. No web-search widening or whole-query Legacy fallback occurred.

Focused verification: 47 provider/admission/supervisor/stream/persistence checks passed; 39 unrelated cases were deselected.
