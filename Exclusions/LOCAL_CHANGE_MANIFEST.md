# Odysseus Local Change Manifest

Last updated: 2026-07-05 16:56:41

Purpose:
This file is the canonical local merge map for this working copy.
It separates upstream tracked project files, local tracked edits, active untracked runtime files, and excluded scaffolding.

Rule:
- Keep this file directly inside Exclusions/.
- Update it after every major checkpoint or feature patch.
- Do not place this manifest inside runtime-searchable source.
- Do not use this file as runtime evidence.
- Use this file before pulling, rebasing, merging, or checkpointing.

---

## Current Branch / Root

```text
D:/AI/odysseus-upstream-test
Branch: 
```

## Tracked Modified Files

```text
 M routes/chat_helpers.py
 M routes/chat_routes.py
 M src/agent_loop.py
```

Expected active tracked local edits at this stage:
- routes/chat_helpers.py
- routes/chat_routes.py
- src/agent_loop.py
- src/agent_runtime/general_tools.py, if Exclusions ignore patch modified it

## Active Untracked Runtime Files

These are local integrated runtime files and should stay in the project source tree.

```text
?? src/agent_runtime/__init__.py
?? src/agent_runtime/action_search.py
?? src/agent_runtime/claim_support.py
?? src/agent_runtime/contract_bridge.py
?? src/agent_runtime/contract_models.py
?? src/agent_runtime/contract_runner.py
?? src/agent_runtime/contract_runtime.py
?? src/agent_runtime/contract_verifier.py
?? src/agent_runtime/daily_driver_preface.py
?? src/agent_runtime/evals/contract_eval_cases.json
?? src/agent_runtime/evals/general_eval_cases.json
?? src/agent_runtime/evals/router_eval_cases.json
?? src/agent_runtime/evals/run_contract_evals.py
?? src/agent_runtime/evals/run_daily_driver_preface_evals.py
?? src/agent_runtime/evals/run_daily_driver_preface_scope_matrix_evals.py
?? src/agent_runtime/evals/run_general_evals.py
?? src/agent_runtime/evals/run_knowledge_chat_bridge_evals.py
?? src/agent_runtime/evals/run_knowledge_graph_router_evals.py
?? src/agent_runtime/evals/run_knowledge_semantic_chat_bridge_evals.py
?? src/agent_runtime/evals/run_knowledge_semantic_rerank_evals.py
?? src/agent_runtime/evals/run_knowledge_vault_index_evals.py
?? src/agent_runtime/evals/run_knowledge_vault_semantic_evals.py
?? src/agent_runtime/evals/run_obsidian_read_evals.py
?? src/agent_runtime/evals/run_router_evals.py
?? src/agent_runtime/evals/run_runtime_trace_evals.py
?? src/agent_runtime/events.py
?? src/agent_runtime/evidence.py
?? src/agent_runtime/evidence_ledger.py
?? src/agent_runtime/exact_terms.py
?? src/agent_runtime/finalizer.py
?? src/agent_runtime/general_bridge.py
?? src/agent_runtime/general_graph.py
?? src/agent_runtime/general_runner.py
?? src/agent_runtime/general_state.py
?? src/agent_runtime/general_tools.py
?? src/agent_runtime/graph.py
?? src/agent_runtime/knowledge_chat_bridge.py
?? src/agent_runtime/knowledge_graph_router.py
?? src/agent_runtime/knowledge_semantic_chat_bridge.py
?? src/agent_runtime/knowledge_vault_index.py
?? src/agent_runtime/knowledge_vault_semantic.py
?? src/agent_runtime/legacy_runner.py
?? src/agent_runtime/legacy_sse_adapter.py
?? src/agent_runtime/obsidian_read_adapter.py
?? src/agent_runtime/obsidian_read_graph.py
?? src/agent_runtime/obsidian_read_runner.py
?? src/agent_runtime/runner.py
?? src/agent_runtime/runtime_router.py
?? src/agent_runtime/source_authority.py
?? src/agent_runtime/sse_adapter.py
?? src/agent_runtime/state.py
?? src/agent_runtime/streaming.py
?? src/agent_runtime/task_contract.py
?? src/agent_runtime/tool_adapters.py
?? src/agent_runtime/tool_policy.py
?? src/agent_runtime/trace_spans.py
```

Classification:
- src/agent_runtime/** = active LangGraph / deterministic runtime system
- src/agent_runtime/evals/** = active runtime regression board

## Excluded Local Scaffolding

These are local patch/debug/report/history artifacts. They should stay inside Exclusions/.

```text
?? Exclusions/Backups/backup-cleanup-manifest-20260705-151910.txt
?? Exclusions/Backups/routes/chat_routes.py.before_v016f3_inverted_tool_intent_gate_20260705-143448
?? Exclusions/Backups/routes/chat_routes.py.before_v016f3b_inverted_tool_intent_gate_20260705-143627
?? Exclusions/Backups/src/agent_loop.py.before_force_local_native_after_decision_20260630_034543
?? Exclusions/Backups/src/agent_loop.py.before_local_native_tools_20260630_034022
?? Exclusions/Backups/tests/test_backup_cli_security.py
?? Exclusions/Backups/tests/test_backup_import_cross_user_dedup.py
?? Exclusions/Backups/tests/test_backup_import_skills.py
?? Exclusions/Backups/tests/test_backup_import_skills_dedup.py
?? Exclusions/Backups/v016f3c/routes/chat_routes.py.before_v016f3c_20260705-151910
?? Exclusions/Backups/v016f3d/routes/chat_routes.py.before_v016f3d_20260705-152300
?? Exclusions/Backups/v016f3debug/routes/chat_routes.py.before_v016f3debug_20260705-152944
?? Exclusions/Backups/v016f3f/routes/chat_routes.py.before_v016f3f_20260705-153813
?? Exclusions/Backups/v016f3g/src/agent_runtime/general_tools.py.before_v016f3g_20260705-160836
?? Exclusions/Backups/v016f3g2/src/agent_runtime/general_tools.py.before_v016f3g2_20260705-161002
?? Exclusions/Backups/v016f3h-ignore-exclusions/src/agent_runtime/general_tools.py.before_v016f3h_ignore_exclusions_20260705-163629
?? Exclusions/LooseRootArtifacts/__force_local_native_after_decision.py
?? Exclusions/LooseRootArtifacts/add_v016f1a_preface_scope_matrix_eval.py
?? Exclusions/LooseRootArtifacts/add_v016f2_semantic_rerank_eval.py
?? Exclusions/LooseRootArtifacts/atlas_inventory_v016a.json
?? Exclusions/LooseRootArtifacts/verify_v016d_save_patch.py
?? Exclusions/LooseRootArtifacts/verify_v016d_session_variable_save.py
?? Exclusions/Patches/InspectScripts/inspect_atlas_inventory_v016a.py
?? Exclusions/Patches/InspectScripts/inspect_embedding_availability_v016e.py
?? Exclusions/Patches/InspectScripts/inspect_general_runtime_search_boundaries.py
?? Exclusions/Patches/InspectScripts/inspect_knowledge_vault_index_v016e1.py
?? Exclusions/Patches/InspectScripts/inspect_knowledge_vault_semantic_v016e2.py
?? Exclusions/Patches/InspectScripts/inspect_langgraph_integration.py
?? Exclusions/Patches/InspectScripts/inspect_latest_db_tool_events.py
?? Exclusions/Patches/InspectScripts/inspect_obsidian_authorized_v016a2.py
?? Exclusions/Patches/InspectScripts/inspect_obsidian_local_rest_v016a2.py
?? Exclusions/Patches/InspectScripts/inspect_obsidian_safe_read_endpoints_v016a3.py
?? Exclusions/Patches/InspectScripts/inspect_preface_model_names_v016f1a.py
?? Exclusions/Patches/InspectScripts/inspect_project_source_index_and_exclusion_plan.py
?? Exclusions/Patches/InspectScripts/inspect_real_message_storage.py
?? Exclusions/Patches/InspectScripts/inspect_runtime_v081.py
?? Exclusions/Patches/InspectScripts/inspect_saved_tool_events.py
?? Exclusions/Patches/InspectScripts/inspect_v013d_contract_vs_general_sse.py
?? Exclusions/Patches/InspectScripts/inspect_v013d_general_bridge_sse.py
?? Exclusions/Patches/InspectScripts/inspect_v015g_persistence_path.py
?? Exclusions/Patches/InspectScripts/inspect_v016d_bridge_frames.py
?? Exclusions/Patches/InspectScripts/inspect_v016d_chat_wiring.py
?? Exclusions/Patches/InspectScripts/inspect_v016d_raw_chat_stream.py
?? Exclusions/Patches/InspectScripts/inspect_v016d_saved_knowledge_response.py
?? Exclusions/Patches/InspectScripts/inspect_v016d_ui_save_issue.py
?? Exclusions/Patches/InspectScripts/inspect_v016e2_chroma_embedding_integration.py
?? Exclusions/Patches/InspectScripts/inspect_v016f1a_preface_failure_shape.py
?? Exclusions/Patches/InspectScripts/inspect_v016f1b_active_note_failure.py
?? Exclusions/Patches/InspectScripts/inspect_v016f1b_simple_chat_route_points.py
?? Exclusions/Patches/InspectScripts/inspect_v016f1c_semantic_bridge_and_ranking.py
?? Exclusions/Patches/InspectScripts/inspect_v016f3_live_routing_failure.py
?? Exclusions/Patches/InspectScripts/inspect_v016f_routing_and_chat_paths.py
?? Exclusions/Patches/InspectScripts/inspect_v08_seam.py
?? Exclusions/Patches/PatchScripts/__patch_local_native_tools.py
?? Exclusions/Patches/PatchScripts/patch_eval_harness_paths.py
?? Exclusions/Patches/PatchScripts/patch_eval_harness_raw_debug.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v010.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0101.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0102.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0111_match_ranking.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0112_contract.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0112a_streaming_emit.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0113_cleanup.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v0113_evidence_snippets.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v011_contract.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v012_contract_bridge.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v02.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v03.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v04.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v05.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v051.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v052.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v06.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v07.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v08.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v081.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v082.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v09.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v091.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v092.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v093.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v094.py
?? Exclusions/Patches/PatchScripts/patch_runtime_v095.py
?? Exclusions/Patches/PatchScripts/patch_utf8_runtime_eval.py
?? Exclusions/Patches/PatchScripts/patch_v012a_bridge_newlines.py
?? Exclusions/Patches/PatchScripts/patch_v012b_agent_branch_bridge.py
?? Exclusions/Patches/PatchScripts/patch_v012b_contract_decision_log.py
?? Exclusions/Patches/PatchScripts/patch_v012c_missing_contract_import.py
?? Exclusions/Patches/PatchScripts/patch_v012c_save_kwargs.py
?? Exclusions/Patches/PatchScripts/patch_v012d_cleanup_labels_logs.py
?? Exclusions/Patches/PatchScripts/patch_v012d_skip_contract_post_tasks.py
?? Exclusions/Patches/PatchScripts/patch_v013a_frontend_backend_ranking.py
?? Exclusions/Patches/PatchScripts/patch_v013a_general_runtime_scaffold.py
?? Exclusions/Patches/PatchScripts/patch_v013a_skip_agent_runtime.py
?? Exclusions/Patches/PatchScripts/patch_v013a_source_hygiene.py
?? Exclusions/Patches/PatchScripts/patch_v013b_general_route_bridge.py
?? Exclusions/Patches/PatchScripts/patch_v013c_add_file_lookup_eval.py
?? Exclusions/Patches/PatchScripts/patch_v013c_file_lookup_behavior.py
?? Exclusions/Patches/PatchScripts/patch_v013c_general_related_files_support.py
?? Exclusions/Patches/PatchScripts/patch_v013d_router_stage1.py
?? Exclusions/Patches/PatchScripts/patch_v013d_router_stage2_chat_routes.py
?? Exclusions/Patches/PatchScripts/patch_v013e_general_agent_step_frames.py
?? Exclusions/Patches/PatchScripts/patch_v013f_general_metrics_state.py
?? Exclusions/Patches/PatchScripts/patch_v014_topic_file_lookup.py
?? Exclusions/Patches/PatchScripts/patch_v014b_architecture_summarizer.py
?? Exclusions/Patches/PatchScripts/patch_v014b_architecture_summarizer_direct.py
?? Exclusions/Patches/PatchScripts/patch_v014c_exact_router_shape.py
?? Exclusions/Patches/PatchScripts/patch_v014c_router_architecture_summary.py
?? Exclusions/Patches/PatchScripts/patch_v014c_router_architecture_summary_direct.py
?? Exclusions/Patches/PatchScripts/patch_v015a_runtime_trace.py
?? Exclusions/Patches/PatchScripts/patch_v015b_ui_runtime_trace.py
?? Exclusions/Patches/PatchScripts/patch_v015b_ui_runtime_trace_exact.py
?? Exclusions/Patches/PatchScripts/patch_v015c_runtime_trace_tool_bubble.py
?? Exclusions/Patches/PatchScripts/patch_v015d_inspected_runtime_trace.py
?? Exclusions/Patches/PatchScripts/patch_v015d_runtime_trace_top_level.py
?? Exclusions/Patches/PatchScripts/patch_v015e_runtime_trace_sse_separators.py
?? Exclusions/Patches/PatchScripts/patch_v015f_legacy_decline_trace.py
?? Exclusions/Patches/PatchScripts/patch_v015g2_history_persistence.py
?? Exclusions/Patches/PatchScripts/patch_v015g3_direct_tool_events_save.py
?? Exclusions/Patches/PatchScripts/patch_v015g3b_direct_tool_events_save.py
?? Exclusions/Patches/PatchScripts/patch_v015g4_round_texts.py
?? Exclusions/Patches/PatchScripts/patch_v015g_persist_runtime_tool_events.py
?? Exclusions/Patches/PatchScripts/patch_v015h_cleanup_dead_runtime_persist_helper.py
?? Exclusions/Patches/PatchScripts/patch_v015i_failure_quality.py
?? Exclusions/Patches/PatchScripts/patch_v016b_obsidian_read_graph.py
?? Exclusions/Patches/PatchScripts/patch_v016c_knowledge_graph_router_hitl.py
?? Exclusions/Patches/PatchScripts/patch_v016d1_knowledge_chat_bridge.py
?? Exclusions/Patches/PatchScripts/patch_v016d2_wire_knowledge_chat.py
?? Exclusions/Patches/PatchScripts/patch_v016e1_structural_vault_index.py
?? Exclusions/Patches/PatchScripts/patch_v016e2_semantic_vault_index.py
?? Exclusions/Patches/PatchScripts/patch_v016e3_semantic_chat_bridge_module.py
?? Exclusions/Patches/PatchScripts/patch_v016f1a_daily_driver_preface_helper.py
?? Exclusions/Patches/PatchScripts/patch_v016f1a_eval_accept_native_ollama.py
?? Exclusions/Patches/PatchScripts/patch_v016f1a_native_ollama_preface.py
?? Exclusions/Patches/PatchScripts/patch_v016f1b_actual_chat_branch_fastpath.py
?? Exclusions/Patches/PatchScripts/patch_v016f1b_scope_eval_destructive_hints.py
?? Exclusions/Patches/PatchScripts/patch_v016f1b_scope_eval_false_positives.py
?? Exclusions/Patches/PatchScripts/patch_v016f1b_scope_eval_simple_chat_bypass.py
?? Exclusions/Patches/PatchScripts/patch_v016f1b_simple_chat_fastpath.py
?? Exclusions/Patches/PatchScripts/patch_v016f1c_semantic_preface_and_rerank.py
?? Exclusions/Patches/PatchScripts/patch_v016f1c_semantic_preface_live.py
?? Exclusions/Patches/PatchScripts/patch_v016f2_scope_eval_word_boundary.py
?? Exclusions/Patches/PatchScripts/patch_v016f3_debug_live_gate_values.py
?? Exclusions/Patches/PatchScripts/patch_v016f3_inverted_tool_intent_gate.py
?? Exclusions/Patches/PatchScripts/patch_v016f3b_inverted_tool_intent_gate_marker_safe.py
?? Exclusions/Patches/PatchScripts/patch_v016f3c_fix_early_research_gate_and_backup_hygiene.py
?? Exclusions/Patches/PatchScripts/patch_v016f3d_normalize_gate_form_booleans.py
?? Exclusions/Patches/PatchScripts/patch_v016f3f_tool_availability_not_tool_intent.py
?? Exclusions/Patches/PatchScripts/patch_v016f3g_general_runtime_search_exclusions.py
?? Exclusions/Patches/PatchScripts/patch_v016f3h_ignore_exclusions.py
?? Exclusions/Patches/ProbeScripts/probe_obsidian_rest_live.py
?? Exclusions/Patches/ProbeScripts/probe_v016f1a_ollama_chat_completions.py
?? Exclusions/Patches/ProbeScripts/probe_v016f1a_ollama_native_think_false.py
?? Exclusions/Patches/ProbeScripts/probe_v016f1a_preface_token_budget.py
?? Exclusions/Patches/RepairScripts/fix_runtime_v08_asyncio.py
?? Exclusions/Patches/RepairScripts/fix_v011_regex.py
?? Exclusions/Patches/RepairScripts/repair_v015_runtime_helper_safe.py
?? Exclusions/Patches/RepairScripts/repair_v015b_runtime_trace.py
?? Exclusions/Patches/RepairScripts/repair_v015e_separators_surgical.py
?? Exclusions/Patches/RepairScripts/repair_v015g_helper_block.py
?? Exclusions/Patches/RepairScripts/repair_v015i_quote_syntax.py
?? Exclusions/Patches/RepairScripts/repair_v016c_knowledge_router_edges.py
?? Exclusions/Patches/RepairScripts/repair_v016d1_bridge_done_eval.py
?? Exclusions/Patches/RepairScripts/repair_v016d_final_answer_lifecycle.py
?? Exclusions/Patches/RepairScripts/repair_v016d_live_delta_shape.py
?? Exclusions/Patches/RepairScripts/repair_v016d_live_render_lifecycle.py
?? Exclusions/Patches/RepairScripts/repair_v016d_save_signature.py
?? Exclusions/Patches/RepairScripts/repair_v016d_session_variable_save.py
?? Exclusions/Patches/RepairScripts/repair_v016e3_router_import.py
?? Exclusions/Patches/RepairScripts/repair_v016f1a_fastpath_regex.py
?? Exclusions/Patches/RepairScripts/repair_v016f1a_honest_preface_fallback.py
?? Exclusions/Patches/RepairScripts/repair_v016f1a_no_hardcoded_preface_fallback.py
?? Exclusions/Patches/RepairScripts/repair_v016f1a_preface_eval_honest_fallback.py
?? Exclusions/Patches/RepairScripts/repair_v016f1a_qwen_reasoning_preface.py
?? Exclusions/Patches/RuntimeNotes/agent_scope_guard_inspect.txt
?? Exclusions/Patches/RuntimeNotes/langgraph_integration_inspect.txt
?? Exclusions/Patches/RuntimeNotes/langgraph_v081_runtime_inspect.txt
?? Exclusions/Patches/RuntimeNotes/langgraph_v08_seam_inspect.txt
?? Exclusions/Patches/RuntimeNotes/native_tool_detection_compare_20260630_034022.txt
?? Exclusions/Patches/RuntimeNotes/verify_native_patch_runtime.txt
?? Exclusions/Patches/SmokeScripts/smoke_v016c_knowledge_router.py
?? Exclusions/Patches/WireScripts/wire_v016e3_semantic_chat_bridge.py
?? Exclusions/Patches/force_patch_v0111.py
?? Exclusions/Patches/install_v0111_ranking_logic.py
?? Exclusions/Patches/install_v0111_ranking_logic_fixed.py
?? Exclusions/Patches/patch-cleanup-manifest-20260705-142739.txt
?? Exclusions/Patches/print_atlas_risk_samples_v016a.py
?? Exclusions/Patches/rewrite_v016f1a_daily_driver_preface_native.py
?? Exclusions/Patches/rewrite_v016f1a_preface_scope_matrix_eval.py
?? Exclusions/Patches/tune_v016f1a_preface_budget_qwen3_8b.py
?? Exclusions/inspection_reports/f3-live-routing-failure-inspection-20260705-152534.txt
?? Exclusions/inspection_reports/f3-lock-eval-board-authenticated-20260705-160131.txt
?? Exclusions/inspection_reports/f3-post-exclusions-full-eval-board-20260705-164531.txt
?? Exclusions/inspection_reports/f3-tool-intent-and-tool-selection-focused-20260705-142455.txt
?? Exclusions/inspection_reports/general-runtime-search-boundaries-20260705-160740.txt
?? Exclusions/inspection_reports/git-source-boundary-and-exclusions-plan-20260705-161923.txt
?? Exclusions/inspection_reports/legacy-agent-loop-and-migration-inspection-20260705-141131.txt
?? Exclusions/inspection_reports/local-change-manifest-before-exclusions-20260705-163052.txt
?? Exclusions/inspection_reports/project-source-index-and-exclusion-plan-20260705-161648.json
?? Exclusions/inspection_reports/project-source-index-and-exclusion-plan-20260705-161648.txt
```

Classification:
- Exclusions/Patches/** = patch/inspect/repair/probe/smoke scripts
- Exclusions/Backups/** = local safety backups
- Exclusions/inspection_reports/** = eval and inspection reports
- Exclusions/LooseRootArtifacts/** = loose helper/eval/debug artifacts moved out of root
- Exclusions/LOCAL_CHANGE_MANIFEST.md = this canonical merge map

## Tracked Diff Summary

```text
 routes/chat_helpers.py |   7 +
 routes/chat_routes.py  | 509 +++++++++++++++++++++++++++++++++++++++++++++----
 src/agent_loop.py      |  11 ++
 3 files changed, 486 insertions(+), 41 deletions(-)
```

## Important Local Blocks / Merge Notes

### routes/chat_helpers.py

Purpose:
- Persist reconstructed assistant text alongside tool events.
- Ensures saved agent/tool-thread responses have both tool bubbles and assistant text bubbles.

Known local block:
- save_assistant_response(...): adds metadata.round_texts when tool events exist but no round text was saved.

### routes/chat_routes.py

Purpose:
- Wires runtime routing, contract/general deterministic runtimes, knowledge runtime, semantic knowledge runtime, runtime trace persistence, and F3 inverted tool-intent gate.

Known local blocks:
- DAILY_DRIVER_INVERTED_TOOL_GATE_V016F3_START/END
- RUNTIME_ROUTER_V013D_DECISION_START/END
- CONTRACT_RUNTIME_V012D_AGENT_BRANCH_START/END
- KNOWLEDGE_RUNTIME_V016D_CHAT_BRANCH_START/END
- GENERAL_RUNTIME_V013_AGENT_BRANCH_START/END
- RUNTIME_TRACE_V015F_LEGACY_DECLINE_START/END

Needs cleanup before final checkpoint:
- Some comments/docstrings contain corrupted separator text like +際釜G킨.
- Treat this as comment hygiene, not runtime behavior.

### src/agent_loop.py

Purpose:
- Keeps local OpenAI-compatible / Ollama-compatible endpoints using native tool schema delivery.
- Prevents local models from falling back to fake fenced pseudo-tool calls.

Known local blocks:
- _LOCAL_NATIVE_TOOLS_PATCH_20260630
- _FORCE_LOCAL_NATIVE_TOOLS_AFTER_DECISION_20260630

### src/agent_runtime/**

Purpose:
- Active local deterministic/LangGraph-style runtime system.
- Includes contract runtime, general runtime, router, knowledge runtime, Obsidian read graph, semantic vault index/search, preface helper, eval board, SSE adapters, trace support, and policy/safety helpers.

Status:
- Active untracked integrated subsystem.
- Do not move into Exclusions.

### Exclusions/**

Purpose:
- Local-only scaffolding and history.
- Must be ignored by General runtime search.
- Should not be used as product source evidence unless explicitly requested.

## Latest Eval Board Summary

Post-exclusions full eval board:
- Router evals: 19/19
- Contract evals: 10/10
- General evals: 18/18
- Runtime trace evals: 5/5
- Knowledge graph router evals: 20/20
- Obsidian read evals: 11/11
- Knowledge chat bridge evals: 5/5
- Knowledge vault index evals: 5/5
- Knowledge vault semantic evals: 6/6
- Knowledge semantic chat bridge evals: 10/10
- Knowledge semantic rerank evals: 6/6
- Daily driver preface evals: 9/9

Checkpoint status:
- F3 behavior is functionally green after Exclusions cleanup.
- Remaining pre-checkpoint hygiene: clean corrupted comment/docstring text in routes/chat_routes.py, then regenerate this manifest.
