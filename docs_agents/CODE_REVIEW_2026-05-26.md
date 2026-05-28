# Code Review Follow-up - 2026-05-26

This is the agent-facing English status for the Korean canonical review in
`docs/CODE_REVIEW_2026-05-26.md`. Keep the Korean file as the full review log.

## Current Status - 2026-05-28

| id | status | current code state |
| --- | --- | --- |
| P0-1 ExpertQGapGate episode reset | fixed | `ControllerGate` is `@runtime_checkable`, `TrainContext.gate` / `build_gate()` use the Protocol type, built gates are validated, and both `ExpertQGapGate` and `RandomGate` expose `reset_episode()`. |
| P0-2 residual_scale train/eval mismatch | fixed | `resolve_residual_scale(context)` is shared by train rollout and context evaluation. |
| P0-3 buffer-underfilled exception string match | intentionally kept | The user rejected a custom `BufferTooSmall` exception. The train loop still skips underfilled replay by matching `"smaller than sequence_length"` in `ValueError`. |
| P1-1 residual rollout hardcoding | partially fixed | Residual learner proposal construction is shared by residual-only and residual+gate rollout. `rollout.execute == "residual"` and implicit metadata keys still remain. |
| P1-2 gate Protocol is expert-agent centric | partially fixed | The gate Protocol is now runtime-checked, but `decide(..., expert_agent=..., action_dim=...)` remains expert-agent centric. Add `GateContext` only when a new gate needs broader context. |
| P1-3 hasattr dispatch | open | Critic-only updates, Q evaluation, and policy sampling still use `hasattr` dispatch. |
| P1-4 residual kind sets | open | `{"residual_rlpd", "residual_td3"}` remains duplicated in actor builder logic. Use a registry/spec before adding another residual family. |
| P1-5 implicit `PolicyOutput.info` schema | open | Keys such as `full_action_chunk`, `base_action`, `residual_action`, and `raw_residual_action` are still implicit contracts. |
| P2-1 critic-loss normalization | fixed | RLPD, residual RLPD, residual TD3, and BC critic losses divide by valid sample counts instead of full batch size. |

## Next Small Coding Unit

Before adding a new residual family or gate family, choose one small cleanup:

1. Add an actor/agent registry spec so residual agent metadata is not repeated as literal kind sets in `il/builders/actors.py`.
2. Promote common `PolicyOutput.info` keys to typed helper output or a small residual proposal dataclass.
3. Run a real-env residual+intervention-gate build-only and short rollout smoke, starting with random gate before `expert_q_gap`.

Do not add `GateContext` only for cleanup; add it when a new gate needs more than the current Protocol signature.

## When To Introduce GateContext

Do not add `GateContext` only for cleanup. Add it when the next gate family
needs one of these inputs:

- learner critic or learner uncertainty
- base actor / residual metadata
- episode or action history
- env info, reward, termination, or success signals
- gate-owned trainable state

At that point, replace the long `decide(..., expert_agent=..., action_dim=...)`
signature with a single context object.
