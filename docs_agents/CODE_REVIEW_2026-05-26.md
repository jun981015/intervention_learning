# Code Review Follow-up - 2026-05-26

This is the agent-facing English status for the Korean canonical review in
`docs/CODE_REVIEW_2026-05-26.md`. Keep the Korean file as the full review log.

## Current Status - 2026-05-27

| id | status | current code state |
| --- | --- | --- |
| P0-1 ExpertQGapGate episode reset | partially fixed | `ExpertQGapGate.reset_episode()` exists, and train rollout calls `reset_rollout_state(..., reset_gate=True)` at episode boundaries. Remaining bug: `RandomGate` still has no `reset_episode()`, so random-gate runtime can fail. |
| P0-2 residual_scale train/eval mismatch | fixed | `resolve_residual_scale(context)` is shared by train rollout and eval. |
| P0-3 buffer-underfilled exception string match | open | Train loop still suppresses underfilled replay by matching `"smaller than sequence_length"` inside `ValueError`. Add a custom exception. |
| P1-1 residual rollout hardcoding | open | `rollout.execute == "residual"` branches and implicit `PolicyOutput.info` keys remain in train/eval/rollout. |
| P1-2 gate Protocol is expert-agent centric | open | `ControllerGate.decide()` still takes `expert_agent`; `ExpertQGapGate` still sniffs `evaluate_q`/`q_values`. |
| P1-3 hasattr dispatch | open | Critic-only updates, Q evaluation, and policy sampling still use `hasattr` dispatch. |
| P1-4 residual kind sets | open | `{"residual_rlpd", "residual_td3"}` remains duplicated in actor builder logic. |
| P1-5 implicit `PolicyOutput.info` schema | open | Keys such as `full_action_chunk`, `base_action`, and `residual_action` are still implicit contracts. |
| P2-1 critic-loss normalization | open | Residual critic losses still average over full batch size instead of valid samples only. |

## Next Small Coding Unit

Before adding a new gate family, clean up the gate contract:

1. Keep `ControllerGate` as a Protocol, but mark it `@runtime_checkable`.
2. Type `TrainContext.gate` and `build_gate()` as `ControllerGate | None`.
3. Validate built gates with `isinstance(gate, ControllerGate)`.
4. Add no-op `reset_episode()` to `RandomGate`.

This is less invasive than converting gates to an ABC and preserves the current
duck-typed style of the repo.

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
