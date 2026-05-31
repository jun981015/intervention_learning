# Debug Note: Residual TD3 Critic Input Mistake

Date: 2026-05-29

## Summary

I made a serious design interpretation mistake while wiring the ToolHang residual TD3 checkpoint as an expert.

I confused the residual actor input contract with the critic / Q-function input contract.

Correct ResFiT-style contract:

```text
actor:  residual = pi(state, base_action)
env:    executed_action = clip(base_action + residual)
critic: Q(state, executed_action)
```

The residual actor should see `state + base_action`. The critic should see `state` and the already-composed executed action.

## What I Got Wrong

I repeatedly described the residual expert as if the Q function also needed `state + base_action`.

That led to a bad implementation direction in:

```text
il/gating/expert_q_gap.py
```

I added residual-specific logic that augments the critic observation with `base_action` before Q evaluation. That is wrong for the intended `expert_q_gap` abstraction. `ExpertQGapGate` should compare:

```text
Q(state, expert_action) - Q(state, learner_action)
```

It should not know about residual actor internals or concatenate `base_action` into its observation.

## Code Reality Checked

Current repo residual critic action input is the executed action:

```text
batch_actions = batch["actions"][..., 0, :]
q = critic(observations, batch_actions)
```

However, current repo residual critic observation is augmented:

```text
observations = concat(state, base_action)
```

So the current implementation is effectively:

```text
Q([state, base_action], executed_action)
```

This differs from the ResFiT reference implementation.

ResFiT reference behavior:

```text
actor input:  state + base_action
critic input: state, executed_action
```

In the reference, `base_action` is used to form the residual actor input and to compose `executed_action`, but it is not passed as an extra critic observation feature.

## Impact

- The ToolHang residual TD3 actor checkpoint can still be used as an action expert, because residual actors do require `state + base_action`.
- The checkpoint's critic should not be treated as a clean `Q(state, action)` expert unless the training implementation is corrected and retrained.
- The bad residual-specific change in `il/gating/expert_q_gap.py` was conceptually wrong and needed to be reverted.
- Any validation that passed with the augmented residual Q path only proves shape/runtime compatibility, not that the gate matches the intended ResFiT Q semantics.

## Required Fix

1. Revert residual-specific observation augmentation from `il/gating/expert_q_gap.py`.
2. Keep residual expert action sampling support in rollout if the expert actor is residual TD3.
3. Decide whether to change residual critic training to ResFiT-style:

```text
Q(state, executed_action)
```

4. If residual critic input is changed, retrain residual TD3 checkpoints. Existing residual critic weights are not compatible with the corrected critic observation shape.

## Fix Applied

The current code now separates residual actor and critic inputs:

```text
actor input:  state + base_action
critic input: state, executed_action
```

`il/gating/expert_q_gap.py` has also been restored to evaluate the expert critic on the plain observation passed to the gate. Existing residual TD3/RLPD checkpoints trained before this fix still have the old critic shape and need retraining before their critics are used for Q-gap gating or value analysis.


## Lesson

Before modifying a shared abstraction like `ExpertQGapGate`, verify the source algorithm's actor and critic contracts separately. Actor input and critic input are not interchangeable.
