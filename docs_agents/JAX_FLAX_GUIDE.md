# JAX/Flax Reading Guide for This Project

This document is for coding agents working in `intervention_learning`. The user knows PyTorch and is learning how the JAX/Flax code is structured. Keep future changes consistent with the patterns described here unless explicitly asked otherwise.

## Core Mental Model

In PyTorch, modules own parameters and autograd tracks operations on those parameters:

```python
q = model.critic(obs, act)
loss = ((q - target) ** 2).mean()
loss.backward()
optimizer.step()
```

In JAX/Flax, parameters are explicit function inputs:

```python
def loss_fn(params):
    q = model.apply({"params": params}, obs, act)
    return ((q - target) ** 2).mean()

grads = jax.grad(loss_fn)(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

The key rule:

```text
JAX differentiates with respect to the inputs passed to `jax.grad`.
```

## Main Files

- `il/utils/flax_utils.py`: `ModuleDict`, `TrainState`, save/load helpers.
- `il/algo/rl/rlpd.py`: SAC/RLPD agent.
- `il/algo/bc/flow.py`: flow-matching BC agent used by the DAgger baseline.
- `il/policies/rlpd.py`: policy wrapper around RLPD checkpoints.
- `il/policies/bc_flow.py`: policy wrapper around BCFlow checkpoints.

## PyTrees

JAX treats nested dict/list/tuple/dataclass structures as PyTrees. Agent parameters are stored as nested parameter trees.

RLPD params look roughly like:

```python
{
    "modules_critic": ...,
    "modules_target_critic": ...,
    "modules_actor": ...,
    "modules_alpha": ...,
}
```

BCFlow params look roughly like:

```python
{
    "modules_actor_bc_flow": ...,
}
```

`jax.grad(loss_fn)(params)` returns a gradient PyTree with the same structure for all parameter leaves that the loss depends on.

## `nn.Module`

Flax `nn.Module` defines computation structure. Parameters are not mutated inside the module like a PyTorch module. They are initialized separately and passed to `apply`.

This project uses `ModuleDict` in `il/utils/flax_utils.py` to hold multiple named modules:

```python
class ModuleDict(nn.Module):
    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            ...
        return self.modules[name](*args, **kwargs)
```

Named modules are selected through:

```python
self.network.select("actor")(obs)
self.network.select("critic")(obs, act)
self.network.select("target_critic")(next_obs, next_act)
```

`select` only chooses the module. It does not decide gradient flow.

## `flax.struct.PyTreeNode`

Agents are immutable PyTree containers:

```python
class ACRLPDAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
```

Use `.replace(...)` to produce updated agents:

```python
return self.replace(network=new_network, rng=new_rng), info
```

Avoid in-place mutation patterns inside JIT-compiled update paths.

## `TrainState`

`TrainState` is the project wrapper around model structure, parameters, optimizer, optimizer state, and step:

```python
class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    grad_clip_norm: Any = nonpytree_field(default=None)
```

Its role is similar to `nn.Module + optimizer + optimizer_state + step`, except parameters live in `params`, not inside `model_def`.

## Forward Calls and Gradient Flow

`TrainState.__call__` wraps Flax `apply`:

```python
def __call__(self, *args, params=None, method=None, **kwargs):
    if params is None:
        params = self.params
    variables = {"params": params}
    return self.apply_fn(variables, *args, method=method_name, **kwargs)
```

Two calls with the same values can have different gradient behavior:

```python
# This module's parameters are part of the differentiated input.
self.network.select("critic")(obs, act, params=grad_params)

# This uses stored params, not the `jax.grad` input.
# The module's parameters are effectively frozen for this loss.
self.network.select("critic")(obs, act)
```

`grad_params` is not a boolean flag. It is the parameter PyTree passed by `jax.grad(loss_fn)(self.params)`.

## RLPD Update

RLPD update path in `il/algo/rl/rlpd.py`:

```python
def loss_fn(grad_params):
    return self.total_loss(batch, grad_params, rng=rng)

new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
self.target_update(new_network, "critic")
```

`TrainState.apply_loss_fn` computes gradients over the full parameter tree:

```python
grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
```

Then it applies optax updates:

```python
updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
new_params = optax.apply_updates(self.params, updates)
```

## RLPD Critic Loss

Target computation uses stored parameters, so actor and target critic are frozen for the critic target:

```python
next_dist = self.network.select("actor")(next_obs)
next_actions = next_dist.sample(seed=sample_rng)
next_qs = self.network.select("target_critic")(next_obs, next_actions)
```

The online critic is trained by passing `params=grad_params`:

```python
q = self.network.select("critic")(obs, actions, params=grad_params)
critic_loss = ((q - target_q) ** 2).mean()
```

## RLPD Actor Loss

The actor is differentiated:

```python
dist = self.network.select("actor")(obs, params=grad_params)
actions = dist.sample(seed=rng)
log_probs = dist.log_prob(actions)
```

The critic is used as a frozen evaluator:

```python
qs = self.network.select("critic")(obs, actions)
q = jnp.mean(qs, axis=0)
actor_loss = (log_probs * alpha - q).mean()
```

Critic parameters do not receive gradients from actor loss, but `dQ/da` still flows through `actions` into the actor.

## BCFlow Update

BCFlow in `il/algo/bc/flow.py` is simpler:

```python
def loss_fn(grad_params):
    return agent.total_loss(batch, grad_params, rng=rng)

new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
```

The DAgger baseline runs learner rollout, stores expert actions, and trains BCFlow on those expert actions as supervised labels.

## Optimizer Grouping

Current RLPD uses one Adam transform:

```python
network_tx = optax.adam(learning_rate=config["lr"])
```

Actor and critic do not have separate optimizer states in the current implementation. Gradient flow is controlled by whether calls use `params=grad_params`.

If adding discriminator, curiosity, or auxiliary value models, choose explicitly between:

```text
1. Single TrainState with optax.multi_transform optimizer groups.
2. Separate TrainState/Agent objects with explicit update order.
```

For teacher/student, target networks, or adversarial objectives, prefer separate update order over a single joint loss unless the math is intentionally joint.

## Checklist for Future Code Reviews

- Identify the agent class and its `ModuleDict`.
- Check every `self.network.select("...")` call.
- Check whether the call passes `params=grad_params`.
- If it does, that module's parameters can receive gradients.
- If not, that module is used as a frozen evaluator for that loss.
- Remember that gradients can still flow through input tensors such as actions.
- Check whether the optimizer is a single transform or `optax.multi_transform`.
- Check whether target networks are updated only after optimizer steps.
