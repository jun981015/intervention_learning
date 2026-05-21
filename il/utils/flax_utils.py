import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    grad_clip_norm: Any = nonpytree_field(default=None)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a train state and initialize optimizer state if an optimizer exists."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Return a callable view of one named module inside `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, update_scale=1.0, **kwargs):
        """Apply optimizer updates, increment step, and return a new train state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        updates = jax.tree_util.tree_map(lambda x: x * update_scale, updates)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn, update_scale=1.0):
        """Differentiate a loss function, apply gradients, and return metrics.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        def tree_grad_stats(tree):
            """Compute max, min, and aggregate norm for a pytree of gradients."""
            grad_max = jax.tree_util.tree_map(jnp.max, tree)
            grad_min = jax.tree_util.tree_map(jnp.min, tree)
            grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, tree)

            grad_max_flat = jnp.concatenate(
                [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0
            )
            grad_min_flat = jnp.concatenate(
                [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0
            )
            grad_norm_flat = jnp.concatenate(
                [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0
            )

            return {
                'grad/max': jnp.max(grad_max_flat),
                'grad/min': jnp.min(grad_min_flat),
                'grad/norm': jnp.linalg.norm(grad_norm_flat, ord=1),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            grads, _ = optax.clip_by_global_norm(self.grad_clip_norm).update(grads, None, self.params)

        info.update(tree_grad_stats(grads))

        # Log grouped gradient stats so joint updates can be interpreted by module.
        # Top-level parameter keys are stable `ModuleDict` entries such as
        # `modules_critic`, `modules_actor_bc_flow`, and `modules_alpha`.
        if hasattr(grads, 'keys'):
            grad_groups = {
                'critic': {},
                'actor': {},
                'alpha': {},
                'value': {},
            }
            for module_key, module_grads in grads.items():
                module_key = str(module_key)
                if module_key == 'modules_critic':
                    grad_groups['critic'][module_key] = module_grads
                elif module_key.startswith('modules_actor'):
                    grad_groups['actor'][module_key] = module_grads
                elif module_key == 'modules_alpha':
                    grad_groups['alpha'][module_key] = module_grads
                elif module_key == 'modules_value':
                    grad_groups['value'][module_key] = module_grads

            for group_name, group_grads in grad_groups.items():
                if not group_grads:
                    continue
                group_info = tree_grad_stats(group_grads)
                info.update({f'{group_name}/{k}': v for k, v in group_info.items()})

        return self.apply_gradients(grads=grads, update_scale=update_scale), info


def save_agent(agent, save_dir, epoch):
    """Serialize an agent as `params_<epoch>.pkl`.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def _merge_state_dict_keep_target_defaults(target_state, loaded_state):
    """Merge a checkpoint state into the current target state.

    Loaded values overwrite matching keys, while keys missing from the checkpoint
    keep their freshly initialized target values. This allows restoring older
    checkpoints after adding optional modules.
    """
    if isinstance(target_state, dict) and isinstance(loaded_state, dict):
        merged = {}
        for key, target_value in target_state.items():
            if key in loaded_state:
                merged[key] = _merge_state_dict_keep_target_defaults(
                    target_value, loaded_state[key]
                )
            else:
                merged[key] = target_value
        return merged
    return loaded_state


def restore_agent_with_file(agent, file_path):
    """Restore an agent from an explicit checkpoint file path."""
    assert os.path.exists(file_path), f'File {file_path} does not exist'
    with open(file_path, 'rb') as f:
        load_dict = pickle.load(f)

    target_state = flax.serialization.to_state_dict(agent)
    merged_state = _merge_state_dict_keep_target_defaults(target_state, load_dict['agent'])
    agent = flax.serialization.from_state_dict(agent, merged_state)

    print(f'Restored from {file_path}')

    return agent

def restore_agent(agent, restore_path, restore_epoch):
    """Restore an agent from a run directory glob and checkpoint epoch.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    target_state = flax.serialization.to_state_dict(agent)
    merged_state = _merge_state_dict_keep_target_defaults(target_state, load_dict['agent'])
    agent = flax.serialization.from_state_dict(agent, merged_state)

    print(f'Restored from {restore_path}')

    return agent
