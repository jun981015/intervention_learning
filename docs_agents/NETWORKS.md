# Network Defaults And Options

## Shared MLP

The shared MLP lives in `il/networks/mlp.py`. RLPD, BC MLP, and BC Flow all use
this class.

Defaults:

| Option | Default | Meaning |
| --- | --- | --- |
| `activations` | `nn.relu` | Hidden-layer activation |
| `activate_final` | `False` | Whether to activate the final layer |
| `kernel_init` | `None` | Explicit Dense kernel initializer for all layers |
| `use_layer_norm` | `False` | Whether to use LayerNorm |
| `layer_norm_after_activation` | `False` | Whether LayerNorm runs after the activation |
| `scale_final` | `None` | Final-layer kernel initializer scale override |
| `dropout_rate` | `None` | Hidden-layer dropout |
| `use_pnorm` | `False` | Normalize final features by their L2 norm |
| `sow_intermediate_feature` | `False` | Store penultimate hidden features in `intermediates/feature` |

`scale_final` controls final-layer weight initialization scale. It does not clip
outputs directly, but smaller initialized weights usually make initial outputs
smaller.

`use_pnorm` is not a loss penalty. It changes the forward activations by
projecting features with `x / ||x||_2`.

`sow_intermediate_feature` does not change forward values. It is a Flax
intermediate hook used when calling the module with `mutable=["intermediates"]`.

## Algorithm Defaults

| Use site | File | Main options |
| --- | --- | --- |
| RLPD critic | `il/algo/rl/rlpd.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["layer_norm"]` |
| RLPD actor | `il/algo/rl/rlpd.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["actor_layer_norm"]` |
| BC MLP actor | `il/algo/bc/mlp.py` | `activations=relu`, `activate_final=True`, `use_layer_norm=config["actor_layer_norm"]` |
| BC Flow vector field | `il/networks/flow.py` | `activations=gelu`, `kernel_init=variance_scaling`, `use_layer_norm=config["actor_layer_norm"]`, `layer_norm_after_activation=True`, `sow_intermediate_feature=True` |

## LayerNorm Order

The shared default is `LayerNorm -> activation`.

BC Flow preserves the qc_base flow-network behavior with
`activation -> LayerNorm`, so it sets `layer_norm_after_activation=True`.

## Why There Is No Separate Flow MLP

qc_base had a local MLP inside its flow-network file. The differences are
expressible as options on the shared MLP, so this project uses one shared MLP.

Preserved BC Flow differences:

- `gelu` activation
- `variance_scaling(scale, fan_avg, uniform)` initializer
- LayerNorm after activation
- `intermediates/feature` hook

## MLPResNet Status

`MLPResNetV2` remains in `il/networks/mlp_resnet.py`, but no current agent uses
it. Treat it as reserved for a future residual-MLP ablation.

## Image Observation TODO

Current policy/network implementations train on low-dimensional state inputs
only. Env wrappers and replay buffers can already carry dict observations such
as `pixels_state`, but `BCFlowAgent`, `BCMLPAgent`, and `ACRLPDAgent` do not yet
attach an image encoder.

If a recipe tries to build a current low-dim agent with image observations, the
actor builder should raise `NotImplementedError` instead of silently flattening
images. Future image support should add:

- a CNN encoder builder driven by `EnvSpec.pixel_keys`
- shared-vs-separate encoder options for multiple camera views
- feature fusion for encoded pixels plus optional low-dimensional state
- JAX PyTree batch handling through agent updates

The current image and multi-camera work is for environment/data plumbing only;
policy learning from pixels is intentionally left as a TODO.
