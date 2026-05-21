"""Intervention-learning package.

Package roles:
- `algo`: trainable agents that own losses, optimizers, and update steps.
- `policies`: action-only wrappers used for rollout, evaluation, and experts.
- `loops`: data-collection and training-loop orchestration.
- `buffers`: replay schemas, storage, routing, and sampling.
- `gating`: controller-selection logic for learner/expert intervention.
- `envs`: environment construction and adaptation.
- `networks` and `distributions`: reusable Flax model components.
- `utils`: shared config, types, TrainState, and small update helpers.

Keep experiment wiring in train/loop code. Keep algorithm math in `algo`.
"""
