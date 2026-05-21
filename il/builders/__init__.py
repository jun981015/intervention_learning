"""Runtime builders for recipe-driven training.

Builder modules translate YAML recipe sections into concrete runtime objects.
Keep this package focused on construction and validation.

Current split:
- `config`: recipe loading, runtime env vars, run path creation.
- `actors`: learner/expert agent construction and optional pretrained loading.
- `components`: env, replay-buffer, and gate construction.
- `types`: small dataclasses shared by the builders.

Env-step loops and algorithm losses live elsewhere.
"""
