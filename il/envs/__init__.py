"""Environment builders and adapters.

This package should hide simulator-specific setup behind small factory
functions. Algorithm files should not import Robomimic/Robosuite directly.
"""

from importlib import import_module
from typing import Callable

ENV_BUILDERS = {
    "robomimic_lowdim": "il.envs.robomimic_lowdim:make_env",
}


def _load_builder(target: str) -> Callable:
    """Load an env factory only when that env kind is requested."""
    module_name, function_name = target.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, function_name)


def make_env(config: dict, *, seed: int = 0):
    """Build an environment from an env config section."""
    kind = config["kind"]
    if kind not in ENV_BUILDERS:
        raise ValueError(f"Unsupported env kind: {kind!r}. Available: {sorted(ENV_BUILDERS)}")
    builder = _load_builder(ENV_BUILDERS[kind])
    kwargs = {
        "seed": seed,
        "render_offscreen": bool(config.get("render_offscreen", False)),
    }
    for key in (
        "observation_mode",
        "render_camera_name",
        "camera_names",
        "image_camera_name",
        "image_camera_names",
    ):
        if key in config:
            kwargs[key] = config[key]
    if "image_size" in config:
        image_size = config["image_size"]
        kwargs["image_hw"] = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size)
    if "render_size" in config:
        render_size = config["render_size"]
        kwargs["render_hw"] = (int(render_size), int(render_size)) if isinstance(render_size, int) else tuple(render_size)
    return builder(config["name"], **kwargs)


def make_robomimic_env(*args, **kwargs):
    """Backward-compatible direct Robomimic low-dim factory."""
    return _load_builder(ENV_BUILDERS["robomimic_lowdim"])(*args, **kwargs)


__all__ = ["ENV_BUILDERS", "make_env", "make_robomimic_env"]
