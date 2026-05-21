from __future__ import annotations

"""Minimal Robomimic low-dimensional Square environment wrapper.

This is copied in spirit from the QC reference, but kept local so the
`intervention_learning` runtime does not import from the QC repository.
"""

import os
import time
from os.path import expanduser

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.spaces import Box, Dict
from robomimic import DATASET_REGISTRY  # noqa: F401  # Keep registry side effects.
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


LOW_DIM_KEYS = {
    "low_dim": (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    )
}
ObsUtils.initialize_obs_modality_mapping_from_dict(LOW_DIM_KEYS)


def is_robomimic_env(env_name: str) -> bool:
    """Return whether `env_name` is one of the supported Robomimic low-dim tasks."""
    if "low_dim" not in env_name:
        return False
    task, dataset_type, _ = env_name.split("-")
    return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in (
        "mh",
        "ph",
    )


def _max_episode_length(env_name: str) -> int:
    """Return Robomimic task horizon used by the QC experiments."""
    if env_name.startswith("lift"):
        return 300
    if env_name.startswith("can"):
        return 300
    if env_name.startswith("square"):
        return 400
    if env_name.startswith("transport"):
        return 800
    if env_name.startswith("tool_hang"):
        return 1000
    raise ValueError(f"Unsupported Robomimic environment: {env_name}")


def _dataset_path(env_name: str) -> str:
    """Return the local Robomimic dataset path and require it to exist."""
    task, dataset_type, _ = env_name.split("-")
    file_name = "low_dim_sparse_v15.hdf5" if dataset_type == "mg" else "low_dim_v15.hdf5"
    path = os.path.join(expanduser("~/.robomimic"), task, dataset_type, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Robomimic dataset not found: {path}")
    return path


class RobomimicLowdimWrapper(gym.Env):
    """Gymnasium wrapper exposing low-dim and optional rendered image observations."""

    def __init__(
        self,
        env,
        *,
        low_dim_keys=LOW_DIM_KEYS["low_dim"],
        observation_mode: str = "lowdim",
        render_hw=(256, 256),
        render_camera_name="agentview",
        camera_names: tuple[str, ...] | list[str] | None = None,
        image_camera_name: str | None = None,
        image_camera_names: tuple[str, ...] | list[str] | None = None,
        image_hw: tuple[int, int] | None = None,
        max_episode_length: int | None = None,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.observation_mode = observation_mode
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        if camera_names is None:
            camera_names = image_camera_names
        if camera_names is None:
            camera_names = (image_camera_name or render_camera_name,)
        self.camera_names = tuple(camera_names)
        if not self.camera_names:
            raise ValueError("At least one image camera must be specified.")
        self.image_hw = image_hw or render_hw
        self.video_writer = None
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0

        low = np.full(env.action_dimension, fill_value=-1.0)
        high = np.full(env.action_dimension, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)

        raw_obs = self.env.get_observation()
        state_example = self._state_from_raw_obs(raw_obs)
        state_space = Box(
            low=np.full_like(state_example, fill_value=-1.0),
            high=np.full_like(state_example, fill_value=1.0),
            shape=state_example.shape,
            dtype=state_example.dtype,
        )
        if observation_mode == "lowdim":
            self.observation_space = state_space
        elif observation_mode == "pixels":
            h, w = self.image_hw
            self.observation_space = Dict(
                {
                    camera_name: Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
                    for camera_name in self.camera_names
                }
            )
        elif observation_mode == "pixels_state":
            h, w = self.image_hw
            pixel_spaces = {
                camera_name: Box(
                    low=0,
                    high=255,
                    shape=(h, w, 3),
                    dtype=np.uint8,
                )
                for camera_name in self.camera_names
            }
            self.observation_space = Dict(
                {
                    "state": state_space,
                    **pixel_spaces,
                }
            )
        else:
            raise ValueError(f"Unsupported observation_mode: {observation_mode!r}")

        self.t = 0
        self.episode_return = 0.0
        self.episode_length = 0

    def _state_from_raw_obs(self, raw_obs):
        """Return concatenated Robomimic low-dimensional state."""
        return np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0).astype(np.float32)

    def _render_camera(self, camera_name: str):
        """Render one image observation camera."""
        h, w = self.image_hw
        return self.env.render(
            mode="rgb_array",
            height=h,
            width=w,
            camera_name=camera_name,
        ).astype(np.uint8)

    def _render_pixels(self):
        """Render configured image observation camera(s)."""
        pixels = {
            camera_name: self._render_camera(camera_name)
            for camera_name in self.camera_names
        }
        return pixels

    def _format_observation(self, raw_obs):
        """Format raw Robomimic observation according to `observation_mode`."""
        state = self._state_from_raw_obs(raw_obs)
        if self.observation_mode == "lowdim":
            return state
        pixels = self._render_pixels()
        if self.observation_mode == "pixels":
            return pixels
        return {"state": state, **pixels}

    def get_observation(self):
        """Return the configured Gymnasium observation."""
        return self._format_observation(self.env.get_observation())

    def seed(self, seed=None):
        """Seed numpy RNG used by Robomimic reset."""
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        """Reset the wrapped Robomimic env and return Gymnasium-style output."""
        del kwargs
        options = {} if options is None else options
        self.t = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes += 1

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        reset_seed = options.get("seed", None)
        if reset_seed is not None:
            self.seed(seed=reset_seed)
        self.env.reset()
        return self.get_observation(), {}

    def step(self, action):
        """Step env with sparse task-success reward and timeout truncation."""
        raw_obs, _, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        task_success = bool(info.get("is_success", {}).get("task", False))
        reward = float(task_success)
        obs = self._format_observation(raw_obs)

        if self.video_writer is not None:
            self.video_writer.append_data(self.render(mode="rgb_array"))

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        done = bool(done or task_success)
        info["success"] = int(task_success)
        info["episode_id"] = int(self.n_episodes - 1)
        info["episode_step"] = int(self.t - 1)

        if done:
            return obs, reward, True, False, info
        if self.max_episode_length is not None and self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        """Render the selected Robomimic camera."""
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        """Return current episode statistics."""
        return {"return": self.episode_return, "length": self.episode_length}


class EpisodeMonitor(gym.Wrapper):
    """Attach episode statistics to `info` at termination/truncation."""

    def __init__(self, env):
        super().__init__(env)
        self.total_timesteps = 0
        self._reset_stats()

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminated or truncated:
            info["episode"] = {
                "final_reward": reward,
                "return": self.reward_sum,
                "length": self.episode_length,
                "duration": time.time() - self.start_time,
                "success": float(info.get("success", 0)),
            }

        return observation, reward, terminated, truncated, info


def make_env(
    env_name: str,
    *,
    seed: int = 0,
    render_offscreen: bool = False,
    observation_mode: str = "lowdim",
    render_camera_name: str = "agentview",
    camera_names: tuple[str, ...] | list[str] | None = None,
    image_camera_name: str | None = None,
    image_camera_names: tuple[str, ...] | list[str] | None = None,
    render_hw: tuple[int, int] = (256, 256),
    image_hw: tuple[int, int] | None = None,
):
    """Create a Robomimic low-dimensional env matching previous Square runs."""
    if observation_mode != "lowdim" and not render_offscreen:
        raise ValueError("Robomimic image observations require render_offscreen=True.")

    dataset_path = _dataset_path(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen,
    )
    env = RobomimicLowdimWrapper(
        env,
        low_dim_keys=LOW_DIM_KEYS["low_dim"],
        observation_mode=observation_mode,
        render_hw=render_hw,
        render_camera_name=render_camera_name,
        camera_names=camera_names,
        image_camera_name=image_camera_name,
        image_camera_names=image_camera_names,
        image_hw=image_hw,
        max_episode_length=_max_episode_length(env_name),
    )
    env.seed(seed)
    return EpisodeMonitor(env)
