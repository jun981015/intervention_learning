from __future__ import annotations

"""Render policy rollout videos with matplotlib action-uncertainty plots."""

import argparse
import copy
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])

import imageio
import jax
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from il.builders.actors import build_actor_bundle
from il.builders.components import build_envs, infer_env_spec
from il.builders.config import load_recipe
from il.gating.action_uncertainty import ActionUncertaintyGate
from il.gating.base import GateContext
from il.loops.rollout import (
    _sample_gate_policy_source,
    policy_observation,
    reset_rollout_state,
    residual_policy_observation,
    sample_base_action,
    uses_residual_composition,
)
from il.utils.types import PolicyOutput


_LINE_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:cyan",
]


def _infer_step(checkpoint: Path) -> int:
    """Infer the training step from a params_<step>.pkl filename."""
    match = re.search(r"params_(\d+)\.pkl$", checkpoint.name)
    if match is None:
        raise ValueError(f"Cannot infer step from checkpoint name: {checkpoint.name}")
    return int(match.group(1))


def _prepare_config(config_path: Path, checkpoint: Path, args) -> dict[str, Any]:
    """Load config and point the selected policy role at `checkpoint`."""
    config = load_recipe(config_path)
    config["run"]["wandb"] = False
    config["env"]["render_offscreen"] = True
    config["env"]["render_camera_name"] = args.camera
    config["env"]["render_hw"] = (args.render_size, args.render_size)
    config["env"]["build_eval_env"] = False

    policy_spec = config.get(args.policy_role)
    if policy_spec is None:
        raise ValueError(f"Config does not define policy role {args.policy_role!r}.")
    config["learner"] = copy.deepcopy(policy_spec)
    if args.policy_kind is not None:
        config["learner"]["kind"] = args.policy_kind
    config["learner"]["pretrained_path"] = str(checkpoint)
    config["learner"]["checkpoint_step"] = None
    config["learner"]["train"] = False
    config["learner"]["policy_view"] = True
    config["learner"].setdefault("config", {})["exploration_noise"] = float(args.exploration_noise)

    if args.force_residual:
        config["rollout"]["execute"] = "residual"
        config["rollout"]["action_composition"] = "residual"
    else:
        config["rollout"]["execute"] = "gate"
        config["rollout"].pop("action_composition", None)
    return config


def _render_frame(env) -> np.ndarray:
    """Render through Gym wrappers."""
    try:
        return env.render(mode="rgb_array")
    except TypeError:
        return env.unwrapped.render(mode="rgb_array")


def _measure_uncertainty(
    context,
    policy_obs,
    *,
    step: int,
    rng_key,
    num_samples: int,
) -> tuple[dict[str, Any], Any]:
    """Compute action-uncertainty gate info for one rollout state."""
    dummy = PolicyOutput(action=np.zeros(context.action_dim, dtype=np.float32))
    gate = ActionUncertaintyGate(
        threshold=float("inf"),
        source="learner",
        estimator="sample_variance",
        num_samples=num_samples,
        score="rms_std",
        intervention_prob=0.0,
        intervention_horizon=1,
    )

    def sample_policy(source: str) -> PolicyOutput:
        nonlocal rng_key
        rng_key, sample_rng = jax.random.split(rng_key)
        return _sample_gate_policy_source(
            context,
            source,
            policy_obs,
            step=step,
            rng=sample_rng,
            learner_output=dummy,
        )

    gate_context = GateContext(
        sample_policy=sample_policy,
        policy_observation=policy_obs,
        action_dim=context.action_dim,
    )
    decision = gate.decide(
        step=step,
        observation=policy_obs,
        learner=dummy,
        expert=dummy,
        rng=np.random.default_rng(step),
        context=gate_context,
    )
    return decision.info, rng_key


def _sample_rollout_action(context, policy_obs, observation, *, step: int, rng_key) -> tuple[np.ndarray, Any]:
    """Sample the action executed in the rollout."""
    if uses_residual_composition(context):
        rng_key, base_rng, residual_rng = jax.random.split(rng_key, 3)
        residual_scale = float(context.learner.config.get("residual_scale", 1.0))
        base_output = sample_base_action(context, observation, rng=base_rng)
        residual_obs = residual_policy_observation(policy_obs, base_output.action)
        residual_output = context.learner.policy.sample_action(residual_obs, rng=residual_rng)
        action = np.asarray(base_output.action, dtype=np.float32) + residual_scale * np.asarray(
            residual_output.action,
            dtype=np.float32,
        )
    else:
        rng_key, action_rng = jax.random.split(rng_key)
        output = context.learner.policy.sample_action(policy_obs, rng=action_rng)
        action = np.asarray(output.action, dtype=np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32), rng_key


def _render_matplotlib_panel(
    *,
    scores: np.ndarray,
    variances: np.ndarray,
    current_idx: int,
    reward_sum: float,
    success: float,
    num_samples: int,
    exploration_noise: float,
    width: int,
    height: int,
    score_y_max: float,
    var_y_max: float,
    dpi: int,
) -> np.ndarray:
    """Render a full-trajectory matplotlib plot with the current time highlighted."""
    scores = np.asarray(scores, dtype=np.float32)
    variances = np.asarray(variances, dtype=np.float32)
    if variances.ndim == 1:
        variances = variances[:, None]
    steps = np.arange(scores.shape[0])

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax_score, ax_var = fig.subplots(2, 1, sharex=True)
    fig.suptitle(
        f"step {current_idx} | return {reward_sum:.1f} | success {success:.0f} | "
        f"samples {num_samples} | exploration_noise {exploration_noise:g}",
        fontsize=9,
    )

    ax_score.plot(steps, scores, color="tab:blue", linewidth=1.6, label="score")
    ax_score.axvline(current_idx, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_score.scatter([current_idx], [scores[current_idx]], color="black", s=18, zorder=4)
    ax_score.set_ylabel("score")
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(loc="upper right", fontsize=7)
    ax_score.set_ylim(0.0, max(float(score_y_max), float(np.nanmax(scores)) * 1.05, 1e-6))

    for dim in range(variances.shape[1]):
        ax_var.plot(
            steps,
            variances[:, dim],
            linewidth=1.1,
            color=_LINE_COLORS[dim % len(_LINE_COLORS)],
            label=f"a{dim}",
        )
        ax_var.scatter(
            [current_idx],
            [variances[current_idx, dim]],
            color=_LINE_COLORS[dim % len(_LINE_COLORS)],
            s=10,
            zorder=4,
        )
    ax_var.axvline(current_idx, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_var.set_ylabel("variance")
    ax_var.set_xlabel("time step")
    ax_var.grid(True, alpha=0.3)
    ax_var.legend(loc="upper right", fontsize=6, ncol=4)
    ax_var.set_ylim(0.0, max(float(var_y_max), float(np.nanmax(variances)) * 1.05, 1e-6))

    x_max = max(int(scores.shape[0]) - 1, 1)
    ax_score.set_xlim(0, x_max)
    ax_var.set_xlim(0, x_max)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    canvas.draw()
    panel = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    panel = panel.reshape(canvas.get_width_height()[1], canvas.get_width_height()[0], 4)
    return np.asarray(panel[..., :3], dtype=np.uint8)


def _compose_frame_with_panel(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Concatenate an RGB environment frame and an RGB matplotlib panel."""
    frame_image = Image.fromarray(frame).convert("RGB")
    panel_image = Image.fromarray(panel).convert("RGB")
    content_h = max(frame_image.height, panel_image.height)
    content_w = frame_image.width + panel_image.width
    canvas_h = ((content_h + 15) // 16) * 16
    canvas_w = ((content_w + 15) // 16) * 16
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(frame_image, (0, 0))
    canvas.paste(panel_image, (frame_image.width, 0))
    return np.asarray(canvas, dtype=np.uint8)


def _rollout_episode(
    *,
    env,
    context,
    seed: int,
    rng_key,
    checkpoint_step: int,
    num_samples: int,
    max_steps: int | None,
) -> tuple[dict[str, Any], Any]:
    """Roll out one episode, collecting frames and full uncertainty arrays first."""
    observation, _ = env.reset(options={"seed": seed})
    reset_rollout_state(context)
    done = False
    frames = []
    rewards = []
    successes = []
    actions = []
    scores = []
    variances = []
    episode_step = 0
    while not done:
        if max_steps is not None and episode_step >= max_steps:
            break
        policy_obs = policy_observation(observation, context)
        global_step = checkpoint_step + episode_step
        rng_key, uncertainty_rng, action_rng = jax.random.split(rng_key, 3)
        uncertainty, _ = _measure_uncertainty(
            context,
            policy_obs,
            step=global_step,
            rng_key=uncertainty_rng,
            num_samples=num_samples,
        )
        action, rng_key = _sample_rollout_action(context, policy_obs, observation, step=global_step, rng_key=action_rng)

        frames.append(_render_frame(env).astype(np.uint8))
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(info.get("success", 0.0)))
        actions.append(action)
        scores.append(float(uncertainty["uncertainty_score"]))
        variances.append(np.asarray(uncertainty["var"]["per_dim"], dtype=np.float32))
        done = bool(terminated or truncated)
        episode_step += 1

    summary = {
        "seed": int(seed),
        "length": int(len(rewards)),
        "return": float(np.sum(rewards)),
        "success": bool(np.max(successes) > 0.0) if successes else False,
        "score_mean": float(np.mean(scores)) if scores else float("nan"),
        "score_max": float(np.max(scores)) if scores else float("nan"),
    }
    arrays = {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "uncertainty_scores": np.asarray(scores, dtype=np.float32),
        "action_variances": np.asarray(variances, dtype=np.float32),
    }
    return {"summary": summary, "arrays": arrays}, rng_key


def _write_episode_video(
    *,
    episode: dict[str, Any],
    video_path: Path,
    fps: int,
    video_skip: int,
    plot_width: int,
    plot_height: int,
    plot_dpi: int,
    score_y_max: float,
    var_y_max: float,
    num_samples: int,
    exploration_noise: float,
) -> None:
    """Write frames with a full matplotlib graph and moving time highlight."""
    arrays = episode["arrays"]
    frames = arrays["frames"]
    rewards = arrays["rewards"]
    successes = arrays["successes"]
    scores = arrays["uncertainty_scores"]
    variances = arrays["action_variances"]
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for idx in range(0, len(frames), video_skip):
            reward_sum = float(np.sum(rewards[:idx]))
            success = float(np.max(successes[:idx])) if idx > 0 else 0.0
            panel = _render_matplotlib_panel(
                scores=scores,
                variances=variances,
                current_idx=idx,
                reward_sum=reward_sum,
                success=success,
                num_samples=num_samples,
                exploration_noise=exploration_noise,
                width=plot_width,
                height=plot_height,
                score_y_max=score_y_max,
                var_y_max=var_y_max,
                dpi=plot_dpi,
            )
            writer.append_data(_compose_frame_with_panel(frames[idx], panel))
    finally:
        writer.close()


def _save_static_plot(
    *,
    episode: dict[str, Any],
    plot_path: Path,
    plot_width: int,
    plot_height: int,
    plot_dpi: int,
    score_y_max: float,
    var_y_max: float,
    num_samples: int,
    exploration_noise: float,
) -> None:
    """Save the full uncertainty plot as a standalone image."""
    arrays = episode["arrays"]
    scores = arrays["uncertainty_scores"]
    variances = arrays["action_variances"]
    rewards = arrays["rewards"]
    successes = arrays["successes"]
    if len(scores) == 0:
        return
    panel = _render_matplotlib_panel(
        scores=scores,
        variances=variances,
        current_idx=len(scores) - 1,
        reward_sum=float(np.sum(rewards)),
        success=float(np.max(successes)) if len(successes) else 0.0,
        num_samples=num_samples,
        exploration_noise=exploration_noise,
        width=plot_width,
        height=plot_height,
        score_y_max=score_y_max,
        var_y_max=var_y_max,
        dpi=plot_dpi,
    )
    imageio.imwrite(plot_path, panel)


def _build_base_if_needed(config: dict[str, Any], *, env_spec, batch_size: int, seed: int):
    """Build a base actor only for residual composition."""
    if config.get("base") is None:
        return None
    if config["rollout"].get("execute") != "residual" and config["rollout"].get("action_composition") != "residual":
        return None
    return build_actor_bundle(
        name="base",
        spec=config["base"],
        env_spec=env_spec,
        batch_size=batch_size,
        seed=seed + 2,
    )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config_path = Path(args.config).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_step = int(args.step) if args.step is not None else _infer_step(checkpoint)

    config = _prepare_config(config_path, checkpoint, args)
    env, _ = build_envs(config)
    env_spec = infer_env_spec(env)
    batch_size = int(config["train"]["batch_size"])
    seed = int(config["run"]["seed"])
    learner = build_actor_bundle(
        name="learner",
        spec=config["learner"],
        env_spec=env_spec,
        batch_size=batch_size,
        seed=seed,
    )
    base = _build_base_if_needed(config, env_spec=env_spec, batch_size=batch_size, seed=seed)
    if learner.policy is None:
        raise ValueError("Uncertainty video requires a learner policy view.")
    if uses_residual_composition(SimpleNamespace(config=config, base=base, learner=learner)) and (base is None or base.policy is None):
        raise ValueError("Residual uncertainty video requires a base policy view.")

    context = SimpleNamespace(
        config=config,
        env_spec=env_spec,
        learner=learner,
        base=base,
        expert=None,
        action_dim=env_spec.action_dim,
        rollout_state={},
    )

    rng_key = jax.random.PRNGKey(args.seed)
    episodes = []
    for episode_idx in range(args.num_episodes):
        episode_seed = args.seed + episode_idx
        video_path = output_dir / f"episode_{episode_idx:02d}_seed{episode_seed}_uncertainty.mp4"
        data_path = output_dir / f"episode_{episode_idx:02d}_seed{episode_seed}_uncertainty.npz"
        plot_path = output_dir / f"episode_{episode_idx:02d}_seed{episode_seed}_uncertainty_plot.png"
        rng_key, episode_rng = jax.random.split(rng_key)
        episode, rng_key = _rollout_episode(
            env=env,
            context=context,
            seed=episode_seed,
            rng_key=episode_rng,
            checkpoint_step=checkpoint_step,
            num_samples=int(args.num_samples),
            max_steps=args.max_steps,
        )
        _write_episode_video(
            episode=episode,
            video_path=video_path,
            fps=int(args.fps),
            video_skip=int(args.video_skip),
            plot_width=int(args.plot_width),
            plot_height=int(args.plot_height),
            plot_dpi=int(args.plot_dpi),
            score_y_max=float(args.score_y_max),
            var_y_max=float(args.var_y_max),
            num_samples=int(args.num_samples),
            exploration_noise=float(args.exploration_noise),
        )
        _save_static_plot(
            episode=episode,
            plot_path=plot_path,
            plot_width=int(args.plot_width),
            plot_height=int(args.plot_height),
            plot_dpi=int(args.plot_dpi),
            score_y_max=float(args.score_y_max),
            var_y_max=float(args.var_y_max),
            num_samples=int(args.num_samples),
            exploration_noise=float(args.exploration_noise),
        )
        np.savez_compressed(
            data_path,
            actions=episode["arrays"]["actions"],
            rewards=episode["arrays"]["rewards"],
            successes=episode["arrays"]["successes"],
            uncertainty_scores=episode["arrays"]["uncertainty_scores"],
            action_variances=episode["arrays"]["action_variances"],
            seed=np.asarray(episode_seed, dtype=np.int64),
            checkpoint=np.asarray(str(checkpoint)),
            config=np.asarray(str(config_path)),
            checkpoint_step=np.asarray(checkpoint_step, dtype=np.int64),
            policy_role=np.asarray(str(args.policy_role)),
        )
        info = {
            **episode["summary"],
            "video": str(video_path),
            "data": str(data_path),
            "plot": str(plot_path),
        }
        episodes.append(info)
        print(f"[save] {video_path} {data_path} {plot_path} {info}", flush=True)

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "policy_role": str(args.policy_role),
        "policy_kind": str(config["learner"]["kind"]),
        "output_dir": str(output_dir),
        "num_episodes": int(args.num_episodes),
        "num_samples": int(args.num_samples),
        "exploration_noise_override": float(args.exploration_noise),
        "plot_width": int(args.plot_width),
        "plot_height": int(args.plot_height),
        "score_y_max": float(args.score_y_max),
        "var_y_max": float(args.var_y_max),
        "episodes": episodes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-role", choices=("learner", "expert"), default="learner")
    parser.add_argument("--policy-kind", default=None)
    parser.add_argument("--force-residual", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--exploration-noise", type=float, default=0.0)
    parser.add_argument("--camera", default="sideview")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--plot-width", type=int, default=464)
    parser.add_argument("--plot-height", type=int, default=432)
    parser.add_argument("--plot-dpi", type=int, default=100)
    parser.add_argument("--score-y-max", type=float, default=0.4)
    parser.add_argument("--var-y-max", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
