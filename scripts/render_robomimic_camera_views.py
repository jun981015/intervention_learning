from __future__ import annotations

"""Render one Robomimic expert trajectory from multiple camera views."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])

import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from robomimic.envs.env_base import EnvBase
from robomimic.utils import env_utils as EnvUtils
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils


DEFAULT_CAMERAS = (
    "frontview",
    "birdview",
    "agentview",
    "sideview",
    "robot0_robotview",
    "robot0_eye_in_hand",
)


def initialize_robomimic_obs_utils() -> None:
    """Initialize robomimic obs utils for state-playback rendering."""
    ObsUtils.initialize_obs_utils_with_obs_specs(
        obs_modality_specs={
            "obs": {
                "low_dim": ["robot0_eef_pos"],
                "rgb": [],
            }
        }
    )


def sorted_demo_keys(dataset: h5py.File) -> list[str]:
    """Return demo keys sorted by numeric suffix."""
    demos = list(dataset["data"].keys())
    return sorted(demos, key=lambda key: int(key[5:]))


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    """Overlay a camera-name label on one RGB frame."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    rect = (0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2)
    draw.rectangle(rect, fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def make_grid(frames: list[np.ndarray], *, columns: int) -> np.ndarray:
    """Make a fixed camera grid from equal-sized RGB frames."""
    if not frames:
        raise ValueError("No frames to place in grid.")
    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / columns))
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    padded = frames + [blank] * (rows * columns - len(frames))
    row_images = []
    for row in range(rows):
        start = row * columns
        row_images.append(np.concatenate(padded[start : start + columns], axis=1))
    return np.concatenate(row_images, axis=0)


def render_demo(args) -> Path:
    """Render a multi-camera video from a stored expert state trajectory."""
    dataset_path = Path(args.dataset).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initialize_robomimic_obs_utils()
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
    )
    if not isinstance(env, EnvBase):
        raise TypeError(f"Expected robomimic EnvBase, got {type(env).__name__}.")

    with h5py.File(dataset_path, "r") as dataset:
        demos = sorted_demo_keys(dataset)
        demo_key = args.demo_key or demos[int(args.demo_index)]
        demo = dataset[f"data/{demo_key}"]
        states = demo["states"][()]
        initial_state = {"states": states[0]}
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

        env.reset_to(initial_state)
        writer = imageio.get_writer(output_path, fps=args.fps)
        try:
            max_frames = min(len(states), args.max_frames) if args.max_frames > 0 else len(states)
            written = 0
            for t in range(max_frames):
                env.reset_to({"states": states[t]})
                if t % args.video_skip != 0:
                    continue
                camera_frames = []
                for camera_name in args.cameras:
                    frame = env.render(
                        mode="rgb_array",
                        height=args.height,
                        width=args.width,
                        camera_name=camera_name,
                    )
                    camera_frames.append(draw_label(frame.astype(np.uint8), camera_name))
                writer.append_data(make_grid(camera_frames, columns=args.columns))
                written += 1
        finally:
            writer.close()

    print(
        f"saved {output_path} demo={demo_key} cameras={list(args.cameras)} "
        f"frames={written} fps={args.fps}",
        flush=True,
    )
    return output_path


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="~/.robomimic/square/mh/low_dim_v15.hdf5",
        help="Robomimic HDF5 dataset path containing expert states.",
    )
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument("--demo-key", default="", help="Explicit demo key such as demo_0.")
    parser.add_argument("--demo-index", type=int, default=0, help="Sorted demo index used when demo-key is unset.")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS), help="Camera names to render.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=240, help="Max dataset timesteps before video-skip.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    render_demo(parse_args())


if __name__ == "__main__":
    main()
