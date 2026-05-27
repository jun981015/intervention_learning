from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotSpec:
    name: str
    title: str
    dataset: Path
    file_prefix: str
    eval_dirs: dict[str, tuple[Path, ...]]


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPECS = {
    "square_mh": PlotSpec(
        name="square_mh",
        title="Square MH BCFlow Top-K",
        dataset=Path("/home/junhyeong/.robomimic/square/mh/low_dim_v141.hdf5"),
        file_prefix="bcflow_square_top",
        eval_dirs={
            "500k": (ROOT / "logs/topk_500k_eval",),
            "1M": (ROOT / "logs/topk_1m_eval",),
        },
    ),
    "square_ph": PlotSpec(
        name="square_ph",
        title="Square PH BCFlow Top-K",
        dataset=Path("/home/junhyeong/.robomimic/square/ph/low_dim_v141.hdf5"),
        file_prefix="bcflow_square_ph_top",
        eval_dirs={
            "500k": (ROOT / "logs/topk_ph_500k_eval",),
            "1M": (ROOT / "logs/topk_ph_1m_eval",),
        },
    ),
    "tool_hang_ph": PlotSpec(
        name="tool_hang_ph",
        title="ToolHang PH BCFlow Top-K",
        dataset=Path("/home/junhyeong/.robomimic/tool_hang/ph/low_dim_v141.hdf5"),
        file_prefix="bcflow_tool_hang_ph_top",
        eval_dirs={
            "500k": (ROOT / "logs/tool_hang_ph_topk_500k_eval",),
            "1M": (ROOT / "logs/tool_hang_ph_topk_1m_eval", ROOT / "logs"),
        },
    ),
}


def demo_sort_key(demo_name: str) -> int:
    """Sort demo names numerically, e.g. demo_10 after demo_9."""
    try:
        return int(demo_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def topk_length_stats(dataset_path: Path) -> dict[int, dict[str, float]]:
    """Return cumulative top-K length stats under the shortest-demo selection rule."""
    with h5py.File(dataset_path, "r") as file:
        lengths = [
            (demo_name, int(file["data"][demo_name]["actions"].shape[0]))
            for demo_name in sorted(file["data"].keys(), key=demo_sort_key)
        ]
    lengths = sorted(lengths, key=lambda item: (item[1], demo_sort_key(item[0])))
    stats = {}
    for top_k in range(1, len(lengths) + 1):
        selected_lengths = np.asarray([length for _, length in lengths[:top_k]], dtype=np.float32)
        stats[top_k] = {
            "avg_demo_length": float(selected_lengths.mean()),
            "total_steps": float(selected_lengths.sum()),
        }
    return stats


def collect_success(eval_dirs: tuple[Path, ...], file_prefix: str) -> dict[int, float]:
    """Collect success_rate values keyed by top_k from eval JSON files."""
    pattern = re.compile(re.escape(file_prefix) + r"(\d+).*\.json$")
    results = {}
    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            continue
        for path in eval_dir.glob("*.json"):
            match = pattern.search(path.name)
            if match is None:
                continue
            data = json.loads(path.read_text())
            if "success_rate" not in data:
                continue
            results[int(match.group(1))] = float(data["success_rate"])
    return results


def plot_spec(spec: PlotSpec, output_dir: Path) -> list[Path]:
    """Plot one task with x coordinates equal to real demo counts."""
    length_stats = topk_length_stats(spec.dataset)
    results = {
        checkpoint: collect_success(eval_dirs, spec.file_prefix)
        for checkpoint, eval_dirs in spec.eval_dirs.items()
    }
    top_ks = sorted({top_k for values in results.values() for top_k in values})
    if not top_ks:
        return []

    fig_width = max(8.0, 0.55 * len(top_ks) + 4.0)
    plt.figure(figsize=(fig_width, 5.0))
    colors = {"500k": "#1f77b4", "1M": "#d62728"}
    for checkpoint in ("500k", "1M"):
        values = results.get(checkpoint, {})
        y = [values.get(top_k, np.nan) for top_k in top_ks]
        if all(np.isnan(value) for value in y):
            continue
        plt.plot(
            top_ks,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            label=checkpoint,
            color=colors.get(checkpoint),
        )
        for top_k, success in zip(top_ks, y):
            if not np.isnan(success):
                plt.text(top_k, success + 0.018, f"{success:.2f}", ha="center", va="bottom", fontsize=8)

    tick_labels = [
        f"{top_k}\n({length_stats.get(top_k, {}).get('avg_demo_length', float('nan')):.0f})"
        for top_k in top_ks
    ]
    plt.xticks(top_ks, tick_labels, fontsize=9)
    plt.yticks(np.linspace(0.0, 1.0, 11))
    plt.ylim(0.0, 1.02)
    plt.grid(True, axis="both", alpha=0.28)
    plt.xlabel("Number of demos (average selected demo length)")
    plt.ylabel("Success rate")
    plt.title(spec.title)
    plt.legend(title="Checkpoint")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{spec.name}_bcflow_topk_success_real_demo_axis"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    csv_path = base.with_suffix(".csv")
    plt.savefig(png_path, dpi=220)
    plt.savefig(pdf_path)
    plt.close()

    with csv_path.open("w") as file:
        file.write("task,top_k,avg_demo_length,total_selected_steps,success_500k,success_1m\n")
        for top_k in top_ks:
            stats = length_stats.get(top_k, {})
            file.write(
                ",".join(
                    [
                        spec.name,
                        str(top_k),
                        f"{stats.get('avg_demo_length', float('nan')):.6f}",
                        f"{stats.get('total_steps', float('nan')):.6f}",
                        "" if top_k not in results.get("500k", {}) else f"{results['500k'][top_k]:.6f}",
                        "" if top_k not in results.get("1M", {}) else f"{results['1M'][top_k]:.6f}",
                    ]
                )
                + "\n"
            )
    return [png_path, pdf_path, csv_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BCFlow top-k eval success with real demo-count x-axis.")
    parser.add_argument("--tasks", nargs="+", default=sorted(DEFAULT_SPECS), choices=sorted(DEFAULT_SPECS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for task_name in args.tasks:
        outputs = plot_spec(DEFAULT_SPECS[task_name], args.output_dir)
        print(task_name)
        for output in outputs:
            print(f"  {output}")


if __name__ == "__main__":
    main()
