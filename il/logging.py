from __future__ import annotations

"""Metric logging utilities for training loops.

The logger records scalar metrics every environment step, but writes only at a
configured interval. Losses, gradients, and rewards are averaged over the
interval. State-like counters such as replay size and episode count keep the
last observed value.
"""

import csv
import json
import time
from numbers import Number
from pathlib import Path
from typing import Any


_LAST_VALUE_KEYS = {
    "train/step",
    "train/online_size",
    "train/demo_size",
    "train/intervention_size",
    "train/episodes",
    "train/interval_sps",
    "train/total_sps",
    "time/elapsed_seconds",
}

_LAST_VALUE_SUFFIXES = (
    "/size",
    "_size",
    "/count",
    "_count",
    "/step",
    "_step",
)

_LAST_VALUE_PREFIXES = (
    "env/recent_",
    "routing/",
)


class MetricLogger:
    """Aggregate scalar metrics and write interval summaries."""

    def __init__(
        self,
        *,
        run_dir: Path,
        config: dict[str, Any],
        stdout_interval: int,
        jsonl_enabled: bool = True,
        csv_enabled: bool = True,
        wandb_enabled: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.stdout_interval = int(stdout_interval)
        self.jsonl_enabled = bool(jsonl_enabled)
        self.csv_enabled = bool(csv_enabled)
        self.wandb_enabled = bool(wandb_enabled)
        self.start_time = time.time()

        self.jsonl_file = None
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_fields: list[str] = []
        self.csv_rows: list[dict[str, float | int | str]] = []
        self.wandb_run = None

        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._last: dict[str, float | int | str] = {}
        self._records = 0

        if self.jsonl_enabled:
            self.jsonl_file = (self.run_dir / "metrics.jsonl").open("a", buffering=1)
        if self.wandb_enabled:
            import wandb

            run_cfg = config["run"]
            self.wandb_run = wandb.init(
                project=run_cfg["project"],
                group=run_cfg["group"],
                name=self.run_dir.name,
                config=_json_safe(config),
                tags=list(run_cfg.get("tags", [])),
            )

    def record(self, metrics: dict[str, Any], *, step: int, force_flush: bool = False) -> None:
        """Accumulate one metric payload and optionally flush an interval row."""
        payload = _flatten_scalar_metrics(metrics)
        payload.setdefault("train/step", int(step))
        payload["time/elapsed_seconds"] = time.time() - self.start_time
        self._accumulate(payload)

        should_flush = force_flush or (self.stdout_interval > 0 and step % self.stdout_interval == 0)
        if should_flush:
            self.flush(step=step, print_stdout=True)

    def log(self, metrics: dict[str, Any], *, step: int, force_stdout: bool = False) -> None:
        """Backward-compatible alias for interval aggregation."""
        self.record(metrics, step=step, force_flush=force_stdout)

    def log_immediate(self, metrics: dict[str, Any], *, step: int, print_stdout: bool = True) -> None:
        """Write metrics immediately without touching the train accumulator."""
        payload = _flatten_scalar_metrics(metrics)
        payload.setdefault("train/step", int(step))
        payload["time/elapsed_seconds"] = time.time() - self.start_time
        self._write_payload(payload, step=step, print_stdout=print_stdout)

    def flush(self, *, step: int, print_stdout: bool = True) -> None:
        """Write one averaged interval row and reset the accumulator."""
        if self._records == 0:
            return
        payload: dict[str, float | int | str] = {}
        for key, value_sum in self._sum.items():
            count = max(self._count.get(key, 0), 1)
            payload[key] = value_sum / count
        payload.update(self._last)
        payload["train/log_interval_records"] = self._records
        payload.setdefault("train/step", int(step))
        payload["time/elapsed_seconds"] = time.time() - self.start_time
        self._write_payload(payload, step=step, print_stdout=print_stdout)
        self._sum.clear()
        self._count.clear()
        self._last.clear()
        self._records = 0

    def _accumulate(self, payload: dict[str, float | int | str]) -> None:
        self._records += 1
        for key, value in payload.items():
            if _is_last_value_key(key) or isinstance(value, str):
                self._last[key] = value
                continue
            if isinstance(value, bool):
                numeric = float(int(value))
            elif isinstance(value, Number):
                numeric = float(value)
            else:
                self._last[key] = value
                continue
            self._sum[key] = self._sum.get(key, 0.0) + numeric
            self._count[key] = self._count.get(key, 0) + 1

    def _write_payload(self, payload: dict[str, float | int | str], *, step: int, print_stdout: bool) -> None:
        if self.jsonl_file is not None:
            self.jsonl_file.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.csv_enabled:
            self._write_csv(payload)
        if self.wandb_run is not None:
            self.wandb_run.log(payload, step=step)
        if print_stdout:
            self._print_summary(payload, step=step)

    def _write_csv(self, payload: dict[str, float | int | str]) -> None:
        """Write CSV rows with a dynamically expanded header."""
        self.csv_rows.append(dict(payload))
        fields = sorted({field for row in self.csv_rows for field in row})
        if fields != self.csv_fields:
            self.csv_fields = fields
        with self.csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fields)
            writer.writeheader()
            writer.writerows(self.csv_rows)

    def _print_summary(self, payload: dict[str, float | int | str], *, step: int) -> None:
        """Print a compact human-readable progress line."""
        pieces = [f"step={step}"]
        for key in (
            "train/log_interval_records",
            "train/online_size",
            "train/demo_size",
            "train/intervention_size",
            "train/episodes",
            "env/recent_success_rate",
            "train/interval_sps",
            "learner_bc/actor/bc_flow_loss",
            "learner_bc/actor/grad/norm",
        ):
            if key not in payload:
                continue
            value = payload[key]
            if isinstance(value, float):
                pieces.append(f"{key}={value:.4g}")
            else:
                pieces.append(f"{key}={value}")
        print("[metrics] " + " ".join(pieces), flush=True)

    def close(self) -> None:
        """Flush pending metrics, close files, and close optional wandb run."""
        self.flush(step=int(self._last.get("train/step", 0) or 0), print_stdout=False)
        if self.wandb_run is not None:
            (self.run_dir / "wandb_url.txt").write_text(self.wandb_run.url)
            self.wandb_run.finish()
        if self.jsonl_file is not None:
            self.jsonl_file.close()


class NullLogger:
    """No-op logger with the same interface as `MetricLogger`."""

    def record(self, metrics: dict[str, Any], *, step: int, force_flush: bool = False) -> None:
        del metrics, step, force_flush

    def log(self, metrics: dict[str, Any], *, step: int, force_stdout: bool = False) -> None:
        del metrics, step, force_stdout

    def log_immediate(self, metrics: dict[str, Any], *, step: int, print_stdout: bool = True) -> None:
        del metrics, step, print_stdout

    def close(self) -> None:
        pass


def _is_last_value_key(key: str) -> bool:
    return (
        key in _LAST_VALUE_KEYS
        or key.endswith(_LAST_VALUE_SUFFIXES)
        or key.startswith(_LAST_VALUE_PREFIXES)
    )


def _flatten_scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str]:
    """Keep only JSON/CSV-friendly scalar metrics."""
    out: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[str(key)] = int(value)
        elif isinstance(value, int):
            out[str(key)] = value
        elif isinstance(value, float):
            out[str(key)] = value
        elif isinstance(value, str):
            out[str(key)] = value
        elif hasattr(value, "item"):
            item = value.item()
            if isinstance(item, (bool, int, float, str)):
                out[str(key)] = int(item) if isinstance(item, bool) else item
    return out


def _json_safe(value: Any) -> Any:
    """Convert config values to JSON-serializable objects for wandb."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
