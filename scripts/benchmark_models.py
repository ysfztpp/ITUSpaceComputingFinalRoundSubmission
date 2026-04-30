from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.model_factory import build_model, build_model_config, normalize_model_type
from scripts.submission_inference import run_inference


CANDIDATES = {
    "c03": ROOT / "configs" / "submission_c03.json",
    "tsvit": ROOT / "configs" / "submission_tsvit.json",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_candidate(config_path: Path, device: torch.device):
    config = json.loads(config_path.read_text())
    checkpoint_path = resolve_path(config["checkpoint"])
    start = time.perf_counter()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    model_config = build_model_config(model_type, payload["model_config"])
    model = build_model(model_type, model_config)
    state = payload["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    synchronize(device)
    load_seconds = time.perf_counter() - start
    return config, checkpoint_path, payload, model, model_type, load_seconds


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synthetic_batch(model, batch_size: int, time_steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    channels = int(model.config.in_channels)
    patch_size = int(model.config.patch_size)
    batch = {
        "patches": torch.randn(batch_size, time_steps, channels, patch_size, patch_size, device=device),
        "time_mask": torch.ones(batch_size, time_steps, dtype=torch.bool, device=device),
        "time_doy": torch.linspace(90.0, 300.0, time_steps, device=device).repeat(batch_size, 1),
        "query_doy": torch.full((batch_size,), 180.0, device=device),
    }
    aux_dim = int(getattr(model.config, "aux_feature_dim", 0))
    if aux_dim > 0:
        batch["aux_features"] = torch.randn(batch_size, aux_dim, device=device)
    return batch


def benchmark_forward(model, *, batch_size: int, time_steps: int, warmup: int, iterations: int, device: torch.device) -> dict[str, Any]:
    batch = synthetic_batch(model, batch_size, time_steps, device)
    with torch.no_grad():
        for _ in range(warmup):
            model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"], batch.get("aux_features"))
        synchronize(device)
        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(batch["patches"], batch["time_mask"], batch["time_doy"], batch["query_doy"], batch.get("aux_features"))
            synchronize(device)
            timings.append(time.perf_counter() - start)
    median_seconds = statistics.median(timings)
    mean_seconds = statistics.mean(timings)
    return {
        "batch_size": batch_size,
        "time_steps": time_steps,
        "warmup": warmup,
        "iterations": iterations,
        "median_forward_ms": median_seconds * 1000.0,
        "mean_forward_ms": mean_seconds * 1000.0,
        "median_rows_per_second": batch_size / median_seconds if median_seconds > 0 else None,
        "mean_rows_per_second": batch_size / mean_seconds if mean_seconds > 0 else None,
    }


def benchmark_candidate(candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    config_path = CANDIDATES[candidate]
    device = select_device(args.device)
    config, checkpoint_path, payload, model, model_type, load_seconds = load_candidate(config_path, device)
    model_config = payload["model_config"]
    report: dict[str, Any] = {
        "candidate": candidate,
        "model_type": model_type,
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metric": payload.get("checkpoint_metric"),
        "parameter_count": count_parameters(model),
        "model_config": {
            "in_channels": model_config.get("in_channels"),
            "patch_size": model_config.get("patch_size"),
            "num_crop_classes": model_config.get("num_crop_classes"),
            "num_phenophase_classes": model_config.get("num_phenophase_classes"),
            "aux_feature_dim": model_config.get("aux_feature_dim", 0),
        },
        "load_seconds": load_seconds,
        "synthetic_forward": benchmark_forward(
            model,
            batch_size=int(args.batch_size or config.get("batch_size", 64)),
            time_steps=int(args.time_steps),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
            device=device,
        ),
    }
    if args.input_root:
        runtime_config = dict(config)
        runtime_config["input_root"] = str(args.input_root)
        runtime_config["output_json"] = str(args.output_dir / candidate / "result.json")
        runtime_config["work_dir"] = str(args.work_dir / candidate)
        start = time.perf_counter()
        inference_report = run_inference(runtime_config)
        total_seconds = time.perf_counter() - start
        report["end_to_end_inference"] = {
            "input_root": str(args.input_root),
            "output_json": runtime_config["output_json"],
            "total_seconds": total_seconds,
            "queries": inference_report.get("queries"),
            "unique_output_keys": inference_report.get("unique_output_keys"),
            "rows_per_second": (inference_report.get("queries") or 0) / total_seconds if total_seconds > 0 else None,
            "patch_report": inference_report.get("patch_report"),
            "result_stats": inference_report.get("result_stats"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C03 and TSViT load/forward/inference timing.")
    parser.add_argument("--candidate", choices=["all", *sorted(CANDIDATES)], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=29)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--input-root", type=Path, default=None, help="Optional platform-shaped input root for end-to-end timing.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "benchmarks")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "benchmark_work")
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "benchmarks" / "benchmark_models.json")
    args = parser.parse_args()

    selected = sorted(CANDIDATES) if args.candidate == "all" else [args.candidate]
    reports = [benchmark_candidate(candidate, args) for candidate in selected]
    payload = {
        "generated_at_unix": time.time(),
        "reports": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
