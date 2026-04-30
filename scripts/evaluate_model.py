from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.query_dataset_npz import QueryDatePatchDataset
from models.model_factory import build_model, build_model_config, normalize_model_type
from training.query_engine import run_query_epoch

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required. Install PyTorch before evaluating.") from exc


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


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = normalize_model_type(payload.get("model_type", "query_cnn_transformer"))
    model_config = build_model_config(model_type, payload["model_config"])
    model = build_model(model_type, model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def build_loader(config: dict[str, Any], split: str, query_shift_days: float, device: torch.device) -> DataLoader:
    dataset = QueryDatePatchDataset(
        npz_path=resolve_path(config["dataset_npz"]),
        split_csv=resolve_path(config["split_csv"]) if config.get("split_csv") else None,
        split=split if config.get("split_csv") else None,
        normalization_json=resolve_path(config["normalization_json"]),
        rice_stage_loss_only=bool(config.get("rice_stage_loss_only", True)),
        include_valid_mask_as_channels=bool(config.get("include_valid_mask_as_channels", False)),
        use_aux_features=bool(config.get("use_aux_features", False)),
        aux_feature_set=str(config.get("aux_feature_set", "summary")),
        use_spectral_indices=bool(config.get("use_spectral_indices", False)),
        spectral_index_stats_json=resolve_path(config["spectral_index_stats_json"]) if config.get("spectral_index_stats_json") else None,
        use_relative_doy=bool(config.get("use_relative_doy", False)),
        fixed_time_shift_days=0.0,
        fixed_query_doy_shift_days=float(query_shift_days),
    )
    kwargs: dict[str, Any] = {
        "batch_size": int(config.get("batch_size", 512)),
        "shuffle": False,
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
    return DataLoader(dataset, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a C03/TSViT checkpoint on train/val query-date slices.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluate_c03_baseline.json")
    args = parser.parse_args()

    config = json.loads(resolve_path(args.config).read_text())
    device = select_device(str(config.get("device", "auto")))
    start = time.perf_counter()
    model, payload = load_model_from_checkpoint(resolve_path(config["checkpoint"]), device)

    results: list[dict[str, Any]] = []
    for split in config.get("splits", ["val"]):
        for query_shift in config.get("query_shift_days", [0]):
            loader = build_loader(config, str(split), float(query_shift), device)
            eval_start = time.perf_counter()
            metrics = run_query_epoch(
                model=model,
                loader=loader,
                optimizer=None,
                device=device,
                train=False,
                stage_loss_weight=float(config.get("stage_loss_weight", 0.6)),
                amp=bool(config.get("amp", False)),
                stage_postprocess=str(config.get("stage_postprocess", "none")),
            )
            results.append(
                {
                    "split": str(split),
                    "query_shift_days": float(query_shift),
                    "rows": int(len(loader.dataset)),
                    "seconds": time.perf_counter() - eval_start,
                    "metrics": metrics,
                }
            )

    report = {
        "status": "ok",
        "generated_at_unix": time.time(),
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint": str(resolve_path(config["checkpoint"])),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metric": payload.get("checkpoint_metric"),
        "model_type": normalize_model_type(payload.get("model_type", "query_cnn_transformer")),
        "model_config": payload.get("model_config", {}),
        "total_seconds": time.perf_counter() - start,
        "results": results,
    }
    output_json = resolve_path(config.get("output_json", "artifacts/evaluations/model_eval.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
