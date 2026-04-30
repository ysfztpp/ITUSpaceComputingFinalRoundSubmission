from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.submission_inference import run_inference


def _load_config(path: Path, *, input_root: Path, output_json: Path, work_dir: Path, device: str, batch_size: int) -> dict[str, Any]:
    config = json.loads(path.read_text())
    config["input_root"] = str(input_root)
    config["output_json"] = str(output_json)
    config["work_dir"] = str(work_dir)
    config["device"] = device
    config["batch_size"] = int(batch_size)
    return config


def _patch_core(report: dict[str, Any]) -> dict[str, Any]:
    patch_report = report.get("patch_report", {})
    return {
        key: patch_report.get(key)
        for key in [
            "samples_kept",
            "samples_dropped",
            "missing_band_cells",
            "file_read_failures",
            "patch_extraction_errors",
            "global_valid_pixel_ratio",
            "mapping_status_counts",
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized C03 data-prep inference paths.")
    parser.add_argument("--input-root", type=Path, default=Path("/tmp/kybelix_pipeline_smoke/input"))
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs" / "submission_c03.json")
    parser.add_argument(
        "--optimized-config",
        type=Path,
        default=ROOT / "configs" / "optimization" / "submission_c03_working.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "benchmarks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_root = args.output_dir / "data_prep_compare"
    compare_root.mkdir(parents=True, exist_ok=True)

    reports: list[tuple[str, Path, dict[str, Any]]] = []
    for name, config_path in [
        ("baseline", args.baseline_config),
        ("optimized", args.optimized_config),
    ]:
        output_json = compare_root / f"{name}_result.json"
        work_dir = compare_root / f"{name}_work"
        config = _load_config(
            config_path,
            input_root=args.input_root,
            output_json=output_json,
            work_dir=work_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        report = run_inference(config)
        (args.output_dir / f"c03_{name}_smoke_inference.json").write_text(json.dumps(report, indent=2))
        reports.append((name, output_json, report))

    baseline_output = json.loads(reports[0][1].read_text())
    optimized_output = json.loads(reports[1][1].read_text())
    comparison = {
        "outputs_identical": baseline_output == optimized_output,
        "baseline_rows": len(baseline_output),
        "optimized_rows": len(optimized_output),
        "baseline_timing": reports[0][2]["timing_seconds"],
        "optimized_timing": reports[1][2]["timing_seconds"],
        "baseline_patch_timing": reports[0][2].get("patch_report", {}).get("timing_seconds", {}),
        "optimized_patch_timing": reports[1][2].get("patch_report", {}).get("timing_seconds", {}),
        "baseline_patch_core": _patch_core(reports[0][2]),
        "optimized_patch_core": _patch_core(reports[1][2]),
    }
    out = args.output_dir / "c03_data_prep_optimization_compare.json"
    out.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    if not comparison["outputs_identical"]:
        raise SystemExit("optimized output differs from baseline output")


if __name__ == "__main__":
    main()
