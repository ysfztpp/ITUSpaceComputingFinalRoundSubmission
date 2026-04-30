from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.submission_inference import run_inference


VARIANTS = {
    "P0": ROOT / "configs" / "optimization" / "versions" / "P0_submission_c03_frozen_reference.json",
    "P1": ROOT / "configs" / "optimization" / "versions" / "P1_submission_c03_file_index_fast.json",
    "P2": ROOT / "configs" / "optimization" / "versions" / "P2_submission_c03_gdal_inmemory.json",
    "P3": ROOT / "configs" / "optimization" / "versions" / "P3_submission_c03_fast_inventory_gdal_inmemory.json",
    "P4": ROOT / "configs" / "optimization" / "versions" / "P4_submission_c03_batched_raster_windows.json",
    "P5": ROOT / "configs" / "optimization" / "versions" / "P5_submission_c03_block_aware_raster_cache.json",
}


def _load_config(config_path: Path, *, input_root: Path, output_json: Path, work_dir: Path, device: str, batch_size: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
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
            "batch_raster_reads",
            "block_raster_reads",
            "raster_read_calls",
            "raster_batch_read_calls",
            "raster_block_read_calls",
            "raster_fallback_patch_read_calls",
            "raster_pixels_read",
        ]
    }


def _timing_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    timing_keys = [
        "patch_extraction_seconds",
        "model_load_seconds",
        "npz_and_query_rows_seconds",
        "batch_prepare_seconds",
        "model_forward_seconds",
        "postprocess_seconds",
        "write_result_seconds",
        "total_seconds",
    ]
    patch_timing_keys = [
        "audit_tiff_files_seconds",
        "select_file_index_seconds",
        "build_region_catalog_seconds",
        "build_lookup_and_allocate_seconds",
        "extract_raster_patches_seconds",
        "write_npz_seconds",
        "total_build_patch_dataset_seconds",
    ]
    patch_count_keys = [
        "raster_read_calls",
        "raster_batch_read_calls",
        "raster_block_read_calls",
        "raster_fallback_patch_read_calls",
        "raster_pixels_read",
    ]
    return {
        "timing_median_seconds": {
            key: median(float(report["timing_seconds"].get(key, 0.0)) for report in reports)
            for key in timing_keys
        },
        "patch_timing_median_seconds": {
            key: median(float(report.get("patch_report", {}).get("timing_seconds", {}).get(key, 0.0)) for report in reports)
            for key in patch_timing_keys
        },
        "patch_count_medians": {
            key: median(int(report.get("patch_report", {}).get(key, 0)) for report in reports)
            for key in patch_count_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run numbered C03 data-prep pipeline experiments.")
    parser.add_argument("--input-root", type=Path, default=Path("/tmp/kybelix_pipeline_smoke/input"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "experiments" / "data_prep_pipeline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = args.output_root / run_name
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    variant_reports: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    variant_outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    orders = [list(VARIANTS.keys()), list(reversed(VARIANTS.keys()))]

    for round_index in range(args.rounds):
        order = orders[round_index % len(orders)]
        for variant_name in order:
            variant_dir = run_root / f"round_{round_index + 1:02d}" / variant_name
            output_json = variant_dir / "output" / "result.json"
            work_dir = variant_dir / "work"
            config = _load_config(
                VARIANTS[variant_name],
                input_root=args.input_root,
                output_json=output_json,
                work_dir=work_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
            report = run_inference(config)
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "report.json").write_text(json.dumps(report, indent=2))
            variant_reports[variant_name].append(report)
            variant_outputs[variant_name].append(json.loads(output_json.read_text()))

    reference_output = variant_outputs["P0"][0]
    output_checks = {
        variant_name: all(output == reference_output for output in outputs)
        for variant_name, outputs in variant_outputs.items()
    }
    core_checks = {
        variant_name: [_patch_core(report) for report in reports]
        for variant_name, reports in variant_reports.items()
    }
    summary = {
        "run_root": str(run_root),
        "input_root": str(args.input_root),
        "device": args.device,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "test_dataset": {
            "description": "platform-shaped local smoke input built from a few known training points",
            "expected_query_rows": 21,
            "expected_unique_points": 3,
        },
        "version_notes": {
            "P0": "Frozen submitted C03 reference config/checkpoint. When run in this codebase it also benefits from safe global code fixes.",
            "P1": "Working C03 checkpoint plus the fast selected-file index implementation.",
            "P2": "P1 plus GDAL open/cache settings and in-memory patch arrays.",
            "P3": "P2 plus inference-only fast TIFF inventory that directly builds the selected-file index.",
            "P4": "P3 plus bounded batched raster-window reads for points sharing the same selected raster file.",
            "P5": "P4 plus adaptive block-aware reads keyed to GeoTIFF internal block layout, with P4/P3 fallback.",
        },
        "outputs_match_P0": output_checks,
        "patch_core_checks": core_checks,
        "summaries": {
            variant_name: _timing_summary(reports)
            for variant_name, reports in variant_reports.items()
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))
    latest = args.output_root / "latest_summary.json"
    latest.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not all(output_checks.values()):
        raise SystemExit("one or more variants produced output different from P0")


if __name__ == "__main__":
    main()
