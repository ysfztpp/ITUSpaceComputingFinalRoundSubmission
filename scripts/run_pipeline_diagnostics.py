from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_ROOT = Path("/tmp/kybelix_pipeline_smoke")


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": args,
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - start,
        "output_tail": proc.stdout[-4000:],
    }


def first_point_rows(points_csv: Path, max_points: int) -> tuple[list[str], list[list[str]]]:
    with points_csv.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        point_id_index = header.index("point_id")
        rows: list[list[str]] = []
        seen: set[str] = set()
        selected: set[str] = set()
        for row in reader:
            point_id = row[point_id_index]
            if point_id not in selected and len(seen) < max_points:
                seen.add(point_id)
                selected.add(point_id)
            if point_id in selected:
                rows.append(row)
        return header, rows


def build_smoke_input(smoke_root: Path, max_points: int) -> dict[str, Any]:
    raw_root = ROOT.parent / "project" / "downloadedRawData"
    points_csv = raw_root / "points_train_label.csv"
    if not points_csv.exists():
        raise FileNotFoundError(f"Missing training points file for smoke input: {points_csv}")

    input_root = smoke_root / "input"
    region_test = input_root / "region_test"
    region_test.mkdir(parents=True, exist_ok=True)
    output_csv = input_root / "test_point.csv"

    header, rows = first_point_rows(points_csv, max_points=max_points)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    symlink_count = 0
    skipped_count = 0
    for region_dir in sorted(raw_root.glob("region_train_*")):
        if not region_dir.is_dir():
            continue
        for src in sorted(region_dir.glob("*.tiff")):
            dst = region_test / src.name
            if dst.exists() or dst.is_symlink():
                skipped_count += 1
                continue
            try:
                dst.symlink_to(src)
                symlink_count += 1
            except FileExistsError:
                skipped_count += 1

    return {
        "input_root": str(input_root),
        "points_csv": str(output_csv),
        "query_rows": len(rows),
        "unique_points": len({row[header.index("point_id")] for row in rows}),
        "new_symlinks": symlink_count,
        "existing_symlinks_or_files": skipped_count,
        "region_test_entries": sum(1 for _ in region_test.iterdir()),
    }


def run_c03_smoke(smoke_root: Path, batch_size: int) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.submission_inference import run_inference

    config = json.loads((ROOT / "configs" / "submission_c03.json").read_text())
    config["input_root"] = str(smoke_root / "input")
    config["output_json"] = str(smoke_root / "output" / "result.json")
    config["work_dir"] = str(smoke_root / "work")
    config["device"] = "cpu"
    config["batch_size"] = batch_size
    report = run_inference(config)

    out = ROOT / "artifacts" / "benchmarks" / "c03_smoke_inference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C03/TSViT local pipeline diagnostics.")
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--max-smoke-points", type=int, default=3)
    parser.add_argument("--smoke-batch-size", type=int, default=16)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "generated_at_unix": time.time(),
        "root": str(ROOT),
        "scope": ["c03", "tsvit"],
    }

    summary["sanity"] = run_command([sys.executable, "scripts/run_sanity_checks.py"])

    compile_env = os.environ.copy()
    compile_env["PYTHONPYCACHEPREFIX"] = "/tmp/project_organized_pyc"
    summary["compileall"] = run_command(
        [sys.executable, "-m", "compileall", "data", "models", "preprocessing", "scripts", "training", "inference.py"],
        env=compile_env,
    )

    if not args.skip_benchmark:
        summary["benchmark"] = run_command(
            [sys.executable, "scripts/benchmark_models.py", "--candidate", "all", "--iterations", "5", "--warmup", "2"]
        )

    summary["smoke_input"] = build_smoke_input(args.smoke_root, max_points=args.max_smoke_points)
    smoke_report = run_c03_smoke(args.smoke_root, batch_size=args.smoke_batch_size)
    summary["c03_smoke"] = {
        "queries": smoke_report.get("queries"),
        "unique_output_keys": smoke_report.get("unique_output_keys"),
        "duplicate_output_key_rows": smoke_report.get("duplicate_output_key_rows"),
        "duplicate_output_key_conflicts": smoke_report.get("duplicate_output_key_conflicts"),
        "device": smoke_report.get("device"),
        "timing_seconds": smoke_report.get("timing_seconds"),
        "result_stats": smoke_report.get("result_stats"),
        "patch_report": {
            "samples_kept": smoke_report.get("patch_report", {}).get("samples_kept"),
            "samples_dropped": smoke_report.get("patch_report", {}).get("samples_dropped"),
            "missing_band_cells": smoke_report.get("patch_report", {}).get("missing_band_cells"),
            "global_valid_pixel_ratio": smoke_report.get("patch_report", {}).get("global_valid_pixel_ratio"),
            "file_read_failures": smoke_report.get("patch_report", {}).get("file_read_failures"),
            "patch_extraction_errors": smoke_report.get("patch_report", {}).get("patch_extraction_errors"),
            "mapping_status_counts": smoke_report.get("patch_report", {}).get("mapping_status_counts"),
        },
    }

    failed = [
        name
        for name in ["sanity", "compileall", "benchmark"]
        if name in summary and summary[name].get("returncode") != 0
    ]
    summary["status"] = "ok" if not failed and summary["c03_smoke"]["duplicate_output_key_conflicts"] == 0 else "fail"
    summary["failed_steps"] = failed

    out = ROOT / "artifacts" / "benchmarks" / "pipeline_diagnostics_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
