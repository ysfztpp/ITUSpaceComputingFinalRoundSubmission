from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CONFIGS = {
    "evaluate_c03_baseline.json",
    "evaluate_c20_parent.json",
    "preprocess.json",
    "submission_c03.json",
    "submission_tsvit.json",
    "train_full_data_c03_epoch75.json",
    "train_full_data_c29_tsvit_relative_aux_crop_submission.json",
}
EXPECTED_MODEL_FILES = {
    "__init__.py",
    "cnn_encoder.py",
    "model_factory.py",
    "query_cnn_transformer.py",
    "query_tsvit.py",
    "temporal_transformer.py",
}
EXPECTED_CHECKPOINTS = {
    "c03_full_data_model.pt",
    "c29_tsvit_model.pt",
}
PDF_REQUIREMENTS = {
    "Round2 Project Submission Manual_en_0407.pdf": [
        "/output/result.json",
        "inference-only",
        "test_point.csv",
        "region_test",
    ],
    "Space Intelligence Empowering Zero Hunger Track_Task Description of Final Round_en.pdf": [
        "Crop Classification Macro",
        "Rice Phenology",
        "0.4",
        "0.6",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add_result(results: list[dict[str, Any]], name: str, status: str, detail: Any) -> None:
    results.append({"check": name, "status": status, "detail": detail})


def macro_f1(y_true: list[int], y_pred: list[int], labels: range) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / max(len(scores), 1)


def competition_score(
    crop_true: list[int],
    crop_pred: list[int],
    stage_true: list[int],
    stage_pred: list[int],
    *,
    rice_id: int = 1,
) -> dict[str, float]:
    crop_macro = macro_f1(crop_true, crop_pred, range(3))
    rice_stage_true: list[int] = []
    rice_stage_pred: list[int] = []
    for truth_crop, pred_crop, truth_stage, pred_stage in zip(crop_true, crop_pred, stage_true, stage_pred):
        if truth_crop == rice_id or pred_crop == rice_id:
            rice_stage_true.append(int(truth_stage) if truth_crop == rice_id else -1)
            rice_stage_pred.append(int(pred_stage) if pred_crop == rice_id else -1)
    rice_stage_macro = macro_f1(rice_stage_true, rice_stage_pred, range(7))
    return {
        "crop_macro_f1": crop_macro,
        "rice_stage_macro_f1": rice_stage_macro,
        "competition_score": 0.4 * crop_macro + 0.6 * rice_stage_macro,
    }


def check_scoring_semantics(results: list[dict[str, Any]]) -> None:
    crop_true = [1, 1, 1, 1, 1, 1, 1, 0, 2]
    crop_pred = [1, 1, 1, 1, 1, 1, 1, 0, 2]
    stage_true = [0, 1, 2, 3, 4, 5, 6, 0, 0]
    stage_pred = [0, 1, 2, 3, 4, 5, 6, 0, 0]
    perfect = competition_score(
        crop_true=crop_true,
        crop_pred=crop_pred,
        stage_true=stage_true,
        stage_pred=stage_pred,
    )
    crop_pred_with_rice_miss = list(crop_pred)
    crop_pred_with_rice_miss[2] = 0
    double_hit_error = competition_score(
        crop_true=crop_true,
        crop_pred=crop_pred_with_rice_miss,
        stage_true=stage_true,
        stage_pred=stage_pred,
    )
    ok = abs(perfect["competition_score"] - 1.0) < 1e-12 and double_hit_error["competition_score"] < 1.0
    add_result(
        results,
        "scoring_formula",
        "ok" if ok else "fail",
        {
            "formula": "0.4 * crop_macro_f1 + 0.6 * rice_only_stage_macro_f1",
            "perfect_case": perfect,
            "rice_double_hit_error_case": double_hit_error,
        },
    )


def check_pdf_requirements(results: list[dict[str, Any]]) -> None:
    try:
        from pypdf import PdfReader
    except ImportError:
        add_result(results, "pdf_requirements", "warn", "pypdf is unavailable; checked PDF file presence only")
        return

    details: dict[str, Any] = {}
    missing_any = False
    for name, keywords in PDF_REQUIREMENTS.items():
        path = ROOT / name
        if not path.exists():
            details[name] = {"exists": False, "missing_keywords": keywords}
            missing_any = True
            continue
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        missing = [keyword for keyword in keywords if keyword not in text]
        details[name] = {"exists": True, "pages": len(reader.pages), "missing_keywords": missing}
        missing_any = missing_any or bool(missing)
    add_result(results, "pdf_requirements", "fail" if missing_any else "ok", details)


def check_scope(results: list[dict[str, Any]]) -> None:
    configs = {path.name for path in (ROOT / "configs").glob("*.json")}
    model_files = {path.name for path in (ROOT / "models").glob("*.py")}
    checkpoints = {path.name for path in (ROOT / "checkpoints").glob("*.pt")}
    detail = {
        "configs_extra": sorted(configs - EXPECTED_CONFIGS),
        "configs_missing": sorted(EXPECTED_CONFIGS - configs),
        "model_files_extra": sorted(model_files - EXPECTED_MODEL_FILES),
        "model_files_missing": sorted(EXPECTED_MODEL_FILES - model_files),
        "checkpoints_extra": sorted(checkpoints - EXPECTED_CHECKPOINTS),
        "checkpoints_missing": sorted(EXPECTED_CHECKPOINTS - checkpoints),
        "note": "temporal_transformer.py is retained because both C03 and TSViT import its time-encoding helpers.",
    }
    ok = not any(value for key, value in detail.items() if key.endswith(("extra", "missing")))
    add_result(results, "candidate_scope", "ok" if ok else "fail", detail)


def check_config_consistency(results: list[dict[str, Any]]) -> None:
    c03_train = load_json(ROOT / "configs" / "train_full_data_c03_epoch75.json")
    tsvit_train = load_json(ROOT / "configs" / "train_full_data_c29_tsvit_relative_aux_crop_submission.json")
    c03_submission = load_json(ROOT / "configs" / "submission_c03.json")
    tsvit_submission = load_json(ROOT / "configs" / "submission_tsvit.json")
    c03_card = load_json(ROOT / "model_cards" / "c03_full_data" / "metrics_summary.json")
    tsvit_card = load_json(ROOT / "model_cards" / "c29_tsvit" / "metrics_summary.json")

    checks = {
        "c03_epoch_matches_card": int(c03_train["epochs"]) == int(c03_card["epoch"]) == 75,
        "tsvit_epoch_matches_card": int(tsvit_train["epochs"]) == int(tsvit_card["epoch"]) == 42,
        "c03_submission_checkpoint_exists": (ROOT / c03_submission["checkpoint"]).exists(),
        "tsvit_submission_checkpoint_exists": (ROOT / tsvit_submission["checkpoint"]).exists(),
        "c03_submission_output_contract": c03_submission["output_json"] == "/output/result.json",
        "tsvit_submission_output_contract": tsvit_submission["output_json"] == "/output/result.json",
        "tsvit_model_type": tsvit_train.get("model_type") == "query_tsvit",
    }
    add_result(results, "config_consistency", "ok" if all(checks.values()) else "fail", checks)


def check_metrics_inventory(results: list[dict[str, Any]]) -> None:
    c03_card = load_json(ROOT / "model_cards" / "c03_full_data" / "metrics_summary.json")
    tsvit_card = load_json(ROOT / "model_cards" / "c29_tsvit" / "metrics_summary.json")
    c03_validation_path = ROOT.parent / "models" / "C03" / "checkpoint" / "metrics_summary.json"
    c03_validation = load_json(c03_validation_path) if c03_validation_path.exists() else None
    detail = {
        "c03_full_data": {
            "epoch": c03_card.get("epoch"),
            "train_queries": c03_card.get("train_queries"),
            "final_train_competition_score": c03_card.get("final_train_competition_score"),
            "final_train_loss": c03_card.get("final_train_loss"),
        },
        "c03_validation_source": str(c03_validation_path) if c03_validation else None,
        "c03_validation": None
        if c03_validation is None
        else {
            "best_epoch": c03_validation.get("best_epoch"),
            "best_val_competition_score": c03_validation.get("best_val_competition_score"),
            "best_val_crop_macro_f1": c03_validation.get("best_val_crop_macro_f1"),
            "best_val_rice_stage_macro_f1": c03_validation.get("best_val_rice_stage_macro_f1"),
            "best_val_loss": c03_validation.get("best_val_loss"),
        },
        "tsvit_full_data": {
            "epoch": tsvit_card.get("epoch"),
            "train_queries": tsvit_card.get("train_queries"),
            "final_train_competition_score": tsvit_card.get("final_train_competition_score"),
            "final_train_loss": tsvit_card.get("final_train_loss"),
        },
        "tsvit_validation": "not present in the local archive; current TSViT evidence is full-data train metrics plus checkpoint metadata",
    }
    add_result(results, "metrics_inventory", "ok" if c03_validation is not None else "warn", detail)


def check_checkpoint_validation(results: list[dict[str, Any]], skip: bool) -> None:
    if skip:
        add_result(results, "checkpoint_validation", "skip", "not requested")
        return
    details: dict[str, Any] = {}
    failed = False
    torch_unavailable = False
    for candidate, config in {"c03": "submission_c03.json", "tsvit": "submission_tsvit.json"}.items():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "validate_submission.py"),
            "--config",
            str(ROOT / "configs" / config),
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        candidate_detail: dict[str, Any] = {"returncode": completed.returncode}
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError:
                candidate_detail["stdout"] = completed.stdout[-1000:]
            else:
                model_config = payload.get("model_config", {})
                candidate_detail.update(
                    {
                        "checkpoint_mb": payload.get("checkpoint_mb"),
                        "checkpoint_epoch": payload.get("checkpoint_epoch"),
                        "checkpoint_metric": payload.get("checkpoint_metric"),
                        "model_type": payload.get("model_type"),
                        "in_channels": model_config.get("in_channels"),
                        "patch_size": model_config.get("patch_size"),
                        "num_crop_classes": model_config.get("num_crop_classes"),
                        "num_phenophase_classes": model_config.get("num_phenophase_classes"),
                        "aux_feature_dim": model_config.get("aux_feature_dim"),
                        "output_json": payload.get("output_json"),
                    }
                )
        else:
            candidate_detail["stdout"] = completed.stdout[-1000:]
            candidate_detail["stderr"] = completed.stderr[-2000:]
        details[candidate] = candidate_detail
        if completed.returncode != 0:
            if "No module named 'torch'" in completed.stderr or "torch is required" in completed.stderr:
                torch_unavailable = True
            else:
                failed = True
    if failed:
        status = "fail"
    elif torch_unavailable:
        status = "warn"
        details["note"] = "Local Python cannot import torch. Run this check inside the competition PyTorch image for checkpoint load validation."
    else:
        status = "ok"
    add_result(results, "checkpoint_validation", status, details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused C03/TSViT sanity checks for the final-round bundle.")
    parser.add_argument("--skip-checkpoint-validation", action="store_true", help="Do not load checkpoints with PyTorch.")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    check_pdf_requirements(results)
    check_scoring_semantics(results)
    check_scope(results)
    check_config_consistency(results)
    check_metrics_inventory(results)
    check_checkpoint_validation(results, skip=args.skip_checkpoint_validation)

    status = "ok"
    if any(row["status"] == "fail" for row in results):
        status = "fail"
    elif any(row["status"] == "warn" for row in results):
        status = "warn"
    print(json.dumps({"status": status, "results": results}, indent=2))
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
